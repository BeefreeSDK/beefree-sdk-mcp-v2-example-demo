import asyncio
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv()  # put .env vars into os.environ so PydanticAI providers can read them

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette import EventSourceResponse

from .agent import (
    EmailPlan,
    EmailStep,
    append_layout_context,
    build_shared_layout,
    generate_plan,
    stream_executor,
)
from .beefree import create_seeded_template
from .config import get_settings

app = FastAPI(title="Beefree Headless MCP v2 Demo")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Expose urlencode as a Jinja2 filter for prompt embedding in SSE URLs
templates.env.filters["urlencode"] = quote

# ─── Routes ──────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


@app.post("/plan", response_class=HTMLResponse)
async def plan(
    request: Request,
    goal: str = Form(...),
):
    """Return the loading partial immediately; SSE streams in the plan."""
    return templates.TemplateResponse(
        "partials/plan_loading.html",
        {"request": request, "goal": goal},
    )


@app.get("/plan-stream")
async def plan_stream(request: Request, goal: str):
    """SSE endpoint: runs the planner LLM and yields the rendered plan HTML."""
    settings = get_settings()

    async def generator():
        try:
            email_plan = await generate_plan(goal, settings)
            plan_json = email_plan.model_dump_json()
            tmpl = templates.env.get_template("partials/plan.html")
            plan_html = tmpl.render(plan=email_plan, plan_json=plan_json)
            yield {"event": "plan", "data": plan_html}
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("Planner error: %s", exc)
            yield {
                "event": "plan-error",
                "data": f"<p class='plan-error'>Planning failed: {exc}</p>",
            }
        finally:
            yield {"event": "close", "data": ""}

    return EventSourceResponse(generator())


@app.post("/execute", response_class=HTMLResponse)
async def execute(request: Request, plan_json: str = Form(...)):
    """Return the loading partial immediately; SSE does the real work."""
    return templates.TemplateResponse(
        "partials/execute_loading.html",
        {"request": request, "plan_json": plan_json},
    )


@app.get("/execute-stream")
async def execute_stream(request: Request, plan_json: str):
    """SSE: build shared layout, seed templates, stream the sequence view."""
    settings = get_settings()
    email_plan = EmailPlan.model_validate_json(plan_json)
    n = len(email_plan.emails)

    async def generator():
        try:
            # Phase 1 — shared layout (blocks until done)
            header_row_id, footer_row_id, layout_json = (
                await build_shared_layout(email_plan.sequence_title, settings)
            )

            # Signal layout complete so the UI flips to phase 2
            s = "s" if n != 1 else ""
            yield {
                "event": "layout-done",
                "data": (
                    '<div class="exec-phase is-done">'
                    '<span class="phase-check">&#10003;</span>'
                    '<div class="phase-content">'
                    '<p class="phase-title">Shared layout ready</p>'
                    "<p class='phase-sub'>"
                    "Header, footer and global styles applied"
                    "</p></div></div>"
                    '<div class="exec-phase is-active">'
                    '<span class="phase-spinner"></span>'
                    '<div class="phase-content">'
                    f"<p class='phase-title'>"
                    f"Launching {n} email agent{s} in parallel"
                    "</p>"
                    "<p class='phase-sub'>"
                    "Previews stream in as each agent builds"
                    "</p></div></div>"
                ),
            }

            # Phase 2 — seed templates + render sequence view
            template_ids: list[str] = await asyncio.gather(
                *[
                    create_seeded_template(settings, layout_json)
                    for _ in email_plan.emails
                ]
            )

            enriched_emails = [
                EmailStep(
                    step=e.step,
                    title=e.title,
                    subject_line=e.subject_line,
                    agent_prompt=append_layout_context(
                        e.agent_prompt, header_row_id, footer_row_id
                    ),
                )
                for e in email_plan.emails
            ]

            emails_with_ids = list(zip(enriched_emails, template_ids))
            tmpl = templates.env.get_template("partials/sequence.html")
            sequence_html = tmpl.render(
                plan=email_plan,
                emails_with_ids=emails_with_ids,
            )
            yield {"event": "sequence", "data": sequence_html}

        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("Execute stream error: %s", exc)
            yield {
                "event": "sequence",
                "data": f"<p class='plan-error'>Execution failed: {exc}</p>",
            }
        finally:
            yield {"event": "close", "data": ""}

    return EventSourceResponse(generator())


@app.get("/stream/{template_id}")
async def stream(template_id: str, prompt: str):
    settings = get_settings()

    async def generator():
        async for event in stream_executor(template_id, prompt, settings):
            yield event

    return EventSourceResponse(generator())
