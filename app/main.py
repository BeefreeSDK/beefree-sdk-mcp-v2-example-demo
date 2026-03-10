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
async def execute(
    request: Request,
    plan_json: str = Form(...),
):
    settings = get_settings()
    email_plan = EmailPlan.model_validate_json(plan_json)

    # Step 1: Build shared header + footer once (sequential — must finish first)
    header_row_id, footer_row_id, layout_json = await build_shared_layout(
        email_plan.sequence_title, settings
    )

    # Step 2: Create N template sessions pre-seeded with the shared layout
    template_ids: list[str] = await asyncio.gather(
        *[create_seeded_template(settings, layout_json) for _ in email_plan.emails]
    )

    # Step 3: Enrich each executor prompt with the protected row IDs
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

    return templates.TemplateResponse(
        "partials/sequence.html",
        {
            "request": request,
            "plan": email_plan,
            "emails_with_ids": emails_with_ids,
        },
    )


@app.get("/stream/{template_id}")
async def stream(template_id: str, prompt: str):
    settings = get_settings()

    async def generator():
        async for event in stream_executor(template_id, prompt, settings):
            yield event

    return EventSourceResponse(generator())
