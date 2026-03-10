import asyncio
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv()  # put .env vars into os.environ so PydanticAI providers can read them

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette import EventSourceResponse

from .agent import EmailPlan, generate_plan, stream_executor
from .beefree import create_template
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

    # Create all template sessions in parallel — one per email
    template_ids: list[str] = await asyncio.gather(
        *[create_template(settings) for _ in email_plan.emails]
    )

    emails_with_ids = list(zip(email_plan.emails, template_ids))

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
