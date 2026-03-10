import asyncio
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

load_dotenv()  # put .env vars into os.environ so PydanticAI can read them

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
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

# Maps template_id → {header_row_id, footer_row_id} for the MCP proxy
template_layouts: dict[str, dict] = {}

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

            # Phase 2 — seed templates + register layouts for the proxy
            template_ids: list[str] = await asyncio.gather(
                *[
                    create_seeded_template(settings, layout_json)
                    for _ in email_plan.emails
                ]
            )

            # Register footer_row_id per template so the proxy can inject it
            for tid in template_ids:
                template_layouts[tid] = {
                    "header_row_id": header_row_id,
                    "footer_row_id": footer_row_id,
                }

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


@app.post("/mcp-proxy")
async def mcp_proxy(request: Request):
    """Transparent MCP proxy that enforces footer position.

    Intercepts every beefree_add_section call and injects
    before_row_id = footer_row_id when the parameter is absent,
    ensuring the footer row always stays last.
    All other calls are forwarded untouched.
    """
    body = await request.json()
    template_id = request.headers.get("x-bee-template-id", "")
    footer_row_id = template_layouts.get(template_id, {}).get("footer_row_id")

    # Intercept beefree_add_section — handle single and batch JSON-RPC
    def _maybe_inject(msg: dict) -> None:
        if (
            footer_row_id
            and msg.get("method") == "tools/call"
            and msg.get("params", {}).get("name") == "beefree_add_section"
        ):
            args = msg["params"].setdefault("arguments", {})
            if "before_row_id" not in args:
                args["before_row_id"] = footer_row_id

    if isinstance(body, list):
        for item in body:
            _maybe_inject(item)
    else:
        _maybe_inject(body)

    # Forward to Beefree — pass all original headers except hop-by-hop ones
    skip_req = {"host", "content-length", "transfer-encoding", "content-type"}
    fwd_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in skip_req
    }

    settings = get_settings()
    client = httpx.AsyncClient(timeout=300.0)
    upstream = client.build_request(
        "POST",
        f"{settings.bee_api_base}/v2/sdk/mcp",
        json=body,
        headers=fwd_headers,
    )
    resp = await client.send(upstream, stream=True)

    skip_resp = {
        "transfer-encoding", "content-encoding",
        "content-length", "content-type",
    }
    resp_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in skip_resp
    }

    async def _stream():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        _stream(),
        status_code=resp.status_code,
        headers=resp_headers,
        media_type=resp.headers.get("content-type", "application/json"),
    )


@app.get("/stream/{template_id}")
async def stream(request: Request, template_id: str, prompt: str):
    settings = get_settings()
    proxy_url = str(request.base_url).rstrip("/") + "/mcp-proxy"

    async def generator():
        async for event in stream_executor(
            template_id, prompt, settings, mcp_url=proxy_url
        ):
            yield event

    return EventSourceResponse(generator())
