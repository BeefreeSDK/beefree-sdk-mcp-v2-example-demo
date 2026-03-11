import asyncio
import html as html_lib
import json
import uuid
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
    stream_single_executor,
    stream_translation_executor,
    stream_palette_executor,
)
from .beefree import create_seeded_template, create_template
from .config import get_settings

app = FastAPI(title="Beefree Headless MCP v2 Demo")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Expose urlencode as a Jinja2 filter for prompt embedding in SSE URLs
templates.env.filters["urlencode"] = quote

# Maps template_id → {header_row_id, footer_row_id} for the MCP proxy
template_layouts: dict[str, dict] = {}

# Temporary store for bulk-translation sessions (session_id → {template_json, languages})
translation_sessions: dict[str, dict] = {}

# Temporary store for palette sessions (session_id → {template_json, palettes})
palette_sessions: dict[str, dict] = {}

AVAILABLE_LANGUAGES = [
    "Spanish", "French", "German", "Italian", "Portuguese (Brazil)",
    "Portuguese (Portugal)", "Dutch", "Russian", "Japanese",
    "Chinese (Simplified)", "Chinese (Traditional)", "Korean",
    "Arabic", "Polish", "Turkish", "Swedish", "Danish", "Norwegian",
    "Finnish", "Czech", "Hungarian", "Romanian", "Greek",
    "Hebrew", "Thai", "Vietnamese", "Indonesian", "Ukrainian", "Hindi",
]

SEQUENCE_PRESETS: dict[str, list[list[str]]] = {
    # sections_per_step[email_index] = ordered list of body section descriptions
    "streaming-onboarding": [
        [  # Email 1 — Welcome
            "Full-width hero: branded 'Welcome!' headline, atmospheric background image placeholder, large 'Start Watching Now' CTA button",
            "Content category grid: 4 genre tiles with cover image placeholders and genre labels (Action, Drama, Comedy, Documentaries)",
            "Featured picks row: 'Hand-picked for you' intro copy + 2–3 show/movie cards with thumbnail placeholder, title, and one-line description",
            "App download row: 'Watch anywhere' copy + App Store and Google Play button placeholders side by side",
        ],
        [  # Email 2 — Password Reset
            "Security alert row: padlock icon + 'Password Reset Request' headline + one line of reassuring intro copy",
            "Reset action row: clear instruction sentence + large high-contrast 'Reset My Password' CTA button + 'This link expires in 24 hours' notice in small muted text",
            "Safety notice row: short paragraph — 'Didn't request this? You can safely ignore this email.' + support contact link",
        ],
        [  # Email 3 — Subscription Confirmation
            "Plan summary row: plan name badge, billing cycle label, and monthly price displayed prominently with icon accents",
            "Benefits grid: 3-column icon list — number of screens, downloads, video quality (HD/4K), and ad-free viewing",
            "Next steps CTA row: 'You're all set!' headline + 'Start exploring now' copy + 'Browse the Catalogue' button + 'Manage your subscription' text link",
        ],
    ],
    "saas-trial-nurture": [
        [  # Email 1 — Trial Started
            "Welcome hero: 'Your trial has started' headline, product dashboard screenshot placeholder, 'Open Your Dashboard' CTA button",
            "Top 3 features grid: 3-column layout — each column has icon, feature name, and one-line benefit (e.g. Task Boards, Time Tracking, Team Collaboration)",
            "Quick-start CTA row: '3 steps to get started' intro + step-count badges with short action labels + 'Start Your First Project' button",
        ],
        [  # Email 2 — Day 3 Check-in
            "Friendly check-in hero: conversational 'How's it going so far?' headline + short personal-tone intro paragraph",
            "3 actionable tips: numbered list with icon accents — each tip is one concrete task the user can complete today in the tool",
            "Tutorial resources row: 2 video tutorial cards with thumbnail placeholder, title, and viewing time",
            "Support CTA row: 'Need a hand?' copy + help docs link + 'Back to Dashboard' button",
        ],
        [  # Email 3 — Day 7 Feature Spotlight
            "Feature hero: bold spotlight headline naming the premium feature, feature screenshot/illustration placeholder, 1-sentence value statement",
            "Benefits breakdown: 2–3 benefit bullets with icons — concrete outcomes the feature enables for the user's team",
            "Upgrade CTA block: contrasting background, 'Unlock this feature today' headline, pricing reminder (monthly/annual), 'Upgrade Now' button",
        ],
        [  # Email 4 — Trial Ending
            "Urgency banner: 'Your trial ends soon' headline with clock icon, warm-but-urgent tone intro paragraph",
            "Value recap: 3 metric/achievement cards showing key outcomes the paid plan unlocks, each with icon accent",
            "Exclusive offer row: highlighted offer box with discount percentage, promo code in large text, and expiry date",
            "Final CTA: 'Don't lose access' copy + large 'Upgrade Now' button + 'Compare plans' text link",
        ],
    ],
    "black-friday-fashion": [
        [  # Email 1 — Teaser
            "Mystery hero: dark atmospheric full-width banner with 'Something extraordinary is coming' headline — no product reveal, pure anticipation",
            "Countdown visual row: '48 HOURS TO GO' copy with large stylised numeral or graphic countdown placeholder",
            "Early access CTA: 'Be the first to shop' copy with exclusive subscriber angle + 'Secure Your Early Access' button",
            "Brand teaser row: two-column layout — fashion editorial image placeholder + 1–2 sentences on craft or exclusivity",
        ],
        [  # Email 2 — Launch Day
            "Sale hero: bold 'BLACK FRIDAY IS HERE' headline over striking product image placeholder, discount percentage badge overlaid",
            "Discount code spotlight: high-contrast box with 'USE CODE:' label and promo code in large monospace-style font + expiry notice + 'Shop Now' CTA",
            "Curated product grid: 3–4 product cards, each with image placeholder, product name, original strikethrough price, and sale price",
            "Closing CTA row: 'Don't miss out — sale ends midnight' urgency copy + 'Browse All Deals' button",
        ],
        [  # Email 3 — Last Chance
            "Urgency hero: high-contrast '24 HOURS LEFT' banner, bold all-caps typography, dramatic fashion image placeholder",
            "Best-sellers grid: 3 product cards with image placeholder, product name, and 'Only X left in stock' scarcity indicator",
            "Final CTA block: 'This is your last chance to save' copy + promo code reminder + large 'Shop Before It's Gone' button",
        ],
    ],
}

PALETTES = [
    {
        "id": "ocean-blue",
        "name": "Ocean Blue",
        "swatches": ["#1E3A5F", "#1E40AF", "#0891B2", "#DBEAFE", "#FFFFFF"],
        "colors": {
            "page_bg": "#EBF4FF", "content_bg": "#FFFFFF",
            "header_bg": "#1E3A5F", "heading": "#1E3A5F", "text": "#374151",
            "primary": "#1E40AF", "accent": "#0891B2",
            "footer_bg": "#DBEAFE", "footer_text": "#475569",
        },
    },
    {
        "id": "forest",
        "name": "Forest",
        "swatches": ["#1A3D2B", "#2D6A4F", "#40916C", "#D8F3DC", "#FFFFFF"],
        "colors": {
            "page_bg": "#F0FFF4", "content_bg": "#FFFFFF",
            "header_bg": "#1A3D2B", "heading": "#1A3D2B", "text": "#374151",
            "primary": "#2D6A4F", "accent": "#40916C",
            "footer_bg": "#D8F3DC", "footer_text": "#3D6B52",
        },
    },
    {
        "id": "sunset",
        "name": "Sunset",
        "swatches": ["#7C2D12", "#EA580C", "#F97316", "#FEF3C7", "#FFFFFF"],
        "colors": {
            "page_bg": "#FFF7ED", "content_bg": "#FFFFFF",
            "header_bg": "#7C2D12", "heading": "#7C2D12", "text": "#44403C",
            "primary": "#EA580C", "accent": "#F97316",
            "footer_bg": "#FEF3C7", "footer_text": "#78716C",
        },
    },
    {
        "id": "rose-gold",
        "name": "Rose Gold",
        "swatches": ["#881337", "#BE185D", "#EC4899", "#FCE7F3", "#FFFFFF"],
        "colors": {
            "page_bg": "#FFF1F2", "content_bg": "#FFFFFF",
            "header_bg": "#881337", "heading": "#881337", "text": "#44403C",
            "primary": "#BE185D", "accent": "#EC4899",
            "footer_bg": "#FCE7F3", "footer_text": "#9F1239",
        },
    },
    {
        "id": "midnight",
        "name": "Midnight",
        "swatches": ["#0F172A", "#1E293B", "#475569", "#334155", "#94A3B8"],
        "colors": {
            "page_bg": "#0F172A", "content_bg": "#1E293B",
            "header_bg": "#020617", "heading": "#F1F5F9", "text": "#CBD5E1",
            "primary": "#6366F1", "accent": "#818CF8",
            "footer_bg": "#0F172A", "footer_text": "#64748B",
        },
    },
    {
        "id": "lavender",
        "name": "Lavender",
        "swatches": ["#3B0764", "#6D28D9", "#8B5CF6", "#EDE9FE", "#FFFFFF"],
        "colors": {
            "page_bg": "#F5F3FF", "content_bg": "#FFFFFF",
            "header_bg": "#3B0764", "heading": "#3B0764", "text": "#374151",
            "primary": "#6D28D9", "accent": "#8B5CF6",
            "footer_bg": "#EDE9FE", "footer_text": "#6D28D9",
        },
    },
    {
        "id": "earthy",
        "name": "Earthy",
        "swatches": ["#431407", "#92400E", "#D97706", "#FEF3C7", "#FFFBF7"],
        "colors": {
            "page_bg": "#FFFBF7", "content_bg": "#FFFFFF",
            "header_bg": "#431407", "heading": "#431407", "text": "#57534E",
            "primary": "#92400E", "accent": "#D97706",
            "footer_bg": "#FEF3C7", "footer_text": "#78716C",
        },
    },
    {
        "id": "arctic",
        "name": "Arctic",
        "swatches": ["#164E63", "#0E7490", "#22D3EE", "#CFFAFE", "#FFFFFF"],
        "colors": {
            "page_bg": "#ECFEFF", "content_bg": "#FFFFFF",
            "header_bg": "#164E63", "heading": "#164E63", "text": "#374151",
            "primary": "#0E7490", "accent": "#22D3EE",
            "footer_bg": "#CFFAFE", "footer_text": "#0E7490",
        },
    },
    {
        "id": "coral",
        "name": "Coral",
        "swatches": ["#9F1239", "#E11D48", "#FB7185", "#FFE4E6", "#FFFFFF"],
        "colors": {
            "page_bg": "#FFF1F2", "content_bg": "#FFFFFF",
            "header_bg": "#9F1239", "heading": "#9F1239", "text": "#44403C",
            "primary": "#E11D48", "accent": "#FB7185",
            "footer_bg": "#FFE4E6", "footer_text": "#9F1239",
        },
    },
    {
        "id": "minimal",
        "name": "Minimal",
        "swatches": ["#111827", "#374151", "#6B7280", "#F3F4F6", "#FFFFFF"],
        "colors": {
            "page_bg": "#F9FAFB", "content_bg": "#FFFFFF",
            "header_bg": "#111827", "heading": "#111827", "text": "#374151",
            "primary": "#374151", "accent": "#6B7280",
            "footer_bg": "#F3F4F6", "footer_text": "#6B7280",
        },
    },
]

# ─── Routes ──────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse(
        "landing.html",
        {"request": request},
    )


@app.get("/sequence", response_class=HTMLResponse)
async def sequence(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


@app.get("/single", response_class=HTMLResponse)
async def single(request: Request):
    return templates.TemplateResponse(
        "single.html",
        {"request": request},
    )


@app.post("/generate", response_class=HTMLResponse)
async def generate(request: Request, brief: str = Form(...)):
    """Return the loading partial immediately; SSE drives the generation."""
    return templates.TemplateResponse(
        "partials/single_loading.html",
        {"request": request, "brief": brief},
    )


@app.get("/generate-stream")
async def generate_stream(request: Request, brief: str):
    """SSE endpoint: creates a template, runs single email agent, streams previews."""
    settings = get_settings()

    async def generator():
        try:
            template_id = await create_template(settings)
            async for event in stream_single_executor(template_id, brief, settings):
                yield event
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("Single gen error: %s", exc)
            yield {
                "event": "preview",
                "data": f"<p class='plan-error'>Generation failed: {exc}</p>",
            }
        finally:
            yield {"event": "close", "data": ""}

    return EventSourceResponse(generator())


@app.post("/preview-template", response_class=HTMLResponse)
async def preview_template(request: Request, template_json: str = Form(...)):
    """Render a pasted/uploaded Beefree template JSON and return a scaled iframe."""
    from .beefree import render_html
    settings = get_settings()
    try:
        data = json.loads(template_json)
        template = data.get("template", data)
        rendered = await render_html(template, settings)
        escaped = html_lib.escape(rendered, quote=True)
        return HTMLResponse(
            '<div class="tpl-preview-wrap">'
            f'<iframe srcdoc="{escaped}" sandbox="allow-same-origin" '
            'style="width:600px;height:1200px;border:none;display:block;" '
            'title="Template preview"></iframe>'
            '</div>'
        )
    except Exception:
        return HTMLResponse(
            '<div class="tpl-preview-placeholder">'
            '<p>Could not render preview — check your JSON.</p>'
            '</div>'
        )


@app.get("/translate", response_class=HTMLResponse)
async def translate_page(request: Request):
    return templates.TemplateResponse(
        "translate.html",
        {"request": request, "languages": AVAILABLE_LANGUAGES},
    )


@app.post("/translate-submit", response_class=HTMLResponse)
async def translate_submit(
    request: Request,
    template_json: str = Form(...),
    languages: str = Form(...),
):
    """Parse template JSON, store session, return the loading partial."""
    try:
        parsed = json.loads(template_json)
    except json.JSONDecodeError as exc:
        return HTMLResponse(
            f"<p class='plan-error'>Invalid JSON: {exc}</p>", status_code=400
        )
    lang_list = [la.strip() for la in languages.split(",") if la.strip()]
    session_id = uuid.uuid4().hex[:12]
    translation_sessions[session_id] = {"template_json": parsed, "languages": lang_list}
    return templates.TemplateResponse(
        "partials/translate_loading.html",
        {"request": request, "session_id": session_id, "languages": lang_list},
    )


@app.get("/translate-stream")
async def translate_stream_route(request: Request, session_id: str):
    """SSE: seed N template copies, render the column grid, agents stream previews."""
    settings = get_settings()
    data = translation_sessions.pop(session_id, None)

    async def generator():
        if not data:
            yield {"event": "close", "data": ""}
            return
        try:
            template_json = data["template_json"]
            languages = data["languages"]

            template_ids: list[str] = await asyncio.gather(
                *[create_seeded_template(settings, template_json) for _ in languages]
            )

            tmpl = templates.env.get_template("partials/translate_sequence.html")
            result_html = tmpl.render(
                languages_with_ids=list(zip(languages, template_ids))
            )
            yield {"event": "ready", "data": result_html}
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("Translate stream error: %s", exc)
            yield {
                "event": "ready",
                "data": f"<p class='plan-error'>Translation failed: {exc}</p>",
            }
        finally:
            yield {"event": "close", "data": ""}

    return EventSourceResponse(generator())


@app.get("/stream-translate/{template_id}")
async def stream_translate(request: Request, template_id: str, language: str):
    settings = get_settings()

    async def generator():
        async for event in stream_translation_executor(template_id, language, settings):
            yield event

    return EventSourceResponse(generator())


@app.get("/palette", response_class=HTMLResponse)
async def palette_page(request: Request):
    return templates.TemplateResponse(
        "palette.html",
        {"request": request, "palettes": PALETTES},
    )


@app.post("/palette-submit", response_class=HTMLResponse)
async def palette_submit(
    request: Request,
    template_json: str = Form(...),
    palette_ids: str = Form(...),
):
    """Parse template JSON, resolve selected palettes, store session, return loading partial."""
    try:
        parsed = json.loads(template_json)
    except json.JSONDecodeError as exc:
        return HTMLResponse(
            f"<p class='plan-error'>Invalid JSON: {exc}</p>", status_code=400
        )
    id_set = {pid.strip() for pid in palette_ids.split(",") if pid.strip()}
    selected = [p for p in PALETTES if p["id"] in id_set]
    session_id = uuid.uuid4().hex[:12]
    palette_sessions[session_id] = {"template_json": parsed, "palettes": selected}
    return templates.TemplateResponse(
        "partials/palette_loading.html",
        {"request": request, "session_id": session_id, "palettes": selected},
    )


@app.get("/palette-stream")
async def palette_stream_route(request: Request, session_id: str):
    """SSE: seed N template copies, render palette column grid."""
    settings = get_settings()
    data = palette_sessions.pop(session_id, None)

    async def generator():
        if not data:
            yield {"event": "close", "data": ""}
            return
        try:
            template_json = data["template_json"]
            palettes = data["palettes"]

            template_ids: list[str] = await asyncio.gather(
                *[create_seeded_template(settings, template_json) for _ in palettes]
            )

            tmpl = templates.env.get_template("partials/palette_sequence.html")
            result_html = tmpl.render(
                palettes_with_ids=list(zip(palettes, template_ids))
            )
            yield {"event": "ready", "data": result_html}
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("Palette stream error: %s", exc)
            yield {
                "event": "ready",
                "data": f"<p class='plan-error'>Palette application failed: {exc}</p>",
            }
        finally:
            yield {"event": "close", "data": ""}

    return EventSourceResponse(generator())


@app.get("/stream-palette/{template_id}")
async def stream_palette(request: Request, template_id: str, palette_id: str):
    settings = get_settings()
    palette = next((p for p in PALETTES if p["id"] == palette_id), None)
    if not palette:
        async def err():
            yield {"event": "close", "data": ""}
        return EventSourceResponse(err())

    async def generator():
        async for event in stream_palette_executor(template_id, palette, settings):
            yield event

    return EventSourceResponse(generator())


@app.post("/plan", response_class=HTMLResponse)
async def plan(
    request: Request,
    goal: str = Form(...),
    preset_id: str = Form(""),
):
    """Return the loading partial immediately; SSE streams in the plan."""
    return templates.TemplateResponse(
        "partials/plan_loading.html",
        {"request": request, "goal": goal, "preset_id": preset_id},
    )


@app.get("/plan-stream")
async def plan_stream(request: Request, goal: str, preset_id: str = ""):
    """SSE endpoint: runs the planner LLM and yields the rendered plan HTML."""
    settings = get_settings()
    sections_per_step = SEQUENCE_PRESETS.get(preset_id) if preset_id else None

    async def generator():
        try:
            email_plan = await generate_plan(goal, settings, sections_per_step)
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
