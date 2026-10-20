import asyncio
import html as html_lib
import json
import re
import logging
import os
import secrets
import socket
import uuid
from urllib.parse import quote

log = logging.getLogger(__name__)

import httpx
from dotenv import load_dotenv

load_dotenv()  # put .env vars into os.environ so PydanticAI can read them

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from sse_starlette import EventSourceResponse

from .agent import (
    EmailPlan,
    EmailStep,
    _tokens_event,
    append_layout_context,
    stream_executor,
    stream_single_executor,
    stream_translation_executor,
    stream_palette_executor,
    stream_edit_executor,
)
from .beefree import (
    BrandRulesRejected, create_seeded_template, create_template, get_template,
    validate_brand_rules,
)
from .brand import BLOCK_TYPES, PRESETS, shape_errors
from .config import Settings, get_settings
from .guards import sequence_guard, validate_sequence
from .plans import build_preset_plan
from .layouts import build_preset_layout
from .samples import build_sample_template
from .payload import chat_session, chat_template, pack, unpack
from .sessions import ExpiringStore


def validate_download(template_id: str, template: dict) -> None:
    policy = template_layouts.get(template_id, {}).get("policy")
    if policy:
        try:
            validate_sequence(template, policy)
        except ValueError as exc:
            log.warning("Exporting sequence %s with layout warning: %s", template_id, exc)


def get_run_settings(request: Request) -> Settings:
    try:
        return get_settings().for_model(request.query_params.get("model"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


app = FastAPI(title="Beefree Headless MCP Demo")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Expose urlencode as a Jinja2 filter for prompt embedding in SSE URLs
templates.env.filters["urlencode"] = quote

# In-memory stores. Entries untouched for an hour are dropped (app/sessions.py),
# so memory stays flat on a public deployment.

# Maps template_id → {policy} for sequence layout validation
template_layouts = ExpiringStore()

# Temporary store for bulk-translation sessions (session_id → {template_json, languages})
translation_sessions = ExpiringStore()

# Temporary store for palette sessions (session_id → {template_json, palettes})
palette_sessions = ExpiringStore()

# Chat sessions (Edit an email, the builder, the integration guide) are signed
# tokens naming the template; the conversation travels with each message
# (chat_history below). Nothing is kept between requests.
CHAT_HISTORY_TURNS = 6
CHAT_TURN_CHARS = 600
CHAT_EXPIRED = ("This chat could not be found. Reload the page to start a new one.")


def chat_secret() -> str:
    return get_settings().bee_api_key or "beefree-demo"


def chat_history(turns) -> list:
    """Rebuild the agent's message history from the [user, agent] text turns the
    page sends back. The email itself lives in the Beefree template, so recent
    wording is all the agent needs for "make it bigger" or "undo that"."""
    history = []
    if not isinstance(turns, list):
        return history
    for turn in turns[-CHAT_HISTORY_TURNS:]:
        if not (isinstance(turn, list) and len(turn) == 2 and all(isinstance(t, str) for t in turn)):
            continue
        user, agent = (t.strip()[:CHAT_TURN_CHARS] for t in turn)
        if user and agent:
            history.append(ModelRequest(parts=[UserPromptPart(content=user)]))
            history.append(ModelResponse(parts=[TextPart(content=agent)]))
    return history

# Stream inputs (briefs, chat messages, campaign plans, brand rules) travel as
# compressed tokens in the stream URL (app/payload.py), not in server memory, so
# any copy of the app behind a load balancer can run them.
RUN_INPUT_FIELDS = {"brief", "message", "history"}
MAX_RUN_INPUT_CHARS = 20_000
BAD_INPUT = "This request could not be read. Reload the page and try again."

# Templates corrected on our side after an agent run (template_id → template
# JSON). Beefree has no update endpoint, so downloads serve these instead.
# Kept for a day so result pages left open still download the fixed version.
template_overrides = ExpiringStore(ttl_seconds=24 * 60 * 60)


async def load_template(template_id: str, settings: Settings) -> dict:
    """The template to download: our corrected copy if there is one, else Beefree's."""
    override = template_overrides.get(template_id)
    if override is not None:
        return override
    data = await get_template(template_id, settings)
    return data.get("template", data)

AVAILABLE_LANGUAGES = [
    "Spanish", "French", "German", "Italian", "Portuguese (Brazil)",
    "Portuguese (Portugal)", "Dutch", "Russian", "Japanese",
    "Chinese (Simplified)", "Chinese (Traditional)", "Korean",
    "Arabic", "Polish", "Turkish", "Swedish", "Danish", "Norwegian",
    "Finnish", "Czech", "Hungarian", "Romanian", "Greek",
    "Hebrew", "Thai", "Vietnamese", "Indonesian", "Ukrainian", "Hindi",
]

SEQUENCE_PRESETS: dict[str, list[list[str]]] = {
    # sections_per_step[email_index] = ordered list of body section descriptions.
    # Each description is a precise layout brief for the executor: layout type,
    # element sizes, copy direction, and colour intent are all specified.
    "black-friday-fashion": [
        [  # Email 1 — Teaser (48 h before launch)
            "Mystery hero (1-col, primary-colour bg, 48px padding): 600×280 dark editorial placehold.co banner; 'Something extraordinary arrives Friday.' headline (30px bold, white, centred, letter-spacing 2px); 'Our biggest sale of the year. Be first.' subtext (15px, #AAAAAA, centred)",
            "Countdown strip (1-col, accent bg, 28px padding): '48 HOURS TO GO' in 36px bold uppercase, jet-black text (#0A0A0A), centred, letter-spacing 3px — the high contrast of yellow bg + black text creates maximum visual impact",
            "Early access CTA (1-col, primary-colour bg, 36px padding): 'VAULTURA subscribers get exclusive early access.' copy (14px, #AAAAAA, centred); 'Secure Early Access' CTA button (accent bg, primary text, 48px tall, bold, centred)",
            "Brand teaser (2-col, page-bg, 28px padding): left column: 280×320 tall fashion editorial placehold.co image; right column: italic brand statement 'Crafted for the few. Coveted by many.' (18px, #0A0A0A) + 1 sentence on craft/exclusivity (14px, #555555) — vertically centred",
        ],
        [  # Email 2 — Launch Day
            "Sale hero (1-col, primary-colour bg, 44px padding): 600×260 bold editorial product placehold.co banner; 'BLACK FRIDAY IS HERE.' headline (32px bold, white, uppercase, centred, letter-spacing 2px); accent-colour badge '30% OFF EVERYTHING' (accent bg, primary text, 16px bold, inline badge centred below headline)",
            "Code spotlight (1-col, accent bg, 36px padding): 'USE CODE:' label (12px bold, #0A0A0A, uppercase, letter-spacing 2px, centred); 'VAULT30' in 40px bold monospace-style, jet-black, centred — maximum visual weight; 'Valid until midnight Friday · one use per customer.' (12px, #333333, centred); 'Shop Now →' CTA (primary bg, accent text, 48px tall, bold, centred)",
            "Product grid (3-col, page-bg, 28px padding): 'Curated picks for you' heading (16px bold, primary, centred); 3 product cards — each with a 190×240 placehold.co product image, product name (14px bold, #0A0A0A), original price struck through (12px, #999999), and sale price (16px bold, #0A0A0A)",
            "Closing urgency strip (1-col, primary-colour bg, 28px padding): 'Sale ends Friday at midnight.' urgency line (13px, #AAAAAA, centred); 'Browse All Deals' CTA (accent bg, primary text, 48px tall, bold, centred)",
        ],
        [  # Email 3 — Last Chance (24 h remaining)
            "Urgency hero (1-col, primary-colour bg, 48px padding): 600×240 dramatic fashion placehold.co image; '24 HOURS LEFT.' headline (36px bold, white, uppercase, centred, letter-spacing 4px); 'This is your final chance to shop the VAULTURA Black Friday sale.' subtext (14px, #AAAAAA, centred)",
            "Best-sellers grid (3-col, page-bg, 28px padding): '3 styles still available' heading (16px bold, primary, centred); 3 product cards — each with 190×220 placehold.co image, product name (14px bold), sale price (15px bold), and a scarcity label 'Only 4 left' (12px, red #DC2626, bold)",
            "Final CTA block (1-col, accent bg, 44px padding): 'Don't leave it too late.' headline (26px bold, #0A0A0A, centred); promo code reminder 'VAULT30' (18px monospace bold, #0A0A0A, centred); 'Shop Before It's Gone' CTA (primary bg, accent text, 48px tall, bold, centred)",
        ],
    ],

    "streaming-onboarding": [
        [  # Email 1 — Welcome & account confirmed
            "Hero (1-col, primary-colour bg, 40px v-padding): 600×220 atmospheric placehold.co banner image; 'Welcome to StreamVault' headline (28px bold, white, centred); 'Your account is ready — start watching in seconds.' subheading (15px, #A0A0B0, centred); 'Start Watching Now' CTA button (accent bg, white text, 48px tall, 6px radius, centred)",
            "Genre grid (2-col, white bg, 28px padding): 'What do you want to watch tonight?' section heading (20px bold, primary colour, centred); 4 genre tiles in a 2×2 grid — Action, Drama, Sci-Fi, Comedy — each tile has a 280×150 placehold.co cover image and a bold genre label (16px) below it",
            "App download strip (1-col, page-bg, 28px padding): 'Watch on phone, tablet, TV, and laptop.' copy (15px, centred); two side-by-side button placeholders — 'App Store' and 'Google Play' — using placehold.co 130×42 images in a neutral dark tone",
        ],
        [  # Email 2 — Password Reset (functional only)
            "Security header (1-col, primary-colour bg, 36px padding): 48×48 padlock icon (placehold.co, white on primary); 'Password Reset Request' headline (22px bold, white, centred); 'We received a request to reset your StreamVault password.' subtext (14px, #A0A0B0, centred)",
            "Action row (1-col, white bg, 44px padding): 'Click the button below — this link expires in 24 hours.' instruction (15px, body colour, centred); 'Reset My Password' CTA button (accent bg, white, 48px tall, centred); 'Didn't request this? You can safely ignore this email — your password will not change.' safety notice (13px, #888888, italic, centred)",
        ],
        [  # Email 3 — Subscription Confirmed
            "Confirmation hero (1-col, accent bg, 36px padding): 48×48 checkmark icon (placehold.co white); 'You're all set — welcome to StreamVault Premium.' headline (24px bold, white, centred); 'Your subscription is active. Next billing: [billing date].' subtext (14px, rgba-white-70, centred)",
            "Plan benefits (3-col, white bg, 32px padding): 'Everything included in your plan' section heading (20px bold, primary, centred); 3 equal columns — col 1: '4K Ultra HD' with 48×48 icon + 13px caption; col 2: 'Watch on 4 screens' with icon + caption; col 3: 'Unlimited downloads' with icon + caption; all icons use placehold.co 48×48",
            "Next steps strip (1-col, primary-colour bg, 36px padding): 'Time to find your next obsession.' headline (22px bold, white, centred); 'Explore the Catalogue' primary CTA button (accent bg, white, 48px, centred); 'Manage your subscription' secondary text link below (13px, #A0A0B0, underlined, centred)",
        ],
    ],

    "saas-trial-nurture": [
        [  # Email 1 — Trial Activated (Day 0)
            "Hero (1-col, primary-colour bg, 40px padding): 600×220 placehold.co product dashboard screenshot; 'Your Taskflow trial is live.' headline (26px bold, white, centred); 'You have 14 days to explore everything — no credit card needed.' subtext (15px, #90B4D4, centred); 'Open Your Dashboard' CTA (accent bg, white, 48px tall, centred)",
            "3-feature grid (3-col, page-bg, 32px padding): 'Start with these three features' section heading (18px bold, primary, centred); 3 equal columns — col 1: Task Boards (48×48 icon, 15px bold title, 13px benefit line); col 2: Time Tracking (icon, title, benefit); col 3: Team Collaboration (icon, title, benefit); icons use placehold.co 48×48",
            "Quick-start checklist (1-col, white bg, 32px padding): 'Get set up in under 5 minutes' heading (18px bold, primary); 3 numbered steps as a tight vertical list — '1. Create your first project', '2. Invite a team member', '3. Set your first deadline' — each step in 15px with a number badge in accent colour; 'Start Your First Project' CTA button (primary bg, white, 44px, below the list)",
        ],
        [  # Email 2 — Day 3 Check-in
            "Friendly intro (1-col, white bg, 36px padding): 'How's Taskflow working for you?' headline (24px bold, primary, centred); 'Here are three things to try today to get the most from your trial.' body copy (15px, #64748B, centred)",
            "Tips list (1-col, page-bg, 32px padding): 3 numbered rows — each row has a large colour-coded number badge (accent bg), a bold tip title (15px, primary), and one sentence of practical context (14px, #64748B); tips: 'Turn on notifications', 'Create a recurring task', 'Use the timeline view'",
            "Tutorial cards (2-col, white bg, 28px padding): 'Watch and learn' section heading (16px bold); 2 video-thumbnail cards side by side — each with a 280×160 placehold.co thumbnail, a video title, and a '3 min' duration label (12px, muted)",
            "Support strip (1-col, accent bg, 28px padding): 'Stuck? We're here.' headline (20px bold, white, centred); two inline text links — 'View Help Docs' and 'Back to Dashboard' — side by side in white, 15px",
        ],
        [  # Email 3 — Day 7 Feature Spotlight (Automation)
            "Feature hero (1-col, primary-colour bg, 40px padding): 600×200 placehold.co feature screenshot; 'Meet Automation — the feature that saves your team 5 hours a week.' headline (24px bold, white, centred); 'Available on the paid plan.' one-line value note (14px, #90B4D4, centred)",
            "Benefits breakdown (1-col, white bg, 32px padding): 'What Automation unlocks for your team' heading (18px bold, primary); 3 benefit rows — each row has a small accent-coloured icon bullet, a bold benefit title (15px), and a concrete one-line outcome (14px, #64748B): 'Auto-assign tasks when a deadline shifts', 'Send instant Slack alerts on status changes', 'Generate weekly progress reports automatically'",
            "Upgrade CTA block (1-col, accent bg, 36px padding): 'Unlock Automation and 12 more premium features' headline (20px bold, white, centred); pricing note '$12/user/month · or $9/user/month billed annually' (14px, white, centred); 'Upgrade Now' CTA (white bg, accent text, 48px tall, bold, centred); 'Compare all plans →' secondary text link (white, 13px, underlined)",
        ],
        [  # Email 4 — Trial Ending (Day 14)
            "Urgency hero (1-col, primary-colour bg, 40px padding): 48×48 clock icon (placehold.co white); 'Your Taskflow trial ends in 3 days.' headline (24px bold, white, centred); 'Everything you've built is here — don't lose access.' subtext (15px, #90B4D4, centred)",
            "Value recap (3-col, page-bg, 32px padding): 'Here's what you've accomplished' section heading (18px bold, primary, centred); 3 stat cards — 'Projects created', 'Tasks completed', 'Team members added' — each card shows a large metric number (32px bold, accent colour) and a caption label (13px, #64748B); use plausible demo numbers",
            "Offer box (1-col, white bg, 28px padding): bordered highlight box (#E8F4FD bg, accent border); '20% off your first 3 months — trial users only.' (18px bold, primary); promo code 'TRIAL20' centred in large monospace style (28px bold, primary); 'Expires when your trial does.' expiry note (12px, muted)",
            "Final CTA (1-col, primary-colour bg, 36px padding): 'Don't lose access to your projects.' copy (16px, white, centred); 'Upgrade Now' primary CTA (accent bg, white, 48px, centred); 'Compare plans →' text link below (13px, #90B4D4, centred)",
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


async def _failed_stream(message: str):
    """A stream that only reports why it could not start."""
    yield {"event": "agent-error", "data": html_lib.escape(message)}
    yield {"event": "outcome", "data": '{"status":"failed"}'}
    yield {"event": "close", "data": ""}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Browsers ask for /favicon.ico on their own; pages link the SVG and PNGs."""
    return FileResponse("static/favicon-32.png", media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})


# Which process answered: hostname (the container) and pid (the worker). Calling
# /healthcheck a few times shows whether requests are spread across copies of the
# app, which breaks the in-memory sessions (see README, "One instance only").
INSTANCE_ID = f"{socket.gethostname()}:{os.getpid()}"


@app.get("/healthcheck")
async def healthcheck():
    """Liveness probe for the hosting platform: no external calls, always fast."""
    return JSONResponse({"status": "ok", "instance": INSTANCE_ID}, headers={"Cache-Control": "no-store"})


@app.get("/ai-models")
async def ai_models():
    return JSONResponse(
        get_settings().model_catalog(),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse(
        "landing.html",
        {"request": request},
    )


@app.get("/editor", response_class=HTMLResponse)
async def editor_modes(request: Request):
    return templates.TemplateResponse(
        "landing.html",
        {"request": request, "section": "editor"},
    )


@app.get("/headless", response_class=HTMLResponse)
async def headless_modes(request: Request):
    return templates.TemplateResponse(
        "landing.html",
        {"request": request, "section": "headless"},
    )


@app.get("/sequence", response_class=HTMLResponse)
async def sequence(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


@app.get("/integration", response_class=HTMLResponse)
async def integration_page(request: Request):
    return templates.TemplateResponse(
        "integration.html",
        {"request": request},
    )


@app.get("/builder", response_class=HTMLResponse)
async def builder_page(request: Request):
    return templates.TemplateResponse("builder.html", {"request": request})


@app.get("/sample-template")
async def sample_template():
    """A ready-made Beefree template for trying the template-based demos."""
    return JSONResponse(build_sample_template())


EDITOR_UID = re.compile(r"^demo-[0-9a-f]{32}$")


@app.get("/integration-auth")
async def integration_auth(uid: str | None = None):
    """Return a fresh Beefree SDK auth token for the embedded editor.

    Each browser gets its own editor user ID, so visitors don't share Beefree's
    per-user state (file manager uploads, co-editing presence). The browser keeps
    the ID it was given and sends it back; anything else gets a new random one.
    """
    if not uid or not EDITOR_UID.match(uid):
        uid = f"demo-{secrets.token_hex(16)}"
    settings = get_settings()
    if not settings.bee_client_id or not settings.bee_client_secret:
        return JSONResponse(
            {"error": "BEE_CLIENT_ID and BEE_CLIENT_SECRET not configured"},
            status_code=500,
        )
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://auth.getbee.io/loginV2",
            json={
                "client_id": settings.bee_client_id,
                "client_secret": settings.bee_client_secret,
                "uid": uid,
            },
        )
        if not resp.is_success:
            return JSONResponse(
                {"error": f"Auth failed: {resp.status_code}"},
                status_code=502,
            )
        return JSONResponse({"token": resp.json(), "uid": uid})


@app.post("/integration-start")
async def integration_start(template_id: str | None = None):
    """Create a chat session for the integration demo.

    - Editor-managed: the editor already owns a template via ``startMcpSession()``;
      pass its ``template_id`` so the agent edits that same template and the
      editor reflects the changes.
    - API-managed: omit ``template_id`` and the backend creates a fresh template
      that the editor then joins via co-editing.
    """
    settings = get_settings()
    if not template_id:
        template_id = await create_template(settings)
    return JSONResponse({"template_id": template_id,
                         "session_id": chat_session(template_id, chat_secret())})


@app.post("/integration-refresh")
async def integration_refresh(session_id: str, template_id: str):
    """Point the chat at a new templateId; returns the session token to use next.

    Editor-managed mode starts a fresh MCP session per chat interaction, so each
    turn the editor hands us a (possibly new) templateId to edit. The page keeps
    the conversation, so only the session token changes.
    """
    if not chat_template(session_id, chat_secret()):
        return JSONResponse({"error": CHAT_EXPIRED}, status_code=404)
    return JSONResponse({"ok": True, "template_id": template_id,
                         "session_id": chat_session(template_id, chat_secret())})


@app.get("/codemode", response_class=HTMLResponse)
async def codemode_page(request: Request):
    return templates.TemplateResponse("codemode.html", {"request": request})


@app.get("/codemode-stream")
async def codemode_stream(
    request: Request, brief: str | None = None, input_id: str | None = None,
    settings: Settings = Depends(get_run_settings),
):
    """SSE endpoint: creates a template, runs single email agent via Code Mode endpoint, streams previews."""
    if input_id:
        brief = (unpack(input_id) or {}).get("brief")
    if not brief:
        return EventSourceResponse(_failed_stream(BAD_INPUT if input_id else "No brief was sent."))

    async def generator():
        try:
            template_id = await create_template(settings)
            yield {"event": "template-id", "data": template_id}
            async for event in stream_single_executor(template_id, brief, settings):
                if event["event"] != "close":
                    yield event
        except Exception as exc:
            log.error("Code Mode gen error: %s", exc)
            yield {
                "event": "agent-error",
                "data": html_lib.escape(f"Generation failed: {exc}"),
            }
            yield {"event": "outcome", "data": '{"status":"failed"}'}
        finally:
            yield {"event": "close", "data": ""}

    return EventSourceResponse(generator())


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
        {"request": request, "brief": brief, "input_id": pack({"brief": brief})},
    )


@app.post("/run-input")
async def run_input(request: Request):
    """Save a long stream input; the page then opens the stream with ?input_id=."""
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"error": "The request was not valid JSON."}, status_code=400)
    fields = body if isinstance(body, dict) else {}
    texts = [v for k, v in fields.items() if k != "history"]
    if (not texts or set(fields) - RUN_INPUT_FIELDS or not all(isinstance(v, str) for v in texts)
            or not isinstance(fields.get("history", []), list)):
        return JSONResponse({"error": "Send a brief or a message."}, status_code=400)
    if fields.get("history"):
        # Only the recent turns are used; trim them here so the token stays short.
        fields["history"] = [
            [t[:CHAT_TURN_CHARS] for t in turn] for turn in fields["history"][-CHAT_HISTORY_TURNS:]
            if isinstance(turn, list) and len(turn) == 2 and all(isinstance(t, str) for t in turn)
        ]
    if len(json.dumps(fields)) > MAX_RUN_INPUT_CHARS:
        return JSONResponse({"error": "That text is too long for this demo."}, status_code=413)
    return JSONResponse({"input_id": pack(fields)})


@app.get("/brand", response_class=HTMLResponse)
async def brand_page(request: Request):
    return templates.TemplateResponse(
        "brand.html",
        {
            "request": request, "presets": PRESETS, "block_types": BLOCK_TYPES,
            # Jinja's tojson sorts keys; keep preset order as written (it is the
            # order Beefree lists them to the agent). Escape "</" for the <script>.
            "presets_json": json.dumps(PRESETS).replace("</", "<\\/"),
        },
    )


@app.post("/brand-rules")
async def brand_rules(request: Request, settings: Settings = Depends(get_run_settings)):
    """Check a brandRules object with Beefree; returns it as a token for the generation."""
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"errors": ["The request was not valid JSON."]}, status_code=400)
    rules = body.get("rules") if isinstance(body, dict) else None
    errors = shape_errors(rules)
    if errors:
        return JSONResponse({"errors": errors}, status_code=422)
    try:
        await validate_brand_rules(settings, rules)
    except BrandRulesRejected as exc:
        return JSONResponse({"errors": exc.details[:20]}, status_code=422)
    except Exception as exc:
        log.error("Brand rules validation failed: %s", exc)
        return JSONResponse(
            {"errors": ["Beefree could not check these rules right now. Try again in a moment."]},
            status_code=502,
        )
    return JSONResponse({"brand_id": pack({"rules": rules})})


@app.post("/brand-generate", response_class=HTMLResponse)
async def brand_generate(request: Request, brief: str = Form(...), brand_id: str = Form(...)):
    """Same result view as Single email, with the brand rules attached to the session."""
    rules = (unpack(brand_id) or {}).get("rules")
    if not isinstance(rules, dict):
        return HTMLResponse(
            "<p class='inline-error' role='alert'>These brand rules could not be read. "
            "Choose them again to continue.</p>",
            status_code=400,
        )
    return templates.TemplateResponse(
        "partials/single_loading.html",
        {"request": request, "brief": brief, "brand_id": brand_id,
         "input_id": pack({"brief": brief, "brand_rules": rules})},
    )


@app.get("/generate-stream")
async def generate_stream(
    request: Request, brief: str | None = None, input_id: str | None = None,
    settings: Settings = Depends(get_run_settings),
):
    """SSE endpoint: creates a template, runs single email agent, streams previews.

    ``input_id`` is a token (app/payload.py) with the brief and, from the brand
    page, the brand rules: the template session then carries them and the stream
    adds "brand" events each time they apply.
    """
    rules = None
    if input_id:
        saved = unpack(input_id) or {}
        brief = saved.get("brief")
        rules = saved.get("brand_rules") if isinstance(saved.get("brand_rules"), dict) else None
    if not brief:
        return EventSourceResponse(_failed_stream(BAD_INPUT if input_id else "No brief was sent."))

    async def generator():
        try:
            template_id = await create_template(settings, brand_rules=rules)
            # Tell the frontend which template was created so it can offer a download
            yield {"event": "template-id", "data": template_id}
            async for event in stream_single_executor(
                template_id, brief, settings, progressive=True, brand=bool(rules),
            ):
                if event["event"] != "close":
                    yield event
        except Exception as exc:
            log.error("Single gen error: %s", exc)
            yield {
                "event": "agent-error",
                "data": html_lib.escape(f"Generation failed: {exc}"),
            }
            yield {"event": "outcome", "data": '{"status":"failed"}'}
        finally:
            yield {"event": "close", "data": ""}

    return EventSourceResponse(generator())


@app.get("/download-template/{template_id}")
async def download_template(template_id: str):
    """Return the Beefree template JSON as a downloadable file."""
    settings = get_settings()
    template = await load_template(template_id, settings)
    validate_download(template_id, template)
    filename = f"email-{template_id[:8]}.json"
    return Response(
        content=json.dumps(template, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/download-html/{template_id}")
async def download_html(template_id: str):
    """Return the rendered email HTML as a downloadable file."""
    from .beefree import render_html as _render_html
    settings = get_settings()
    template = await load_template(template_id, settings)
    validate_download(template_id, template)
    html_content = await _render_html(template, settings)
    filename = f"email-{template_id[:8]}.html"
    return Response(
        content=html_content,
        media_type="text/html",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


MAX_ZIP_EMAILS = 12


def zip_entries(ids: str, steps: str | None) -> list[tuple[int, str]]:
    """Pair each template id with its email number for zip file names.

    ``steps`` keeps the original numbering when only some emails are
    downloaded (emails 1 and 3 stay "email-01" and "email-03").
    """
    template_ids = [tid.strip() for tid in ids.split(",") if tid.strip()]
    if not template_ids or len(template_ids) > MAX_ZIP_EMAILS:
        raise HTTPException(status_code=400, detail=f"Choose between 1 and {MAX_ZIP_EMAILS} emails to download.")
    numbers = list(range(1, len(template_ids) + 1))
    if steps:
        try:
            given = [int(n) for n in steps.split(",")]
        except ValueError:
            given = []
        if len(given) == len(template_ids) and all(n > 0 for n in given):
            numbers = given
    return list(zip(numbers, template_ids))


@app.get("/download-all")
async def download_all(ids: str, steps: str | None = None):
    """Return a ZIP file containing all template JSONs for a sequence."""
    import io
    import zipfile

    settings = get_settings()
    entries = zip_entries(ids, steps)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, tid in entries:
            template = await load_template(tid, settings)
            validate_download(tid, template)
            zf.writestr(f"email-{i:02d}-{tid[:8]}.json", json.dumps(template, indent=2))
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="email-sequence.zip"'},
    )


@app.get("/download-all-html")
async def download_all_html(ids: str, steps: str | None = None):
    """Return a ZIP file containing all rendered HTML emails for a sequence."""
    import io
    import zipfile

    from .beefree import render_html as _render_html

    settings = get_settings()
    entries = zip_entries(ids, steps)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, tid in entries:
            template = await load_template(tid, settings)
            validate_download(tid, template)
            html_content = await _render_html(template, settings)
            zf.writestr(f"email-{i:02d}-{tid[:8]}.html", html_content)
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="email-sequence.zip"'},
    )


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
            'style="width:700px;height:1200px;border:none;display:block;" '
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
            f"<p class='inline-error' role='alert'>The template JSON is not valid: {html_lib.escape(str(exc))}</p>", status_code=400
        )
    lang_list = [la.strip() for la in languages.split(",") if la.strip()]
    session_id = uuid.uuid4().hex[:12]
    translation_sessions[session_id] = {"template_json": parsed, "languages": lang_list}
    return templates.TemplateResponse(
        "partials/translate_loading.html",
        {"request": request, "session_id": session_id, "languages": lang_list},
    )


@app.get("/translate-stream")
async def translate_stream_route(
    request: Request, session_id: str,
    settings: Settings = Depends(get_run_settings),
):
    """SSE: seed N template copies, render the column grid, agents stream previews."""
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
                languages_with_ids=list(zip(languages, template_ids)),
                llm_model=settings.llm_model,
            )
            yield {"event": "ready", "data": result_html}
        except Exception as exc:
            log.error("Translate stream error: %s", exc)
            yield {
                "event": "ready",
                "data": f"<p class='inline-error' role='alert'>We couldn't start the translations: {html_lib.escape(str(exc))}</p>",
            }
        finally:
            yield {"event": "close", "data": ""}

    return EventSourceResponse(generator())


@app.get("/stream-translate/{template_id}")
async def stream_translate(
    request: Request, template_id: str, language: str,
    settings: Settings = Depends(get_run_settings),
):

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
            f"<p class='inline-error' role='alert'>The template JSON is not valid: {html_lib.escape(str(exc))}</p>", status_code=400
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
async def palette_stream_route(
    request: Request, session_id: str,
    settings: Settings = Depends(get_run_settings),
):
    """SSE: seed N template copies, render palette column grid."""
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
                palettes_with_ids=list(zip(palettes, template_ids)),
                llm_model=settings.llm_model,
            )
            yield {"event": "ready", "data": result_html}
        except Exception as exc:
            log.error("Palette stream error: %s", exc)
            yield {
                "event": "ready",
                "data": f"<p class='inline-error' role='alert'>We couldn't start the color versions: {html_lib.escape(str(exc))}</p>",
            }
        finally:
            yield {"event": "close", "data": ""}

    return EventSourceResponse(generator())


@app.get("/stream-palette/{template_id}")
async def stream_palette(
    request: Request, template_id: str, palette_id: str,
    settings: Settings = Depends(get_run_settings),
):
    palette = next((p for p in PALETTES if p["id"] == palette_id), None)
    if not palette:
        async def err():
            yield {"event": "close", "data": ""}
        return EventSourceResponse(err())

    async def generator():
        async for event in stream_palette_executor(template_id, palette, settings):
            if event["event"] == "fixed-template":
                template_overrides[template_id] = event["data"]  # internal, not sent to the browser
                continue
            yield event

    return EventSourceResponse(generator())


@app.post("/plan", response_class=HTMLResponse)
async def plan(
    request: Request,
    goal: str = Form(...),
    preset_id: str = Form(""),
):
    """Render a predefined plan immediately, without AI or an SSE wait."""
    sections = SEQUENCE_PRESETS.get(preset_id)
    if sections is None:
        raise HTTPException(status_code=400, detail="Choose a supported campaign preset.")
    email_plan = build_preset_plan(preset_id, goal, sections, secrets.randbelow(3))
    return templates.TemplateResponse(
        "partials/preset_plan.html",
        {"request": request, "plan": email_plan,
         "plan_json": email_plan.model_dump_json()},
    )


@app.post("/execute", response_class=HTMLResponse)
async def execute(request: Request, plan_json: str = Form(...)):
    """Show email cards immediately while SSE prepares their sessions."""
    try:
        email_plan = EmailPlan.model_validate_json(plan_json)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid campaign plan.") from exc
    return templates.TemplateResponse(
        "partials/execute_loading.html",
        {"request": request, "input_id": pack({"plan": email_plan.model_dump()}), "plan": email_plan,
         "emails_with_ids": [(email, None) for email in email_plan.emails],
         "pending_sessions": True},
    )


@app.get("/execute-stream")
async def execute_stream(
    request: Request, plan_json: str | None = None, input_id: str | None = None,
    settings: Settings = Depends(get_run_settings),
):
    """SSE: load a static preset layout, seed templates, stream the sequence view."""
    try:
        if input_id:
            email_plan = EmailPlan.model_validate((unpack(input_id) or {}).get("plan"))
        else:
            email_plan = EmailPlan.model_validate_json(plan_json or "")
    except ValueError:
        async def _unreadable():
            yield {"event": "sequence", "data": (
                f"<p class='inline-error' role='alert'>{html_lib.escape(BAD_INPUT)}</p>")}
            yield {"event": "close", "data": ""}
        return EventSourceResponse(_unreadable())
    num_sections = max((e.body_section_count for e in email_plan.emails), default=3)
    try:
        header_row_id, footer_row_id, placeholder_row_ids, layout_json = (
            build_preset_layout(email_plan.preset_id, num_sections)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    async def generator():
        try:
            policy = sequence_guard(layout_json, header_row_id, footer_row_id)
            template_ids: list[str] = await asyncio.gather(
                *[
                    create_seeded_template(settings, layout_json, brand_rules={
                        "limits": {"rows": {"max": len(policy["ids"]), "min": 2}},
                    })
                    for _ in email_plan.emails
                ]
            )

            seeded_templates = await asyncio.gather(
                *[get_template(tid, settings) for tid in template_ids]
            )
            policies = []
            for data in seeded_templates:
                seeded = data.get("template", data)
                child_policy = sequence_guard(seeded, header_row_id, footer_row_id)
                if child_policy["ids"] != policy["ids"]:
                    raise ValueError("Seeded sequence row IDs differ from the shared layout.")
                policies.append(child_policy)

            # Register each template's layout policy; the executor and downloads
            # validate against it. Code Mode scripts must also follow the
            # protected-row instructions in their executor prompt.
            for tid, child_policy in zip(template_ids, policies):
                template_layouts[tid] = {"policy": child_policy}

            enriched_emails = [
                EmailStep(
                    step=e.step,
                    title=e.title,
                    subject_line=e.subject_line,
                    body_section_count=e.body_section_count,
                    agent_prompt=append_layout_context(
                        e.agent_prompt, header_row_id, footer_row_id, placeholder_row_ids
                    ),
                )
                for e in email_plan.emails
            ]

            emails_with_ids = list(zip(enriched_emails, template_ids))
            tmpl = templates.env.get_template("partials/sequence.html")
            sequence_html = tmpl.render(
                plan=email_plan,
                emails_with_ids=emails_with_ids,
                # Each column's stream URL carries an id, not the (long) agent prompt
                prompt_inputs={tid: pack({"prompt": email.agent_prompt})
                               for email, tid in emails_with_ids},
                llm_model=settings.llm_model,
            )
            yield {"event": "sequence", "data": sequence_html}

        except Exception as exc:
            log.error("Execute stream error: %s", exc)
            yield {
                "event": "sequence",
                "data": f"<p class='inline-error' role='alert'>We couldn't build the emails: {html_lib.escape(str(exc))}</p>",
            }
        finally:
            yield {"event": "close", "data": ""}

    return EventSourceResponse(generator())


@app.get("/edit", response_class=HTMLResponse)
async def edit_page(request: Request):
    return templates.TemplateResponse("edit.html", {"request": request})


@app.post("/edit-start", response_class=HTMLResponse)
async def edit_start(request: Request, template_json: str = Form(...)):
    """Parse template JSON, seed a Beefree template, create a session, return chat view."""
    from .beefree import render_html

    settings = get_settings()
    try:
        parsed = json.loads(template_json)
    except json.JSONDecodeError as exc:
        return HTMLResponse(
            f"<p class='inline-error' role='alert'>The template JSON is not valid: {html_lib.escape(str(exc))}</p>", status_code=400
        )

    template = parsed.get("template", parsed)
    try:
        template_id = await create_seeded_template(settings, template)
        rendered = await render_html(template, settings)
    except Exception as exc:
        return HTMLResponse(
            f"<p class='inline-error' role='alert'>We couldn't load the template: {html_lib.escape(str(exc))}</p>", status_code=500
        )

    escaped = html_lib.escape(rendered, quote=True)
    initial_preview = (
        f'<iframe srcdoc="{escaped}" sandbox="allow-same-origin" '
        'style="width:700px;height:1200px;border:none;display:block;" '
        'title="Email preview"></iframe>'
    )

    session_id = chat_session(template_id, chat_secret())

    return templates.TemplateResponse(
        "partials/edit_workspace.html",
        {
            "request": request,
            "session_id": session_id,
            "template_id": template_id,
            "initial_preview": initial_preview,
        },
    )


SELECTION_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")


def with_selection(message: str, selection_type: str | None, selection_id: str | None) -> str:
    """Tell the agent which editor element the user has selected, if any.

    The Beefree editor reports selections through its onSelectElement callback
    ({type: "row" | "module", uuid}); the chat sends them along with the message.
    """
    if selection_type not in ("row", "module") or not selection_id or not SELECTION_ID.match(selection_id):
        return message
    kind = "row" if selection_type == "row" else "content block (module)"
    return (
        f"{message}\n\n"
        f"[Editor selection] The user currently has a {kind} selected in the editor "
        f"(uuid: {selection_id}). Unless the message says otherwise, words like \"this\", "
        f"\"it\" or \"here\" refer to that element: find it in the template by its uuid "
        f"and apply the change to it, leaving the rest of the email unchanged."
    )


@app.get("/edit-stream")
async def edit_stream_route(
    request: Request, session_id: str, message: str | None = None, input_id: str | None = None,
    selection_type: str | None = None, selection_id: str | None = None,
    settings: Settings = Depends(get_run_settings),
):
    """SSE: run one edit-agent turn, stream previews + the agent's reply."""
    turns = None
    if input_id:
        saved = unpack(input_id) or {}
        message, turns = saved.get("message"), saved.get("history")
    if not message:
        async def _missing():
            yield {"event": "agent-error", "data": html_lib.escape(
                BAD_INPUT if input_id else "No message was sent.")}
            yield {"event": "close", "data": ""}
        return EventSourceResponse(_missing())
    template_id = chat_template(session_id, chat_secret())
    if not template_id:
        async def _err():
            yield {"event": "agent-message", "data": f"<p>{html_lib.escape(CHAT_EXPIRED)}</p>"}
            yield {"event": "close", "data": ""}
        return EventSourceResponse(_err())

    async def generator():
        async for event in stream_edit_executor(
            template_id=template_id,
            message=with_selection(message, selection_type, selection_id),
            settings=settings,
            message_history=chat_history(turns),
        ):
            yield event

    return EventSourceResponse(generator())


@app.get("/stream/{template_id}")
async def stream(
    request: Request, template_id: str, prompt: str | None = None, input_id: str | None = None,
    settings: Settings = Depends(get_run_settings),
):
    if input_id:
        prompt = (unpack(input_id) or {}).get("prompt")
    if not prompt:
        return EventSourceResponse(_failed_stream(BAD_INPUT if input_id else "No prompt was sent."))
    async def generator():
        async for event in stream_executor(
            template_id, prompt, settings,
            sequence_policy=template_layouts.get(template_id, {}).get("policy"),
        ):
            yield event

    return EventSourceResponse(generator())
