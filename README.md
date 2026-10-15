# Beefree SDK MCP — Demo App

A FastAPI web application that demonstrates AI-powered email campaign generation using the **Beefree MCP (Model Context Protocol) API**. The AI agent communicates with the Beefree editor over MCP to autonomously build, translate, re-palette, and edit email templates — all streamed live to the browser.

---

## Features

- **Generation modes** — five distinct ways to create and manipulate email templates:
  - **Campaign generator** — choose a preset and immediately review a predefined plan, then let AI build the emails in parallel with live previews. Each of the 3 presets (streaming onboarding, SaaS trial nurture, Black Friday fashion) has 3 predefined variations, selected randomly per submission. Planning and shared-layout preparation require no LLM calls: each preset has a fixed branded header, footer, colors, typography, and placeholder body rows.
  - **Single email generator** — generate a standalone, production-ready email from a free-form prompt with a single AI agent.
  - **Bulk translation** — upload a template and translate it into any combination of 29 languages simultaneously; parallel agents replace text only, leaving layout and design intact.
  - **Palette swap** — apply one or more of 10 built-in color palettes to an existing template in parallel; colors change, everything else stays identical.
  - **Email editor** — load any template and chat with an AI agent to iteratively refine it; each message triggers a live MCP edit.
- **In-editor experience** — at `/integration`, the AI agent works alongside the embedded **visual Beefree editor**: changes the agent makes over MCP appear live in the editor. Demonstrates both session models — **Editor-Managed** (a fresh MCP session per chat turn) and **API-Managed + co-editing** (a persistent shared session). Requires `BEE_CLIENT_ID` / `BEE_CLIENT_SECRET`.
- **Export** — download any result as Beefree SDK JSON, rendered HTML, or a full sequence as a ZIP archive.
- **Token counter** — live usage tracker in the page header showing input, output, cache-write (↑cache), and cache-read (↓cache) tokens; accumulated across all agent calls and persisted across mode switches within the same browser session.
- **Multi-provider AI** — switch between Anthropic, OpenAI, and Google Gemini by changing a single env var; prompt caching is enabled automatically for all three.

---

## Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (package manager)
- A **Beefree CSAPI key** — get one at [developers.beefree.io](https://developers.beefree.io)
- An API key for at least one LLM provider (Anthropic, OpenAI, or Google AI Studio)
- *(Optional)* **Beefree SDK editor credentials** (`BEE_CLIENT_ID` / `BEE_CLIENT_SECRET`) — only needed for the in-editor experience at `/integration`

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd beefree-sdk-mcp-v2-example-demo
uv sync
```

### 2. Configure environment variables

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

Open `.env` and set:

```dotenv
# Required — your Beefree CSAPI key
BEE_API_KEY=your_csapi_key_here

# Optional — Beefree SDK editor credentials. Required ONLY for the in-editor
# experience at /integration, where the AI agent works alongside
# the embedded visual Beefree editor. Get them from developers.beefree.io.
BEE_CLIENT_ID=your_client_id_here
BEE_CLIENT_SECRET=your_client_secret_here

# Choose your AI provider: anthropic | openai | google
AI_PROVIDER=anthropic

# Add the key for your chosen provider:
ANTHROPIC_API_KEY=your_anthropic_key_here
# OPENAI_API_KEY=your_openai_key_here
# GOOGLE_API_KEY=your_google_ai_studio_key_here

# Optional — override the model used by all agents
# LLM_MODEL=anthropic:claude-sonnet-5-5
```

#### Provider defaults

All agents use a single model per provider. Prompt caching is enabled automatically for all providers (explicit cache markers for Anthropic, implicit/automatic for Google and OpenAI).

| `AI_PROVIDER` | Model | Notes |
|---|---|---|
| `anthropic` | `claude-sonnet-5-5` | Explicit prompt caching via cache markers |
| `openai` | `gpt-6.1-sol` | Automatic prompt caching; pricing depends on model |
| `google` | `gemini-3.8-flash` | Provider-default thinking configuration and implicit caching |

You can override the model for all agents with a single env var:

```dotenv
LLM_MODEL=anthropic:claude-sonnet-5-5
```

#### Model selection in the browser

Every demo has Provider and Model selectors in its header. Providers appear only
when their `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY` is non-empty
in the server environment or `.env`. API keys never appear in the catalog sent
to the browser. Restart the server after changing environment configuration.
Google/Gemini also accepts `GEMINI_API_KEY`; either key enables that provider.
When both keys are configured, the Google SDK prefers `GOOGLE_API_KEY`.

`AI_PROVIDER` and `LLM_MODEL` still define the initial model. An `LLM_MODEL`
override using a supported provider prefix is included alongside the built-in
model choices. If the initial provider has no key, explicitly choose an available
model; the app does not silently fall back to another provider.

The curated catalog was refreshed against official provider documentation on
October 15, 2026. Current choices include Claude Sonnet/Opus/Haiku 5.5 and Fable
5.1, GPT-6 Astra / GPT-6.1 Sol / GPT-6 Luna, and Gemini 4 Argon / 3.8 Flash /
3.5 Flash-Lite. Older choices remain for compatibility; in particular Gemini 2.5
access may be restricted to existing users. Catalog inclusion is not a guarantee
of account access or a live generation benchmark for that model.

Selections persist within the browser tab and apply to new runs or chat turns,
not agents already running. Campaign and bulk transformation child agents retain
their parent run's model. Choices are validated server-side against the catalog
in `app/config.py`; key presence does not guarantee model access or valid credentials.

#### Embedded editor (optional — in-editor experience)

The headless modes (single, sequence, translate, palette, edit, code mode) only
need `BEE_API_KEY`. The **in-editor experience** at `/integration` — where the AI
agent edits the email live inside the embedded visual Beefree editor — additionally
requires editor credentials:

```dotenv
BEE_CLIENT_ID=your_client_id_here
BEE_CLIENT_SECRET=your_client_secret_here
```

These are used by the `/integration-auth` endpoint to mint a short-lived editor
token (`auth.getbee.io/loginV2`). Both **Editor-Managed** and **API-Managed +
co-editing** session models are demonstrated on that page. If they are not set,
the `/integration` demo returns a configuration error while every headless mode
keeps working.

---

## Running the app

```bash
uv run python main.py
```

The server starts at [http://localhost:8000](http://localhost:8000) with hot-reload enabled.

Alternatively, you can start it directly with uvicorn:

```bash
uv run uvicorn app.main:app --reload
```

---

## Project structure

```
.
├── main.py              # Entrypoint — starts uvicorn
├── pyproject.toml       # Project metadata and dependencies
├── .env                 # Local config (git-ignored)
├── .env.example         # Config template
├── app/
│   ├── main.py          # FastAPI application and routes
│   ├── agent.py         # PydanticAI agents
│   ├── beefree.py       # Beefree REST API client
│   └── config.py        # Settings (pydantic-settings)
├── templates/
│   ├── landing.html     # Home page — mode selection
│   ├── index.html       # Campaign sequence generator UI
│   ├── single.html      # Single email generator UI
│   ├── translate.html   # Bulk translation UI
│   ├── palette.html     # Palette swap UI
│   ├── edit.html        # Email editor UI
│   ├── codemode.html    # Code Mode UI (single TypeScript-script generation)
│   ├── integration.html # In-editor (co-editing) experience with the embedded editor
│   └── partials/        # HTMX partial templates (SSE targets, loading states)
└── static/
    ├── style.css        # Application styles
    └── tokens.js        # Live token usage counter (sessionStorage-backed)
```

---

## Code Mode

All MCP-backed agents use `/v2/sdk/mcp/codemode`: campaign
executors, single emails, translations, palette changes, and chat editing in
both headless and embedded-editor modes. Campaign planning uses predefined data
and does not call an LLM or connect to MCP. Shared layouts are built locally
from the preset definitions in `app/layouts.py`; each email starts from a seeded
copy of that layout. Only the per-email body agents use AI in campaign execution.
Plans include a `preset_id` so layouts are selected explicitly, not inferred
from generated titles. Regenerate proposals created before this field existed.

The endpoint exposes a TypeScript execution tool. Agents use its schema to batch
related operations, inspect partial failures before retrying, and validate their
results. Session creation, template retrieval, and HTML rendering still use REST.
The dedicated Code Mode page remains a script-inspection demo using the same
executor as single-email generation.

The sequence proxy forwards to Code Mode while retaining its legacy guards for
direct tool calls. Those guards do not inspect operations inside TypeScript.
Sequence sessions carry server-side Brand Rules row limits. Header and footer
baselines are captured separately from each seeded session, after Beefree
normalizes the shared layout. This avoids treating server-added metadata as an
agent edit. Protected header and
footer contents, existing row IDs, and row order are checked during sequence
previews, at completion, and before exports. Preview checks are advisory and never
cancel running agent work. Final mismatches produce a `needs_review` outcome,
with previews retained. Layout warnings are advisory and do not block JSON or
HTML downloads, including ZIP exports; the validator logs warnings on export.
Sequences with completed and needs-review results expose both download actions.
These checks detect
invalid mutations but do not roll them back;
the session may retain partial work. Brand Rules require account support; session
creation fails rather than silently dropping requested guards.

Long-running streams emit status events and poll previews while scripts execute.
Agents are instructed to split work into meaningful phases. Streams distinguish
`outcome` (completed/failed) from transport `close`; the Code Mode page treats
failed scripts as failed/partial results, not successful generations.

Translation-only and palette-only field restrictions are still prompt guidance;
they are not a general permission boundary.

Code Mode requires a Beefree account with Code Mode access and a model that can
generate valid TypeScript. There is no automatic fallback to standard MCP tools.

Run local regression tests with `uv run python -m unittest discover -s tests -v`.
For opt-in live discovery and sequence guard smoke tests (create temporary
Beefree sessions and edit disposable templates, no LLM calls), run
`RUN_LIVE_MCP=1 uv run python -m unittest discover -s tests -p test_live_code_mode.py -v`.

## How it works

```
Browser → FastAPI (app/main.py)
              │
              ├─ Predefined campaign plans
              │     └─ instantly selects one of three variants per preset
              │
              ├─ Predefined campaign layouts
              │     └─ loads fixed branded header/footer and placeholder rows locally,
              │        then seeds a separate template per email via REST
              │
              ├─ PydanticAI Executor agents  (one per email, run concurrently)
              │     └─ each agent calls Beefree MCP tools to build the email body,
              │        streaming HTML previews back to the browser via SSE
              │
              └─ Beefree REST API  (api.getbee.io)
                    ├─ POST /v2/sdk/mcp/template   — create / seed a template session
                    ├─ GET  /v2/sdk/mcp/template/:id — read current template JSON
                    └─ POST /v1/message/html         — render template to email HTML
```

### Key modules

| File | Responsibility |
|------|---------------|
| `app/main.py` | FastAPI routes, SSE endpoints, session stores |
| `app/agent.py` | PydanticAI generation/editing agents and stream monitoring |
| `app/plans.py` | Three predefined campaign plan variations per preset, no LLM calls |
| `app/layouts.py` | Predefined campaign branding and shared Beefree template JSON |
| `app/beefree.py` | Thin async HTTP wrapper around the Beefree REST API |
| `app/config.py` | `pydantic-settings` config — reads `.env`, resolves model defaults per provider |
| `templates/` | Jinja2 HTML templates for the web UI |
| `static/` | CSS and JS assets (including `tokens.js` for live token tracking) |

