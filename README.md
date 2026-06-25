# Beefree SDK MCP v2 — Demo App

A FastAPI web application that demonstrates AI-powered email campaign generation using the **Beefree MCP (Model Context Protocol) v2 API**. The AI agent communicates with the Beefree editor over MCP to autonomously build, translate, re-palette, and edit email templates — all streamed live to the browser.

---

## Features

- **Generation modes** — five distinct ways to create and manipulate email templates:
  - **Campaign generator** — describe a campaign and choose a preset; the AI plans and builds a full multi-email sequence in parallel, with live previews streamed per email. Includes 3 built-in presets (streaming onboarding, SaaS trial nurture, Black Friday fashion).
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
# LLM_MODEL=anthropic:claude-sonnet-4-6
```

#### Provider defaults

All agents use a single model per provider. Prompt caching is enabled automatically for all providers (explicit cache markers for Anthropic, implicit/automatic for Google and OpenAI).

| `AI_PROVIDER` | Model | Notes |
|---|---|---|
| `anthropic` | `claude-sonnet-4-6` | Explicit prompt caching via cache markers |
| `openai` | `gpt-5.2` | Automatic 50% input token discount on cached prompts |
| `google` | `gemini-2.5-pro` | Implicit caching + `thinking_budget=256` |

You can override the model for all agents with a single env var:

```dotenv
LLM_MODEL=anthropic:claude-sonnet-4-6
```

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

## How it works

```
Browser → FastAPI (app/main.py)
              │
              ├─ PydanticAI Planner agent
              │     └─ generates an email sequence skeleton (titles, subject lines)
              │
              ├─ PydanticAI Layout agent
              │     └─ creates a blank Beefree template via REST, then calls MCP tools
              │        to add a shared header/footer layout
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
| `app/agent.py` | PydanticAI agents (planner, layout, executor, translation, palette, edit, single) |
| `app/beefree.py` | Thin async HTTP wrapper around the Beefree REST API |
| `app/config.py` | `pydantic-settings` config — reads `.env`, resolves model defaults per provider |
| `templates/` | Jinja2 HTML templates for the web UI |
| `static/` | CSS and JS assets (including `tokens.js` for live token tracking) |

