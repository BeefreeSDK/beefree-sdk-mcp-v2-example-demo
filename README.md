# Beefree SDK MCP v2 — Demo App

A FastAPI web application that demonstrates AI-powered email campaign generation using the **Beefree MCP (Model Context Protocol) v2 API**. The AI agent communicates with the Beefree editor over MCP to autonomously build, translate, re-palette, and edit email templates — all streamed live to the browser.

---

## How it works

```
Browser → FastAPI (app/main.py)
              │
              ├─ PydanticAI Planner agent  (fast LLM: Haiku / Flash / GPT-4o-mini)
              │     └─ generates an email sequence skeleton (titles, subject lines)
              │
              ├─ PydanticAI Layout agent   (main LLM: Sonnet / Gemini Pro / o4-mini)
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

### Features

- **Campaign generator** — describe a campaign and choose a preset; the AI builds a full multi-email sequence simultaneously, with live previews streamed per email. Includes 3 built-in presets (streaming onboarding, SaaS trial nurture, Black Friday fashion).
- **Single email generator** — generate a standalone email from a free-form prompt.
- **Bulk translation** — translate an existing Beefree template into multiple languages in parallel. Supports 33 languages.
- **Palette swap** — apply one or more of 10 built-in color palettes to an existing template in parallel.
- **Email editor** — chat with an AI agent to iteratively edit an existing template through multi-turn conversation.
- **Export** — download generated templates as Beefree JSON, rendered HTML, or a full sequence as a ZIP file.
- **Token counter** — live token usage tracker in the page header, accumulated across all agent calls and persisted across mode switches within the same browser session.
- **Multi-provider AI** — switch between Anthropic, OpenAI, and Google Gemini by changing one env var.

---

## Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (package manager)
- A **Beefree CSAPI key** — get one at [developers.beefree.io](https://developers.beefree.io)
- An API key for at least one LLM provider (Anthropic, OpenAI, or Google AI Studio)

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

# Choose your AI provider: anthropic | openai | google
AI_PROVIDER=anthropic

# Add the key for your chosen provider:
ANTHROPIC_API_KEY=your_anthropic_key_here
# OPENAI_API_KEY=your_openai_key_here
# GOOGLE_API_KEY=your_google_ai_studio_key_here
```

#### Provider defaults

| `AI_PROVIDER` | Main model (layout + executor) | Fast model (planner) |
|---|---|---|
| `anthropic` | `claude-sonnet-4-6` | `claude-haiku-4-5-20251001` |
| `openai` | `o4-mini` | `gpt-4o-mini` |
| `google` | `gemini-2.5-pro` | `gemini-2.5-flash` |

You can override individual models without changing the provider:

```dotenv
LLM_EXECUTOR_MODEL=openai:o3
LLM_PLANNER_MODEL=openai:gpt-4o-mini
```

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
│   └── partials/        # HTMX partial templates (SSE targets, loading states)
└── static/
    ├── style.css        # Application styles
    └── tokens.js        # Live token usage counter (sessionStorage-backed)
```
