# Beefree SDK MCP — Demo App

A FastAPI web application that demonstrates AI-powered email campaign generation using the **Beefree MCP (Model Context Protocol) API**. The AI agent communicates with the Beefree editor over MCP to autonomously build, translate, re-palette, and edit email templates — all streamed live to the browser.

---

## Features

The home page offers two paths: **In the editor** (`/editor`) and **Headless** (`/headless`).

### In the editor

Both need the editor credentials `BEE_CLIENT_ID` / `BEE_CLIENT_SECRET`.

- **Agent in the builder** (`/builder`): a marketer-facing page with the visual Beefree editor and an AI chat side by side, sized to fit the window. Notable behaviour:
  - **Starter ideas:** three short buttons. Each fills the composer with a detailed brief (brand, colors, layout, copy) that can be edited before sending.
  - **Selection-aware chat:** selecting a row or block in the editor shows a "Selected row/block" chip above the composer. The selection is sent with the next message, so "make this bigger" applies to that element. This uses the SDK's `onSelectElement` callback.
  - **Plain-language replies:** the agent is instructed to answer without code, JSON or element IDs, and `clean_reply()` strips any that slip through.
- **Integration guide** (`/integration`): the developer page. It runs live demos of both session models, **Editor-managed** (a fresh MCP session per chat turn) and **API-managed + co-editing** (a persistent shared session), with implementation code for each.

### Headless

These only need `BEE_API_KEY`.

**Create a new email**
- **Single email** (`/single`): generate one email from a preset or a free-form brief.
  - The agent builds it step by step (one script per section), so the preview fills in progressively.
  - An Activity log describes each step.
  - The toolbar shows the run's tokens and estimated cost.
  - Desktop/Mobile preview toggle.
- **Code Mode** (`/codemode`): the same generation, but the agent batches everything into as few TypeScript scripts as possible. The script it writes is shown next to the result.
- **Email sequence** (`/sequence`): pick a campaign preset, review a predefined plan, then build every email at the same time.
  - Each of the 3 presets (streaming onboarding, SaaS trial nurture, Black Friday fashion) has 3 predefined variations, selected randomly.
  - Planning and the shared layout need no LLM calls.
- **On-brand email** (`/brand`): set [Brand Rules](https://docs.beefree.io/beefree-sdk/ai-and-mcp/brand-rules-for-ai), then generate an email inside them.
  - Three ways to get the rules: a preset (Harbor Bank, Bloom & Co., Voltage, in `app/brand.py`), a four-step guided setup (colors, type, buttons, guardrails), or pasting your own `brandRules` JSON.
  - A live preview, a plain-language summary and the JSON update as the rules change.
  - Beefree validates the rules before they are used (`POST /brand-rules`); its error list is shown as is.
  - The generation uses the Single email result view. The template session carries the rules, and the toolbar counts presets used, values corrected and calls refused. Each correction and refusal appears in the Activity log.
  - Brand Rules require a Superpowers or Enterprise plan.

**Change an existing template.** Each of these has a **Use a sample template** button (served by `/sample-template`), so they can be tried without bringing your own Beefree JSON.
- **Edit an email** (`/edit`): chat with an agent to refine a template.
- **Bulk translations** (`/translate`): translate a template into any of 29 languages at once. Only the text changes.
- **Color palettes** (`/palette`): recolor a template with any of 10 palettes at once.
  - The server builds a complete color plan from the template (`app/recolor.py`), and the agent applies it.
  - Anything the agent misses is completed automatically, and the result column says so.

### Across all pages

- **Export:** download any result as Beefree JSON or rendered HTML, or a whole sequence as a ZIP. Partial sequences keep each email's number.
- **Model selection:** pick the provider and model in the header for each run. Anthropic, OpenAI and Google are supported, with prompt caching on all three.
- **Usage:** the header counts input, output and cache tokens for the browser tab. The Single email result shows the cost of that run, using prices in `app/pricing.py`.

---

## Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (package manager)
- A **Beefree CSAPI key** — get one at [developers.beefree.io](https://developers.beefree.io)
- An API key for at least one LLM provider (Anthropic, OpenAI, or Google AI Studio)
- *(Optional)* **Beefree SDK editor credentials** (`BEE_CLIENT_ID` / `BEE_CLIENT_SECRET`) — only needed for the in-editor pages (`/builder` and `/integration`)

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
# pages (/builder and /integration), where the AI agent works alongside
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
need `BEE_API_KEY`. The **in-editor pages** (`/builder` and `/integration`), where the AI
agent edits the email live inside the embedded visual Beefree editor, additionally
require editor credentials:

```dotenv
BEE_CLIENT_ID=your_client_id_here
BEE_CLIENT_SECRET=your_client_secret_here
```

These are used by the `/integration-auth` endpoint to mint a short-lived editor
token (`auth.getbee.io/loginV2`). Both **Editor-Managed** and **API-Managed +
co-editing** session models are demonstrated on `/integration`. If they are not set,
the in-editor pages show a configuration error while every headless mode keeps
working.

Each browser gets its own random editor user ID (`demo-` followed by 32 hex
characters), issued by `/integration-auth` and kept in `localStorage`. Visitors
never share an editor user, so file manager uploads and co-editing sessions stay
separate. Clearing site data, or a browser without storage, starts a new ID.

---

## Running the app

### Local development

```bash
uv run python main.py
```

The server starts at [http://localhost:8000](http://localhost:8000) with hot reload.
Hot reload restarts the server on every Python change, which clears all open chats
and runs (see [Sessions](#sessions)).

### Production

Run a single worker without reload:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

- **Health check:** `GET /healthcheck` returns `{"status": "ok"}` instantly, with no
  external calls. Point your platform's liveness or readiness probe at it.
- **One instance only:** sessions, saved briefs and plans live in memory, so every
  request from a visitor must reach the same process. Run one worker (the
  Dockerfile pins `--workers 1`) and one container, or enable sticky sessions on the
  load balancer. With several instances, some requests land on a copy that never saw
  the data: chats say "no longer available" and sequence emails fail at random.
  `GET /healthcheck` returns an `instance` id (hostname and pid); call it a few times
  and check it never changes.
- **Streaming responses:** results stream over Server-Sent Events. Make sure your
  proxy or gateway doesn't buffer them and allows connections of several minutes.
- **Short stream URLs:** briefs, chat messages and campaign plans are sent with a
  POST and kept for 30 minutes (`run_inputs` in `app/main.py`); the stream URL only
  carries an `input_id`. This keeps URLs well under proxy limits (no 414 errors).
- **Shared limits:** every visitor's agents use your LLM and Beefree API keys, so many
  simultaneous runs can hit provider rate limits. Set spending alerts with your
  providers.
- **No built-in login:** the app has no authentication of its own. Put it behind
  your SSO or another access control before sharing the URL.

### Sessions

Chat sessions, pending runs and sequence layout checks are kept in memory
(`app/sessions.py`):
- **Expiry:** an entry is dropped after an hour without activity, so memory stays
  flat. Using a chat keeps it alive.
- **Restarts:** a restart or redeploy clears every session. Visitors see "This chat
  is no longer available" and need to reload the page.
- **Templates are safe:** the emails themselves live in Beefree, so already-generated
  results can still be downloaded.
- **Corrected palette versions:** these are kept for a day.

---

## Project structure

```
.
├── main.py              # Development entrypoint (uvicorn with hot reload)
├── pyproject.toml       # Project metadata and dependencies
├── .env.example         # Config template (copy to .env)
├── app/
│   ├── main.py          # FastAPI app: pages, SSE endpoints, downloads, /healthcheck
│   ├── agent.py         # PydanticAI agents, prompts, stream monitoring, reply cleaning
│   ├── beefree.py       # Beefree REST API client
│   ├── config.py        # Settings and the model catalog
│   ├── pricing.py       # Per-model token prices for the cost readout
│   ├── recolor.py       # Color plan for palette changes (plus the safety net)
│   ├── samples.py       # Sample template for the template-based demos
│   ├── brand.py         # Brand rule presets and Code Mode result → brand event parsing
│   ├── sessions.py      # In-memory stores that expire after an hour of inactivity
│   ├── plans.py         # Predefined campaign plans
│   ├── layouts.py       # Predefined campaign layouts
│   └── guards.py        # Sequence layout checks
├── templates/
│   ├── landing.html     # Home, /editor and /headless (one template, `section` variable)
│   ├── builder.html     # /builder (renders integration.html in builder mode)
│   ├── integration.html # /integration guide and the shared editor + chat workspace
│   ├── single.html      # Single email
│   ├── codemode.html    # Code Mode
│   ├── index.html       # Email sequence
│   ├── translate.html   # Bulk translations
│   ├── palette.html     # Color palettes
│   ├── edit.html        # Edit an email
│   ├── brand.html       # On-brand email
│   └── partials/        # htmx partials: loading states and results views
├── static/
│   ├── style.css        # Site styles
│   ├── results.css      # Shared styles for the sequence, palette and translation results
│   ├── brand.css        # On-brand email page
│   ├── ui.js            # Shared UI: inline server errors, collapse state, sample loader
│   ├── model-selection.js  # Provider/model picker
│   ├── tokens.js        # Header token counter (sessionStorage)
│   ├── favicon.svg      # Beefree mark (plus favicon-32/-192 and apple-touch-icon PNGs)
│   └── vendor/          # Self-hosted htmx and its SSE extension
└── tests/               # unittest suite
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

On-brand sessions (`/brand`) pass the whole `brandRules` object when the
template is created. Beefree then adds a "Session constraints" section to the
`beefree_execute` description, so the agent sees the preset names and limits.
Calls that skip a required preset, add a forbidden block or exceed a limit are
refused with a message the agent recovers from (these runs get 6 tool retries
instead of 3). Preset values that replace the agent's own are reported in each
result's `presetApplication`, which `app/brand.py` turns into `brand` events.

Long-running streams emit status events and poll previews while scripts execute.
Agents are instructed to split work into meaningful phases. Streams distinguish
`outcome` (completed/failed) from transport `close`; the Code Mode page treats
failed scripts as failed/partial results, not successful generations.

Translation-only restrictions are prompt guidance, not a permission boundary.
Palette runs are verified: after the agent finishes, the server compares the
template with the color plan, applies any missed change, and serves the corrected
version for the final preview and downloads (Beefree has no template update
endpoint, so the corrected copy is kept server-side).

The Single email page asks the agent to build one section per script
(`PROGRESSIVE_BUILD_INSTRUCTIONS`) so its preview fills in step by step, and to
start each script with a one-line comment that the Activity log shows. Code Mode
keeps the "as few scripts as possible" behaviour.

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
| `app/pricing.py` | Per-model token prices (update when providers change prices) |
| `app/recolor.py` | Color plan for palette runs, plus the safety net that applies missed changes |
| `app/samples.py` | Sample template served at `/sample-template` |
| `app/brand.py` | Brand rule presets, a local shape check, and the parser that turns `presetApplication` results and refusals into `brand` SSE events |
| `app/sessions.py` | `ExpiringStore`: in-memory stores that forget entries after an hour of inactivity |
| `templates/` | Jinja2 HTML templates for the web UI |
| `static/` | CSS and JS assets (including `tokens.js` for live token tracking) |

