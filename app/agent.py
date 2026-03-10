"""PydanticAI agents: planner (structured output) and executor (MCP tools + SSE).

Planner strategy (single fast LLM call)
-----------------------------------------
One Haiku call generates the sequence skeleton: titles and subject lines only
(minimal token output, ~1-3 s).  The executor prompt is then built from a
code template — zero additional LLM calls, instant.

Executor streaming strategy
----------------------------
We use `Agent.iter()` (the graph-node iterator) which yields each node
BEFORE it executes. This means:

- When `CallToolsNode` is yielded the LLM has just decided to call tools
  but the tools have NOT yet run -> we emit a status update.
- When the *next* `ModelRequestNode` is yielded the tools have completed ->
  we fetch the current template, render it to HTML, and emit a "preview" SSE
  event so the user sees the email being built progressively.

Because each SSE endpoint (/stream/{template_id}) is an independent async
generator, N email agents run as N concurrent asyncio tasks, each streaming
events to their own card in the browser.
"""

import html as html_module
from typing import AsyncIterator

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRequestNode
from pydantic_ai.mcp import MCPServerStreamableHTTP

from .config import Settings

# --- Data models -------------------------------------------------------------


class EmailSkeleton(BaseModel):
    """Lightweight plan item — no agent_prompt yet (generated in parallel)."""
    step: int
    title: str
    subject_line: str


class EmailSkeletonPlan(BaseModel):
    sequence_title: str
    emails: list[EmailSkeleton]


class EmailStep(BaseModel):
    step: int
    title: str
    subject_line: str
    agent_prompt: str


class EmailPlan(BaseModel):
    sequence_title: str
    emails: list[EmailStep]


class LayoutRowIds(BaseModel):
    """Structured output from the layout agent: IDs of the header and footer rows."""
    header_row_id: str
    footer_row_id: str


# --- System prompts -----------------------------------------------------------

PLANNER_SKELETON_PROMPT = """You are an expert email marketing strategist.

Given a campaign brief, output ONLY the sequence structure — no detailed copy.
Determine the right number of emails from the brief: follow the user exactly
if they specify a count, otherwise decide based on campaign needs.

Rules:
- `sequence_title` must be <= 50 characters.
- `title` for each email should be short and descriptive (<= 40 characters).
- `subject_line` must be concise and compelling (<= 60 characters).
- Do NOT write body copy or design instructions — only titles and subjects.
"""

LAYOUT_AGENT_SYSTEM_PROMPT = """You are building the shared header and footer for an email campaign sequence.
Your ONLY job is to create exactly TWO rows using the MCP tools:

1. HEADER row — placed at the top:
   - Full-width, dark branded background (e.g. deep navy or brand primary colour)
   - Centred logo/brand image placeholder and campaign name as text
   - Add a subtle bottom divider for separation

2. FOOTER row — placed at the bottom:
   - Full-width, light neutral background (e.g. #F5F5F5)
   - Centred placeholder text: company name, mailing address, unsubscribe link
   - Small, muted typography (12–13 px)

RULES — follow exactly:
- Create EXACTLY these two rows. Nothing else — no hero, no body, no CTA.
- Do NOT call beefree_check_template.
- After both rows are created, return your structured output with the row IDs
  you received from the tool responses as header_row_id and footer_row_id.
"""

EXECUTOR_SYSTEM_PROMPT = """You are an expert email designer working inside the Beefree headless editor.
The template already contains a shared header (top row) and a shared footer (bottom row)
built by the layout agent — do NOT recreate, modify, or delete them.

Your task: build the BODY content rows that go between the existing header and footer.

CRITICAL — ROW INSERTION ORDER:
Every body row you add MUST be inserted BEFORE the footer row (use the footer row ID
as the position reference). Never append rows to the end of the template, as that
places them after the footer. The footer must always remain the last row.

CRITICAL — PROTECTED ROWS:
The protected row IDs listed in your prompt must never be passed to any tool that
modifies, moves, or deletes rows. Only add NEW rows for the body.

You MUST complete ALL of these steps — skipping any step is a failure:
1. Add body rows BEFORE the footer row: hero title/image, body paragraphs, CTA button, dividers, spacers
2. Apply typography and colour styles consistent with the campaign brand
3. beefree_check_template — validate the final result

CRITICAL: The email is NOT done until you have called beefree_check_template.
Never leave rows empty. Build a complete body — hero, copy, CTA, and supporting content.
"""


# --- Planner -----------------------------------------------------------------


def _build_executor_prompt(
    s: EmailSkeleton,
    sequence_title: str,
    campaign_goal: str,
) -> str:
    """Build the executor prompt from a template — no LLM, instant."""
    return (
        f"Campaign: {sequence_title}\n"
        f"Email {s.step}: {s.title}\n"
        f"Subject line: {s.subject_line}\n\n"
        f"Brief: {campaign_goal}\n\n"
        "Build a complete, professional email that matches the campaign brand "
        "and tone. Include all sections: header with logo/banner, hero message, "
        "body copy, primary CTA button, supporting content, and a footer with "
        "unsubscribe link. Write all text content appropriate to this specific "
        "email's purpose and apply consistent typography, colours, and spacing."
    )


async def generate_plan(
    goal: str,
    settings: Settings,
) -> EmailPlan:
    """Single LLM call for skeleton, then template-built executor prompts."""

    skeleton_agent: Agent[None, EmailSkeletonPlan] = Agent(
        model=settings.llm_planner_model,
        output_type=EmailSkeletonPlan,
        system_prompt=PLANNER_SKELETON_PROMPT,
        retries=3,
    )
    result = await skeleton_agent.run(goal)
    skeleton = result.output

    emails = [
        EmailStep(
            step=s.step,
            title=s.title,
            subject_line=s.subject_line,
            agent_prompt=_build_executor_prompt(s, skeleton.sequence_title, goal),
        )
        for s in skeleton.emails
    ]
    return EmailPlan(sequence_title=skeleton.sequence_title, emails=emails)


# --- Shared layout agent -----------------------------------------------------


async def build_shared_layout(
    sequence_title: str,
    settings: Settings,
) -> tuple[str, str, dict]:
    """Run the layout agent once to build a shared header + footer.

    Returns (header_row_id, footer_row_id, template_json).
    The template_json can be used to seed each email template so they all
    start with the identical header and footer.
    """
    from .beefree import create_template, get_template

    layout_template_id = await create_template(settings)

    mcp = MCPServerStreamableHTTP(
        url=f"{settings.bee_api_base}/v2/sdk/mcp",
        headers={
            "Authorization": f"Bearer {settings.bee_api_key}",
            "x-bee-template-id": layout_template_id,
        },
        max_retries=3,
    )
    layout_agent: Agent[None, LayoutRowIds] = Agent(
        model=settings.llm_executor_model,
        output_type=LayoutRowIds,
        toolsets=[mcp],
        system_prompt=LAYOUT_AGENT_SYSTEM_PROMPT,
        retries=2,
    )

    result = await layout_agent.run(
        f"Build the shared header and footer for the '{sequence_title}' campaign."
    )
    layout_ids = result.output

    template_data = await get_template(layout_template_id, settings)
    template_json = template_data.get("template", template_data)

    return layout_ids.header_row_id, layout_ids.footer_row_id, template_json


def append_layout_context(
    prompt: str,
    header_row_id: str,
    footer_row_id: str,
) -> str:
    """Append protected row IDs and insertion instructions to an executor prompt."""
    return (
        prompt + "\n\n"
        "SHARED LAYOUT — already in the template:\n"
        f"  Header row ID (top):    {header_row_id}\n"
        f"  Footer row ID (bottom): {footer_row_id}\n\n"
        "Protected row IDs — NEVER delete, modify, or pass to destructive tools.\n\n"
        "INSERTION RULE — this is mandatory:\n"
        f"Every body row you add MUST be inserted BEFORE row {footer_row_id}.\n"
        "Never append rows to the end — that places them after the footer.\n"
        "The footer must always be the last row in the template."
    )


# --- SSE helpers --------------------------------------------------------------


def _preview_event(email_html: str) -> dict:
    """Wrap rendered email in an iframe (via srcdoc) so styles don't bleed."""
    escaped = html_module.escape(email_html, quote=True)
    iframe = (
        f'<iframe srcdoc="{escaped}" '
        f'sandbox="allow-same-origin" '
        f'style="width:600px;height:800px;border:none;display:block;" '
        f'title="Email preview"></iframe>'
    )
    return {"event": "preview", "data": iframe}


# --- Executor SSE stream -----------------------------------------------------


async def _fetch_preview(template_id: str, settings: Settings) -> str | None:
    """Fetch the current template and render it to HTML. Returns None on error."""
    try:
        from .beefree import get_template, render_html

        template_data = await get_template(template_id, settings)
        template_json = template_data.get("template", template_data)
        return await render_html(template_json, settings)
    except Exception:
        return None


async def stream_executor(
    template_id: str,
    prompt: str,
    settings: Settings,
) -> AsyncIterator[dict]:
    """Run one MCP executor agent and yield SSE events.

    Emits:
    - "status" events: single-line agent status (tool calls, progress)
    - "preview" events: rendered email HTML in an iframe
    - "close" event: tells HTMX to stop reconnecting
    """
    mcp = MCPServerStreamableHTTP(
        url=f"{settings.bee_api_base}/v2/sdk/mcp",
        headers={
            "Authorization": f"Bearer {settings.bee_api_key}",
            "x-bee-template-id": template_id,
        },
        max_retries=3,
    )
    executor: Agent[None, str] = Agent(
        model=settings.llm_executor_model,
        toolsets=[mcp],
        system_prompt=EXECUTOR_SYSTEM_PROMPT,
        retries=3,
    )

    try:
        async with executor.iter(prompt) as agent_run:
            async for node in agent_run:
                if isinstance(node, ModelRequestNode):
                    has_tool_returns = any(
                        getattr(p, "part_kind", "") == "tool-return"
                        for p in node.request.parts
                    )
                    if has_tool_returns:
                        preview_html = await _fetch_preview(template_id, settings)
                        if preview_html:
                            yield _preview_event(preview_html)

    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Agent error for %s: %s", template_id, exc)

    # Final preview
    preview_html = await _fetch_preview(template_id, settings)
    if preview_html:
        yield _preview_event(preview_html)

    yield {"event": "close", "data": ""}
