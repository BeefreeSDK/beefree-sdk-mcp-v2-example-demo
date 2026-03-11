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

LAYOUT_AGENT_SYSTEM_PROMPT = """You are building the shared foundation for an email campaign sequence.
Your job is to set up three things, in this exact order:

STEP 1 — GLOBAL STYLES:
Set the template-level styles so every email in the sequence shares the same base:
- Page background colour (e.g. light grey #F4F4F4)
- Content area background colour (white #FFFFFF)
- Default font family (a clean web-safe sans-serif stack)
- Default body text colour and size
- Default link colour
- Content area width (600 px)

STEP 2 — HEADER ROW (top of every email):
- Full-width, dark branded background (e.g. deep navy #26045D or similar)
- Centred logo/brand image placeholder and campaign name as text
- Subtle bottom divider for visual separation

STEP 3 — FOOTER ROW (bottom of every email):
- Full-width, light neutral background (e.g. #F5F5F5)
- Centred placeholder text: company name, mailing address, unsubscribe link
- Small, muted typography (12-13 px)

RULES — follow exactly:
- Complete the steps in order: global styles first, then header row, then footer row.
- Do NOT add any body content, hero sections, CTAs, or anything between header and footer.
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

SINGLE_EMAIL_AGENT_SYSTEM_PROMPT = """You are an expert email designer working inside the Beefree headless editor.
Your task is to build one complete, production-ready email from scratch.

Complete ALL of these steps in this exact order:

STEP 1 — GLOBAL STYLES:
Set template-level styles:
- Page background colour (light grey #F4F4F4)
- Content area background colour (white #FFFFFF)
- Default font family (a clean web-safe sans-serif stack)
- Default body text colour and size
- Default link colour
- Content area width (600 px)

STEP 2 — HEADER ROW:
- Full-width branded background (deep navy #26045D or similar)
- Centred logo/brand image placeholder and email title as text
- Subtle bottom divider for visual separation

STEP 3 — BODY ROWS:
Build compelling body content:
- Hero section: large headline and supporting image placeholder
- Body copy: 2–3 paragraphs relevant to the brief
- Primary CTA button: clear, action-oriented label
- Supporting content sections as appropriate (features, benefits, highlights)
- Dividers and spacers for visual breathing room

STEP 4 — FOOTER ROW:
- Full-width light neutral background (#F5F5F5)
- Centred placeholder text: company name, mailing address, unsubscribe link
- Small, muted typography (12–13 px)

STEP 5 — VALIDATE:
Call beefree_check_template to validate the final result.

RULES:
- Complete all steps in order. Do NOT skip any step.
- Do NOT leave any content areas empty.
- Apply consistent typography, colours, and spacing throughout.
- Write all text content appropriate to the email's purpose.
- The email is NOT done until beefree_check_template has been called.
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
        model=settings.llm_layout_model,
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


def _single_preview_event(email_html: str) -> dict:
    """Wrap rendered email in a scalable iframe div for the single-generation view."""
    escaped = html_module.escape(email_html, quote=True)
    content = (
        '<div class="single-iframe-wrap">'
        f'<iframe srcdoc="{escaped}" '
        f'sandbox="allow-same-origin" '
        f'style="width:600px;height:1400px;border:none;display:block;" '
        f'title="Email preview"></iframe>'
        '</div>'
    )
    return {"event": "preview", "data": content}


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
    mcp_url: str | None = None,
) -> AsyncIterator[dict]:
    """Run one MCP executor agent and yield SSE events.

    Emits:
    - "status" events: single-line agent status (tool calls, progress)
    - "preview" events: rendered email HTML in an iframe
    - "close" event: tells HTMX to stop reconnecting
    """
    effective_url = mcp_url or f"{settings.bee_api_base}/v2/sdk/mcp"
    mcp = MCPServerStreamableHTTP(
        url=effective_url,
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


# --- Translation executor SSE stream -----------------------------------------

TRANSLATION_AGENT_SYSTEM_PROMPT = """You are a professional translation agent working inside the Beefree headless editor.
The email template is already fully built. Your only job is to translate every visible text string into {language}.

WHAT TO TRANSLATE:
- Headings, titles, and subheadings
- All paragraph and body copy
- Button labels and CTA text
- Link text
- Image alt text

STRICT RULES — you must NEVER:
- Add, remove, reorder, or restructure any rows, columns, or content blocks
- Change any colors, fonts, spacing, padding, borders, or any visual/style property
- Call beefree_add_section, beefree_delete_section, or beefree_set_template_styles
- Leave any original-language text untranslated (brand names and proper nouns are the only exception)

Workflow:
1. Examine the template to identify all text content blocks
2. Translate each block's text into {language}, preserving formatting and intent
3. Update each block using the appropriate text-editing tool
4. Call beefree_check_template to validate

The task is NOT complete until beefree_check_template has been called.
"""


async def stream_translation_executor(
    template_id: str,
    language: str,
    settings: Settings,
) -> AsyncIterator[dict]:
    """Run one translation agent for a specific language and yield SSE events.

    The agent only edits existing text content — it never adds or removes rows.

    Emits:
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
    agent: Agent[None, str] = Agent(
        model=settings.llm_executor_model,
        toolsets=[mcp],
        system_prompt=TRANSLATION_AGENT_SYSTEM_PROMPT.format(language=language),
        retries=3,
    )

    try:
        async with agent.iter(
            f"Translate all text content in this email template into {language}."
        ) as agent_run:
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
        logging.getLogger(__name__).error(
            "Translation agent error for %s (%s): %s", template_id, language, exc
        )

    # Final preview
    preview_html = await _fetch_preview(template_id, settings)
    if preview_html:
        yield _preview_event(preview_html)

    yield {"event": "close", "data": ""}


# --- Color palette executor SSE stream ---------------------------------------

PALETTE_AGENT_SYSTEM_PROMPT = """You are a color palette agent working inside the Beefree headless editor.
The email template is already fully built. Your only task is to restyle it using the exact color palette below.

TARGET PALETTE — {palette_name}:
  Page background:      {page_bg}
  Content area:         {content_bg}
  Header row bg:        {header_bg}
  Heading text:         {heading}
  Body text:            {text}
  Buttons / CTAs:       {primary}
  Links / accents:      {accent}
  Footer row bg:        {footer_bg}
  Footer text:          {footer_text}

WORKFLOW — follow this order exactly, calling beefree_check_template after each phase:
1. Call beefree_set_template_styles to set page background to {page_bg}, content area background to {content_bg}, default text color to {text}, and link color to {accent}. Then call beefree_check_template.
2. Identify the first row (header) and update its background color to {header_bg}. Update any text inside it to a contrasting light color if needed. Then call beefree_check_template.
3. Update all heading elements throughout the template to use {heading}. Update all button and CTA elements to use {primary} as the background color. Then call beefree_check_template.
4. Identify the last row (footer) and update its background to {footer_bg} and its text to {footer_text}. Then call beefree_check_template.

STRICT RULES — you must NEVER:
- Change any text content, copy, or wording
- Add, remove, reorder, or restructure rows, columns, or content blocks
- Change fonts, font sizes, spacing, padding, or any layout property
- Call beefree_add_section or beefree_delete_section

beefree_check_template MUST be called after every phase, not just at the end.
"""


async def stream_palette_executor(
    template_id: str,
    palette: dict,
    settings: "Settings",
) -> "AsyncIterator[dict]":
    """Run one palette agent and yield SSE events.

    The agent applies the target color palette to an existing template —
    changing only colors, never layout, text content, or structure.

    Emits:
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
    system_prompt = PALETTE_AGENT_SYSTEM_PROMPT.format(
        palette_name=palette["name"],
        **palette["colors"],
    )
    agent: Agent[None, str] = Agent(
        model=settings.llm_executor_model,
        toolsets=[mcp],
        system_prompt=system_prompt,
        retries=3,
    )

    try:
        async with agent.iter(
            f"Apply the '{palette['name']}' color palette to this email template."
        ) as agent_run:
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
        logging.getLogger(__name__).error(
            "Palette agent error for %s (%s): %s", template_id, palette["name"], exc
        )

    # Final preview
    preview_html = await _fetch_preview(template_id, settings)
    if preview_html:
        yield _preview_event(preview_html)

    yield {"event": "close", "data": ""}


# --- Single email executor SSE stream ----------------------------------------


async def stream_single_executor(
    template_id: str,
    brief: str,
    settings: Settings,
) -> AsyncIterator[dict]:
    """Run one single-email agent and yield SSE events.

    The agent builds everything from scratch: global styles, header, body
    content, footer, and final validation — all in one pass.

    Emits:
    - "preview" events: rendered email HTML in a scalable iframe wrapper
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
    agent: Agent[None, str] = Agent(
        model=settings.llm_executor_model,
        toolsets=[mcp],
        system_prompt=SINGLE_EMAIL_AGENT_SYSTEM_PROMPT,
        retries=3,
    )

    try:
        async with agent.iter(brief) as agent_run:
            async for node in agent_run:
                if isinstance(node, ModelRequestNode):
                    has_tool_returns = any(
                        getattr(p, "part_kind", "") == "tool-return"
                        for p in node.request.parts
                    )
                    if has_tool_returns:
                        preview_html = await _fetch_preview(template_id, settings)
                        if preview_html:
                            yield _single_preview_event(preview_html)

    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Single agent error for %s: %s", template_id, exc)

    # Final preview
    preview_html = await _fetch_preview(template_id, settings)
    if preview_html:
        yield _single_preview_event(preview_html)

    yield {"event": "close", "data": ""}
