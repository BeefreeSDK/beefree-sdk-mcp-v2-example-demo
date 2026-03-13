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
import json as _json
import logging
from typing import AsyncIterator

log = logging.getLogger(__name__)

from pydantic import BaseModel
from pydantic_ai import Agent, CallToolsNode, ModelRequestNode
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
    body_section_count: int = 3


class EmailPlan(BaseModel):
    sequence_title: str
    emails: list[EmailStep]



# --- System prompts -----------------------------------------------------------

PLANNER_SKELETON_PROMPT = """Given a campaign brief, output the email sequence structure.

Rules:
- sequence_title: max 50 chars
- title: max 40 chars
- subject_line: max 60 chars
- No body copy, no design instructions
- Follow the stated email count exactly
"""

def _build_layout_agent_system_prompt(num_sections: int) -> str:
    """Build the layout-agent system prompt with the exact number of placeholder body rows."""
    placeholder_ids = ", ".join(f'"<id of placeholder {i+1}>"' for i in range(num_sections))
    return f"""You are building a shared base template for an email sequence.
The final template must have exactly {2 + num_sections} rows in this order:
  Row 1: header
  Rows 2-{num_sections + 1}: body placeholders ({num_sections} rows)
  Row {num_sections + 2}: footer

Read the campaign brief and decide before calling any tool:
- BRAND: the product/company name (from the brief)
- SHORT: 2-4 char uppercase abbreviation (e.g. "ACME", "STVT") — used in the logo image
- primary, accent, page_bg: pick from this table based on the campaign domain:
    B2B/SaaS:   primary=#1A3A6B  accent=#0EA5E9  page_bg=#EFF6FF
    Streaming:  primary=#0F1729  accent=#E50914   page_bg=#F4F4F4
    E-commerce: primary=#1A1A2E  accent=#F97316   page_bg=#FFF8F0
    Health:     primary=#0A4D68  accent=#22C55E   page_bg=#F0FFF4
    Finance:    primary=#0F2A4A  accent=#0EA5E9   page_bg=#F0F7FF
    Fashion:    primary=#1C1033  accent=#C084FC   page_bg=#FDF4FF
    Education:  primary=#312E81  accent=#F59E0B   page_bg=#FFFBEF
    Travel:     primary=#0C3547  accent=#06B6D4   page_bg=#F0FDFF

STEP 1 — SET GLOBAL STYLES
Call beefree_set_template_styles with:
  page_bg, content_bg=#FFFFFF, width=600, font="Helvetica Neue, Helvetica, Arial, sans-serif", text=#1A1A2E, links=accent
Call beefree_check_template.

STEP 2 — ADD HEADER ROW
Add one full-width single-column row with these properties:
  background: primary
  image (centred): https://placehold.co/160x44/XXXXXX/FFFFFF?text=SHORT
    where XXXXXX = primary colour WITHOUT the # (e.g. primary #0F1729 → XXXXXX is 0F1729)
    alt text: "BRAND logo"
  text (centred, below image): BRAND · colour #FFFFFF · bold · 18px
  column padding: 36px top, 36px bottom, 20px left, 20px right
  row border-bottom: 3px solid accent
Call beefree_check_template.

STEP 3 — ADD FOOTER ROW
Add one full-width single-column row with these properties:
  background: page_bg
  row border-top: 1px solid #E2E8F0
  text (centred): "BRAND · 123 Main St, City · © 2025 · Unsubscribe" · 12px · #64748B
  column padding: 24px top, 24px bottom
Call beefree_check_template.

STEP 4 — ADD {num_sections} PLACEHOLDER ROWS
Add {num_sections} rows, one at a time, each inserted before the footer row.
For each row (n = 1, 2, … {num_sections}):
  1. Call beefree_add_section with before_row_id = <the footer row ID from step 3>
  2. background: #FFFFFF
  3. Add one text element: "Placeholder n" · centred · 12px · colour #CBD5E1 · italic · 40px top padding · 40px bottom padding
  4. Write down the row ID returned — you need all {num_sections} IDs for the output
Call beefree_check_template after all {num_sections} rows are added.

STEP 5 — VERIFY STRUCTURE
Call beefree_get_content_hierarchy.
Count the rows. There must be exactly {2 + num_sections} rows.
If there are more, delete the extra rows. Call beefree_check_template.

OUTPUT — last line of your response, nothing after it:
{{"header_row_id": "<id>", "footer_row_id": "<id>", "placeholder_row_ids": [{placeholder_ids}]}}
"""

EXECUTOR_SYSTEM_PROMPT = """You are filling body content into an email template.

The template already has:
  - Row 1: shared header (do NOT touch)
  - Middle rows: placeholder rows (you will replace these)
  - Last row: shared footer (do NOT touch)

Your prompt tells you:
  - The subject line and campaign brief
  - The placeholder row IDs in order
  - The exact content to build in each row (BODY SECTIONS list)

Follow these steps in order:

STEP 1 — SET METADATA
Call beefree_set_email_metadata:
  subject: copy the subject line from your prompt exactly
  preheader: write a short preview text (50-90 chars) that complements the subject

STEP 2 — FILL EACH PLACEHOLDER ROW
Process each placeholder row ID from your prompt, in order:
  a. Call beefree_get_content_hierarchy — find the text element ID inside that placeholder row
  b. Delete that text element with beefree_delete_section
  c. Build the section content described in the BODY SECTIONS list into that row
     Follow the section description exactly — layout, colours, sizes, copy
  d. Call beefree_check_template

If there are more placeholder rows than body sections, delete the extra rows with beefree_delete_section.

IMPORTANT:
  - Do NOT call beefree_add_section — row count is fixed
  - Do NOT modify the header row or footer row
  - Build one section per placeholder row, in the listed order

STEP 3 — VERIFY
Call beefree_get_content_hierarchy.
When beefree_check_template passes, respond with: "Done."
"""

SINGLE_EMAIL_AGENT_SYSTEM_PROMPT = """You are an expert email design and copy assistant powered by the Beefree SDK. Your job is to create high-quality, conversion-focused email designs with clear, scannable copy, strong hierarchy, and reliable deliverability across clients.

## Core Principles (Quality First)
- **Clarity**: One primary message and one primary CTA per email.
- **Scannability**: Short paragraphs, strong headings, generous spacing.
- **Value > Features**: Lead with benefits, support with features.
- **Consistency**: Match tone and brand voice across all sections.
- **Accessibility**: Descriptive alt text, strong contrast, 14px+ body text, 44px+ buttons.
- **Compliance**: Include unsubscribe + physical address where appropriate.

## Copy & Content Standards
- Always set **subject** and **preheader** using `beefree_set_email_metadata`.
- Use a clear structure: **Header → Hero → Value Props → Proof → CTA → Footer**.
- Write crisp headlines (6–10 words) and benefit-led subheadlines.
- Use bullet lists for feature/value sections when possible.
- Include social proof or credibility cues when appropriate.
- If no copy is provided, generate industry-appropriate placeholder copy.
- Never leave empty image blocks.
- When calling `beefree_add_image`, always pass `src`. If the user
  does not provide a URL, use the placeholder URL such as e.g.
    `https://placehold.co/600x300?text=600x300`.

## Tool Usage Patterns (New Email)
1. `beefree_get_content_hierarchy`
2. `beefree_set_email_default_styles` (content width, fonts, link color)
3. Add sections, then content blocks
4. After each **major section** (hero, value props, proof, CTA, footer):
   - `beefree_check_section` on the new section
   - `beefree_get_content_hierarchy` to confirm no unexpected sections or block types were added
5. Apply styling after structure is in place
6. Final validation: `beefree_check_template`, fix issues, re-run


## Validation Workflow
- Fix critical issues first: missing alt text, broken links, insufficient contrast.
- Address warnings and suggestions after critical issues.
- Re-run validation to confirm fixes.
- Use `beefree_get_content_hierarchy` intermittently to detect accidental structure drift.
- Use `beefree_check_section` for major sections, and continue if it fails.
- If any tool fails (e.g., validation tools), continue with alternative checks
  and report the limitation.

Remember: prioritize the recipient experience and the sender's goals. Build emails that look great, read well, and perform."""


# --- Planner -----------------------------------------------------------------


def _build_executor_prompt(
    s: EmailSkeleton,
    sequence_title: str,
    campaign_goal: str,
    total_emails: int = 1,
    sections: list[str] | None = None,
) -> str:
    """Build the executor prompt — no LLM call, instant."""
    base = (
        f"Campaign: {sequence_title}\n"
        f"Email {s.step} of {total_emails}: {s.title}\n"
        f"Subject line: {s.subject_line}\n"
        f"Brief: {campaign_goal}\n"
    )
    if sections:
        numbered = "\n".join(f"  {i + 1}. {sec}" for i, sec in enumerate(sections))
        base += (
            f"\nBODY SECTIONS — fill the {len(sections)} placeholder rows in this exact order:\n"
            f"{numbered}"
        )
    else:
        base += "\nBuild a hero section, a value/features section, and a closing CTA section."
    return base


async def generate_plan(
    goal: str,
    settings: Settings,
    sections_per_step: list[list[str]] | None = None,
) -> EmailPlan:
    """Single LLM call for skeleton, then template-built executor prompts.

    If sections_per_step is provided (one list of section descriptions per email),
    those are injected verbatim into each executor prompt so the agent builds
    exactly the prescribed rows and nothing else.
    """
    skeleton_agent: Agent[None, EmailSkeletonPlan] = Agent(
        model=settings.llm_planner_model,
        output_type=EmailSkeletonPlan,
        system_prompt=PLANNER_SKELETON_PROMPT,
        retries=6,
    )
    result = await skeleton_agent.run(goal)
    skeleton = result.output

    total_emails = len(skeleton.emails)
    emails = [
        EmailStep(
            step=s.step,
            title=s.title,
            subject_line=s.subject_line,
            body_section_count=(
                len(sections_per_step[s.step - 1])
                if sections_per_step and s.step - 1 < len(sections_per_step)
                else 3
            ),
            agent_prompt=_build_executor_prompt(
                s,
                skeleton.sequence_title,
                goal,
                total_emails=total_emails,
                sections=sections_per_step[s.step - 1]
                if sections_per_step and s.step - 1 < len(sections_per_step)
                else None,
            ),
        )
        for s in skeleton.emails
    ]
    return EmailPlan(sequence_title=skeleton.sequence_title, emails=emails), result.usage()


# --- Shared layout agent -----------------------------------------------------


async def build_shared_layout(
    sequence_title: str,
    settings: Settings,
    num_sections: int = 3,
) -> tuple[str, str, list[str], dict]:
    """Run the layout agent once to build a shared header, placeholder body rows, and footer.

    Returns (header_row_id, footer_row_id, placeholder_row_ids, template_json).
    The template_json can be used to seed each email template so they all
    start with the identical structure.

    The agent returns plain text ending with a JSON block containing all row IDs.
    We parse that block rather than using structured output, which avoids a
    Gemini API limitation that rejects the combination of tool schemas + response
    schema when the total constraint state count is too large.
    """
    import json as _json
    from .beefree import create_template, get_template

    layout_template_id = await create_template(settings)

    mcp = MCPServerStreamableHTTP(
        url=f"{settings.bee_api_base}/v2/sdk/mcp",
        headers={
            "Authorization": f"Bearer {settings.bee_api_key}",
            "x-bee-template-id": layout_template_id,
        },
        max_retries=5,
        timeout=60,
        read_timeout=300,
    )
    layout_agent: Agent[None, str] = Agent(
        model=settings.llm_layout_model,
        toolsets=[mcp],
        system_prompt=_build_layout_agent_system_prompt(num_sections),
        retries=6,
    )

    result = await layout_agent.run(
        f"Build the shared header, {num_sections} placeholder body row(s), and footer "
        f"for the '{sequence_title}' campaign."
    )
    layout_usage = result.usage()
    text = result.output

    # Extract the row-ID JSON block the agent was instructed to append.
    # Use JSONDecoder.raw_decode scanning backwards for the last valid JSON object
    # that contains 'header_row_id' — handles the nested array in placeholder_row_ids.
    decoder = _json.JSONDecoder()
    ids: dict | None = None
    idx = len(text) - 1
    while idx >= 0:
        idx = text.rfind("{", 0, idx + 1)
        if idx < 0:
            break
        try:
            obj, _ = decoder.raw_decode(text, idx)
            if isinstance(obj, dict) and "header_row_id" in obj:
                ids = obj
                break
        except _json.JSONDecodeError:
            pass
        idx -= 1

    if ids is None:
        raise ValueError(
            f"Layout agent did not return row IDs in the expected format. "
            f"Response tail: {text[-300:]!r}"
        )

    header_row_id: str = ids["header_row_id"]
    footer_row_id: str = ids["footer_row_id"]
    placeholder_row_ids: list[str] = ids.get("placeholder_row_ids", [])

    template_data = await get_template(layout_template_id, settings)
    template_json = template_data.get("template", template_data)

    return header_row_id, footer_row_id, placeholder_row_ids, template_json, layout_usage


def append_layout_context(
    prompt: str,
    header_row_id: str,
    footer_row_id: str,
    placeholder_row_ids: list[str],
) -> str:
    """Append protected row IDs and placeholder editing instructions to an executor prompt."""
    numbered = "\n".join(
        f"  Placeholder {i + 1}: {rid}" for i, rid in enumerate(placeholder_row_ids)
    )
    return (
        prompt + "\n\n"
        "SHARED LAYOUT — already in the template:\n"
        f"  Header row ID (top):    {header_row_id}\n"
        f"  Footer row ID (bottom): {footer_row_id}\n\n"
        "Protected row IDs — NEVER delete, modify, or pass to destructive tools.\n\n"
        f"PLACEHOLDER BODY ROWS — fill these {len(placeholder_row_ids)} row(s) with real content:\n"
        f"{numbered}\n\n"
        "RULES — this is mandatory:\n"
        "- Work through the placeholder rows in the order listed above.\n"
        "- Delete each placeholder's existing text element, then add real content to that row.\n"
        "- NEVER call beefree_add_section — the row structure is fixed.\n"
        "- If you have fewer REQUIRED BODY SECTIONS than placeholder rows, delete the extra\n"
        "  placeholder rows using beefree_delete_section."
    )


# --- SSE helpers --------------------------------------------------------------


def _tokens_event(usage) -> dict | None:
    """Build a tokens SSE event from a PydanticAI usage object. Returns None if no usage."""
    if usage is None:
        return None
    return {
        "event": "tokens",
        "data": _json.dumps({
            "input": usage.input_tokens or 0,
            "output": usage.output_tokens or 0,
        }),
    }


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
        max_retries=5,
        timeout=60,
        read_timeout=300,
    )
    executor: Agent[None, str] = Agent(
        model=settings.llm_executor_model,
        toolsets=[mcp],
        system_prompt=EXECUTOR_SYSTEM_PROMPT,
        retries=6,
    )

    failed = False
    try:
        async with executor.iter(prompt) as agent_run:
            async for node in agent_run:
                try:
                    if isinstance(node, CallToolsNode):
                        tok = _tokens_event(node.model_response.usage)
                        if tok:
                            yield tok
                    if isinstance(node, ModelRequestNode):
                        has_tool_returns = any(
                            getattr(p, "part_kind", "") == "tool-return"
                            for p in node.request.parts
                        )
                        if has_tool_returns:
                            preview_html = await _fetch_preview(template_id, settings)
                            if preview_html:
                                yield _preview_event(preview_html)
                except Exception as node_exc:
                    log.warning("Executor node error (continuing): %s", node_exc)
    except Exception as exc:
        log.error("Executor agent error for %s: %s", template_id, exc)
        failed = True

    # Always fetch a final preview — partial work is still useful to show
    try:
        preview_html = await _fetch_preview(template_id, settings)
        if preview_html:
            yield _preview_event(preview_html)
        elif failed:
            yield {
                "event": "preview",
                "data": "<div class='plan-error' style='padding:1.5rem'>Agent failed — retries exhausted. The template may be partially built.</div>",
            }
    except Exception:
        pass

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

When beefree_check_template has confirmed all translations are applied,
stop calling tools and respond with: "Done."
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
        timeout=60,
        read_timeout=300,
    )
    agent: Agent[None, str] = Agent(
        model=settings.llm_planner_model,  # fast/cheap model for the active provider
        toolsets=[mcp],
        system_prompt=TRANSLATION_AGENT_SYSTEM_PROMPT.format(language=language),
        retries=3,
    )

    try:
        async with agent.iter(
            f"Translate all text content in this email template into {language}."
        ) as agent_run:
            async for node in agent_run:
                try:
                    if isinstance(node, CallToolsNode):
                        tok = _tokens_event(node.model_response.usage)
                        if tok:
                            yield tok
                    if isinstance(node, ModelRequestNode):
                        has_tool_returns = any(
                            getattr(p, "part_kind", "") == "tool-return"
                            for p in node.request.parts
                        )
                        if has_tool_returns:
                            preview_html = await _fetch_preview(template_id, settings)
                            if preview_html:
                                yield _preview_event(preview_html)
                except Exception as node_exc:
                    log.warning("Translation node error (continuing): %s", node_exc)
    except Exception as exc:
        log.error("Translation agent error for %s (%s): %s", template_id, language, exc)

    try:
        preview_html = await _fetch_preview(template_id, settings)
        if preview_html:
            yield _preview_event(preview_html)
    except Exception:
        pass

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
When all four phases are complete and the final beefree_check_template has passed,
stop calling tools and respond with: "Done."
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
        timeout=60,
        read_timeout=300,
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
                try:
                    if isinstance(node, CallToolsNode):
                        tok = _tokens_event(node.model_response.usage)
                        if tok:
                            yield tok
                    if isinstance(node, ModelRequestNode):
                        has_tool_returns = any(
                            getattr(p, "part_kind", "") == "tool-return"
                            for p in node.request.parts
                        )
                        if has_tool_returns:
                            preview_html = await _fetch_preview(template_id, settings)
                            if preview_html:
                                yield _preview_event(preview_html)
                except Exception as node_exc:
                    log.warning("Palette node error (continuing): %s", node_exc)
    except Exception as exc:
        log.error("Palette agent error for %s (%s): %s", template_id, palette["name"], exc)

    try:
        preview_html = await _fetch_preview(template_id, settings)
        if preview_html:
            yield _preview_event(preview_html)
    except Exception:
        pass

    yield {"event": "close", "data": ""}


# --- Edit email executor SSE stream ------------------------------------------

EDIT_AGENT_SYSTEM_PROMPT = """You are an expert email design and copy assistant with full access to the Beefree SDK tools.
An email template is already loaded in the editor. The user will give you instructions to modify it.

You can perform any modification the user requests:
- Add or remove sections using beefree_add_section and beefree_delete_section
- Change global styles (colors, fonts, spacing) using beefree_set_template_styles
- Update email metadata (subject line, preheader) using beefree_set_email_metadata
- Inspect the current template structure using beefree_get_content_hierarchy
- Validate the template using beefree_check_template

RULES:
- Always call beefree_check_template after making structural or style changes to validate the result.
- When the user asks you to inspect or describe the email, call beefree_get_content_hierarchy first.
- Keep changes focused on what the user asked — don't redesign the whole email unless explicitly told to.
- When done, respond with a short, friendly summary of what you changed (1-3 sentences).
- If you cannot fulfil a request with the available tools, explain why clearly.
"""


async def stream_edit_executor(
    template_id: str,
    message: str,
    settings: Settings,
    message_history=None,
    out_messages: list | None = None,
) -> AsyncIterator[dict]:
    """Run one edit-agent turn and yield SSE events.

    Emits:
    - "preview" events: rendered email HTML in an iframe
    - "agent-message" event: the agent's text reply for the chat UI
    - "close" event: signals the client to stop listening

    If *out_messages* (a mutable list) is supplied, it is populated with the full
    conversation history after the run so the caller can persist it for the next turn.
    """
    mcp = MCPServerStreamableHTTP(
        url=f"{settings.bee_api_base}/v2/sdk/mcp",
        headers={
            "Authorization": f"Bearer {settings.bee_api_key}",
            "x-bee-template-id": template_id,
        },
        max_retries=3,
        timeout=60,
        read_timeout=300,
    )
    agent: Agent[None, str] = Agent(
        model=settings.llm_executor_model,
        toolsets=[mcp],
        system_prompt=EDIT_AGENT_SYSTEM_PROMPT,
        retries=2,
    )

    agent_text = ""
    try:
        async with agent.iter(
            message, message_history=message_history or []
        ) as agent_run:
            async for node in agent_run:
                try:
                    if isinstance(node, CallToolsNode):
                        tok = _tokens_event(node.model_response.usage)
                        if tok:
                            yield tok
                    if isinstance(node, ModelRequestNode):
                        has_tool_returns = any(
                            getattr(p, "part_kind", "") == "tool-return"
                            for p in node.request.parts
                        )
                        if has_tool_returns:
                            preview_html = await _fetch_preview(template_id, settings)
                            if preview_html:
                                yield _preview_event(preview_html)
                except Exception as node_exc:
                    log.warning("Edit agent node error (continuing): %s", node_exc)

            if agent_run.result:
                agent_text = agent_run.result.output or "Done."
                if out_messages is not None:
                    out_messages.extend(agent_run.result.all_messages())

    except Exception as exc:
        log.error("Edit agent error for %s: %s", template_id, exc)
        agent_text = f"I encountered an issue while editing: {exc}"

    # Final preview after agent finishes
    try:
        preview_html = await _fetch_preview(template_id, settings)
        if preview_html:
            yield _preview_event(preview_html)
    except Exception:
        pass

    # Escape and format the agent reply as safe HTML
    escaped = html_module.escape(agent_text)
    formatted = escaped.replace("\n\n", "</p><p>").replace("\n", "<br>")
    yield {"event": "agent-message", "data": f"<p>{formatted}</p>"}
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
    - "preview" events: rendered email HTML in an iframe (same as other executors)
    - "close" event: tells HTMX to stop reconnecting
    """
    mcp = MCPServerStreamableHTTP(
        url=f"{settings.bee_api_base}/v2/sdk/mcp",
        headers={
            "Authorization": f"Bearer {settings.bee_api_key}",
            "x-bee-template-id": template_id,
        },
        max_retries=3,
        timeout=60,
        read_timeout=300,
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
                try:
                    if isinstance(node, CallToolsNode):
                        tok = _tokens_event(node.model_response.usage)
                        if tok:
                            yield tok
                    if isinstance(node, ModelRequestNode):
                        has_tool_returns = any(
                            getattr(p, "part_kind", "") == "tool-return"
                            for p in node.request.parts
                        )
                        if has_tool_returns:
                            preview_html = await _fetch_preview(template_id, settings)
                            if preview_html:
                                yield _preview_event(preview_html)
                except Exception as node_exc:
                    log.warning("Single agent node error (continuing): %s", node_exc)
    except Exception as exc:
        log.error("Single agent error for %s: %s", template_id, exc)

    try:
        preview_html = await _fetch_preview(template_id, settings)
        if preview_html:
            yield _preview_event(preview_html)
    except Exception:
        pass

    yield {"event": "close", "data": ""}
