import html as html_module
import asyncio
import json as _json
import logging
import re
from functools import wraps
from typing import AsyncIterator

log = logging.getLogger(__name__)

from pydantic import BaseModel
from pydantic_ai import Agent, CallToolsNode, ModelRequestNode
from pydantic_ai.mcp import MCPServerStreamableHTTP
# Applied to every agent — provider-prefixed keys are silently ignored by other providers.
_CACHE_SETTINGS: dict = {
    # Anthropic — explicit prompt caching
    "anthropic_cache_instructions":    True,  # cache the (large) system prompt
    "anthropic_cache_tool_definitions": True,  # cache MCP tool schema list
    "anthropic_cache_messages":        True,  # cache conversation history between rounds
}

from .config import Settings
from .guards import validate_sequence
from .pricing import request_cost


CODE_MODE_INSTRUCTIONS = """

EXECUTION MODE: Beefree Code Mode.
The MCP server exposes beefree_execute with a code string rather than individual
beefree_* tools. Read its description and schema for the supported script API.
All beefree_* operations named above or in the user request refer to operations
to perform INSIDE that script using the documented API, not standalone MCP calls.
Do not invent imports, API names, or method signatures.
Batch related operations in a script to reduce round trips.
Prefer separate scripts for inspection, each major design phase, and validation
so the host can show meaningful progress and intermediate previews.
The sandbox API is synchronous. Use documented positional addressing when
building new content; do not assume mutation calls return live IDs. Use out(data)
for results. Respect the tool's operation, code-size, and timeout limits. Inspect
execution results before continuing. After a partial failure, inspect the current
template before retrying; do not blindly repeat mutations that already succeeded.
Keep every operation-specific restriction above, including protected rows and
text-only or color-only editing rules, inside scripts too. Validate the result.
Script output is for you, not the user: never paste script results, JSON or
element IDs into your final reply.
"""


def monitored_stream(function):
    @wraps(function)
    async def wrapped(template_id, *args, **kwargs):
        settings = kwargs.get("settings")
        if settings is None:
            settings = args[1]
        guard = kwargs.pop("sequence_policy", None)
        source = function(template_id, *args, **kwargs)
        events = asyncio.Queue(maxsize=1)
        producer = None
        pending = None
        failed = False
        review = False

        async def produce():
            try:
                try:
                    async for event in source:
                        await events.put(event)
                finally:
                    await source.aclose()
            except Exception as exc:
                await events.put(exc)
            else:
                await events.put(None)

        async def check_layout():
            if guard:
                from .beefree import get_template
                current = await get_template(template_id, settings)
                validate_sequence(current.get("template", current), guard)

        try:
            yield {"event": "status", "data": "Agent running"}
            producer = asyncio.create_task(produce())
            pending = asyncio.create_task(events.get())
            while True:
                done, _ = await asyncio.wait({pending}, timeout=5)
                if not done:
                    yield {"event": "status", "data": "Agent working; checking preview"}
                    try:
                        await check_layout()
                    except Exception as exc:
                        log.warning("Preview check for %s: %s", template_id, exc)
                        yield {"event": "validation-warning", "data": (
                            "Layout check warning on this email; generation continues."
                        )}
                    else:
                        if guard:
                            yield {"event": "validation-warning", "data": ""}
                    preview = await _fetch_preview(template_id, settings)
                    if preview:
                        yield _preview_event(preview)
                    continue
                event = pending.result()
                if event is None:
                    break
                if isinstance(event, Exception):
                    raise event
                if event["event"] == "agent-error":
                    failed = True
                    event = {**event, "data": html_module.escape(event["data"])}
                if event["event"] == "preview" and guard:
                    try:
                        await check_layout()
                    except Exception as exc:
                        log.warning("Preview check for %s: %s", template_id, exc)
                        yield {"event": "validation-warning", "data": (
                            "Layout check warning on this email; generation continues."
                        )}
                    else:
                        yield {"event": "validation-warning", "data": ""}
                if event["event"] != "close":
                    yield event
                pending = asyncio.create_task(events.get())
            try:
                await check_layout()
            except Exception as exc:
                review = True
                log.warning("Final sequence check for %s: %s", template_id, exc)
                yield {"event": "validation-warning", "data": (
                    "Generation finished with a layout warning. "
                    + html_module.escape(str(exc))
                    + " You can still download the result."
                )}
            else:
                yield {"event": "validation-warning", "data": ""}
            yield {"event": "outcome", "data": _json.dumps({
                "status": "failed" if failed else "needs_review" if review else "completed",
            })}
        except Exception as exc:
            yield {"event": "agent-error", "data": html_module.escape(_format_exc(exc))}
            yield {"event": "outcome", "data": '{"status":"failed"}'}
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            if producer is not None:
                if not producer.done():
                    producer.cancel()
                await asyncio.gather(producer, return_exceptions=True)
        yield {"event": "close", "data": ""}
    return wrapped


def create_code_mode_mcp(
    template_id: str,
    settings: Settings,
    *,
    max_retries: int = 3,
) -> MCPServerStreamableHTTP:
    return MCPServerStreamableHTTP(
        url=f"{settings.bee_api_base.rstrip('/')}/v2/sdk/mcp/codemode",
        headers={
            "Authorization": f"Bearer {settings.bee_api_key}",
            "x-bee-template-id": template_id,
        },
        max_retries=max_retries,
        timeout=60,
        read_timeout=300,
    )


def _format_exc(exc: BaseException) -> str:
    """Flatten an exception into a readable message, unwrapping ExceptionGroups.

    asyncio/anyio TaskGroups (used by the MCP transport) raise an ExceptionGroup
    whose ``str()`` is just "unhandled errors in a TaskGroup (N sub-exception)" —
    which hides the real cause. Recurse into the group to surface the underlying
    error(s).
    """
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(_format_exc(sub) for sub in exc.exceptions)
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


# --- Data models -------------------------------------------------------------


class EmailSkeleton(BaseModel):
    """Lightweight plan item used to build an executor prompt."""
    step: int
    title: str
    subject_line: str


class EmailStep(BaseModel):
    step: int
    title: str
    subject_line: str
    agent_prompt: str
    body_section_count: int = 3


class EmailPlan(BaseModel):
    sequence_title: str
    emails: list[EmailStep]
    preset_id: str = ""



# --- System prompts -----------------------------------------------------------

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


# Appended after CODE_MODE_INSTRUCTIONS so it overrides the general "batch
# related operations" advice for pages that show a live preview.
PROGRESSIVE_BUILD_INSTRUCTIONS = """

PROGRESSIVE BUILD: the user is watching a live preview that refreshes after
every script you run. Build the email in several small scripts, one visible
step at a time, never as one large script:
1. First script: set the subject and preheader, the default styles, and the
   header. Keep it short so the first preview appears within seconds.
2. Then one script per body section, in reading order (hero, value
   propositions, proof, closing CTA, and so on). Each script adds one
   finished section, with its content and styling, and runs that section's
   checks.
3. Then one script for the footer.
4. Finish with validation and any fixes.
Never put more than one major section in a script.
Start every script with a single-line comment that tells the user, in plain
language, what this step does, for example:
// Adding the hero section with the headline and main button
"""


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


_FENCED = re.compile(r"```.*?(```|$)", re.S)
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_TECH_SENTENCE = re.compile(
    r"[^.\n]*\b(content hierarchy|hierarchy json|block ids?|section ids?|element ids?|json)\b[^.\n]*[.:]?", re.I)


def _strip_json_blocks(text: str) -> str:
    """Remove multi-line {...} / [...] blocks (pasted JSON) from a reply."""
    out, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch in "{[" and (i == 0 or text[i - 1] == "\n"):
            close = "}" if ch == "{" else "]"
            depth, j = 0, i
            while j < len(text):
                if text[j] in "{[":
                    depth += 1
                elif text[j] in "}]":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            block = text[i:j + 1]
            if "\n" in block and len(block) > 40 and text[j:j + 1] == close:
                i = j + 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def clean_reply(text: str) -> str:
    """Keep the chat reply readable for non-developers: drop code, JSON and IDs."""
    cleaned = _FENCED.sub("", text or "")
    cleaned = _strip_json_blocks(cleaned)
    cleaned = _TECH_SENTENCE.sub("", cleaned)
    cleaned = _UUID.sub("", cleaned)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)          # inline code -> plain text
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or "Done. The email in the editor has been updated."


def _tokens_event(usage, model: str) -> dict | None:
    """Build a tokens SSE event from a PydanticAI usage object. Returns None if no usage.

    `cost` is the request's USD cost, or null when `model` has no known price.
    """
    if usage is None:
        return None
    counts = {
        "input": usage.input_tokens or 0,
        "output": usage.output_tokens or 0,
        "cache_write": usage.cache_write_tokens or 0,
        "cache_read": usage.cache_read_tokens or 0,
    }
    cost = request_cost(
        model,
        input_tokens=counts["input"],
        output_tokens=counts["output"],
        cache_write_tokens=counts["cache_write"],
        cache_read_tokens=counts["cache_read"],
    )
    return {"event": "tokens", "data": _json.dumps({**counts, "cost": cost})}


def _preview_event(email_html: str) -> dict:
    """Wrap rendered email in an iframe (via srcdoc) so styles don't bleed."""
    escaped = html_module.escape(email_html, quote=True)
    iframe = (
        f'<iframe srcdoc="{escaped}" '
        f'sandbox="allow-same-origin" '
        f'style="width:700px;height:800px;border:none;display:block;" '
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


@monitored_stream
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
    mcp = create_code_mode_mcp(
        template_id, settings, max_retries=5,
    )
    executor: Agent[None, str] = Agent(
        model=settings.agent_model,
        toolsets=[mcp],
        system_prompt=EXECUTOR_SYSTEM_PROMPT + CODE_MODE_INSTRUCTIONS,
        model_settings=_CACHE_SETTINGS,
        retries=6,
    )

    failed = False
    try:
        async with executor.iter(prompt) as agent_run:
            async for node in agent_run:
                try:
                    if isinstance(node, CallToolsNode):
                        tok = _tokens_event(node.model_response.usage, settings.llm_model)
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
        detail = _format_exc(exc)
        log.error("Executor agent error for %s: %s", template_id, detail)
        failed = True
        yield {"event": "agent-error", "data": detail}

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


@monitored_stream
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
    mcp = create_code_mode_mcp(template_id, settings)
    agent: Agent[None, str] = Agent(
        model=settings.agent_model,
        toolsets=[mcp],
        system_prompt=TRANSLATION_AGENT_SYSTEM_PROMPT.format(language=language) + CODE_MODE_INSTRUCTIONS,
        model_settings=_CACHE_SETTINGS,
        retries=3,
    )

    try:
        async with agent.iter(
            f"Translate all text content in this email template into {language}."
        ) as agent_run:
            async for node in agent_run:
                try:
                    if isinstance(node, CallToolsNode):
                        tok = _tokens_event(node.model_response.usage, settings.llm_model)
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
        detail = _format_exc(exc)
        log.error("Translation agent error for %s (%s): %s", template_id, language, detail)
        yield {"event": "agent-error", "data": detail}

    try:
        preview_html = await _fetch_preview(template_id, settings)
        if preview_html:
            yield _preview_event(preview_html)
    except Exception:
        pass

    yield {"event": "close", "data": ""}


# --- Color palette executor SSE stream ---------------------------------------

PALETTE_AGENT_SYSTEM_PROMPT = """You are a color palette agent working inside the Beefree headless editor.
The email template is already fully built. Your only task is to recolor it with
the "{palette_name}" palette by applying the color plan below.

The plan was computed from the template and lists EVERY color that must change:
the page background, each row's full-width container background AND its content
area background, text colors, link colors, buttons, and colors written inline
inside text HTML (style="color:..."). Rows are numbered from the top; the first
row is the header and the last row is the footer.

COLOR PLAN ({change_count} changes):
{plan}

HOW TO WORK:
1. Read the template structure first so you can find each row and block in order.
2. Apply the plan in a few scripts, roughly top to bottom (global styles and the
   header first, then the body rows, then the footer), so the preview updates as
   you go. For inline colors, edit the text's HTML and replace only the color value.
3. Re-read the template and compare it with the plan. Fix anything still using an
   old color, then run the template check.

STRICT RULES — you must NEVER:
- Change any text content, copy, or wording
- Add, remove, reorder, or restructure rows, columns, or content blocks
- Change fonts, font sizes, spacing, padding, or any layout property
- Use colors that are not in the plan

When every change in the plan is applied, stop calling tools and respond with: "Done."
"""


@monitored_stream
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
    from .beefree import get_template, render_html
    from .recolor import apply_plan, build_plan, describe, missed

    def _template_of(data: dict) -> dict:
        return data.get("template", data)

    # Compute the full color plan from the template before the agent starts.
    plan = []
    try:
        plan = build_plan(_template_of(await get_template(template_id, settings)), palette["colors"])
    except Exception as exc:
        log.warning("Could not build a color plan for %s: %s", template_id, exc)

    mcp = create_code_mode_mcp(template_id, settings)
    system_prompt = PALETTE_AGENT_SYSTEM_PROMPT.format(
        palette_name=palette["name"],
        change_count=len(plan),
        plan=describe(plan) or "(no color plan could be computed: recolor backgrounds, text and links with the palette)",
    )
    agent: Agent[None, str] = Agent(
        model=settings.agent_model,
        toolsets=[mcp],
        system_prompt=system_prompt + CODE_MODE_INSTRUCTIONS,
        model_settings=_CACHE_SETTINGS,
        retries=3,
    )

    try:
        async with agent.iter(
            f"Apply the '{palette['name']}' color palette to this email template."
        ) as agent_run:
            async for node in agent_run:
                try:
                    if isinstance(node, CallToolsNode):
                        tok = _tokens_event(node.model_response.usage, settings.llm_model)
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
        detail = _format_exc(exc)
        log.error("Palette agent error for %s (%s): %s", template_id, palette["name"], detail)
        yield {"event": "agent-error", "data": detail}

    # Safety net: apply whatever part of the plan the agent missed. Beefree has no
    # update endpoint, so the fixed template is handed back to the route
    # ("fixed-template", internal) and used for the preview and the downloads.
    final_preview = None
    try:
        final = _template_of(await get_template(template_id, settings))
        left = missed(final, plan) if plan else []
        if left:
            fixed = apply_plan(final, plan)
            yield {"event": "fixed-template", "data": fixed}
            final_preview = await render_html(fixed, settings)
            done = len(plan) - len(left)
            yield {"event": "fix-note", "data": html_module.escape(
                f"The agent applied {done} of {len(plan)} color changes. "
                f"The remaining {len(left)} were completed automatically."
            )}
            log.info("Palette %s on %s: safety net applied %d of %d changes",
                     palette["name"], template_id, len(left), len(plan))
    except Exception as exc:
        log.warning("Palette safety net failed for %s: %s", template_id, exc)

    try:
        preview_html = final_preview or await _fetch_preview(template_id, settings)
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
- When done, reply with a short, friendly summary of what you changed (1-3 sentences),
  written for a marketer rather than a developer. You may end with one short
  suggestion for a next step.
- Never include code, scripts, JSON, the content hierarchy, element or section IDs,
  tool or function names, or color hex codes in your reply. Describe changes in
  plain words ("made the header dark green", "added a three-column benefits row").
- If you cannot fulfil a request with the available tools, explain why clearly.
"""


@monitored_stream
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
    mcp = create_code_mode_mcp(template_id, settings)
    agent: Agent[None, str] = Agent(
        model=settings.agent_model,
        toolsets=[mcp],
        system_prompt=EDIT_AGENT_SYSTEM_PROMPT + CODE_MODE_INSTRUCTIONS,
        model_settings=_CACHE_SETTINGS,
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
                        tok = _tokens_event(node.model_response.usage, settings.llm_model)
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
        detail = _format_exc(exc)
        log.error("Edit agent error for %s: %s", template_id, detail)
        yield {"event": "agent-error", "data": detail}
        agent_text = f"I encountered an issue while editing: {detail}"

    # Final preview after agent finishes
    try:
        preview_html = await _fetch_preview(template_id, settings)
        if preview_html:
            yield _preview_event(preview_html)
    except Exception:
        pass

    # Escape and format the agent reply as safe HTML
    escaped = html_module.escape(clean_reply(agent_text))
    formatted = escaped.replace("\n\n", "</p><p>").replace("\n", "<br>")
    yield {"event": "agent-message", "data": f"<p>{formatted}</p>"}
    yield {"event": "close", "data": ""}


# --- Single email executor SSE stream ----------------------------------------


@monitored_stream
async def stream_single_executor(
    template_id: str,
    brief: str,
    settings: Settings,
    *,
    progressive: bool = False,
) -> AsyncIterator[dict]:
    """Run one single-email agent and yield SSE events.

    The agent builds everything from scratch: global styles, header, body
    content, footer, and final validation. With ``progressive`` it builds one
    section per script so the live preview fills in step by step; without it
    the agent is free to batch everything into as few scripts as possible
    (the Code Mode page relies on that).

    Emits:
    - "preview" events: rendered email HTML in an iframe (same as other executors)
    - "close" event: tells HTMX to stop reconnecting
    """
    mcp = create_code_mode_mcp(template_id, settings)
    system_prompt = SINGLE_EMAIL_AGENT_SYSTEM_PROMPT + CODE_MODE_INSTRUCTIONS
    if progressive:
        system_prompt += PROGRESSIVE_BUILD_INSTRUCTIONS
    agent: Agent[None, str] = Agent(
        model=settings.agent_model,
        toolsets=[mcp],
        system_prompt=system_prompt,
        model_settings=_CACHE_SETTINGS,
        retries=3,
    )

    try:
        async with agent.iter(brief) as agent_run:
            async for node in agent_run:
                try:
                    if isinstance(node, CallToolsNode):
                        tok = _tokens_event(node.model_response.usage, settings.llm_model)
                        if tok:
                            yield tok
                        for part in node.model_response.parts:
                            args = getattr(part, "args", None)
                            if isinstance(args, str):
                                try:
                                    args = _json.loads(args)
                                except _json.JSONDecodeError:
                                    args = None
                            if isinstance(args, dict):
                                code = (
                                    args.get("script")
                                    or args.get("code")
                                    or args.get("typescript")
                                )
                                if isinstance(code, str) and code:
                                    yield {"event": "code", "data": code}
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
        detail = _format_exc(exc)
        log.error("Single agent error for %s: %s", template_id, detail)
        yield {"event": "agent-error", "data": detail}

    try:
        preview_html = await _fetch_preview(template_id, settings)
        if preview_html:
            yield _preview_event(preview_html)
    except Exception:
        pass

    yield {"event": "close", "data": ""}
