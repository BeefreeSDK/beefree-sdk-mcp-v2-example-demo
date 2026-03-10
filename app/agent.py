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

EXECUTOR_SYSTEM_PROMPT = """You are an expert email designer working inside the Beefree headless editor.
Use the available MCP tools to build a complete, professional email template.
The template starts empty. Build it from scratch by adding rows and content elements.

You MUST complete ALL of these steps — skipping any step is a failure:
1. Add rows and columns to create the email structure
2. Add EVERY content element: header image, titles, body paragraphs, buttons, footer text, dividers, spacers
3. Apply typography and colour styles
4. beefree_check_template — validate the final result

CRITICAL: The email is NOT done until you have called beefree_check_template.
If the email has a footer section, you MUST populate it. Never leave rows empty.
Never stop after just the header — keep going until the entire email is built and validated.
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
