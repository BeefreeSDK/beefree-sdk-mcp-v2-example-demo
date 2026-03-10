"""PydanticAI agents: planner (structured output) and executor (MCP tools + SSE).

Streaming strategy
------------------
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


class EmailStep(BaseModel):
    step: int
    title: str
    subject_line: str
    agent_prompt: str


class EmailPlan(BaseModel):
    sequence_title: str
    emails: list[EmailStep]


# --- System prompts -----------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """You are an expert email marketing strategist.

Given a campaign brief, generate a structured email sequence plan.
Determine the right number of emails from the brief — if the user specifies
how many, follow that exactly. Otherwise decide based on the campaign needs.

Rules:
- Make `agent_prompt` highly detailed so an AI agent can build a complete email
  using only Beefree MCP tools. Include: layout guidance, content to write,
  visual style, and CTA instructions.
- Keep subject lines concise and compelling (<= 60 characters).
- `sequence_title` should be short (<= 50 characters).
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


async def generate_plan(
    goal: str,
    settings: Settings,
) -> EmailPlan:
    planner: Agent[None, EmailPlan] = Agent(
        model=settings.llm_planner_model,
        output_type=EmailPlan,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        retries=3,
    )
    result = await planner.run(goal)
    return result.output


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
