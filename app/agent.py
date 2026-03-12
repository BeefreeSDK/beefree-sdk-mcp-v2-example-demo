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
import logging
from typing import AsyncIterator

log = logging.getLogger(__name__)

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

LAYOUT_AGENT_SYSTEM_PROMPT = """You are building the shared visual identity for an email sequence.
Choose ONE bold, distinctive colour palette from the campaign brief and apply it in every step.

COLOUR FORMULA — decide all six values before calling any tool.
Match the campaign domain using the table below as a guide:

  Domain              primary    accent     page_bg
  ─────────────────────────────────────────────────────
  B2B SaaS / tech     #1E3A5F    #0EA5E9    #F0F6FF
  E-commerce/retail   #1A1A2E    #F97316    #FFF8F0
  Food & beverage     #7C2D12    #FB923C    #FFF7ED
  Eco / wellness      #1A3D2B    #40916C    #F0FFF4
  Healthcare          #0A4D68    #0EA5E9    #F0FAFF
  Finance             #0F2A4A    #22C55E    #F0F7FF
  Fashion / luxury    #2D1B69    #EC4899    #FDF4FF
  Education           #1E3A5F    #F59E0B    #FFFBF0
  Travel              #0C3547    #06B6D4    #F0FDFF

  text      : #1A1A2E (never change)
  text_muted: #64748B (never change)

Pick the closest domain match; adjust hue/saturation to better fit the brief.
Never use generic grey, beige, or unmodified default navy.

STEP 1 — GLOBAL STYLES:
Call beefree_set_template_styles with:
- Page background  → page_bg
- Content area bg  → #FFFFFF
- Content width    → 600 px
- Font family      → "Helvetica Neue, Helvetica, Arial, sans-serif"
- Body text colour → #1A1A2E
- Link colour      → accent
Then call beefree_check_template.

STEP 2 — HEADER ROW:
Add ONE full-width single-column header row. This is the brand banner — make it visually striking.

Construction rules:
1. Set the ROW background to the primary colour.
2. Logo image, centred:
   - Build the placehold.co URL so the image background matches primary and the text is white.
     Format: https://placehold.co/160x44/{PRIMARY_HEX_NO_HASH}/FFFFFF?text=LOGO
     Example — if primary is #1E3A5F → https://placehold.co/160x44/1E3A5F/FFFFFF?text=LOGO
   - This makes the placeholder invisible against the row background, simulating a white logo.
   - Alt text: "Brand logo"
3. Brand / company name (inferred from the brief) as a text element below the logo:
   - Colour: #FFFFFF · Bold · 18 px · Centred · Letter-spacing: 1 px
4. Column padding: 36 px top · 36 px bottom · 20 px left/right.
5. Add a 3 px solid border in the accent colour at the bottom of the row.
Then call beefree_check_template.

STEP 3 — FOOTER ROW:
Add ONE clean compliance footer row.

Construction rules:
1. Set the ROW background to page_bg.
2. Add a 1 px solid #E2E8F0 top border to the row.
3. Two text elements, both centred:
   LINE 1 — Company name (inferred from brief), 13 px, #1A1A2E, semi-bold.
   LINE 2 — Address and legal: "123 Campaign St, City, Country  ·  © 2025  ·  Unsubscribe"
             12 px, text_muted (#64748B), normal weight.
4. Column padding: 24 px top · 24 px bottom.
Then call beefree_check_template.

STEP 4 — DUPLICATE CHECK:
Call beefree_get_content_hierarchy and verify:
- The template contains exactly 2 rows (header + footer) — no extra rows.
- The header row contains exactly 1 image (the logo) and exactly 1 text block (brand name).
- The footer row contains exactly 1 text block (the two-line company/address/unsubscribe copy).
If any element appears more than once inside the same row, it was inserted twice.
Remove every duplicate by deleting the extra element, keeping only the first occurrence.
Then call beefree_check_template once more.

RULES:
- Header and footer only — never add body content, hero sections, or CTAs.
- When done, end your response with ONLY this JSON block (no markdown, no extra text after it):
  {"header_row_id": "<id of the header row>", "footer_row_id": "<id of the footer row>"}
"""

EXECUTOR_SYSTEM_PROMPT = """You are an expert email design and copy assistant building the body of one email in a sequence.
The template already has a shared header (top) and footer (bottom) — never touch them.

YOUR ONLY JOB: set email metadata, then add the required body rows between header and footer.

━━━ CORE PRINCIPLES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Clarity: one primary message, one primary CTA per email.
- Brevity: 3 body rows maximum — every row must earn its place.
- Scannability: short copy, strong headings, 24–32 px row padding.
- Value first: lead every section with benefits, not features.
- Accessibility: 14 px+ body text, 44 px+ CTA buttons, descriptive alt text on all images.

━━━ PROTECTED ROWS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The header row ID and footer row ID in your prompt are protected.
NEVER pass them to any tool. NEVER recreate, edit, or delete them.

━━━ STEP 0 — EMAIL METADATA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before adding any rows, call beefree_set_email_metadata:
- subject:   use the subject line provided in the prompt (verbatim)
- preheader: write a compelling 40–90 character preview that complements the subject

━━━ INSERTION RULE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every body row MUST be inserted BEFORE the footer row using the footer row ID.
Never append to the end of the template.

━━━ REQUIRED SECTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the prompt lists "REQUIRED BODY SECTIONS", build exactly those rows — one row
per item, in order. No extra rows. No skipped rows. No reordering.

━━━ ROW TYPES — pick the right layout for each row ━━━━━━━━━━━━━━━━━━━━━━━━━

HERO ROW (always the first body row):
  Layout: 1 full-width column
  - Full-width image: https://placehold.co/600x200?text=Topic (max 200 px tall, descriptive alt)
  - Headline: ≤8 words, benefit-led, 26 px bold, primary colour, centred
  - Subheadline: 1 sentence, 15 px, #64748B, centred
  - CTA button: centred, primary bg, white text, 14 px top/bottom · 32 px left/right,
    border-radius 6 px, 15 px bold

FEATURE ROW (for showcasing 2–3 benefits/items side by side):
  Layout: 2 or 3 equal columns
  - Each column: icon (https://placehold.co/48x48?text=✓, descriptive alt) + bold title
    (16 px) + 1-line description (14 px, #64748B)
  - Row has a section heading above: 20 px bold, primary colour, centred
  - Background: white or page_bg

TEXT + CTA ROW (for narrative + conversion):
  Layout: 1 column, centred content
  - Heading: ≤8 words, 22 px bold, primary colour
  - Body: 2–3 sentences max, 15 px, #1A1A2E, line-height 1.6
  - CTA button (same spec as hero CTA)
  - Background: primary colour (makes it stand out) with all text in white

PROOF ROW (for a testimonial or key stat):
  Layout: 1 column, centred
  - Option A — Quote: large opening quotation mark (decorative), 1–2 sentence quote,
    attributed name + role/company in small muted text (13 px, #64748B)
  - Option B — Stat: metric in large bold text (36 px, primary colour) + 1-line caption
  - Background: page_bg · 28 px top/bottom padding — keep it compact

━━━ COPY STANDARDS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Headlines: ≤8 words, benefit-led ("Start watching in 60 seconds")
- Body copy: 2–3 sentences per row — never more. Prefer bullet lists for features.
- CTAs: action verb + specific outcome ("Start Free Trial", "Get 30% Off")
- Never lorem ipsum. Generate real, campaign-appropriate copy.

━━━ DESIGN STANDARDS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPACT: 24–32 px top/bottom padding per row. No standalone spacer rows.

COLOURS: use the global palette — never introduce new colours:
- Section backgrounds: white (#FFFFFF) or page_bg tint, except TEXT+CTA rows (primary)
- Headlines: primary colour (white if on primary background)
- Body copy: #1A1A2E, 15–16 px, line-height 1.6 (white if on primary background)

━━━ WORKFLOW ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Call beefree_set_email_metadata (subject + preheader).
2. Add each body row in order before the footer. Call beefree_check_template after each row.
3. DUPLICATE CHECK — call beefree_get_content_hierarchy and verify:
   - The number of body rows equals exactly the number of REQUIRED BODY SECTIONS.
   - No row contains more than one image with the same src URL.
   - No row contains more than one button with identical label text.
   - No two adjacent rows are structurally identical (same layout + same content type).
   If any duplicate is found, delete the extra element/row before proceeding.
4. When the hierarchy looks clean and beefree_check_template passes, respond with: "Done."
"""

SINGLE_EMAIL_AGENT_SYSTEM_PROMPT = """You are an expert email design and copy assistant powered by the Beefree SDK.
Build one complete, conversion-focused email from scratch following the steps below.

━━━ CORE PRINCIPLES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Clarity: one primary message, one primary CTA.
- Brevity: 4 body rows maximum — readable in under 90 seconds.
- Scannability: short copy, strong headings, tight padding.
- Value first: lead with benefits, support with features.
- Accessibility: 14 px+ body text, 44 px+ CTA buttons, descriptive alt text on every image.
- Compliance: unsubscribe link + physical address in the footer.

Follow ALL steps in order — never skip, never reorder.

── STEP 1 · EMAIL METADATA ─────────────────────────────────────────────────
Call beefree_set_email_metadata:
- subject:   compelling subject line (≤60 chars) derived from the brief
- preheader: 40–90 character preview text that complements the subject

── STEP 2 · GLOBAL STYLES ──────────────────────────────────────────────────
Choose a bold, distinctive palette from the brief context using this table:

  Domain              primary    accent     page_bg
  ─────────────────────────────────────────────────────
  B2B SaaS / tech     #1E3A5F    #0EA5E9    #F0F6FF
  E-commerce/retail   #1A1A2E    #F97316    #FFF8F0
  Food & beverage     #7C2D12    #FB923C    #FFF7ED
  Eco / wellness      #1A3D2B    #40916C    #F0FFF4
  Healthcare          #0A4D68    #0EA5E9    #F0FAFF
  Finance             #0F2A4A    #22C55E    #F0F7FF
  Fashion / luxury    #2D1B69    #EC4899    #FDF4FF
  Education           #1E3A5F    #F59E0B    #FFFBF0
  Travel              #0C3547    #06B6D4    #F0FDFF

  text: #1A1A2E · text_muted: #64748B (never change these)

Pick the closest domain match; adjust hue/saturation to fit the brief.

Call beefree_set_template_styles:
- Page bg: page_bg · Content area: #FFFFFF · Width: 600 px
- Font: "Helvetica Neue, Helvetica, Arial, sans-serif"
- Body text: #1A1A2E · Link colour: accent
Then call beefree_check_template.

── STEP 3 · HEADER ──────────────────────────────────────────────────────────
One full-width single-column header row — the brand banner, make it striking.
1. Set the ROW background to the primary colour.
2. Logo image, centred:
   - Build the URL so image bg matches primary and text is white:
     https://placehold.co/160x44/{PRIMARY_HEX_NO_HASH}/FFFFFF?text=LOGO
     Example — primary #1E3A5F → https://placehold.co/160x44/1E3A5F/FFFFFF?text=LOGO
   - Alt text: "Brand logo"
3. Brand name text below logo: white, bold, 18 px, centred, letter-spacing 1 px.
4. Column padding: 36 px top · 36 px bottom · 20 px left/right.
5. 3 px solid accent-colour border at the bottom of the row.
Then call beefree_check_template.

── STEP 4 · HERO ────────────────────────────────────────────────────────────
One full-width single-column hero row — the primary conversion moment.
1. Full-width image: https://placehold.co/600x200?text=Hero+Image
   - Max 200 px tall · alt text must describe the actual email topic (not "hero image")
2. Headline: ≤8 words, benefit-led, 28 px bold, primary colour, centred
3. Subheadline: 1 sentence only, 15 px, #64748B, centred
4. CTA button: centred, action verb + outcome ("Get 30% Off", "Start Free Trial")
   Primary bg · white text · border-radius 6 px · 14 px top/bottom · 32 px left/right
5. Row padding: 32 px top/bottom · background white
Then call beefree_check_template.

── STEP 5 · VALUE PROPS ─────────────────────────────────────────────────────
One row with 2–3 equal columns — scannable benefits at a glance.
1. Section heading above columns: ≤6 words, 20 px bold, primary colour, centred
2. Each column contains:
   - Icon: https://placehold.co/48x48?text=Icon (descriptive alt text, e.g. "Fast delivery icon")
   - Title: 2–4 words, 15 px bold, #1A1A2E
   - Description: 1 tight sentence, 14 px, #64748B
3. Lead with the most compelling benefit first
4. Background: page_bg · row padding 28 px top/bottom
Then call beefree_check_template.

── STEP 6 · SOCIAL PROOF ────────────────────────────────────────────────────
One compact proof row — maximum 3 lines of text total.
Choose the format that fits the brief best:
  A) Quote: opening large " mark (decorative) + 1–2 sentence customer quote
     + attribution line: "— Name, Role at Company" (13 px, #64748B)
  B) Stat: one large metric (36 px bold, primary colour) + 1-line caption below (14 px, #64748B)
Background: page_bg · 24 px top/bottom padding · no image needed
Then call beefree_check_template.

── STEP 7 · CLOSING CTA ─────────────────────────────────────────────────────
One punchy closing strip — drive the final conversion.
1. ROW background: primary colour (full width)
2. Headline: ≤6 words, white, 22 px bold, centred — create urgency or reinforce the offer
3. CTA button: same spec as hero, white bg with primary text OR inverted accent style
4. Row padding: 32 px top/bottom — no other copy, no distractions
Then call beefree_check_template.

── STEP 8 · FOOTER ──────────────────────────────────────────────────────────
One compliance footer row:
1. ROW background: page_bg · 1 px solid #E2E8F0 top border.
2. LINE 1 — Company name (from the brief): 13 px, #1A1A2E, semi-bold, centred.
3. LINE 2 — "123 Campaign St, City, Country  ·  © 2025  ·  Unsubscribe"
   12 px, #64748B, normal, centred.
4. Column padding: 24 px top · 24 px bottom.
Then call beefree_check_template.

── STEP 9 · FINAL VALIDATION ────────────────────────────────────────────────
1. Call beefree_check_template.
2. Call beefree_get_content_hierarchy and scan for duplicates:
   - Total rows: exactly 6 (Header · Hero · Value Props · Social Proof · Closing CTA · Footer).
   - Logo image appears exactly once (inside the header row only).
   - Brand name text appears exactly once (inside the header row only).
   - Each CTA button label appears at most twice (hero + closing strip share the same action — that's intentional and fine).
   - No row contains more than one image with the same src URL.
   - No two rows have identical content type AND identical heading text.
   If any unintended duplicate is found, remove the extra occurrence, then call beefree_check_template again.
3. When hierarchy is clean and the final check passes, respond with: "Done."

━━━ DESIGN RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Total email body: 4 rows (Hero + Value Props + Social Proof + Closing CTA). No extras.
- All images: https://placehold.co with descriptive alt text. Never leave src empty.
- All copy: real, campaign-relevant content — no lorem ipsum.
- Body copy: 15–16 px, line-height 1.6, max 3 sentences per row.
- Feature / value sections: 2–3 column layout, not bullet lists in prose.
- Every CTA: action verb + specific outcome ("Get 30% Off" not "Learn More").
- One primary CTA per email — hero and closing strip share the same action.
- No standalone spacer rows — use row padding for breathing room.
- If in doubt, cut it — a shorter email always outperforms a long one.
"""


# --- Planner -----------------------------------------------------------------


def _build_executor_prompt(
    s: EmailSkeleton,
    sequence_title: str,
    campaign_goal: str,
    sections: list[str] | None = None,
) -> str:
    """Build the executor prompt from a template — no LLM, instant."""
    base = (
        f"Campaign: {sequence_title}\n"
        f"Email {s.step}: {s.title}\n"
        f"Subject line: {s.subject_line}\n\n"
        f"Brief: {campaign_goal}\n\n"
        "Build a complete, professional email that matches the campaign brand and tone."
    )
    if sections:
        numbered = "\n".join(f"  {i + 1}. {sec}" for i, sec in enumerate(sections))
        base += (
            f"\n\nREQUIRED BODY SECTIONS — build exactly {len(sections)} rows in this order:\n"
            f"{numbered}\n\n"
            "You MUST build every section listed above, in the exact order shown. "
            "Do NOT add extra rows. Do NOT skip any row. Do NOT change the order. "
            "Each listed section corresponds to exactly one body row inserted before the footer."
        )
    else:
        base += (
            " Include a hero section, body copy, primary CTA button, and supporting "
            "content. Apply consistent typography, colours, and spacing throughout."
        )
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
        retries=3,
    )
    result = await skeleton_agent.run(goal)
    skeleton = result.output

    emails = [
        EmailStep(
            step=s.step,
            title=s.title,
            subject_line=s.subject_line,
            agent_prompt=_build_executor_prompt(
                s,
                skeleton.sequence_title,
                goal,
                sections=sections_per_step[s.step - 1]
                if sections_per_step and s.step - 1 < len(sections_per_step)
                else None,
            ),
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

    The agent returns plain text ending with a JSON block containing the row IDs.
    We parse that block rather than using structured output, which avoids a
    Gemini API limitation that rejects the combination of tool schemas + response
    schema when the total constraint state count is too large.
    """
    import json as _json
    import re as _re
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
    layout_agent: Agent[None, str] = Agent(
        model=settings.llm_layout_model,
        toolsets=[mcp],
        system_prompt=LAYOUT_AGENT_SYSTEM_PROMPT,
        retries=2,
    )

    result = await layout_agent.run(
        f"Build the shared header and footer for the '{sequence_title}' campaign."
    )
    text = result.output

    # Extract the row-ID JSON block the agent was instructed to append.
    match = _re.search(
        r'\{\s*"header_row_id"\s*:\s*"([^"]+)"\s*,\s*"footer_row_id"\s*:\s*"([^"]+)"\s*\}',
        text,
    ) or _re.search(
        r'\{\s*"footer_row_id"\s*:\s*"([^"]+)"\s*,\s*"header_row_id"\s*:\s*"([^"]+)"\s*\}',
        text,
    )
    if match:
        # Prefer the named-group approach: just parse the whole JSON object
        json_str = match.group(0)
        ids = _json.loads(json_str)
        header_row_id = ids["header_row_id"]
        footer_row_id = ids["footer_row_id"]
    else:
        raise ValueError(
            f"Layout agent did not return row IDs in the expected format. "
            f"Response tail: {text[-300:]!r}"
        )

    template_data = await get_template(layout_template_id, settings)
    template_json = template_data.get("template", template_data)

    return header_row_id, footer_row_id, template_json


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
                try:
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

    try:
        preview_html = await _fetch_preview(template_id, settings)
        if preview_html:
            yield _preview_event(preview_html)
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
