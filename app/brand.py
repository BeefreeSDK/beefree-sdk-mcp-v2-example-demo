"""Brand Rules for the on-brand email demo.

A `brandRules` object travels with the template session (POST /v2/sdk/mcp/template).
Beefree enforces it on every Code Mode call: blocks, sections and email defaults
must name one of the session's presets, preset values replace whatever the agent
asked for, forbidden block types are refused, and row/column limits are capped.

This module holds the ready-made rule sets shown on the page, a cheap local
shape check, and the parser that turns Code Mode results into "brand" events,
so the page can show each time the rules shaped the email.
"""

import json
import re

BLOCK_TYPES = [
    "paragraph", "title", "list", "image", "button", "social",
    "icon", "spacer", "divider", "table", "menu", "video",
]
MAX_RULES_BYTES = 32_000


def _button(bg, fg, radius, font, *, outline=None, description):
    props = {
        "backgroundColor": bg, "color": fg, "borderRadius": radius,
        "fontFamily": font, "fontWeight": "700", "fontSize": "16px",
        "padding": "12px 28px",
    }
    if outline:
        for side in ("Top", "Right", "Bottom", "Left"):
            props[f"border{side}"] = f"2px solid {outline}"
    return {"description": description, "properties": props}


def _rules(*, page_bg, content_bg, tint, heading, text, muted, primary, on_primary,
           link, heading_font, body_font, radius, heading_spacing="0px",
           no_add, max_rows, max_columns, max_buttons):
    return {
        "presets": {
            "settings": {
                "base": {
                    "description": "Email defaults. Use for setEmailDefaultStyles.",
                    "properties": {
                        "backgroundColor": page_bg, "contentAreaBackgroundColor": content_bg,
                        "contentAreaWidth": "600px", "fontFamily": body_font, "linkColor": link,
                    },
                },
            },
            "rows": {
                "standard": {
                    "description": "Default section for header, body content and footer.",
                    "properties": {
                        "rowBackgroundColor": page_bg, "contentAreaBackgroundColor": content_bg,
                        "contentAreaPadding": "24px",
                    },
                },
                "highlight": {
                    "description": "Tinted section to make one block of content stand out "
                                   "(an offer, a quote, key benefits). At most one or two per email.",
                    "properties": {
                        "rowBackgroundColor": page_bg, "contentAreaBackgroundColor": tint,
                        "contentAreaPadding": "32px",
                    },
                },
            },
            "blocks": {
                "title": {
                    "headline": {
                        "description": "Main headline and section headings.",
                        "properties": {
                            "color": heading, "fontFamily": heading_font, "fontWeight": "700",
                            "letterSpacing": heading_spacing, "lineHeight": "120%",
                        },
                    },
                },
                "paragraph": {
                    "body": {
                        "description": "Body copy.",
                        "properties": {
                            "color": text, "fontFamily": body_font, "fontSize": "16px",
                            "lineHeight": "160%", "linkColor": link,
                        },
                    },
                    "finePrint": {
                        "description": "Footer, legal text and small notes.",
                        "properties": {
                            "color": muted, "fontFamily": body_font, "fontSize": "12px",
                            "lineHeight": "150%", "linkColor": muted,
                        },
                    },
                },
                "button": {
                    "primary": _button(primary, on_primary, radius, body_font,
                                       description="The main call to action. One per email."),
                    "secondary": _button(content_bg, primary, radius, body_font, outline=primary,
                                         description="Secondary actions such as 'Learn more'."),
                },
            },
        },
        "permissions": {"blocks": {"noAdd": no_add}},
        "limits": {
            "rows": {"max": max_rows},
            "columns": {"maxPerRow": max_columns},
            "blocks": {"maxPerType": {"button": max_buttons}},
        },
    }


PRESETS = [
    {
        "id": "harbor",
        "name": "Harbor Bank",
        "tagline": "Calm and trustworthy. Navy, serif headlines, square buttons.",
        "swatches": ["#0B2545", "#13A89E", "#F2F5F9", "#FFFFFF"],
        "rules": _rules(
            page_bg="#F2F5F9", content_bg="#FFFFFF", tint="#E3F4F2",
            heading="#0B2545", text="#33415C", muted="#6B7A90",
            primary="#0B2545", on_primary="#FFFFFF", link="#13A89E",
            heading_font="Georgia, 'Times New Roman', serif",
            body_font="Arial, 'Helvetica Neue', Helvetica, sans-serif",
            radius="2px", no_add=["video", "menu", "social"],
            max_rows=8, max_columns=2, max_buttons=2,
        ),
    },
    {
        "id": "bloom",
        "name": "Bloom & Co.",
        "tagline": "Warm and editorial. Plum and blush, pill buttons.",
        "swatches": ["#5B2A4E", "#C9A227", "#FBF1EE", "#FFFFFF"],
        "rules": _rules(
            page_bg="#FBF1EE", content_bg="#FFFFFF", tint="#F7E3E3",
            heading="#5B2A4E", text="#4A3B45", muted="#8A7682",
            primary="#5B2A4E", on_primary="#FFFFFF", link="#A0801A",
            heading_font="Georgia, 'Times New Roman', serif",
            body_font="'Trebuchet MS', 'Lucida Grande', Tahoma, sans-serif",
            radius="30px", no_add=["video", "table", "menu"],
            max_rows=10, max_columns=3, max_buttons=3,
        ),
    },
    {
        "id": "voltage",
        "name": "Voltage",
        "tagline": "Bold dark tech. Violet on near-black, wide-tracked headlines.",
        "swatches": ["#8B5CF6", "#22D3EE", "#0F1020", "#181A2E"],
        "rules": _rules(
            page_bg="#0F1020", content_bg="#181A2E", tint="#24204A",
            heading="#FFFFFF", text="#C9CCE0", muted="#8D91AD",
            primary="#8B5CF6", on_primary="#FFFFFF", link="#22D3EE",
            heading_font="'Helvetica Neue', Helvetica, Arial, sans-serif",
            body_font="'Helvetica Neue', Helvetica, Arial, sans-serif",
            radius="0px", heading_spacing="1px", no_add=["video", "table"],
            max_rows=9, max_columns=3, max_buttons=2,
        ),
    },
]
PRESETS_BY_ID = {p["id"]: p for p in PRESETS}


def shape_errors(rules) -> list[str]:
    """Cheap checks before asking Beefree to validate the object."""
    if not isinstance(rules, dict):
        return ["Brand rules must be a JSON object, like {\"presets\": {...}}."]
    if not rules:
        return ["The brand rules object is empty."]
    if len(json.dumps(rules)) > MAX_RULES_BYTES:
        return [f"Brand rules are limited to {MAX_RULES_BYTES // 1000} KB in this demo."]
    known = {"presets", "permissions", "limits"}
    if not known & rules.keys():
        return ["Add at least one of presets, permissions or limits."]
    return []


# ── Code Mode results → brand events ────────────────────────────────────────

_FUNCTION_TARGET = re.compile(r"^(?:add|update)(\w+)$")
_TARGET_NAMES = {
    "Section": "section", "SectionStyle": "section", "ColumnStyle": "column",
    "Title": "title", "Paragraph": "paragraph", "Button": "button", "Image": "image",
    "List": "list", "Social": "social", "Icon": "icon", "Spacer": "spacer",
    "Divider": "divider", "Table": "table", "Menu": "menu", "Video": "video",
}
_SCRIPT_PREFIX = re.compile(r"^(?:script error:\s*)?(?:Error:\s*)?(?:\w+:\s+)?")
_PRESETS_LIST = re.compile(r"\s*Available presets: ([^.]+)\.?")


def _target(function: str) -> str:
    if function == "setEmailDefaultStyles":
        return "email defaults"
    m = _FUNCTION_TARGET.match(function or "")
    return _TARGET_NAMES.get(m.group(1), m.group(1).lower()) if m else "element"


def _as_data(content):
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


def events_from_result(content) -> list[dict]:
    """Brand events in one successful beefree_execute result."""
    data = _as_data(content)
    if not isinstance(data, dict):
        return []
    events = []
    for step in data.get("completed") or []:
        if not isinstance(step, dict):
            continue
        applied = (step.get("result") or {}).get("presetApplication")
        if not isinstance(applied, dict) or not applied.get("appliedPreset"):
            continue
        target = _target(step.get("function", ""))
        events.append({"kind": "applied", "target": target, "preset": applied["appliedPreset"]})
        for change in applied.get("presetOverrides") or []:
            if isinstance(change, dict) and change.get("property"):
                events.append({
                    "kind": "override", "target": target, "preset": applied["appliedPreset"],
                    "property": change["property"],
                    "requested": str(change.get("requested", "")),
                    "applied": str(change.get("applied", "")),
                })
    return events


def events_from_error(message) -> list[dict]:
    """A brand event for a script the server refused, if a brand rule refused it."""
    data = _as_data(message)
    text = data.get("error", "") if isinstance(data, dict) else str(data)
    text = _SCRIPT_PREFIX.sub("", str(text).strip())
    presets = _PRESETS_LIST.search(text)
    text = _PRESETS_LIST.sub("", text).strip()
    if "not allowed in this session" in text:
        return [{"kind": "blocked", "message": text}]
    if "requires a preset" in text:
        event = {"kind": "preset-required", "message": text}
        if presets:
            event["presets"] = [p.strip() for p in presets.group(1).split(",")]
        return [event]
    if text.startswith("Cannot add another") or "maximum of" in text:
        return [{"kind": "limit", "message": text}]
    return []
