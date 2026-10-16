"""Build a complete color plan for recoloring a Beefree template with a palette.

A Beefree template keeps colors in many places: the page background, each
row's full-width container and its 600px content area, columns, every text
block, link colors, buttons, and inline styles inside the text HTML. Telling
an agent "change the header and the headings" leaves most of them untouched.

This module finds every color occurrence, decides which palette color should
replace it (based on where it is and how dark it is), and can apply that plan
deterministically. The palette agent gets the plan as instructions; anything
it misses is fixed afterwards with apply_plan().
"""

import colorsys
import copy
import re
from dataclasses import dataclass

BACKGROUND_KEYS = {"background-color", "messageBackgroundColor"}
TEXT_KEYS = {"color", "linkColor"}
HEX = re.compile(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
RGB = re.compile(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*[\d.]+\s*)?\)")
INLINE_COLOR = re.compile(r"(?<![\w-])(color|background-color)\s*:\s*(#[0-9a-fA-F]{3,6}\b|rgba?\([^)]*\))", re.I)
WHITE = "#FFFFFF"


def normalize(value) -> str | None:
    """Return a color as #RRGGBB, or None if it isn't a solid color."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    m = HEX.fullmatch(value)
    if m:
        h = value[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return "#" + h.upper()
    m = RGB.fullmatch(value)
    if m:
        return "#" + "".join(f"{int(c):02X}" for c in m.groups())
    return None


def luminance(hex_color: str) -> float:
    """WCAG relative luminance, 0 (black) to 1 (white)."""
    def channel(c):
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def saturation(hex_color: str) -> float:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)[2]


def readable(preferred: str, background: str, fallbacks: list[str]) -> str:
    """Use `preferred` unless it is hard to read on `background`."""
    for color in [preferred, *fallbacks]:
        if contrast(color, background) >= 3:
            return color
    return max([preferred, *fallbacks, WHITE, "#111111"], key=lambda c: contrast(c, background))


@dataclass
class Change:
    path: tuple            # where the color lives in the template JSON
    where: str             # human description for the agent
    source: str            # current color (#RRGGBB)
    target: str            # palette color to use
    inline: bool = False   # True: the color is inside an HTML string at `path`


def _get(obj, path):
    for key in path:
        obj = obj[key]
    return obj


def _set(obj, path, value):
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


def _row_label(row, index, last):
    if index == 0:
        return "header row"
    if index == last:
        return "footer row"
    return f"row {index + 1}"


def _snippet(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html or "")
    text = re.sub(r"\s+", " ", text).strip()
    return (text[:40] + "…") if len(text) > 40 else text


def _row_background(row) -> str | None:
    for path in (("content", "style", "background-color"), ("container", "style", "background-color")):
        try:
            color = normalize(_get(row, path))
        except (KeyError, TypeError, IndexError):
            color = None
        if color:
            return color
    return None


def build_plan(template: dict, palette: dict) -> list[Change]:
    """Return every color change needed to apply `palette` (the PALETTES[i]["colors"] dict)."""
    page = template.get("page", template)
    root = ("page",) if "page" in template else ()
    changes: list[Change] = []

    def add(path, where, source, target):
        if source and target and source != target.upper():
            changes.append(Change(root + path, where, source, target.upper()))

    body = page.get("body", {})
    add(("body", "container", "style", "background-color"), "page background",
        normalize(body.get("container", {}).get("style", {}).get("background-color")), palette["page_bg"])
    content = body.get("content", {})
    add(("body", "content", "computedStyle", "messageBackgroundColor"), "email content area",
        normalize(content.get("computedStyle", {}).get("messageBackgroundColor")), palette["content_bg"])
    add(("body", "content", "style", "color"), "default text color",
        normalize(content.get("style", {}).get("color")), palette["text"])
    add(("body", "content", "computedStyle", "linkColor"), "default link color",
        normalize(content.get("computedStyle", {}).get("linkColor")), palette["accent"])

    rows = page.get("rows", [])
    last = len(rows) - 1
    for index, row in enumerate(rows):
        label = _row_label(row, index, last)
        source_bg = _row_background(row)
        if index == 0:
            row_bg = palette["header_bg"]
        elif index == last:
            row_bg = palette["footer_bg"]
        elif source_bg and luminance(source_bg) < 0.25:
            row_bg = palette["header_bg"]
        elif source_bg and luminance(source_bg) > 0.97:
            row_bg = palette["content_bg"]   # white sections stay the content color
        else:
            row_bg = palette["page_bg"]
        dark_row = luminance(row_bg) < 0.25
        text_color = palette["footer_text"] if index == last else palette["text"]
        heading_color = text_color if index == last else palette["heading"]
        # Keep the palette's own text colors where they are readable on this row
        # (dark palettes have light text); otherwise switch to a light color.
        text_color = readable(text_color, row_bg, [palette["text"], palette["heading"], palette["content_bg"], WHITE])
        heading_color = readable(heading_color, row_bg, [text_color])
        link_color = readable(palette["accent"], row_bg, [palette["primary"], text_color])

        base = ("rows", index)
        for part in ("container", "content"):
            add(base + (part, "style", "background-color"), f"{label} {part} background",
                normalize(row.get(part, {}).get("style", {}).get("background-color")), row_bg)
        add(base + ("content", "style", "color"), f"{label} default text color",
            normalize(row.get("content", {}).get("style", {}).get("color")), text_color)

        for ci, column in enumerate(row.get("columns", [])):
            cpath = base + ("columns", ci)
            col_bg = normalize(column.get("style", {}).get("background-color"))
            if col_bg:
                add(cpath + ("style", "background-color"), f"{label} column {ci + 1} background", col_bg, row_bg)
            for mi, module in enumerate(column.get("modules", [])):
                mpath = cpath + ("modules", mi)
                descriptor = module.get("descriptor", {})
                kind = (module.get("type") or "").rsplit("-", 1)[-1]
                block_bg = normalize(descriptor.get("style", {}).get("background-color"))
                if block_bg:
                    add(mpath + ("descriptor", "style", "background-color"), f"{label} block background",
                        block_bg, row_bg)
                for key, block in descriptor.items():
                    if not isinstance(block, dict) or key in ("style", "computedStyle", "mobileStyle"):
                        continue
                    style = block.get("style", {})
                    snippet = _snippet(block.get("html") or block.get("label") or block.get("text") or "")
                    where = f'{label}, {kind} "{snippet}"' if snippet else f"{label}, {kind}"
                    if kind == "button" or key == "button":
                        btn_bg = normalize(style.get("background-color"))
                        add(mpath + ("descriptor", key, "style", "background-color"), f"{where} button background",
                            btn_bg, palette["primary"])
                        add(mpath + ("descriptor", key, "style", "color"), f"{where} button text",
                            normalize(style.get("color")), readable(WHITE, palette["primary"], [palette["content_bg"]]))
                        continue
                    size = style.get("font-size", "")
                    big = bool(re.match(r"(\d+)", size)) and int(re.match(r"(\d+)", size).group(1)) >= 22
                    bold = str(style.get("font-weight", "")) in ("600", "700", "bold") or kind in ("heading", "title")
                    role_color = heading_color if (big or (bold and kind in ("heading", "title"))) else text_color
                    source_text = normalize(style.get("color"))
                    accent_text = (source_text and saturation(source_text) > 0.35
                                   and 0.03 < luminance(source_text) < 0.6 and not dark_row)
                    add(mpath + ("descriptor", key, "style", "color"), f"{where} text color",
                        source_text, link_color if accent_text else role_color)
                    add(mpath + ("descriptor", key, "computedStyle", "linkColor"), f"{where} link color",
                        normalize(block.get("computedStyle", {}).get("linkColor")), link_color)
                    html = block.get("html")
                    if isinstance(html, str):
                        for m in INLINE_COLOR.finditer(html):
                            source = normalize(m.group(2))
                            if not source:
                                continue
                            if m.group(1).lower() == "background-color":
                                target = row_bg
                            elif saturation(source) > 0.35 and 0.03 < luminance(source) < 0.6:
                                target = link_color
                            else:
                                target = role_color
                            if source != target.upper():
                                changes.append(Change(root + mpath + ("descriptor", key, "html"),
                                                      f"{where} inline {m.group(1).lower()} in the text",
                                                      source, target.upper(), inline=True))
    return changes


def _replace_inline(html: str, source: str, target: str) -> str:
    def swap(m):
        return m.group(0).replace(m.group(2), target) if normalize(m.group(2)) == source else m.group(0)
    return INLINE_COLOR.sub(swap, html)


def apply_plan(template: dict, plan: list[Change]) -> dict:
    """Return a copy of `template` with every change in `plan` applied."""
    result = copy.deepcopy(template)
    for change in plan:
        try:
            if change.inline:
                _set(result, change.path, _replace_inline(_get(result, change.path), change.source, change.target))
            else:
                _set(result, change.path, change.target)
        except (KeyError, IndexError, TypeError):
            continue  # the structure changed at this spot; leave it
    return result


def missed(template: dict, plan: list[Change]) -> list[Change]:
    """Changes from `plan` that are not reflected in `template`."""
    left = []
    for change in plan:
        try:
            value = _get(template, change.path)
        except (KeyError, IndexError, TypeError):
            continue
        if change.inline:
            colors = {normalize(m.group(2)) for m in INLINE_COLOR.finditer(value or "")}
            if change.source in colors:
                left.append(change)
        elif normalize(value) != change.target:
            left.append(change)
    return left


def describe(plan: list[Change]) -> str:
    """The plan as numbered instructions for the agent."""
    lines = []
    for n, change in enumerate(plan, 1):
        lines.append(f"{n}. {change.where}: {change.source} → {change.target}")
    return "\n".join(lines)
