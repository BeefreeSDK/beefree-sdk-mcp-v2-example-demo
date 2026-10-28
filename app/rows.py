"""Reusable Rows for the Email from saved rows page (/rows).

A `reusableRows` library travels with the template session, in the body of
POST /v2/sdk/mcp/template. Beefree stores it for the session and, on Core plans and above,
adds four functions to the Code Mode script API: searchReusableRows,
getReusableRowsFacets, getReusableRowsDetails and addReusableRow.

This module holds the sample library shown on the pages, a cheap local shape
check, the parser that turns Code Mode results into "rows" events, and the lookup
that names the saved rows a finished template contains.
"""

import json
from pathlib import Path

# Four rows saved from the Beefree builder, as onSaveRow produces them.
SAMPLE_LIBRARY = {
    "entries": json.loads((Path(__file__).parent / "data" / "sample-reusable-rows.json").read_text()),
}

BLOCK_NAMES = {
    "text": "text", "paragraph": "text", "heading": "title", "title": "title", "list": "list",
    "image": "image", "button": "button", "social": "social", "icons": "icons",
    "spacer": "spacer", "divider": "divider", "table": "table", "menu": "menu",
    "video": "video", "html": "html",
}

ROW_FUNCTIONS = {
    "searchReusableRows": "search",
    "getReusableRowsFacets": "facets",
    "getReusableRowsDetails": "details",
    "addReusableRow": "placed",
}


def tags_of(tags) -> list[str]:
    """A row's tags as a list: an array, or one comma-separated string."""
    if isinstance(tags, list):
        return [tag for tag in tags if isinstance(tag, str)]
    if isinstance(tags, str):
        return [tag.strip() for tag in tags.split(",") if tag.strip()]
    return []


def _blocks(row: dict) -> list[str]:
    seen = []
    for column in row.get("columns") or []:
        for module in (column or {}).get("modules") or []:
            kind = str((module or {}).get("type", "")).replace("mailup-bee-newsletter-modules-", "")
            name = BLOCK_NAMES.get(kind, kind)
            if name and name not in seen:
                seen.append(name)
    return seen


def library_cards(library: dict) -> list[dict]:
    """What the page shows for each row of a library."""
    cards = []
    for row in library.get("entries") or []:
        metadata = row.get("metadata") or {}
        cards.append({
            "name": metadata.get("name", ""),
            "category": metadata.get("category"),
            "description": metadata.get("description") or "",
            "tags": tags_of(metadata.get("tags")),
            "synced": row.get("synced") is True,
            "blocks": _blocks(row),
        })
    return cards


def shape_errors(library) -> list[str]:
    """Cheap checks before asking Beefree to validate the library."""
    if not isinstance(library, dict):
        return ['A row library must be a JSON object, like {"entries": [...]}.']
    entries = library.get("entries")
    if not isinstance(entries, list):
        return ['The library needs an "entries" array of saved rows.']
    if not entries:
        return ["The library has no rows: add at least one saved row to entries."]
    errors = []
    for index, row in enumerate(entries):
        if not isinstance(row, dict):
            errors.append(f"Row {index + 1} is not an object.")
            continue
        if not (row.get("metadata") or {}).get("name"):
            errors.append(f"Row {index + 1} needs a metadata.name.")
        if not row.get("columns"):
            errors.append(f"Row {index + 1} needs its columns.")
    return errors


def placed_rows(template: dict) -> list[str]:
    """The names of the saved rows a template contains, in page order.

    A placed row keeps its metadata, while the sections the agent builds have
    none, so on a blank template every named row is a saved one.
    """
    page = template.get("page") if isinstance(template, dict) else None
    rows = (page or {}).get("rows") or []
    names = []
    for row in rows:
        name = ((row or {}).get("metadata") or {}).get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


# ── Code Mode results → rows events ─────────────────────────────────────────

def _as_data(content):
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


def events_from_result(content) -> list[dict]:
    """A "rows" event for each reusable rows call in one beefree_execute result.

    Every successful call is listed in `completed`, with its function name and the
    ids it returned: a placed row's sectionId, but not its name.
    """
    data = _as_data(content)
    if not isinstance(data, dict):
        return []
    events = []
    for step in data.get("completed") or []:
        if not isinstance(step, dict):
            continue
        kind = ROW_FUNCTIONS.get(step.get("function", ""))
        if not kind:
            continue
        event = {"kind": kind}
        section = (step.get("result") or {}).get("sectionId")
        if kind == "placed" and section:
            event["sectionId"] = str(section)
        events.append(event)
    return events
