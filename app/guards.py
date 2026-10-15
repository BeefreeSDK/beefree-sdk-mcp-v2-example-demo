from copy import deepcopy


def sequence_guard(template: dict, header_id: str, footer_id: str) -> dict:
    rows = template.get("page", {}).get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("Shared layout must contain header and footer rows.")
    ids = [row.get("uuid") for row in rows]
    if None in ids or len(set(ids)) != len(ids):
        raise ValueError("Shared layout has missing or duplicate row IDs.")
    if ids[0] != header_id or ids[-1] != footer_id:
        raise ValueError("Shared layout row IDs do not match its header/footer.")
    return {"rows": deepcopy(rows), "ids": ids}


def validate_sequence(template: dict, guard: dict) -> None:
    rows = template.get("page", {}).get("rows", [])
    ids = [row.get("uuid") for row in rows]
    original = guard["ids"]
    if len(rows) < 2 or rows[0] != guard["rows"][0] or rows[-1] != guard["rows"][-1]:
        raise ValueError("Sequence validation failed: protected header/footer changed.")
    if ids != [row_id for row_id in original if row_id in ids]:
        raise ValueError("Sequence validation failed: rows added, duplicated, or reordered.")