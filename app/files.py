"""File Manager for AI, for the Email with your File Manager page (/files).

A `fileManager` object travels with the template session, in the body of
POST /v2/sdk/mcp/template. It names the end user (`uid`) and the folders the
agent may list and search, save files to, and delete files from. Beefree then
adds up to four functions to the Code Mode script API: listFiles, searchFiles,
uploadFile and removeFile.

This module holds the access presets and sample files shown on the page, a cheap
local shape check, the parser that turns Code Mode results into "files" events,
and the helpers that compare the File Manager before and after a run.
"""

import json

UPLOAD_FOLDER = "/AI Generated Images"
SAMPLE_FOLDER = "/Brand"

PRESETS = [
    {
        "id": "save",
        "tag": "Save",
        "name": "Read everything, save to one folder",
        "description": f"The agent reads the whole File Manager and saves new images to {UPLOAD_FOLDER}. It deletes nothing.",
        "config": {"readPaths": ["/"], "uploadPaths": [UPLOAD_FOLDER], "deletePaths": []},
    },
    {
        "id": "cleanup",
        "tag": "Save + clean up",
        "name": "Also clean up its own folder",
        "description": f"As above, and the agent may delete files in {UPLOAD_FOLDER}, for example an image it replaced.",
        "config": {"readPaths": ["/"], "uploadPaths": [UPLOAD_FOLDER], "deletePaths": [UPLOAD_FOLDER]},
    },
    {
        "id": "brand",
        "tag": "Read-only",
        "name": "Brand folder only, read-only",
        "description": f"The agent can only find and reuse the images in {SAMPLE_FOLDER}. It saves and deletes nothing.",
        "config": {"readPaths": [SAMPLE_FOLDER], "uploadPaths": [], "deletePaths": []},
    },
]

# Fieldnote's brand images, the same ones the sample reusable rows use. Beefree
# fetches each URL server-side and stores the file in SAMPLE_FOLDER.
SAMPLE_FILES = [
    {"name": "fieldnote-logo.png", "url": "https://placehold.co/360x96/1F3A5F/FFFFFF/png?text=Fieldnote"},
    {"name": "school-desk.jpg", "url": "https://picsum.photos/id/20/1200/640"},
    {"name": "cart-item.jpg", "url": "https://picsum.photos/id/26/600/400"},
    {"name": "desk-setup.jpg", "url": "https://picsum.photos/id/60/600/400"},
]

FILE_FUNCTIONS = {
    "listFiles": "list",
    "searchFiles": "search",
    "uploadFile": "upload",
    "removeFile": "remove",
}

MAX_FOLDERS = 20
OPERATIONS = ("readPaths", "uploadPaths", "deletePaths")


def shape_errors(config) -> list[str]:
    """Cheap checks before asking Beefree to validate the access."""
    if not isinstance(config, dict):
        return ['The File Manager access must be a JSON object, like {"readPaths": ["/"]}.']
    errors = []
    for key in OPERATIONS:
        if key not in config:
            continue
        paths = config[key]
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            errors.append(f"{key} must be a list of folder paths.")
            continue
        if len(paths) > MAX_FOLDERS:
            errors.append(f"{key} can list at most {MAX_FOLDERS} folders.")
        errors.extend(f'{key}: "{p}" must start with "/".' for p in paths if not p.startswith("/"))
    return errors


def with_uid(config: dict, uid: str) -> dict:
    """The access as Beefree gets it: the operations the visitor chose, for this browser's uid."""
    return {"uid": uid, **{key: config[key] for key in OPERATIONS if key in config}}


def summary(config: dict) -> dict:
    """What the page shows of an access: each operation's folders, or the default."""
    return {key: config[key] if key in config else [UPLOAD_FOLDER] for key in OPERATIONS}


def folders_to_watch(config: dict) -> list[str]:
    """The folders whose files the run may change or place: read and upload folders."""
    folders = []
    for key in ("uploadPaths", "readPaths"):
        for path in summary(config)[key]:
            if path not in folders:
                folders.append(path)
    return folders[:MAX_FOLDERS]


def file_entry(item: dict) -> dict | None:
    """A file from a Content Services API listing, as the page shows it; None for a folder."""
    if not isinstance(item, dict) or item.get("mime-type") == "application/directory":
        return None
    path = str(item.get("path") or "")
    return {
        "name": item.get("name") or path.rsplit("/", 1)[-1],
        "path": path,
        "url": item.get("public-url") or "",
        "thumbnail": item.get("thumbnail") or item.get("public-url") or "",
    }


def changes(before: list[dict], after: list[dict]) -> dict:
    """The files a run saved and removed, from two snapshots of the same folders."""
    old = {f["path"] for f in before}
    new = {f["path"] for f in after}
    return {
        "saved": [f for f in after if f["path"] not in old],
        "removed": [f for f in before if f["path"] not in new],
    }


def image_sources(template: dict) -> list[str]:
    """The src of every image block in a template, in page order."""
    sources = []
    page = template.get("page") if isinstance(template, dict) else None
    for row in (page or {}).get("rows") or []:
        for column in (row or {}).get("columns") or []:
            for module in (column or {}).get("modules") or []:
                src = (((module or {}).get("descriptor") or {}).get("image") or {}).get("src")
                if isinstance(src, str) and src:
                    sources.append(src)
    return sources


# ── Code Mode results → files events ────────────────────────────────────────

def _as_data(content):
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


def events_from_result(content) -> list[dict]:
    """A "files" event for each file call in one beefree_execute result.

    `completed` names each successful call but keeps only element ids from its
    result, so the events say what the agent did, not which files: the summary
    at the end of the run names those.
    """
    data = _as_data(content)
    if not isinstance(data, dict):
        return []
    return [
        {"kind": FILE_FUNCTIONS[step["function"]]}
        for step in data.get("completed") or []
        if isinstance(step, dict) and step.get("function") in FILE_FUNCTIONS
    ]
