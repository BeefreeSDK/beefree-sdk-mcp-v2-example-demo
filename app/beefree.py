"""Thin async wrapper around the Beefree REST endpoints used before/after MCP."""

from urllib.parse import quote

import httpx

from .config import Settings


class BrandRulesRejected(Exception):
    """Beefree refused a brandRules object; ``details`` lists each problem."""

    def __init__(self, details: list[str]):
        super().__init__("; ".join(details))
        self.details = details


class ReusableRowsRejected(Exception):
    """Beefree refused a reusableRows library; ``details`` lists each problem."""

    def __init__(self, details: list[str]):
        super().__init__("; ".join(details))
        self.details = details


class FileManagerRejected(Exception):
    """Beefree refused a fileManager object; ``details`` lists each problem."""

    def __init__(self, details: list[str]):
        super().__init__("; ".join(details))
        self.details = details


async def create_template(
    settings: Settings,
    brand_rules: dict | None = None,
    reusable_rows: dict | None = None,
    file_manager: dict | None = None,
) -> str:
    """Create a blank template session and return the templateId.

    With ``brand_rules``, Beefree enforces them on every Code Mode call in the session.
    With ``reusable_rows`` (a ``{"entries": [...]}`` library of reusable rows), the agent
    can search the library and place its rows (Core plans and above).
    With ``file_manager`` (a ``uid`` and the folders for each operation), the agent can
    list, search, save and delete files in that user's File Manager.
    """
    body = {}
    if brand_rules:
        body["brandRules"] = brand_rules
    if reusable_rows:
        body["reusableRows"] = reusable_rows
    if file_manager:
        body["fileManager"] = file_manager
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{settings.bee_api_base}/v2/sdk/mcp/template",
            headers={"Authorization": f"Bearer {settings.bee_api_key}"},
            json=body,
        )
        if resp.status_code == 400:
            if file_manager:
                _raise_file_manager(resp)
            if reusable_rows:
                _raise_reusable_rows(resp)
            if brand_rules:
                _raise_brand_rules(resp)
        if not resp.is_success:
            raise httpx.HTTPStatusError(
                f"{resp.status_code} — {resp.text}",
                request=resp.request,
                response=resp,
            )
        return resp.json()["templateId"]


async def create_seeded_template(
    settings: Settings,
    template_json: dict,
    brand_rules: dict | None = None,
) -> str:
    """Create a template session pre-seeded with existing template content."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{settings.bee_api_base}/v2/sdk/mcp/template",
            headers={"Authorization": f"Bearer {settings.bee_api_key}"},
            json={"template": template_json, **(
                {"brandRules": brand_rules} if brand_rules else {}
            )},
        )
        if not resp.is_success:
            raise httpx.HTTPStatusError(
                f"{resp.status_code} — {resp.text}",
                request=resp.request,
                response=resp,
            )
        return resp.json()["templateId"]


def _raise_brand_rules(resp: httpx.Response) -> None:
    try:
        body = resp.json()
    except ValueError:
        return
    details = body.get("details") if isinstance(body, dict) else None
    if isinstance(details, list) and details:
        raise BrandRulesRejected([str(d) for d in details])


def _raise_reusable_rows(resp: httpx.Response) -> None:
    try:
        body = resp.json()
    except ValueError:
        return
    if not isinstance(body, dict) or body.get("error") != "Invalid reusableRows":
        return
    details = body.get("details")
    if isinstance(details, list) and details:
        raise ReusableRowsRejected([str(d) for d in details])


def _raise_file_manager(resp: httpx.Response) -> None:
    try:
        body = resp.json()
    except ValueError:
        return
    if not isinstance(body, dict) or body.get("error") != "Invalid fileManager":
        return
    details = body.get("details")
    if isinstance(details, list) and details:
        raise FileManagerRejected([str(d) for d in details])


async def validate_file_manager(settings: Settings, file_manager: dict) -> None:
    """Raise FileManagerRejected if Beefree would refuse this access (it needs a uid)."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{settings.bee_api_base}/v2/sdk/mcp/template/validate",
            headers={"Authorization": f"Bearer {settings.bee_api_key}"},
            json={"fileManager": file_manager},
        )
        if not resp.is_success:
            raise httpx.HTTPStatusError(
                f"{resp.status_code} — {resp.text}",
                request=resp.request,
                response=resp,
            )
        result = resp.json()
        if not result.get("valid"):
            raise FileManagerRejected([str(e) for e in result.get("errors") or ["Invalid fileManager"]])


def _file_url(settings: Settings, path: str) -> str:
    """A Content Services API file route; each segment is encoded, the slashes kept."""
    return f"{settings.bee_api_base}/v1/file/" + quote(path.strip("/"), safe="/")


async def list_folder(settings: Settings, uid: str, folder: str) -> list[dict]:
    """The items directly inside a folder of the user's File Manager; [] if it doesn't exist."""
    url = _file_url(settings, folder).rstrip("/") + "/"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            url, headers={"Authorization": f"Bearer {settings.bee_api_key}", "X-BEE-Uid": uid},
        )
    if resp.status_code in (400, 404):
        return []
    resp.raise_for_status()
    items = ((resp.json() or {}).get("data") or {}).get("items")
    return items if isinstance(items, list) else []


async def upload_from_url(settings: Settings, uid: str, path: str, source_url: str) -> None:
    """Store the file at ``source_url`` (fetched by Beefree) at ``path``, replacing any file there."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            _file_url(settings, path),
            params={"conflict_strategy": "replace"},
            headers={"Authorization": f"Bearer {settings.bee_api_key}", "X-BEE-Uid": uid},
            json={"url": source_url},
        )
    if not resp.is_success or (resp.json() or {}).get("status") != "success":
        raise httpx.HTTPStatusError(
            f"{resp.status_code} — {resp.text}", request=resp.request, response=resp,
        )


async def delete_file(settings: Settings, uid: str, path: str) -> None:
    """Delete one file of the user's File Manager; a file already gone counts as deleted."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.delete(
            _file_url(settings, path),
            headers={"Authorization": f"Bearer {settings.bee_api_key}", "X-BEE-Uid": uid},
        )
    if resp.is_success:
        return
    # Beefree answers 400, not 404, for a path that doesn't exist.
    parent, _, name = path.rstrip("/").rpartition("/")
    if any(item.get("name") == name for item in await list_folder(settings, uid, parent or "/")):
        resp.raise_for_status()


async def validate_reusable_rows(settings: Settings, reusable_rows: dict) -> None:
    """Raise ReusableRowsRejected if Beefree would refuse this library.

    Uses Beefree's validation endpoint, which checks the payload against the same
    rules as session creation without creating a session.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{settings.bee_api_base}/v2/sdk/mcp/template/validate",
            headers={"Authorization": f"Bearer {settings.bee_api_key}"},
            json={"reusableRows": reusable_rows},
        )
        if not resp.is_success:
            raise httpx.HTTPStatusError(
                f"{resp.status_code} — {resp.text}",
                request=resp.request,
                response=resp,
            )
        result = resp.json()
        if not result.get("valid"):
            raise ReusableRowsRejected([str(e) for e in result.get("errors") or ["Invalid reusableRows"]])


async def validate_brand_rules(settings: Settings, brand_rules: dict) -> None:
    """Raise BrandRulesRejected if Beefree would refuse these rules.

    Beefree validates brandRules when a session is created, so this creates a
    throwaway blank session; it is never used and expires on its own.
    """
    await create_template(settings, brand_rules)


async def get_template(template_id: str, settings: Settings) -> dict:
    """Retrieve the current state of a template."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{settings.bee_api_base}/v2/sdk/mcp/template/{template_id}",
            headers={"Authorization": f"Bearer {settings.bee_api_key}"},
        )
        resp.raise_for_status()
        return resp.json()


async def render_html(template: dict, settings: Settings) -> str:
    """Render a template JSON to email-ready HTML."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{settings.bee_api_base}/v1/message/html",
            json=template,
            headers={"Authorization": f"Bearer {settings.bee_api_key}"},
        )
        resp.raise_for_status()
        return resp.text
