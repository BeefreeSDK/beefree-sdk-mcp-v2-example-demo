"""Thin async wrapper around the Beefree REST endpoints used before/after MCP."""

import httpx

from .config import Settings


class BrandRulesRejected(Exception):
    """Beefree refused a brandRules object; ``details`` lists each problem."""

    def __init__(self, details: list[str]):
        super().__init__("; ".join(details))
        self.details = details


async def create_template(settings: Settings, brand_rules: dict | None = None) -> str:
    """Create a blank template session and return the templateId.

    With ``brand_rules``, Beefree enforces them on every Code Mode call in the session.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{settings.bee_api_base}/v2/sdk/mcp/template",
            headers={"Authorization": f"Bearer {settings.bee_api_key}"},
            json={"brandRules": brand_rules} if brand_rules else {},
        )
        if brand_rules and resp.status_code == 400:
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
