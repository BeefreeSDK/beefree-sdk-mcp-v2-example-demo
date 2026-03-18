"""Thin async wrapper around the Beefree REST endpoints used before/after MCP."""

import httpx

from .config import Settings


async def create_template(settings: Settings) -> str:
    """Create a blank template session and return the templateId."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{settings.bee_api_base}/v2/sdk/mcp/template",
            headers={"Authorization": f"Bearer {settings.bee_api_key}"},
            json={},
        )
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
) -> str:
    """Create a template session pre-seeded with existing template content."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{settings.bee_api_base}/v2/sdk/mcp/template",
            headers={"Authorization": f"Bearer {settings.bee_api_key}"},
            json={"template": template_json},
        )
        if not resp.is_success:
            raise httpx.HTTPStatusError(
                f"{resp.status_code} — {resp.text}",
                request=resp.request,
                response=resp,
            )
        return resp.json()["templateId"]


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
