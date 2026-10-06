"""Notify host discovery when a native harness observes skill changes."""

import logging

import httpx

_logger = logging.getLogger(__name__)


async def refresh_host_skills(client: httpx.AsyncClient, session_id: str) -> None:
    """Keep discovery failures independent of native turn delivery."""
    try:
        response = await client.post(f"/v1/sessions/{session_id}/skills/refresh", json={})
        response.raise_for_status()
    except httpx.HTTPError:
        _logger.warning("Could not refresh host skills for session %s", session_id, exc_info=True)
