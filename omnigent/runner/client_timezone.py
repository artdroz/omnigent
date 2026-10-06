"""The viewer's local timezone, remembered per session.

A web client reports its IANA zone on every user message it sends
(``client_timezone`` on ``POST /v1/sessions/{id}/events``); the server
validates it and forwards it on the turn body. The runner keeps the latest
value per session so the tools and prompt of that session can speak in the
user's wall-clock time: ``sys_scheduled_task_create`` evaluates a schedule in
this zone when the model omits one, and the composed instructions tell the
model which zone the user is in. Nothing is remembered for sessions whose
clients never report a zone (CLI, SDK), so those keep the existing UTC
defaults.
"""

from __future__ import annotations

from omnigent.util.timezones import is_valid_timezone

_session_client_timezones: dict[str, str] = {}


def remember_client_timezone(conversation_id: str, timezone: object) -> None:
    """
    Record the zone a client reported for *conversation_id*.

    :param conversation_id: Session/conversation id.
    :param timezone: The reported value; anything but a valid IANA key is ignored.
    """
    if is_valid_timezone(timezone):
        _session_client_timezones[conversation_id] = str(timezone)


def client_timezone_for(conversation_id: str | None) -> str | None:
    """
    Return the zone most recently reported for *conversation_id*.

    :param conversation_id: Session/conversation id, or ``None`` when unknown.
    :returns: An IANA zone key, or ``None`` when no client reported one.
    """
    if conversation_id is None:
        return None
    return _session_client_timezones.get(conversation_id)


def forget_client_timezone(conversation_id: str) -> None:
    """
    Drop the remembered zone when a session is torn down.

    :param conversation_id: Session/conversation id.
    """
    _session_client_timezones.pop(conversation_id, None)
