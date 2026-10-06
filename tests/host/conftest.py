"""Isolate tunnel tests from inherited proxy settings."""

import pytest


@pytest.fixture(autouse=True)
def _no_ambient_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
