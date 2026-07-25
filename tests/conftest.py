"""Shared fixtures.

The whole suite runs offline: a local mock API stands in for the three vendors,
so no API key and no network access is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mock_api  # noqa: E402  (tests/ is on the path via rootdir conftest)
import providers  # noqa: E402


@pytest.fixture(scope="session")
def api_server() -> Iterator[str]:
    """Run the mock API for the session; yields its base URL."""
    server = mock_api.serve()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture
def mock_env(api_server: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point every provider SDK at the mock server with fake credentials."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", api_server)
    monkeypatch.setenv("OPENAI_BASE_URL", f"{api_server}/v1")
    monkeypatch.setenv("GOOGLE_GENAI_BASE_URL", api_server)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)

    # Clients are cached per provider and capture the key/base URL at build
    # time, so drop them between tests.
    providers._clients.clear()  # noqa: SLF001
    mock_api.reset()
    yield api_server
    providers._clients.clear()  # noqa: SLF001


@pytest.fixture
def no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every provider credential from the environment."""
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "JUDGE_MODEL"):
        monkeypatch.delenv(var, raising=False)
    providers._clients.clear()  # noqa: SLF001
