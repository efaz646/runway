"""Shared test setup: import the app modules, a simulated Gemini, and a throwaway cache."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import forecast  # noqa: E402
import llm  # noqa: E402
from fake_gemini import FakeGemini  # noqa: E402


@pytest.fixture
def data():
    """(statement, invoices, forecast) for Dave's sample data."""
    return forecast.load_and_forecast()


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    """Never read or overwrite the real cache/last_agent_run.json during tests."""
    monkeypatch.setattr(config, "CACHE_FILE", tmp_path / "last_agent_run.json")


@pytest.fixture
def no_env_file(monkeypatch, tmp_path):
    """Ignore the real .env, so offline tests behave the same on a machine that has keys."""
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "missing.env")
    monkeypatch.setattr(config, "SECRETS_FILES", ())


@pytest.fixture
def fake_gemini(monkeypatch):
    """Install a simulated Gemini. Call it with options, e.g. fake_gemini(plan_mistake=True)."""

    def install(**options) -> FakeGemini:
        fake = FakeGemini(**options)
        monkeypatch.setattr(llm, "_client", lambda: fake)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        return fake

    return install


@pytest.fixture
def no_keys(monkeypatch, no_env_file):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
