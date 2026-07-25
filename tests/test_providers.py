"""Registry behaviour and the three provider adapters, against the mock API."""

from __future__ import annotations

import json

import pytest

import mock_api
import providers


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def test_unknown_model_lists_the_known_ones():
    with pytest.raises(providers.UnknownModel) as exc:
        providers.get_model("gpt-9-ultra")
    assert "gpt-4o" in str(exc.value)


def test_missing_key_is_reported_not_retried(no_keys):
    with pytest.raises(providers.MissingCredentials):
        providers.generate("gpt-4o", "hello")


def test_available_model_keys_tracks_the_environment(monkeypatch, no_keys):
    assert providers.available_model_keys() == []
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    available = providers.available_model_keys()
    assert "gpt-4o" in available
    assert "claude-opus-5" not in available


def test_models_json_patches_a_builtin_spec_field_by_field(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps([{"key": "gpt-4o", "label": "Renamed"}]), encoding="utf-8")
    original = providers.MODELS["gpt-4o"]
    try:
        providers._load_models_json(path)  # noqa: SLF001
        patched = providers.MODELS["gpt-4o"]
        assert patched.label == "Renamed"
        assert patched.model_id == original.model_id  # untouched fields survive
    finally:
        providers.MODELS["gpt-4o"] = original


def test_models_json_can_add_a_new_model(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps([{
            "key": "brand-new", "provider": "openai", "model_id": "x",
            "label": "Brand New", "api_key_env": "OPENAI_API_KEY",
        }]),
        encoding="utf-8",
    )
    try:
        providers._load_models_json(path)  # noqa: SLF001
        assert providers.get_model("brand-new").label == "Brand New"
    finally:
        providers.MODELS.pop("brand-new", None)


# --------------------------------------------------------------------------- #
# Adapters (real SDKs -> mock HTTP)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("model_key", ["claude-opus-5", "gpt-4o", "gemini-2.0-flash"])
def test_every_provider_round_trips(mock_env, model_key):
    text = providers.generate(model_key, "What is the population of Wakanda?")
    assert text and isinstance(text, str)
    assert text == text.strip()
    assert all(call["authenticated"] for call in mock_api.CALLS)


def test_temperature_is_withheld_from_models_that_reject_it(mock_env):
    """claude-opus-5 rejects sampling params; the mock 400s if we send one."""
    providers.generate("claude-opus-5", "hi", temperature=0.7)
    body = mock_api.CALLS[-1]["body"]
    assert "temperature" not in body
    assert body["max_tokens"] == 2048


def test_temperature_is_sent_to_models_that_accept_it(mock_env):
    providers.generate("claude-haiku-4-5", "hi", temperature=0.7)
    assert mock_api.CALLS[-1]["body"]["temperature"] == 0.7


def test_openai_uses_the_configured_token_parameter(mock_env):
    providers.generate("gpt-4o", "hi", max_tokens=256)
    assert mock_api.CALLS[-1]["body"]["max_tokens"] == 256


@pytest.mark.parametrize("model_key", ["claude-opus-5", "gpt-4o", "gemini-2.0-flash"])
def test_no_text_raises_empty_response_rather_than_returning_blank(mock_env, model_key):
    """A budget-exhausted reply must surface as an error, never as an empty answer.

    Scoring "" would fabricate a 0/10 for a model that was simply cut off.
    """
    mock_api.EMPTY_MODELS.add(providers.get_model(model_key).model_id)
    with pytest.raises(providers.EmptyResponse):
        providers.generate(model_key, "hi", max_tokens=16)
