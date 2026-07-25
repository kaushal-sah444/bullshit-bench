"""Model registry and provider adapters for Bullshit-Bench.

Every model the benchmark can talk to is described by a :class:`ModelSpec` and
reached through a single function, :func:`generate`. Core benchmark logic
(``run_bench.py``, ``scorer.py``) never imports a vendor SDK directly, so adding
a model is a registry change and nothing else.

Three ways to add a model, in increasing order of effort:

1. Drop an entry into ``models.json`` next to this file (no code change).
2. Call :func:`register_model` from your own script.
3. Add a ``ModelSpec`` to ``_BUILTIN_MODELS`` below.

Adding a whole new *provider* means writing one ``_call_*`` function and adding
it to ``_DISPATCH``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List

from dotenv import load_dotenv

load_dotenv()

MODELS_JSON = Path(__file__).resolve().parent / "models.json"


class ProviderError(RuntimeError):
    """Base class for provider problems."""


class MissingDependency(ProviderError):
    """The provider's SDK is not installed. Not worth retrying."""


class MissingCredentials(ProviderError):
    """No API key in the environment for this provider. Not worth retrying."""


class UnknownModel(ProviderError):
    """The requested model key is not in the registry."""


class EmptyResponse(ProviderError):
    """The call succeeded but carried no usable text.

    Usually means the output budget was exhausted before any visible text was
    produced (models that think by default spend ``max_tokens`` on reasoning
    first), or the provider blocked the response. Deterministic, so it is not
    retried — raise ``--max-tokens`` instead. Recorded as an error rather than
    scored, because grading an empty string as 0/10 would invent a result.
    """


@dataclass(frozen=True)
class ModelSpec:
    """Everything the benchmark needs to know about one model.

    Attributes:
        key: Stable identifier used in CLI flags, result files and the
            leaderboard. Keep it human-readable.
        provider: One of ``anthropic``, ``openai``, ``google``.
        model_id: The identifier the provider's API expects.
        label: Display name for charts and tables.
        api_key_env: Environment variable holding this provider's API key.
        supports_temperature: ``False`` for models that reject sampling
            parameters (current Anthropic frontier models return HTTP 400 for
            ``temperature``/``top_p``/``top_k``; several OpenAI reasoning models
            behave the same way). The temperature is silently omitted for those.
        max_tokens_param: Name of the output-cap argument. OpenAI's newer models
            renamed ``max_tokens`` to ``max_completion_tokens``.
        enabled_by_default: Whether ``run_bench.py`` includes it when no
            ``--models`` flag is given.
    """

    key: str
    provider: str
    model_id: str
    label: str
    api_key_env: str
    supports_temperature: bool = True
    max_tokens_param: str = "max_tokens"
    enabled_by_default: bool = False


# The default roster is deliberately small: one current model per provider.
# Everything else is opt-in via `--models`.
_BUILTIN_MODELS: List[ModelSpec] = [
    # --- Anthropic -----------------------------------------------------------
    ModelSpec(
        key="claude-opus-5",
        provider="anthropic",
        model_id="claude-opus-5",
        label="Claude Opus 5",
        api_key_env="ANTHROPIC_API_KEY",
        supports_temperature=False,  # sampling params return 400 on this model
        enabled_by_default=True,
    ),
    ModelSpec(
        key="claude-sonnet-5",
        provider="anthropic",
        model_id="claude-sonnet-5",
        label="Claude Sonnet 5",
        api_key_env="ANTHROPIC_API_KEY",
        supports_temperature=False,
    ),
    ModelSpec(
        key="claude-haiku-4-5",
        provider="anthropic",
        model_id="claude-haiku-4-5",
        label="Claude Haiku 4.5",
        api_key_env="ANTHROPIC_API_KEY",
        supports_temperature=True,
    ),
    # --- OpenAI --------------------------------------------------------------
    ModelSpec(
        key="gpt-4o",
        provider="openai",
        model_id="gpt-4o",
        label="GPT-4o",
        api_key_env="OPENAI_API_KEY",
        enabled_by_default=True,
    ),
    ModelSpec(
        key="gpt-4o-mini",
        provider="openai",
        model_id="gpt-4o-mini",
        label="GPT-4o mini",
        api_key_env="OPENAI_API_KEY",
    ),
    # --- Google --------------------------------------------------------------
    ModelSpec(
        key="gemini-2.0-flash",
        provider="google",
        model_id="gemini-2.0-flash",
        label="Gemini 2.0 Flash",
        api_key_env="GOOGLE_API_KEY",
        enabled_by_default=True,
    ),
    ModelSpec(
        key="gemini-1.5-pro",
        provider="google",
        model_id="gemini-1.5-pro",
        label="Gemini 1.5 Pro",
        api_key_env="GOOGLE_API_KEY",
    ),
]

MODELS: Dict[str, ModelSpec] = {spec.key: spec for spec in _BUILTIN_MODELS}


def register_model(spec: ModelSpec) -> None:
    """Add (or replace) a model in the registry."""
    MODELS[spec.key] = spec


def _load_models_json(path: Path = MODELS_JSON) -> None:
    """Merge user-supplied model definitions from ``models.json``, if present.

    The file is a JSON list of objects with the same field names as
    :class:`ModelSpec`. Entries whose ``key`` already exists override the
    built-in definition field by field, so you can flip one flag without
    restating the whole spec.
    """
    if not path.exists():
        return
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - config typo path
        raise ProviderError(f"{path.name} is not valid JSON: {exc}") from exc

    for entry in entries:
        key = entry.get("key")
        if not key:
            raise ProviderError(f"{path.name}: every entry needs a 'key'")
        if key in MODELS:
            MODELS[key] = replace(MODELS[key], **entry)
        else:
            MODELS[key] = ModelSpec(**entry)


_load_models_json()


def get_model(key: str) -> ModelSpec:
    """Look up a model spec by key."""
    try:
        return MODELS[key]
    except KeyError:
        known = ", ".join(sorted(MODELS))
        raise UnknownModel(f"Unknown model {key!r}. Known models: {known}") from None


def default_model_keys() -> List[str]:
    """Model keys used when the caller does not pass ``--models``."""
    return [spec.key for spec in MODELS.values() if spec.enabled_by_default]


def available_model_keys() -> List[str]:
    """Model keys whose provider API key is actually present in the env."""
    return [key for key, spec in MODELS.items() if os.getenv(spec.api_key_env)]


def _require_key(spec: ModelSpec) -> str:
    key = os.getenv(spec.api_key_env)
    if not key:
        raise MissingCredentials(
            f"{spec.api_key_env} is not set; cannot call {spec.key}. "
            "Copy .env.example to .env and fill it in."
        )
    return key


# --------------------------------------------------------------------------- #
# Provider adapters
#
# Clients are cached per provider: constructing them is cheap but not free, and
# the SDKs pool HTTP connections internally.
# --------------------------------------------------------------------------- #

_clients: Dict[str, Any] = {}


def _anthropic_client(spec: ModelSpec):
    if "anthropic" not in _clients:
        try:
            import anthropic
        except ImportError as exc:
            raise MissingDependency("pip install anthropic") from exc
        _clients["anthropic"] = anthropic.Anthropic(api_key=_require_key(spec))
    return _clients["anthropic"]


def _call_anthropic(spec: ModelSpec, prompt: str, temperature: float, max_tokens: int) -> str:
    """Single-turn completion via the Anthropic Messages API."""
    client = _anthropic_client(spec)
    kwargs: Dict[str, Any] = {
        "model": spec.model_id,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if spec.supports_temperature:
        kwargs["temperature"] = temperature

    message = client.messages.create(**kwargs)
    # `content` is a list of blocks; thinking blocks (when the model thinks by
    # default) carry no visible text, so filter to text blocks.
    text = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    ).strip()
    if not text:
        raise EmptyResponse(
            f"{spec.key} returned no text (stop_reason="
            f"{getattr(message, 'stop_reason', None)!r}). Models that think by "
            "default spend max_tokens on reasoning first - retry with a larger "
            "--max-tokens."
        )
    return text


def _openai_client(spec: ModelSpec):
    if "openai" not in _clients:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise MissingDependency("pip install openai") from exc
        _clients["openai"] = OpenAI(api_key=_require_key(spec))
    return _clients["openai"]


def _call_openai(spec: ModelSpec, prompt: str, temperature: float, max_tokens: int) -> str:
    """Single-turn completion via the OpenAI Chat Completions API."""
    client = _openai_client(spec)
    kwargs: Dict[str, Any] = {
        "model": spec.model_id,
        "messages": [{"role": "user", "content": prompt}],
        spec.max_tokens_param: max_tokens,
    }
    if spec.supports_temperature:
        kwargs["temperature"] = temperature

    response = client.chat.completions.create(**kwargs)
    if not response.choices:
        raise EmptyResponse(f"{spec.key} returned no choices")

    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        raise EmptyResponse(
            f"{spec.key} returned no text (finish_reason="
            f"{getattr(choice, 'finish_reason', None)!r}). If it is 'length', "
            "retry with a larger --max-tokens."
        )
    return text


def _google_client(spec: ModelSpec):
    """Return ``(sdk_kind, handle)`` for Google, preferring the current SDK.

    ``google-generativeai`` reached end of life and no longer receives updates,
    so ``google-genai`` is tried first and the old package is only a fallback for
    environments still pinned to it. The two have different call shapes, hence
    the tag.
    """
    cache_key = f"google:{spec.model_id}"
    if cache_key in _clients:
        return _clients[cache_key]

    api_key = _require_key(spec)
    try:
        from google import genai  # google-genai (current)

        # The Anthropic and OpenAI SDKs pick up ANTHROPIC_BASE_URL / OPENAI_BASE_URL
        # on their own; google-genai has no equivalent, so wire one up for parity
        # (useful for proxies and gateways, and for pointing tests at a stub).
        base_url = os.getenv("GOOGLE_GENAI_BASE_URL")
        http_options = {"base_url": base_url} if base_url else None
        _clients[cache_key] = (
            "genai",
            genai.Client(api_key=api_key, http_options=http_options),
        )
    except ImportError:
        try:
            import google.generativeai as legacy  # google-generativeai (EOL)
        except ImportError as exc:
            raise MissingDependency("pip install google-genai") from exc
        legacy.configure(api_key=api_key)
        _clients[cache_key] = ("legacy", legacy.GenerativeModel(spec.model_id))

    return _clients[cache_key]


def _call_google(spec: ModelSpec, prompt: str, temperature: float, max_tokens: int) -> str:
    """Single-turn completion via the Google GenAI SDK."""
    kind, handle = _google_client(spec)
    config: Dict[str, Any] = {"max_output_tokens": max_tokens}
    if spec.supports_temperature:
        config["temperature"] = temperature

    if kind == "genai":
        response = handle.models.generate_content(
            model=spec.model_id, contents=prompt, config=config
        )
    else:
        response = handle.generate_content(prompt, generation_config=config)

    text = ""
    try:
        text = (response.text or "").strip()
    except (ValueError, AttributeError):
        # `.text` raises when the response was blocked or has no text parts;
        # fall back to walking the first candidate's parts by hand.
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []
            text = "".join(getattr(part, "text", "") or "" for part in parts).strip()

    if not text:
        reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        raise EmptyResponse(
            f"{spec.key} returned no text"
            + (f" (block_reason={reason!r})" if reason else "")
            + ". If the finish reason was the token cap, retry with a larger --max-tokens."
        )
    return text


_DISPATCH: Dict[str, Callable[[ModelSpec, str, float, int], str]] = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "google": _call_google,
}


def generate(
    model_key: str,
    prompt: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Send ``prompt`` to ``model_key`` and return the response text.

    Args:
        model_key: A key from the registry (see :data:`MODELS`).
        prompt: The user-turn text.
        temperature: Sampling temperature. Ignored for models whose spec sets
            ``supports_temperature=False``.
        max_tokens: Output cap.

    Raises:
        UnknownModel: ``model_key`` is not registered.
        MissingDependency: The provider SDK is not installed.
        MissingCredentials: The provider's API key is not in the environment.
        ProviderError: The provider returned nothing usable.
    """
    spec = get_model(model_key)
    call = _DISPATCH.get(spec.provider)
    if call is None:
        raise ProviderError(f"No adapter for provider {spec.provider!r}")
    return call(spec, prompt, temperature, max_tokens)
