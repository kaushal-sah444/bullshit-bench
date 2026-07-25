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
from typing import Any, Callable, Dict, List, Optional, Tuple

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


class EndpointUnreachable(ProviderError):
    """A keyless local endpoint is not accepting connections.

    Almost always means the server simply is not running. Deterministic, so it
    is not retried: backing off five times cannot start Ollama for you, and
    doing so wastes minutes per model before the real message appears.
    """


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
        provider: Adapter to use — ``anthropic``, ``google``, or
            ``openai_compatible`` (which covers OpenAI itself plus Ollama,
            OpenRouter, Groq, Together, DeepSeek, Mistral, xAI, LM Studio, vLLM
            and anything else speaking the Chat Completions API).
        model_id: The identifier the provider's API expects.
        label: Display name for charts and tables.
        api_key_env: Environment variable holding the API key. **Empty means the
            endpoint needs no key** — the normal case for a local server.
        base_url: API root. ``None`` uses the SDK default.
        base_url_env: Environment variable that overrides ``base_url``, so a
            local server on a different host or port needs no code change.
        supports_temperature: ``False`` for models that reject sampling
            parameters (current Anthropic frontier models return HTTP 400 for
            ``temperature``/``top_p``/``top_k``; several OpenAI reasoning models
            behave the same way). The temperature is silently omitted for those.
        supports_json_mode: Whether the endpoint accepts OpenAI-style
            ``response_format={"type": "json_object"}``. Used to hold small local
            judges to valid JSON.
        max_tokens_param: Name of the output-cap argument. OpenAI's newer models
            renamed ``max_tokens`` to ``max_completion_tokens``.
        request_delay_s: Seconds to wait after each call. Free tiers meter
            aggressively; pacing beats burning retries on 429s.
        free: Whether running this model costs nothing (local, or a free tier).
            Display only.
        enabled_by_default: Whether ``run_bench.py`` includes it when no
            ``--models`` flag is given.
    """

    key: str
    provider: str
    model_id: str
    label: str
    api_key_env: str = ""
    base_url: Optional[str] = None
    base_url_env: str = ""
    supports_temperature: bool = True
    supports_json_mode: bool = False
    max_tokens_param: str = "max_tokens"
    request_delay_s: float = 0.0
    free: bool = False
    enabled_by_default: bool = False

    @property
    def needs_key(self) -> bool:
        """Whether this endpoint requires an API key at all."""
        return bool(self.api_key_env)

    def resolved_base_url(self) -> Optional[str]:
        """Base URL after applying the environment override."""
        return (os.getenv(self.base_url_env) if self.base_url_env else None) or self.base_url


@dataclass(frozen=True)
class ProviderPreset:
    """Connection defaults for one platform, used by ``provider:model`` refs."""

    provider: str
    api_key_env: str = ""
    base_url: Optional[str] = None
    base_url_env: str = ""
    supports_json_mode: bool = False
    request_delay_s: float = 0.0
    free: bool = False
    label: str = ""


# Anything speaking the OpenAI Chat Completions API is one entry here. This is
# what makes `--models ollama:llama3.2` or `--models groq:<id>` work with no code
# change: the alias before the colon picks the preset, the rest is the model id.
PROVIDER_PRESETS: Dict[str, ProviderPreset] = {
    # --- first-party -------------------------------------------------------- #
    "openai": ProviderPreset(
        provider="openai_compatible", api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL", supports_json_mode=True, label="OpenAI",
    ),
    "anthropic": ProviderPreset(
        provider="anthropic", api_key_env="ANTHROPIC_API_KEY",
        base_url_env="ANTHROPIC_BASE_URL", label="Anthropic",
    ),
    "google": ProviderPreset(
        provider="google", api_key_env="GOOGLE_API_KEY",
        base_url_env="GOOGLE_GENAI_BASE_URL", label="Google",
    ),
    # --- free: local, no key, no cost --------------------------------------- #
    "ollama": ProviderPreset(
        provider="openai_compatible", base_url="http://localhost:11434/v1",
        base_url_env="OLLAMA_BASE_URL", supports_json_mode=True, free=True,
        label="Ollama",
    ),
    "lmstudio": ProviderPreset(
        provider="openai_compatible", base_url="http://localhost:1234/v1",
        base_url_env="LMSTUDIO_BASE_URL", supports_json_mode=True, free=True,
        label="LM Studio",
    ),
    "vllm": ProviderPreset(
        provider="openai_compatible", base_url="http://localhost:8000/v1",
        base_url_env="VLLM_BASE_URL", supports_json_mode=True, free=True,
        label="vLLM",
    ),
    "llamacpp": ProviderPreset(
        provider="openai_compatible", base_url="http://localhost:8080/v1",
        base_url_env="LLAMACPP_BASE_URL", supports_json_mode=True, free=True,
        label="llama.cpp",
    ),
    # --- hosted, with free tiers -------------------------------------------- #
    "openrouter": ProviderPreset(
        provider="openai_compatible", api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1", base_url_env="OPENROUTER_BASE_URL",
        supports_json_mode=True, request_delay_s=3.0, label="OpenRouter",
    ),
    "groq": ProviderPreset(
        provider="openai_compatible", api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1", base_url_env="GROQ_BASE_URL",
        supports_json_mode=True, request_delay_s=2.0, label="Groq",
    ),
    # --- other paid, OpenAI-compatible -------------------------------------- #
    "deepseek": ProviderPreset(
        provider="openai_compatible", api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1", base_url_env="DEEPSEEK_BASE_URL",
        supports_json_mode=True, label="DeepSeek",
    ),
    "mistral": ProviderPreset(
        provider="openai_compatible", api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1", base_url_env="MISTRAL_BASE_URL",
        supports_json_mode=True, label="Mistral",
    ),
    "together": ProviderPreset(
        provider="openai_compatible", api_key_env="TOGETHER_API_KEY",
        base_url="https://api.together.xyz/v1", base_url_env="TOGETHER_BASE_URL",
        supports_json_mode=True, label="Together",
    ),
    "xai": ProviderPreset(
        provider="openai_compatible", api_key_env="XAI_API_KEY",
        base_url="https://api.x.ai/v1", base_url_env="XAI_BASE_URL",
        supports_json_mode=True, label="xAI",
    ),
    # --- escape hatch: anything else ---------------------------------------- #
    "custom": ProviderPreset(
        provider="openai_compatible", api_key_env="CUSTOM_API_KEY",
        base_url_env="CUSTOM_BASE_URL", supports_json_mode=True, label="Custom",
    ),
}


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
        provider="openai_compatible",
        model_id="gpt-4o",
        label="GPT-4o",
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        supports_json_mode=True,
        enabled_by_default=True,
    ),
    ModelSpec(
        key="gpt-4o-mini",
        provider="openai_compatible",
        model_id="gpt-4o-mini",
        label="GPT-4o mini",
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        supports_json_mode=True,
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
    # --- Free: local models via Ollama --------------------------------------- #
    # No key, no cost, no network. `ollama pull <model>` then benchmark it.
    # Any other tag works too, without touching this file: `--models ollama:phi4`.
    ModelSpec(
        key="ollama-llama3.2",
        provider="openai_compatible",
        model_id="llama3.2",
        label="Llama 3.2 (Ollama)",
        base_url="http://localhost:11434/v1",
        base_url_env="OLLAMA_BASE_URL",
        supports_json_mode=True,
        free=True,
    ),
    ModelSpec(
        key="ollama-qwen2.5",
        provider="openai_compatible",
        model_id="qwen2.5",
        label="Qwen 2.5 (Ollama)",
        base_url="http://localhost:11434/v1",
        base_url_env="OLLAMA_BASE_URL",
        supports_json_mode=True,
        free=True,
    ),
    ModelSpec(
        key="ollama-mistral",
        provider="openai_compatible",
        model_id="mistral",
        label="Mistral 7B (Ollama)",
        base_url="http://localhost:11434/v1",
        base_url_env="OLLAMA_BASE_URL",
        supports_json_mode=True,
        free=True,
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


def spec_from_ref(ref: str) -> ModelSpec:
    """Build a spec from an ad-hoc ``provider:model`` reference.

    This is what lets the benchmark reach *any* model on a supported platform
    without editing the registry::

        ollama:llama3.2
        openrouter:meta-llama/llama-3.3-70b-instruct:free
        groq:llama-3.3-70b-versatile
        custom:my-model            # with CUSTOM_BASE_URL set

    Only the first colon splits — model ids may contain their own (OpenRouter's
    ``:free`` suffix, for one).

    Args:
        ref: A ``alias:model_id`` string whose alias is in
            :data:`PROVIDER_PRESETS`.

    Returns:
        A :class:`ModelSpec` keyed by the reference itself.

    Raises:
        UnknownModel: The alias is not a known platform, or the model id is
            empty.
    """
    alias, _, model_id = ref.partition(":")
    preset = PROVIDER_PRESETS.get(alias.lower())
    if preset is None:
        known = ", ".join(sorted(PROVIDER_PRESETS))
        raise UnknownModel(f"Unknown platform {alias!r} in {ref!r}. Known: {known}")
    if not model_id:
        raise UnknownModel(f"{ref!r} is missing a model id (expected '{alias}:<model>')")

    platform = preset.label or alias
    return ModelSpec(
        key=ref,
        provider=preset.provider,
        model_id=model_id,
        label=f"{model_id} ({platform})",
        api_key_env=preset.api_key_env,
        base_url=preset.base_url,
        base_url_env=preset.base_url_env,
        supports_json_mode=preset.supports_json_mode,
        request_delay_s=preset.request_delay_s,
        # OpenRouter marks its no-cost models with a `:free` suffix.
        free=preset.free or model_id.endswith(":free"),
    )


def get_model(key: str) -> ModelSpec:
    """Look up a model spec by key, or build one from a ``provider:model`` ref.

    Registry keys win over ad-hoc references, so a curated entry can shadow the
    generic form.
    """
    if key in MODELS:
        return MODELS[key]

    if ":" in key:
        spec = spec_from_ref(key)
        MODELS[key] = spec  # cache so later lookups (and results) agree
        return spec

    known = ", ".join(sorted(MODELS))
    raise UnknownModel(
        f"Unknown model {key!r}.\nRegistry: {known}\n"
        f"Or use a platform reference like 'ollama:llama3.2' "
        f"({', '.join(sorted(PROVIDER_PRESETS))})."
    )


def default_model_keys() -> List[str]:
    """Model keys used when the caller does not pass ``--models``."""
    return [spec.key for spec in MODELS.values() if spec.enabled_by_default]


def available_model_keys() -> List[str]:
    """Model keys that are usable right now.

    A model counts as usable when its API key is present, or when the endpoint
    needs no key at all (a local server). Whether that local server is actually
    running is a separate question — see :func:`endpoint_is_reachable`.
    """
    return [
        key
        for key, spec in MODELS.items()
        if not spec.needs_key or os.getenv(spec.api_key_env)
    ]


def free_model_keys() -> List[str]:
    """Registry keys that cost nothing to run."""
    return [key for key, spec in MODELS.items() if spec.free]


def _require_key(spec: ModelSpec) -> str:
    """Return the API key for ``spec``, or a placeholder when none is needed.

    Keyless endpoints (a local Ollama or LM Studio server) still need *something*
    in the header for the OpenAI client, which rejects an empty api_key.
    """
    if not spec.needs_key:
        return "not-needed"

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


def _require_local_server(spec: ModelSpec) -> None:
    """Fail fast, and helpfully, when a keyless local endpoint is not up.

    Without this the SDK raises a connection error that the caller then retries
    with backoff — minutes of silence per model before the user learns the
    server simply is not running.
    """
    if spec.needs_key or endpoint_is_reachable(spec):
        return

    hint = f"set {spec.base_url_env}" if spec.base_url_env else "point it elsewhere"
    raise EndpointUnreachable(
        f"Nothing is listening at {spec.resolved_base_url()} for {spec.key!r}. "
        f"Start the server (for Ollama: `ollama serve`, then "
        f"`ollama pull {spec.model_id}`), or {hint} to a running one."
    )


def _create_completion(spec: ModelSpec, client: Any, kwargs: Dict[str, Any]):
    """Call the endpoint, turning a dead local server into a clear error."""
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 - re-raised either way
        # A remote endpoint may be having a transient blip, which is worth
        # retrying. A refused local port is not.
        if not spec.needs_key and _looks_like_connection_error(exc):
            _reachable_cache[spec.resolved_base_url() or ""] = False
            raise EndpointUnreachable(
                f"Could not reach {spec.resolved_base_url()} for {spec.key!r}: {exc}. "
                "Is the server running?"
            ) from exc
        raise


def _looks_like_connection_error(exc: BaseException) -> bool:
    """Whether an SDK exception means 'nothing answered', not 'it said no'."""
    if isinstance(exc, (ConnectionError, OSError)):
        return True
    return type(exc).__name__ in {
        "APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout",
    }


def _looks_like_bad_request(exc: BaseException) -> bool:
    """Whether an SDK exception represents a 4xx the server chose to reject."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return 400 <= status < 500
    return "unsupported" in str(exc).lower() or "unrecognized" in str(exc).lower()


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
    """Client for any Chat Completions endpoint.

    Cached per (base URL, key variable) so OpenAI, a local Ollama server and
    OpenRouter can all be in the same run without clobbering each other.
    """
    base_url = spec.resolved_base_url()
    cache_key = f"openai:{base_url}:{spec.api_key_env}"
    if cache_key not in _clients:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise MissingDependency("pip install openai") from exc
        kwargs: Dict[str, Any] = {"api_key": _require_key(spec)}
        if base_url:
            kwargs["base_url"] = base_url
        if not spec.needs_key:
            # A refused connection to a local port is final; the SDK's own
            # connection retries only add delay before the useful error.
            kwargs["max_retries"] = 0
        _clients[cache_key] = OpenAI(**kwargs)
    return _clients[cache_key]


def _call_openai(
    spec: ModelSpec,
    prompt: str,
    temperature: float,
    max_tokens: int,
    *,
    json_mode: bool = False,
) -> str:
    """Single-turn completion via any OpenAI-compatible Chat Completions API.

    Covers OpenAI itself plus Ollama, LM Studio, vLLM, llama.cpp, OpenRouter,
    Groq, Together, DeepSeek, Mistral and xAI — they all speak this shape.

    Args:
        json_mode: Ask the endpoint to constrain output to valid JSON. Only used
            for judge calls, and only where the spec says it is supported.
    """
    _require_local_server(spec)

    client = _openai_client(spec)
    kwargs: Dict[str, Any] = {
        "model": spec.model_id,
        "messages": [{"role": "user", "content": prompt}],
        spec.max_tokens_param: max_tokens,
    }
    if spec.supports_temperature:
        kwargs["temperature"] = temperature
    if json_mode and spec.supports_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = _create_completion(spec, client, kwargs)
    except Exception as exc:  # noqa: BLE001 - narrow retry for one known case
        # Some self-hosted servers reject response_format outright. Losing JSON
        # mode is far better than losing the call.
        if "response_format" not in kwargs or not _looks_like_bad_request(exc):
            raise
        kwargs.pop("response_format")
        response = _create_completion(spec, client, kwargs)
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


_DISPATCH: Dict[str, Callable[..., str]] = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,  # kept as an alias for older models.json files
    "openai_compatible": _call_openai,
    "google": _call_google,
}


def generate(
    model_key: str,
    prompt: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    json_mode: bool = False,
) -> str:
    """Send ``prompt`` to ``model_key`` and return the response text.

    Args:
        model_key: A registry key (see :data:`MODELS`) or a ``provider:model``
            reference such as ``ollama:llama3.2``.
        prompt: The user-turn text.
        temperature: Sampling temperature. Ignored for models whose spec sets
            ``supports_temperature=False``.
        max_tokens: Output cap.
        json_mode: Request JSON-constrained output where the endpoint supports
            it. Used by the judge; ignored by providers without the feature.

    Raises:
        UnknownModel: ``model_key`` is neither registered nor a valid reference.
        MissingDependency: The provider SDK is not installed.
        MissingCredentials: The endpoint needs an API key and none is set.
        EmptyResponse: The call returned no usable text.
        ProviderError: Any other provider-level problem.
    """
    spec = get_model(model_key)
    call = _DISPATCH.get(spec.provider)
    if call is None:
        raise ProviderError(f"No adapter for provider {spec.provider!r}")

    if spec.provider in ("openai", "openai_compatible"):
        return call(spec, prompt, temperature, max_tokens, json_mode=json_mode)
    return call(spec, prompt, temperature, max_tokens)


_reachable_cache: Dict[str, bool] = {}


def endpoint_is_reachable(spec: ModelSpec, timeout: float = 1.0) -> bool:
    """Cheap liveness probe for a keyless local endpoint.

    Used to decide whether a local model can act as the judge. Keyed endpoints
    are assumed reachable — the presence of a key is the signal there, and
    probing a paid API just to look would be rude. Results are cached per base
    URL so a per-response judge lookup does not reopen a socket every time.
    """
    base_url = spec.resolved_base_url()
    if spec.needs_key or not base_url:
        return True
    if base_url in _reachable_cache:
        return _reachable_cache[base_url]

    import socket
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            reachable = True
    except OSError:
        reachable = False

    _reachable_cache[base_url] = reachable
    return reachable
