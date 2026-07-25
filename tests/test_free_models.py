"""Free and OpenAI-compatible endpoints: local servers, free tiers, ad-hoc refs.

The mock API already speaks Chat Completions, which is exactly what Ollama,
LM Studio, vLLM, OpenRouter, Groq and friends serve — so pointing a preset's
base-URL override at it exercises those paths for real.
"""

from __future__ import annotations

import pytest

import mock_api
import providers
import run_bench
import scorer


@pytest.fixture
def local_server(api_server: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """Pretend the mock API is a local Ollama server, with no key anywhere."""
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                "GROQ_API_KEY", "OPENROUTER_API_KEY", "JUDGE_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", f"{api_server}/v1")
    providers._clients.clear()  # noqa: SLF001
    providers._reachable_cache.clear()  # noqa: SLF001
    mock_api.reset()
    yield api_server
    providers._clients.clear()  # noqa: SLF001
    providers._reachable_cache.clear()  # noqa: SLF001


# --------------------------------------------------------------------------- #
# Ad-hoc provider:model references
# --------------------------------------------------------------------------- #

def test_a_reference_builds_a_spec_without_touching_the_registry():
    spec = providers.spec_from_ref("ollama:llama3.2")
    assert spec.provider == "openai_compatible"
    assert spec.model_id == "llama3.2"
    assert spec.free is True
    assert spec.needs_key is False


def test_model_ids_may_contain_colons():
    """OpenRouter marks no-cost models with a `:free` suffix."""
    spec = providers.spec_from_ref("openrouter:meta-llama/llama-3.3-70b-instruct:free")
    assert spec.model_id == "meta-llama/llama-3.3-70b-instruct:free"
    assert spec.free is True  # the suffix means no cost


def test_a_paid_platform_reference_is_not_marked_free():
    assert providers.spec_from_ref("groq:llama-3.3-70b-versatile").free is False


@pytest.mark.parametrize(
    "ref, expected_key_env",
    [
        ("openai:gpt-4o", "OPENAI_API_KEY"),
        ("anthropic:claude-haiku-4-5", "ANTHROPIC_API_KEY"),
        ("google:gemini-2.0-flash", "GOOGLE_API_KEY"),
        ("groq:x", "GROQ_API_KEY"),
        ("deepseek:x", "DEEPSEEK_API_KEY"),
        ("together:x", "TOGETHER_API_KEY"),
        ("mistral:x", "MISTRAL_API_KEY"),
        ("xai:x", "XAI_API_KEY"),
    ],
)
def test_every_hosted_platform_maps_to_its_own_key_variable(ref, expected_key_env):
    assert providers.spec_from_ref(ref).api_key_env == expected_key_env


@pytest.mark.parametrize("ref", ["lmstudio:x", "vllm:x", "llamacpp:x", "ollama:x"])
def test_local_platforms_need_no_key(ref):
    assert providers.spec_from_ref(ref).needs_key is False


def test_an_unknown_platform_is_rejected_with_the_known_list():
    with pytest.raises(providers.UnknownModel, match="ollama"):
        providers.spec_from_ref("notaplatform:some-model")


def test_a_reference_without_a_model_id_is_rejected():
    with pytest.raises(providers.UnknownModel, match="missing a model id"):
        providers.spec_from_ref("ollama:")


def test_get_model_accepts_references_and_caches_them():
    spec = providers.get_model("ollama:tinyllama")
    assert providers.get_model("ollama:tinyllama") is spec
    providers.MODELS.pop("ollama:tinyllama", None)


def test_registry_keys_take_precedence_over_references():
    assert providers.get_model("claude-opus-5").model_id == "claude-opus-5"


# --------------------------------------------------------------------------- #
# Calling a keyless local endpoint
# --------------------------------------------------------------------------- #

def test_a_local_model_answers_with_no_api_key_set(local_server):
    text = providers.generate("ollama:llama3.2", "Population of Wakanda?")
    assert text
    assert mock_api.CALLS[-1]["path"].endswith("/chat/completions")


def test_the_base_url_override_is_honoured(local_server, monkeypatch):
    """A local server on a different port needs no code change."""
    spec = providers.get_model("ollama:llama3.2")
    assert spec.resolved_base_url() == f"{local_server}/v1"
    monkeypatch.delenv("OLLAMA_BASE_URL")
    assert spec.resolved_base_url() == "http://localhost:11434/v1"


def test_a_free_tier_platform_shares_the_chat_completions_adapter(
    api_server, monkeypatch
):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("GROQ_BASE_URL", f"{api_server}/v1")
    providers._clients.clear()  # noqa: SLF001
    mock_api.reset()

    assert providers.generate("groq:llama-3.3-70b-versatile", "hi")
    assert mock_api.CALLS[-1]["authenticated"]


def test_endpoints_are_cached_separately_so_they_do_not_collide(
    local_server, monkeypatch
):
    """A local model and a hosted one in the same run must not share a client."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", f"{local_server}/v1")

    providers.generate("ollama:llama3.2", "hi")
    providers.generate("gpt-4o", "hi")

    clients = [k for k in providers._clients if k.startswith("openai:")]  # noqa: SLF001
    assert len(clients) == 2


# --------------------------------------------------------------------------- #
# Judging for free
# --------------------------------------------------------------------------- #

def test_a_reachable_local_server_becomes_the_judge_when_no_keys_exist(local_server):
    """The whole benchmark, LLM judge included, runs at zero cost."""
    assert scorer.resolve_judge() == "ollama-llama3.2"


def test_a_dead_local_server_fails_fast_with_an_actionable_message(monkeypatch):
    """The common beginner case: no key, and Ollama isn't running.

    This must be one quick error, not minutes of connection retries.
    """
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")  # nothing listens
    providers._reachable_cache.clear()  # noqa: SLF001

    with pytest.raises(providers.EndpointUnreachable) as exc:
        providers.generate("ollama:llama3.2", "hi")

    message = str(exc.value)
    assert "Nothing is listening" in message
    assert "ollama serve" in message  # tells them how to fix it
    assert "OLLAMA_BASE_URL" in message


def test_an_unreachable_server_is_not_retried(monkeypatch):
    """Backing off five times cannot start a server that is not running."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    providers._reachable_cache.clear()  # noqa: SLF001
    monkeypatch.setattr(run_bench.time, "sleep", lambda _s: None)

    calls = {"n": 0}

    def attempt():
        calls["n"] += 1
        return providers.generate("ollama:llama3.2", "hi")

    with pytest.raises(providers.EndpointUnreachable):
        run_bench.with_retry(attempt, description="t", verbose=False)
    assert calls["n"] == 1


def test_a_whole_run_against_a_dead_server_still_finishes(monkeypatch, tmp_path):
    """Every prompt errors, the run completes, and the file is written."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    providers._reachable_cache.clear()  # noqa: SLF001

    run = run_bench.run_benchmark(["ollama:llama3.2"], run_bench.load_prompts()[:3])
    assert all("EndpointUnreachable" in r["error"] for r in run["records"])
    assert run_bench.save_run(run, tmp_path).exists()


def test_the_leaderboard_explains_a_run_where_everything_failed(monkeypatch, tmp_path):
    """The message must point at the cause, not blame scoring."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    providers._reachable_cache.clear()  # noqa: SLF001
    import leaderboard

    run = run_bench.run_benchmark(["ollama:llama3.2"], run_bench.load_prompts()[:2])
    run_bench.save_run(run, tmp_path)

    with pytest.raises(SystemExit) as exc:
        leaderboard.aggregate(leaderboard.load_runs(tmp_path))

    message = str(exc.value)
    assert "Every call in this run failed" in message
    assert "--dry-run" in message  # offers a way forward


def test_an_unreachable_local_server_falls_back_to_the_heuristic(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "JUDGE_MODEL"):
        monkeypatch.delenv(var, raising=False)
    # Nothing is listening on this port.
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    providers._reachable_cache.clear()  # noqa: SLF001

    assert scorer.resolve_judge() is None
    assert scorer.score("q", "an answer", "unanswerable").method == "heuristic"


def test_paid_judges_still_win_over_free_ones(local_server, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert scorer.resolve_judge() == "claude-haiku-4-5"


def test_a_local_judge_is_asked_for_json_mode(local_server):
    scorer.score("q", mock_api.HONEST, "trick_fictional", judge_model="ollama-llama3.2")
    judge_call = mock_api.CALLS[-1]["body"]
    assert judge_call["response_format"] == {"type": "json_object"}


def test_json_mode_is_dropped_if_the_server_rejects_it(local_server, monkeypatch):
    """Self-hosted servers vary; losing JSON mode beats losing the call."""
    spec = providers.get_model("ollama:llama3.2")
    calls = {"n": 0}
    real_create = providers._openai_client(spec).chat.completions.create  # noqa: SLF001

    def picky(**kwargs):
        calls["n"] += 1
        if "response_format" in kwargs:
            raise RuntimeError("unsupported parameter: response_format")
        return real_create(**kwargs)

    monkeypatch.setattr(
        providers._openai_client(spec).chat.completions, "create", picky  # noqa: SLF001
    )
    assert providers.generate("ollama:llama3.2", "hi", json_mode=True)
    assert calls["n"] == 2  # rejected once, retried without it


# --------------------------------------------------------------------------- #
# Full run on free models
# --------------------------------------------------------------------------- #

def test_the_whole_benchmark_runs_on_local_models_alone(local_server, tmp_path):
    prompts = run_bench.load_prompts()[:3]
    run = run_bench.run_benchmark(["ollama:llama3.2", "ollama:qwen2.5"], prompts)

    assert len(run["records"]) == 6
    assert all(r["response"] and not r["error"] for r in run["records"])
    assert run["config"]["judge_model"] == "ollama-llama3.2"
    assert all(r["scores"]["method"] == "judge" for r in run["records"])

    import leaderboard

    run_bench.save_run(run, tmp_path / "results")
    assert leaderboard.main(["--results", str(tmp_path / "results"),
                             "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "leaderboard.png").stat().st_size > 0


def test_free_and_paid_models_can_be_mixed_in_one_run(local_server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", f"{local_server}/v1")
    providers._clients.clear()  # noqa: SLF001

    run = run_bench.run_benchmark(
        ["ollama:llama3.2", "gpt-4o"], run_bench.load_prompts()[:2]
    )
    assert all(r["response"] and not r["error"] for r in run["records"])
    assert {r["model"] for r in run["records"]} == {"ollama:llama3.2", "gpt-4o"}


def test_pacing_is_applied_between_calls(local_server, monkeypatch):
    """Free tiers rate-limit; the run paces itself instead of eating 429s."""
    slept: list[float] = []
    monkeypatch.setattr(run_bench.time, "sleep", lambda s: slept.append(s))

    run_bench.run_benchmark(
        ["ollama:llama3.2"], run_bench.load_prompts()[:2], delay=0.25
    )
    assert slept == [0.25, 0.25]
