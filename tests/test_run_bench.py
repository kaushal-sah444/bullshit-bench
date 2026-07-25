"""Prompt loading, retry semantics, and the benchmark loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mock_api
import providers
import run_bench

PROMPTS = [
    {"id": "p1", "type": "trick_fictional", "prompt": "Population of Wakanda?"},
    {"id": "p2", "type": "unanswerable", "prompt": "Bitcoin price in a year?"},
]


# --------------------------------------------------------------------------- #
# Prompt file validation
# --------------------------------------------------------------------------- #

def test_the_shipped_prompt_set_loads_and_covers_all_four_types():
    prompts = run_bench.load_prompts()
    assert len(prompts) >= 15  # the spec's floor
    assert {p["type"] for p in prompts} == {
        "trick_fictional", "fabrication_bait", "unanswerable", "absurd_precision",
    }
    assert len({p["id"] for p in prompts}) == len(prompts)


@pytest.mark.parametrize(
    "content, expected",
    [
        ("not json at all", "not valid JSON"),
        ("[]", "non-empty"),
        ('[{"id": "a", "prompt": "q"}]', "missing field"),
        ('[{"id": "a", "type": "t", "prompt": "q"}, {"id": "a", "type": "t", "prompt": "r"}]',
         "duplicate"),
    ],
)
def test_malformed_prompt_files_fail_loudly(tmp_path: Path, content, expected):
    path = tmp_path / "prompts.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        run_bench.load_prompts(path)
    assert expected in str(exc.value)


def test_a_missing_prompt_file_fails_loudly(tmp_path: Path):
    with pytest.raises(SystemExit):
        run_bench.load_prompts(tmp_path / "nope.json")


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #

def test_transient_failures_are_retried_until_one_succeeds(monkeypatch):
    monkeypatch.setattr(run_bench.time, "sleep", lambda _s: None)
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("rate limited")
        return "ok"

    assert run_bench.with_retry(flaky, description="t", verbose=False) == "ok"
    assert attempts["n"] == 3


def test_retries_are_bounded_and_the_last_error_propagates(monkeypatch):
    monkeypatch.setattr(run_bench.time, "sleep", lambda _s: None)
    attempts = {"n": 0}

    def always_fails():
        attempts["n"] += 1
        raise RuntimeError("still down")

    with pytest.raises(RuntimeError, match="still down"):
        run_bench.with_retry(always_fails, description="t", max_attempts=4, verbose=False)
    assert attempts["n"] == 4


@pytest.mark.parametrize(
    "error",
    [
        providers.MissingCredentials("no key"),
        providers.MissingDependency("no sdk"),
        providers.UnknownModel("no such model"),
        providers.EmptyResponse("budget exhausted"),
    ],
)
def test_deterministic_failures_are_not_retried(monkeypatch, error):
    """Retrying these can only waste time - the outcome cannot change."""
    monkeypatch.setattr(run_bench.time, "sleep", lambda _s: None)
    attempts = {"n": 0}

    def fails():
        attempts["n"] += 1
        raise error

    with pytest.raises(type(error)):
        run_bench.with_retry(fails, description="t", verbose=False)
    assert attempts["n"] == 1


# --------------------------------------------------------------------------- #
# The benchmark loop
# --------------------------------------------------------------------------- #

def test_dry_run_makes_no_api_calls(mock_env):
    run = run_bench.run_benchmark(["gpt-4o"], PROMPTS, dry_run=True)
    assert mock_api.CALLS == []
    assert all(r["response"] for r in run["records"])


def test_a_full_run_records_responses_and_scores(mock_env):
    run = run_bench.run_benchmark(["claude-opus-5", "gpt-4o"], PROMPTS)

    assert run["config"]["judge_model"] == "claude-haiku-4-5"
    assert len(run["records"]) == 4
    for record in run["records"]:
        assert record["error"] is None
        assert record["response"]
        assert record["scores"]["method"] == "judge"
        assert 0 <= record["scores"]["total"] <= 10
        assert record["latency_s"] is not None


def test_one_broken_model_does_not_sink_the_run(mock_env, monkeypatch):
    """A model without a key is recorded as an error; the others still run."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    run = run_bench.run_benchmark(["gpt-4o", "gemini-2.0-flash"], PROMPTS)

    failed = [r for r in run["records"] if r["error"]]
    succeeded = [r for r in run["records"] if r["response"] and not r["error"]]
    assert len(failed) == 2 and len(succeeded) == 2
    assert all(r["model"] == "gemini-2.0-flash" for r in failed)
    assert "MissingCredentials" in failed[0]["error"]


def test_no_score_collects_raw_answers_only(mock_env):
    run = run_bench.run_benchmark(["gpt-4o"], PROMPTS, do_score=False)
    assert run["config"]["scored"] is False
    assert all(r["scores"] is None and r["response"] for r in run["records"])


def test_a_saved_run_round_trips_through_json(mock_env, tmp_path: Path):
    run = run_bench.run_benchmark(["gpt-4o"], PROMPTS, dry_run=True)
    path = run_bench.save_run(run, tmp_path)
    assert path.name.startswith("run_") and path.suffix == ".json"
    assert json.loads(path.read_text(encoding="utf-8"))["records"] == run["records"]


def test_cli_rejects_an_unknown_model_before_spending_anything(mock_env):
    with pytest.raises(providers.UnknownModel):
        run_bench.main(["--models", "not-a-real-model", "--dry-run"])
