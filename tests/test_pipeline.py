"""End-to-end: real vendor SDKs -> mock API -> run file -> leaderboard artifacts.

This is the test that answers "does the whole thing actually work". Everything
runs for real except the vendors' servers: the genuine ``anthropic``, ``openai``
and ``google-genai`` clients build requests, speak HTTP to a local stub, and
their responses are parsed by the same code a live run would use.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import leaderboard
import mock_api
import run_bench

MODELS = ["claude-opus-5", "gpt-4o", "gemini-2.0-flash", "claude-haiku-4-5"]


@pytest.fixture
def full_run(mock_env, tmp_path: Path):
    """Benchmark four models across a slice of the real prompt set."""
    prompts = run_bench.load_prompts()[:4]
    run = run_bench.run_benchmark(MODELS, prompts, temperature=0.7, max_tokens=2048)
    results_dir = tmp_path / "results"
    run_bench.save_run(run, results_dir)
    return run, results_dir


def test_every_model_answers_every_prompt(full_run):
    run, _ = full_run
    assert len(run["records"]) == len(MODELS) * 4
    assert all(r["response"] and not r["error"] for r in run["records"])


def test_every_response_is_graded_by_the_judge(full_run):
    run, _ = full_run
    for record in run["records"]:
        scores = record["scores"]
        assert scores["method"] == "judge"
        assert scores["judge_model"] == "claude-haiku-4-5"
        assert scores["error"] is None
        assert 0 <= scores["total"] <= 10


def test_every_request_carried_credentials(full_run):
    assert mock_api.CALLS
    assert all(call["authenticated"] for call in mock_api.CALLS)


def test_scores_separate_the_honest_model_from_the_fabricating_one(full_run):
    run, _ = full_run
    totals: dict[str, list[float]] = {}
    for record in run["records"]:
        totals.setdefault(record["model"], []).append(record["scores"]["total"])
    average = {model: sum(v) / len(v) for model, v in totals.items()}

    # claude-opus-5 gets the honest persona, gpt-4o the fabricating one.
    assert average["claude-opus-5"] > average["gpt-4o"]


def test_the_pipeline_produces_a_ranked_leaderboard_and_artifacts(full_run, tmp_path: Path):
    _, results_dir = full_run
    out = tmp_path / "out"
    assert leaderboard.main(["--results", str(results_dir), "--out", str(out)]) == 0

    agg = leaderboard.aggregate(leaderboard.load_runs(results_dir))
    assert agg.iloc[0]["model"] == "claude-opus-5"  # the honest persona wins
    assert list(agg["rank"]) == sorted(agg["rank"])
    assert agg["total"].is_monotonic_decreasing

    assert (out / "leaderboard.png").read_bytes()[:4] == b"\x89PNG"
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "Claude Opus 5" in summary
    assert "Worst answer of the week" in summary


def test_a_run_with_no_keys_still_completes_using_the_heuristic(no_keys, tmp_path: Path):
    """The full pipeline has to work for someone who has not set any key yet."""
    prompts = run_bench.load_prompts()[:3]
    run = run_bench.run_benchmark(["gpt-4o"], prompts, dry_run=True)
    results_dir = tmp_path / "results"
    run_bench.save_run(run, results_dir)

    assert all(r["scores"]["method"] == "heuristic" for r in run["records"])
    out = tmp_path / "out"
    assert leaderboard.main(["--results", str(results_dir), "--out", str(out)]) == 0
    assert (out / "leaderboard.csv").stat().st_size > 0


def test_dry_run_spends_nothing_at_all(mock_env, tmp_path: Path):
    """Generation *and* judging must be skipped - the judge is a billable call."""
    run_bench.run_benchmark(["gpt-4o"], run_bench.load_prompts()[:2], dry_run=True)
    assert mock_api.CALLS == []
