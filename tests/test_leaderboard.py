"""Aggregation and artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import leaderboard


def _record(model, prompt_id, total, *, error=None, response="an answer"):
    scores = None
    if total is not None:
        scores = {
            "directness": total, "honesty": total, "padding": total,
            "hallucination_risk": total, "total": total,
            "verdict": f"scored {total}", "method": "judge",
            "judge_model": "claude-haiku-4-5", "error": None,
        }
    return {
        "model": model, "model_label": model.title(), "provider": "test",
        "prompt_id": prompt_id, "prompt_type": "trick_fictional",
        "prompt": f"prompt {prompt_id}", "timestamp": "2026-01-01T00:00:00Z",
        "response": None if error else response, "error": error,
        "latency_s": 1.0, "scores": scores,
    }


def _write_run(results_dir: Path, run_id: str, records) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"run_{run_id}.json"
    path.write_text(json.dumps({"run_id": run_id, "records": records}), encoding="utf-8")
    return path


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    d = tmp_path / "results"
    _write_run(d, "A", [
        _record("good", "p1", 9.0), _record("good", "p2", 8.0),
        _record("bad", "p1", 2.0), _record("bad", "p2", 3.0),
    ])
    return d


def test_no_run_files_is_a_clear_error(tmp_path: Path):
    with pytest.raises(SystemExit, match="run_bench"):
        leaderboard.load_runs(tmp_path)


def test_all_runs_are_combined_by_default(results_dir: Path):
    _write_run(results_dir, "B", [_record("good", "p3", 7.0)])
    assert len(leaderboard.load_runs(results_dir)) == 5
    assert len(leaderboard.load_runs(results_dir, latest_only=True)) == 1


def test_a_corrupt_run_file_is_skipped_not_fatal(results_dir: Path, capsys):
    (results_dir / "run_BROKEN.json").write_text("{not json", encoding="utf-8")
    df = leaderboard.load_runs(results_dir)
    assert len(df) == 4
    assert "Skipping" in capsys.readouterr().err


def test_models_are_ranked_most_honest_first(results_dir: Path):
    agg = leaderboard.aggregate(leaderboard.load_runs(results_dir))
    assert list(agg["model"]) == ["good", "bad"]
    assert list(agg["rank"]) == [1, 2]
    assert agg.iloc[0]["total"] == pytest.approx(8.5)
    assert agg.iloc[0]["responses"] == 2


def test_unscored_results_are_a_clear_error(tmp_path: Path):
    d = tmp_path / "results"
    _write_run(d, "A", [_record("m", "p1", None)])
    with pytest.raises(SystemExit, match="No scored responses"):
        leaderboard.aggregate(leaderboard.load_runs(d))


def test_errors_are_counted_against_the_model_that_produced_them(tmp_path: Path):
    d = tmp_path / "results"
    _write_run(d, "A", [
        _record("m", "p1", 5.0),
        _record("m", "p2", None, error="RateLimitError: slow down"),
    ])
    agg = leaderboard.aggregate(leaderboard.load_runs(d))
    assert agg.iloc[0]["errors"] == 1
    assert agg.iloc[0]["responses"] == 1  # only the scored one is averaged


def test_a_model_that_never_succeeded_is_reported_not_dropped(tmp_path: Path):
    """Without this it would silently vanish from the leaderboard entirely."""
    d = tmp_path / "results"
    _write_run(d, "A", [
        _record("works", "p1", 6.0),
        _record("broken", "p1", None, error="MissingCredentials: no key"),
        _record("broken", "p2", None, error="MissingCredentials: no key"),
    ])
    df = leaderboard.load_runs(d)
    assert "broken" not in set(leaderboard.aggregate(df)["model"])

    dead = leaderboard.failed_models(df)
    assert list(dead["model"]) == ["broken"]
    assert dead.iloc[0]["failures"] == 2
    assert "MissingCredentials" in dead.iloc[0]["first_error"]


def test_no_dead_models_yields_an_empty_frame(results_dir: Path):
    assert leaderboard.failed_models(leaderboard.load_runs(results_dir)).empty


def test_csv_has_a_row_per_model_and_the_expected_columns(results_dir: Path, tmp_path: Path):
    agg = leaderboard.aggregate(leaderboard.load_runs(results_dir))
    path = leaderboard.write_csv(agg, tmp_path / "leaderboard.csv")
    written = pd.read_csv(path)
    assert len(written) == 2
    for column in ("rank", "model", "total", "directness", "hallucination_risk", "errors"):
        assert column in written.columns


def test_chart_renders_to_a_real_png(results_dir: Path, tmp_path: Path):
    agg = leaderboard.aggregate(leaderboard.load_runs(results_dir))
    path = leaderboard.save_chart(leaderboard.build_chart(agg), tmp_path / "leaderboard.png")
    assert path.suffix == ".png", "kaleido missing - PNG export fell back to HTML"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_chart_orders_bars_with_the_most_honest_on_top(results_dir: Path):
    agg = leaderboard.aggregate(leaderboard.load_runs(results_dir))
    fig = leaderboard.build_chart(agg)
    # Plotly draws the first horizontal-bar category at the bottom.
    assert list(fig.data[0].y) == ["Bad", "Good"]
    assert fig.layout.showlegend is False  # single series needs no legend


def test_summary_quotes_the_worst_answer_for_each_prompt(results_dir: Path, tmp_path: Path):
    df = leaderboard.load_runs(results_dir)
    agg = leaderboard.aggregate(df)
    text = leaderboard.write_summary(df, agg, tmp_path / "summary.md").read_text(encoding="utf-8")

    assert "# Bullshit-Bench summary" in text
    assert "Worst answer of the week" in text
    assert "**Most honest:** Good" in text
    assert "**Most bullshit:** Bad" in text
    # The worst answer for each prompt is the low scorer, not the high one.
    assert text.count("**Bad** scored") == 2


def test_summary_calls_out_models_with_no_usable_responses(tmp_path: Path):
    d = tmp_path / "results"
    _write_run(d, "A", [
        _record("works", "p1", 6.0),
        _record("broken", "p1", None, error="MissingCredentials: no key"),
    ])
    df = leaderboard.load_runs(d)
    text = leaderboard.write_summary(df, leaderboard.aggregate(df), tmp_path / "s.md").read_text(
        encoding="utf-8"
    )
    assert "no usable responses" in text
    assert "Broken" in text


def test_main_writes_all_three_artifacts(results_dir: Path, tmp_path: Path):
    out = tmp_path / "out"
    assert leaderboard.main(["--results", str(results_dir), "--out", str(out)]) == 0
    for name in ("leaderboard.csv", "leaderboard.png", "summary.md"):
        assert (out / name).stat().st_size > 0
