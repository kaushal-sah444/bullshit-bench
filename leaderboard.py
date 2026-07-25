"""Aggregate Bullshit-Bench runs into a leaderboard.

Reads every ``results/run_*.json``, flattens it into a pandas DataFrame, and
writes three artifacts:

* ``results/leaderboard.csv``  — per-model averages, most honest first.
* ``results/leaderboard.png``  — horizontal bar chart of the honesty ranking.
* ``results/summary.md``       — shareable writeup, including the worst answer
  for every prompt (the bit worth turning into short-form content).

Usage::

    python leaderboard.py
    python leaderboard.py --results results --out results --latest-only
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

SCORE_COLUMNS = ["directness", "honesty", "padding", "hallucination_risk", "total"]

# Chart palette. Single-series ranking: one hue does the whole chart, so color
# carries no information and needs no legend. Values are the dark-surface steps
# of the reference palette (validated: L-band, chroma floor, >=3:1 on surface).
SURFACE = "#1a1a19"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#8a8981"
GRID = "#383835"
SERIES = "#3987e5"


def load_runs(results_dir: Path = RESULTS_DIR, latest_only: bool = False) -> pd.DataFrame:
    """Flatten ``run_*.json`` files into one row per model/prompt pair.

    Args:
        results_dir: Directory holding the run files.
        latest_only: Use only the most recent run file instead of all of them.

    Returns:
        A DataFrame with one row per scored response. Records that errored out
        are kept (with null scores) so failures stay visible.

    Raises:
        SystemExit: No run files found, or none of them contained records.
    """
    files = sorted(results_dir.glob("run_*.json"))
    if not files:
        sys.exit(
            f"No run files in {results_dir}. Run `python run_bench.py` first."
        )
    if latest_only:
        files = files[-1:]

    rows: List[Dict[str, object]] = []
    for path in files:
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"Skipping {path.name}: not valid JSON ({exc})", file=sys.stderr)
            continue

        for record in run.get("records", []):
            row: Dict[str, object] = {
                "run_id": run.get("run_id", path.stem),
                "run_file": path.name,
                "model": record.get("model"),
                "model_label": record.get("model_label") or record.get("model"),
                "provider": record.get("provider"),
                "prompt_id": record.get("prompt_id"),
                "prompt_type": record.get("prompt_type"),
                "prompt": record.get("prompt"),
                "response": record.get("response"),
                "latency_s": record.get("latency_s"),
                "error": record.get("error"),
            }
            scores = record.get("scores") or {}
            for column in SCORE_COLUMNS:
                row[column] = scores.get(column)
            row["verdict"] = scores.get("verdict")
            row["score_method"] = scores.get("method")
            row["judge_model"] = scores.get("judge_model")
            rows.append(row)

    if not rows:
        sys.exit(f"Found {len(files)} run file(s) but no records inside them.")

    df = pd.DataFrame(rows)
    for column in [*SCORE_COLUMNS, "latency_s"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Average each model's scores across every prompt and run.

    Args:
        df: Row-level frame from :func:`load_runs`.

    Returns:
        One row per model, sorted most honest first. ``responses`` counts scored
        responses; ``errors`` counts calls that failed outright.

    Raises:
        SystemExit: Nothing in the frame was scored.
    """
    scored = df[df["total"].notna()]
    if scored.empty:
        sys.exit(
            "No scored responses found. Re-run without --no-score, or set an API "
            "key so the judge (or heuristic fallback) can grade the answers."
        )

    agg = (
        scored.groupby(["model", "model_label", "provider"], as_index=False)
        .agg(
            directness=("directness", "mean"),
            honesty=("honesty", "mean"),
            padding=("padding", "mean"),
            hallucination_risk=("hallucination_risk", "mean"),
            total=("total", "mean"),
            responses=("total", "size"),
            avg_latency_s=("latency_s", "mean"),
        )
        .round(2)
    )

    errors = (
        df[df["error"].notna()].groupby("model").size().rename("errors").reset_index()
    )
    agg = agg.merge(errors, on="model", how="left")
    agg["errors"] = agg["errors"].fillna(0).astype(int)

    agg = agg.sort_values("total", ascending=False).reset_index(drop=True)
    agg.insert(0, "rank", agg.index + 1)
    return agg


def failed_models(df: pd.DataFrame) -> pd.DataFrame:
    """Models that produced no scored response at all.

    These are invisible in :func:`aggregate` — a model whose every call failed
    has nothing to average — so they are surfaced separately rather than quietly
    dropped off the leaderboard.
    """
    scored_models = set(df.loc[df["total"].notna(), "model"].unique())
    dead = df[~df["model"].isin(scored_models)]
    if dead.empty:
        return dead.iloc[0:0]
    return (
        dead.groupby(["model", "model_label"], as_index=False)
        .agg(failures=("error", "size"), first_error=("error", "first"))
    )


def write_csv(agg: pd.DataFrame, path: Path) -> Path:
    """Write the aggregated leaderboard to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(path, index=False)
    return path


def build_chart(agg: pd.DataFrame):
    """Build the honesty-ranking bar chart.

    Horizontal bars because the job is ranking magnitude across named categories
    with long labels. One series, so one flat hue and no legend — the title says
    what the bars measure. Values are labeled directly, which also satisfies the
    relief rule for anyone who cannot separate bar length from the axis alone.

    Args:
        agg: Aggregated leaderboard from :func:`aggregate`.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    import plotly.graph_objects as go

    # Plotly draws the first category at the bottom of a horizontal bar chart,
    # so reverse to put the most honest model on top.
    plot_df = agg.sort_values("total", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=plot_df["total"],
            y=plot_df["model_label"],
            orientation="h",
            marker=dict(color=SERIES),
            text=[f"{v:.1f}" for v in plot_df["total"]],
            textposition="outside",
            textfont=dict(color=INK_PRIMARY, size=13),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Honesty score: %{x:.2f} / 10<br>"
                "Directness: %{customdata[0]:.1f}<br>"
                "Honesty: %{customdata[1]:.1f}<br>"
                "Info density: %{customdata[2]:.1f}<br>"
                "Hallucination safety: %{customdata[3]:.1f}"
                "<extra></extra>"
            ),
            customdata=plot_df[
                ["directness", "honesty", "padding", "hallucination_risk"]
            ].to_numpy(),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text="Bullshit-Bench: honesty score by model",
            font=dict(color=INK_PRIMARY, size=22),
            subtitle=dict(
                text="Higher is more honest. Averaged over all trick prompts.",
                font=dict(color=INK_MUTED, size=12),
            ),
            x=0,
            xanchor="left",
            pad=dict(b=16),
        ),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(
            family="Inter, Segoe UI, Helvetica, Arial, sans-serif",
            color=INK_SECONDARY,
            size=13,
        ),
        showlegend=False,
        bargap=0.5,  # thin marks, generous gap between bars
        barcornerradius=4,  # rounded data-end, square against the baseline
        margin=dict(l=20, r=70, t=110, b=60),
        height=170 + 62 * len(plot_df),
        width=900,
    )
    fig.update_xaxes(
        range=[0, 10.6],
        title=dict(
            text="Honesty score (0 = pure bullshit, 10 = straight answer)",
            font=dict(color=INK_MUTED, size=12),
            standoff=14,
        ),
        gridcolor=GRID,
        griddash="dot",
        zeroline=False,
        showline=False,
        ticks="",
        tickfont=dict(color=INK_MUTED),
    )
    fig.update_yaxes(
        title=None,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
        tickfont=dict(color=INK_PRIMARY, size=14),
    )
    return fig


def save_chart(fig, path: Path) -> Path:
    """Export the chart to PNG, falling back to interactive HTML.

    PNG export needs ``kaleido``. If it is missing or fails, an HTML file is
    written next to the requested path instead and the actual path returned.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(path), scale=2)
        return path
    except Exception as exc:  # noqa: BLE001 - kaleido surfaces several types
        html_path = path.with_suffix(".html")
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        print(
            f"PNG export failed ({type(exc).__name__}: {exc}).\n"
            f"Wrote {html_path.name} instead. `pip install kaleido` for PNG output.",
            file=sys.stderr,
        )
        return html_path


def _worst_per_prompt(df: pd.DataFrame) -> pd.DataFrame:
    """The lowest-scoring answer for each prompt — the shareable material."""
    scored = df[df["total"].notna()]
    if scored.empty:
        return scored
    idx = scored.groupby("prompt_id")["total"].idxmin()
    return scored.loc[idx].sort_values("total")


def _quote(text: Optional[str], limit: int = 400) -> str:
    """Format a response as an indented markdown blockquote."""
    if not text:
        return "> *(no response)*"
    flat = " ".join(str(text).split())
    if len(flat) > limit:
        flat = flat[:limit].rstrip() + "…"
    return "\n".join(f"> {line}" for line in textwrap.wrap(flat, 90)) or "> …"


def write_summary(df: pd.DataFrame, agg: pd.DataFrame, path: Path) -> Path:
    """Write the shareable markdown summary.

    Contains the ranking, per-category averages, and the worst answer for each
    prompt — quoted verbatim, with the judge's one-line verdict.
    """
    lines: List[str] = []
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    scored = df[df["total"].notna()]

    lines.append("# Bullshit-Bench summary")
    lines.append("")
    lines.append(
        f"_Generated {generated} · {len(scored)} scored responses · "
        f"{scored['prompt_id'].nunique()} prompts · {agg.shape[0]} models_"
    )
    lines.append("")

    lines.append("## Leaderboard")
    lines.append("")
    lines.append("| # | Model | Honesty | Direct | Honest | Density | No-hallucination |")
    lines.append("|---|-------|---------|--------|--------|---------|------------------|")
    for row in agg.itertuples():
        lines.append(
            f"| {row.rank} | {row.model_label} | **{row.total:.2f}** | "
            f"{row.directness:.1f} | {row.honesty:.1f} | {row.padding:.1f} | "
            f"{row.hallucination_risk:.1f} |"
        )
    lines.append("")
    lines.append("All dimensions run 0-10 and point the same way: higher is better.")
    lines.append("")

    best, worst = agg.iloc[0], agg.iloc[-1]
    lines.append(
        f"**Most honest:** {best.model_label} ({best.total:.2f}/10)  \n"
        f"**Most bullshit:** {worst.model_label} ({worst.total:.2f}/10)"
    )
    lines.append("")

    by_type = (
        scored.groupby(["prompt_type"])["total"].mean().round(2).sort_values()
    )
    if not by_type.empty:
        lines.append("## Which trick works best")
        lines.append("")
        lines.append("| Prompt type | Avg score across all models |")
        lines.append("|-------------|-----------------------------|")
        for prompt_type, value in by_type.items():
            lines.append(f"| `{prompt_type}` | {value:.2f} |")
        lines.append("")
        lines.append(f"Lowest-scoring category: **`{by_type.index[0]}`**.")
        lines.append("")

    lines.append("## Worst answer of the week, per prompt")
    lines.append("")
    for row in _worst_per_prompt(df).itertuples():
        lines.append(f"### {row.prompt}")
        lines.append("")
        lines.append(
            f"`{row.prompt_type}` · **{row.model_label}** scored "
            f"**{row.total:.1f}/10**"
        )
        lines.append("")
        lines.append(_quote(row.response))
        lines.append("")
        if isinstance(row.verdict, str) and row.verdict:
            lines.append(f"*Judge:* {row.verdict}")
            lines.append("")

    dead = failed_models(df)
    if not dead.empty:
        lines.append("## Models with no usable responses")
        lines.append("")
        lines.append(
            "These are absent from the leaderboard above — every call failed, so "
            "there is nothing to average."
        )
        lines.append("")
        for row in dead.itertuples():
            lines.append(f"- **{row.model_label}** ({row.failures} failures): {row.first_error}")
        lines.append("")

    failures = df[df["error"].notna()]
    if not failures.empty:
        lines.append("## Failures")
        lines.append("")
        for row in failures.itertuples():
            lines.append(f"- `{row.model}` / `{row.prompt_id}`: {row.error}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Bullshit-Bench leaderboard.")
    parser.add_argument(
        "--results", type=Path, default=RESULTS_DIR, help="Directory holding run_*.json."
    )
    parser.add_argument(
        "--out", type=Path, default=RESULTS_DIR, help="Directory for generated artifacts."
    )
    parser.add_argument(
        "--latest-only", action="store_true", help="Aggregate only the newest run file."
    )
    parser.add_argument(
        "--no-chart", action="store_true", help="Skip chart rendering (CSV + summary only)."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    df = load_runs(args.results, latest_only=args.latest_only)
    agg = aggregate(df)

    csv_path = write_csv(agg, args.out / "leaderboard.csv")
    print(f"Wrote {csv_path}")

    if not args.no_chart:
        chart_path = save_chart(build_chart(agg), args.out / "leaderboard.png")
        print(f"Wrote {chart_path}")

    summary_path = write_summary(df, agg, args.out / "summary.md")
    print(f"Wrote {summary_path}")

    print()
    print(agg[["rank", "model_label", "total", "responses", "errors"]].to_string(index=False))

    dead = failed_models(df)
    if not dead.empty:
        print("\nNot ranked (no usable responses):")
        for row in dead.itertuples():
            print(f"  {row.model_label}: {row.failures} failures - {row.first_error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
