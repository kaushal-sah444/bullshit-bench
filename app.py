"""Streamlit dashboard for Bullshit-Bench.

Reads ``results/leaderboard.csv`` plus the raw run files and shows the ranking,
the chart, and a "Worst Answer of the Week" callout.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import leaderboard

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
CSV_PATH = RESULTS_DIR / "leaderboard.csv"

st.set_page_config(page_title="Bullshit-Bench", page_icon="🧢", layout="wide")


@st.cache_data(show_spinner=False)
def _load_leaderboard(mtime: float) -> pd.DataFrame:
    """Load the aggregated CSV. ``mtime`` busts the cache when the file changes."""
    return pd.read_csv(CSV_PATH)


@st.cache_data(show_spinner=False)
def _load_rows(signature: tuple) -> pd.DataFrame:
    """Load row-level results. ``signature`` busts the cache when runs change."""
    return leaderboard.load_runs(RESULTS_DIR)


def _run_signature() -> tuple:
    return tuple(
        (p.name, p.stat().st_mtime) for p in sorted(RESULTS_DIR.glob("run_*.json"))
    )


st.title("🧢 Bullshit-Bench")
st.caption(
    "How much do chatbots bullshit when a question can't be honestly answered? "
    "Every score runs 0-10 and points the same way — higher is more honest."
)

if not CSV_PATH.exists():
    st.warning(
        "No leaderboard yet. Run `python run_bench.py` and then "
        "`python leaderboard.py`, and reload this page."
    )
    st.stop()

agg = _load_leaderboard(CSV_PATH.stat().st_mtime)

try:
    rows = _load_rows(_run_signature())
except SystemExit as exc:  # load_runs exits when there is nothing to read
    st.error(str(exc))
    st.stop()

scored = rows[rows["total"].notna()]

# --- Headline numbers ------------------------------------------------------- #
best = agg.iloc[0]
worst = agg.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Most honest", best["model_label"], f"{best['total']:.2f} / 10")
col2.metric("Most bullshit", worst["model_label"], f"{worst['total']:.2f} / 10")
col3.metric("Models tested", len(agg))
col4.metric("Scored responses", len(scored))

# --- Chart ------------------------------------------------------------------ #
st.subheader("Ranking")
st.plotly_chart(leaderboard.build_chart(agg), width="stretch")

st.dataframe(
    agg[
        [
            "rank", "model_label", "provider", "total", "directness", "honesty",
            "padding", "hallucination_risk", "responses", "errors",
        ]
    ].rename(
        columns={
            "model_label": "model",
            "total": "honesty score",
            "padding": "info density",
            "hallucination_risk": "no-hallucination",
        }
    ),
    width="stretch",
    hide_index=True,
)

# --- Worst Answer of the Week ----------------------------------------------- #
st.subheader("🏆 Worst Answer of the Week")

if scored.empty:
    st.info("Nothing scored yet.")
else:
    hall_of_shame = scored.nsmallest(1, "total").iloc[0]
    st.error(
        f"**{hall_of_shame['model_label']}** scored "
        f"**{hall_of_shame['total']:.1f}/10** on a `{hall_of_shame['prompt_type']}` prompt."
    )
    st.markdown(f"**Prompt:** {hall_of_shame['prompt']}")
    st.markdown("**Answer:**")
    st.markdown(f"> {' '.join(str(hall_of_shame['response']).split())}")
    if isinstance(hall_of_shame.get("verdict"), str) and hall_of_shame["verdict"]:
        st.caption(f"Judge: {hall_of_shame['verdict']}")

    with st.expander("Worst answer for every prompt"):
        for row in leaderboard._worst_per_prompt(rows).itertuples():  # noqa: SLF001
            st.markdown(
                f"**{row.prompt}**  \n"
                f"`{row.prompt_type}` · {row.model_label} · {row.total:.1f}/10"
            )
            st.markdown(f"> {' '.join(str(row.response).split())}")
            st.divider()

# --- Explorer --------------------------------------------------------------- #
st.subheader("Browse every response")
left, right = st.columns(2)
model_filter = left.multiselect(
    "Model", sorted(scored["model_label"].unique()), default=None
)
type_filter = right.multiselect(
    "Prompt type", sorted(scored["prompt_type"].unique()), default=None
)

view = scored
if model_filter:
    view = view[view["model_label"].isin(model_filter)]
if type_filter:
    view = view[view["prompt_type"].isin(type_filter)]

st.dataframe(
    view[["model_label", "prompt_type", "prompt", "total", "verdict", "response"]]
    .sort_values("total")
    .rename(columns={"model_label": "model", "total": "score"}),
    width="stretch",
    hide_index=True,
)
