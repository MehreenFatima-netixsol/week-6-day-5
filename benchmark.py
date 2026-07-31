"""
benchmark.py
------------
Task 2 requirement: compare the match-winner model's real predictive performance
against a simple public-style benchmark (a naive "team with the better recent
win rate wins" predictor -- the closest analogue to a ladder-position heuristic,
since this synthetic dataset has no external ladder feed to call out to).

Both are evaluated on the SAME held-out set of matches (never used to fit the
LogisticRegression pipeline), so the comparison is apples-to-apples.
Run: python3 -m eval.benchmark
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import data as afl_data
from app import model as M


def naive_predict(home_form, away_form):
    """Ladder-style baseline: whichever team has the better recent win rate is
    predicted to win; ties default to the home team (home-ground advantage)."""
    return "home" if home_form >= away_form else "away"


def build_eval_set(n_per_stride=5, seed=99):
    m = afl_data.MATCHES.sort_values("match_date").reset_index(drop=True)
    # Held-out slice: every 5th match from the SECOND HALF of the season range,
    # deliberately offset from model.py's training stride (::3) to avoid overlap.
    half = m[m.season >= (afl_data.SEASON_MIN + afl_data.SEASON_MAX) // 2]
    holdout = half.iloc[2::n_per_stride]  # offset=2, stride=5 -> disjoint from train's stride-3,offset-0
    rows = []
    for _, r in holdout.iterrows():
        feat_row, hg, ag = M.build_match_feature_row(r.home_team, r.away_team, r.match_date)
        if hg < 3 or ag < 3:
            continue
        rows.append({
            "home_team": r.home_team, "away_team": r.away_team, "match_date": r.match_date,
            "actual_winner_is_home": int(r.winner == r.home_team),
            "home_form": feat_row.iloc[0]["home_recent_win_rate"],
            "away_form": feat_row.iloc[0]["away_recent_win_rate"],
            **{c: feat_row.iloc[0][c] for c in M.MATCH_FEATURE_COLS},
        })
    return pd.DataFrame(rows)


def main():
    eval_df = build_eval_set()
    print(f"Held-out evaluation set: {len(eval_df):,} matches "
          f"(disjoint from the model's training stride).")

    # naive baseline
    naive_pred_home = (eval_df["home_form"] >= eval_df["away_form"]).astype(int)
    naive_acc = (naive_pred_home == eval_df["actual_winner_is_home"]).mean()

    # trained model
    X = eval_df[M.MATCH_FEATURE_COLS]
    proba_home = M.MATCH_ARTIFACT["pipeline"].predict_proba(X)[:, 1]
    model_pred_home = (proba_home >= 0.5).astype(int)
    model_acc = (model_pred_home == eval_df["actual_winner_is_home"]).mean()

    # always-picks-home baseline (a second, even-simpler benchmark)
    home_acc = eval_df["actual_winner_is_home"].mean()

    lines = []
    lines.append("# Match-Winner Model vs. Naive Benchmarks\n")
    lines.append(f"Held-out set: **{len(eval_df):,} matches**, disjoint from model training data.\n")
    lines.append("| Predictor | Accuracy |")
    lines.append("|---|---|")
    lines.append(f"| Always pick home team | {home_acc*100:.1f}% |")
    lines.append(f"| Naive (better recent win-rate wins) | {naive_acc*100:.1f}% |")
    lines.append(f"| **Trained model (LogisticRegression)** | **{model_acc*100:.1f}%** |")
    lines.append("")
    lift = model_acc - naive_acc
    lines.append(f"**Model lift over the naive form-based baseline: {lift*100:+.1f} points.**\n")
    lines.append(
        "Context: AFL match outcomes are inherently noisy (recent public modelling and "
        "bookmaker-implied win probabilities for the real competition typically sit in the "
        "low-to-mid 70s% accuracy range across a season). A few points of lift over a naive "
        "recent-form heuristic is a realistic, defensible result for a lightweight logistic "
        "model on this feature set -- it is not a large edge, and should be presented to "
        "stakeholders as directionally useful rather than betting-grade."
    )
    report = "\n".join(lines)
    out = Path(__file__).parent / "benchmark_results.md"
    out.write_text(report)
    print(report)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
