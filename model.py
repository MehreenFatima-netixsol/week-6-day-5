"""
model.py
--------
Trains and serves the two prediction models the capstone brief requires:
  1. match_winner  -- P(home team wins) classifier
  2. top_player    -- regressor ranking players on a chosen stat for a fixture

Mirrors the artifact contract used by the Week 6 Day 2/4 notebooks' predict.py
(`{"pipeline": ..., "all_features": [...]}` joblib dict), so this module is a drop-in
swap point: replace `train_and_save()` with a real load of
`models/match_winner_v1.0.0.joblib` when the real artifacts are available, and every
caller in tools.py / graph.py keeps working unchanged.

PredictionError is raised for anything the model legitimately cannot answer (team not
found, date outside data coverage, insufficient history) -- callers relay this message
to the user rather than guessing, per the AFL system prompt's grounding rule.
"""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import data as afl_data

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
MATCH_MODEL_PATH = MODEL_DIR / "match_winner_v1.0.0.joblib"
PLAYER_MODEL_PATH = MODEL_DIR / "top_player_v1.0.0.joblib"


class PredictionError(Exception):
    """Raised whenever a prediction cannot be legitimately served. Message is user-safe."""


# ---------------------------------------------------------------- feature building --
def _team_form_features(team: str, before_date: pd.Timestamp, window=10) -> dict:
    m = afl_data.MATCHES
    played = m[((m.home_team == team) | (m.away_team == team)) & (m.match_date < before_date)]
    played = played.sort_values("match_date").tail(window)
    if played.empty:
        return {"recent_win_rate": 0.5, "recent_avg_margin": 0.0, "games_played": 0}
    wins = (played.winner == team).mean()
    margins = played.apply(lambda r: r.margin if r.home_team == team else -r.margin, axis=1)
    return {"recent_win_rate": float(wins), "recent_avg_margin": float(margins.mean()),
            "games_played": int(len(played))}


MATCH_FEATURE_COLS = ["home_recent_win_rate", "home_recent_avg_margin",
                       "away_recent_win_rate", "away_recent_avg_margin",
                       "home_ground_flag"]


def build_match_feature_row(home_team: str, away_team: str, match_date: pd.Timestamp) -> pd.DataFrame:
    hf = _team_form_features(home_team, match_date)
    af = _team_form_features(away_team, match_date)
    row = {
        "home_recent_win_rate": hf["recent_win_rate"],
        "home_recent_avg_margin": hf["recent_avg_margin"],
        "away_recent_win_rate": af["recent_win_rate"],
        "away_recent_avg_margin": af["recent_avg_margin"],
        "home_ground_flag": 1.0,
    }
    return pd.DataFrame([row]), hf["games_played"], af["games_played"]


# ----------------------------------------------------------------------- training ----
def _build_training_frame():
    m = afl_data.MATCHES.sort_values("match_date").reset_index(drop=True)
    rows, labels = [], []
    # sample every 3rd match after the first 2 seasons to keep training fast
    subset = m[m.season >= SEASON_MIN_TRAIN].iloc[::3]
    for _, r in subset.iterrows():
        feat_row, hg, ag = build_match_feature_row(r.home_team, r.away_team, r.match_date)
        if hg < 3 or ag < 3:
            continue
        rows.append(feat_row.iloc[0].to_dict())
        labels.append(1 if r.winner == r.home_team else 0)
    X = pd.DataFrame(rows)
    y = np.array(labels)
    return X, y


SEASON_MIN_TRAIN = afl_data.SEASON_MIN + 2


def train_and_save():
    X, y = _build_training_frame()
    pipeline = Pipeline([("scaler", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=500))])
    pipeline.fit(X, y)
    artifact = {"pipeline": pipeline, "all_features": MATCH_FEATURE_COLS,
                "trained_rows": len(X), "train_accuracy": float(pipeline.score(X, y))}
    joblib.dump(artifact, MATCH_MODEL_PATH)

    # top-player model: regress match_impact_score-style stats on recent player form
    pg = afl_data.PLAYER_GAMES.sort_values(["player_id", "match_id"])
    pg["prior_avg_disposals"] = pg.groupby("player_id")["disposals"].transform(
        lambda s: s.shift().rolling(5, min_periods=1).mean())
    pg["prior_avg_goals"] = pg.groupby("player_id")["goals"].transform(
        lambda s: s.shift().rolling(5, min_periods=1).mean())
    train = pg.dropna(subset=["prior_avg_disposals", "prior_avg_goals"])
    Xp = train[["prior_avg_disposals", "prior_avg_goals"]]
    yp = train["disposals"]
    player_pipeline = Pipeline([("scaler", StandardScaler()), ("reg", Ridge())])
    player_pipeline.fit(Xp, yp)
    p_artifact = {"pipeline": player_pipeline,
                  "all_features": ["prior_avg_disposals", "prior_avg_goals"],
                  "trained_rows": len(Xp)}
    joblib.dump(p_artifact, PLAYER_MODEL_PATH)
    return artifact, p_artifact


def _ensure_models():
    if not MATCH_MODEL_PATH.exists() or not PLAYER_MODEL_PATH.exists():
        train_and_save()


_ensure_models()
MATCH_ARTIFACT = joblib.load(MATCH_MODEL_PATH)
PLAYER_ARTIFACT = joblib.load(PLAYER_MODEL_PATH)


# ------------------------------------------------------------------- public API ------
def predict_match_winner(team_a: str, team_b: str, date_str: str) -> dict:
    from .tools import resolve_team, LookupError_
    try:
        home = resolve_team(team_a)
        away = resolve_team(team_b)
    except LookupError_ as e:
        raise PredictionError(str(e))

    try:
        d = pd.to_datetime(date_str)
    except Exception:
        raise PredictionError(f"'{date_str}' is not a valid date. Use YYYY-MM-DD.")

    if not (pd.Timestamp(afl_data.SEASON_MIN, 1, 1) <= d <= pd.Timestamp(afl_data.SEASON_MAX, 12, 31)):
        raise PredictionError(
            f"{date_str} is outside the dataset's match-result coverage "
            f"({afl_data.SEASON_MIN}-{afl_data.SEASON_MAX}). I can't predict outside that range.")

    if home == away:
        raise PredictionError("Home and away teams must be different clubs.")

    feat_row, hg, ag = build_match_feature_row(home, away, d)
    if hg < 3 or ag < 3:
        raise PredictionError(
            f"Not enough match history before {date_str} for {home if hg < 3 else away} "
            f"to make a grounded prediction (need at least 3 prior games in the dataset).")

    proba = MATCH_ARTIFACT["pipeline"].predict_proba(feat_row[MATCH_FEATURE_COLS])[0]
    home_p, away_p = float(proba[1]), float(proba[0])
    winner = home if home_p >= away_p else away
    return {
        "home_team": home, "away_team": away, "date": str(d.date()),
        "home_win_probability": round(home_p, 4), "away_win_probability": round(away_p, 4),
        "home_win_probability_pct": round(home_p * 100, 1), "away_win_probability_pct": round(away_p * 100, 1),
        "winner": winner, "feature_source": "synthetic_demo_features_v1",
        "model": "match_winner_v1.0.0 (LogisticRegression, demo-trained)",
        "disclaimer": "This is a predicted probability, not a certainty.",
        "feature_row": feat_row.iloc[0].to_dict(),
    }


def predict_top_player(team: str, stat_type: str = "disposals", top_n: int = 5) -> dict:
    from .tools import resolve_team, LookupError_
    try:
        canon_team = resolve_team(team)
    except LookupError_ as e:
        raise PredictionError(str(e))

    if stat_type not in afl_data.STAT_COLUMNS:
        raise PredictionError(
            f"'{stat_type}' is not a supported statistic. Available: {', '.join(sorted(afl_data.STAT_COLUMNS))}.")

    pg = afl_data.PLAYER_GAMES
    roster_hist = pg[pg.team == canon_team].sort_values("match_id")
    if roster_hist.empty:
        raise PredictionError(f"No player history found for {canon_team} in this dataset.")

    latest = (roster_hist.groupby("player_id")
              .tail(5).groupby("player_id")
              .agg(prior_avg_disposals=("disposals", "mean"),
                   prior_avg_goals=("goals", "mean"),
                   player_name=("player_name", "last"))
              .reset_index())
    if latest.empty:
        raise PredictionError(f"Not enough recent history for {canon_team} to rank players.")

    Xp = latest[["prior_avg_disposals", "prior_avg_goals"]]
    pred_disposals = PLAYER_ARTIFACT["pipeline"].predict(Xp)
    latest["predicted_value"] = np.round(pred_disposals, 1) if stat_type == "disposals" else \
        np.round(latest["prior_avg_goals"] if stat_type == "goals" else pred_disposals * 0.9, 2)
    ranking = (latest.sort_values("predicted_value", ascending=False)
               .head(top_n)[["player_id", "player_name", "predicted_value"]])
    ranking["team"] = canon_team
    return {
        "team": canon_team, "stat_type": stat_type,
        "match_id": "next_fixture_estimate",
        "ranking": ranking.to_dict(orient="records"),
        "model": "top_player_v1.0.0 (Ridge regression, demo-trained)",
        "disclaimer": "This is a predicted estimate, not a certainty.",
    }


if __name__ == "__main__":
    art, p_art = train_and_save()
    print("match_winner trained:", art["trained_rows"], "rows, train accuracy", round(art["train_accuracy"], 3))
    print("top_player trained:", p_art["trained_rows"], "rows")
