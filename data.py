"""
data.py
-------
Synthetic AFL data layer.

The capstone brief assumes a production feature store (match_features_v2.0.0.parquet,
player_features_v2.0.0.parquet, players.parquet, and joblib model artifacts) built in
Weeks 1-5. Those binary artifacts were not included in the files uploaded for this
capstone, so this module generates a deterministic, seeded synthetic dataset that
respects the SAME schema and coverage the system prompt promises:

    - 19 clubs
    - match results 1983-2025
    - player statistics 1999-2025
    - stats: disposals, fantasy_points, goals, match_impact_score

Swap this module for real parquet loads (see the commented `load_real_data()` stub at
the bottom) to go to production. Every downstream node (tools.py, model.py, graph.py)
only depends on the DataFrames this module exposes, not on how they were built -- so
that swap is a one-file change.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SEASON_MIN, SEASON_MAX = 1983, 2025
PSEASON_MIN, PSEASON_MAX = 1999, 2025

CLUBS = [
    "Adelaide Crows", "Brisbane Lions", "Carlton Blues", "Collingwood Magpies",
    "Essendon Bombers", "Fremantle Dockers", "Geelong Cats",
    "Gold Coast Suns", "Greater Western Sydney Giants", "Hawthorn Hawks",
    "Melbourne Demons", "North Melbourne Kangaroos", "Port Adelaide Power",
    "Richmond Tigers", "St Kilda Saints", "Sydney Swans",
    "West Coast Eagles", "Western Bulldogs", "Fitzroy Lions",
]
assert len(CLUBS) == 19, len(CLUBS)

STAT_COLUMNS = {"disposals", "fantasy_points", "goals", "match_impact_score"}

_rng = np.random.default_rng(7)

# --- persistent per-club "strength" so results are internally consistent ---
CLUB_STRENGTH = {c: float(_rng.normal(0, 1)) for c in CLUBS}

FIRST_NAMES = ["Jack", "Tom", "Sam", "Josh", "Nick", "Charlie", "Max", "Will",
               "Jordan", "Isaac", "Noah", "Zach", "Ryan", "Ben", "Lachlan",
               "Ethan", "Harry", "Oscar", "Liam", "Cody"]
LAST_NAMES = ["Smith", "Daicos", "Cripps", "Neale", "Petracca", "Bontempelli",
              "Oliver", "Walsh", "Rowell", "Miller", "Green", "Hawkins",
              "Warner", "Heeney", "De Goey", "Sheed", "Butters", "Yeo",
              "Wines", "Docherty"]


def _make_players(n=520):
    names = set()
    rows = []
    i = 0
    while len(rows) < n:
        fn, ln = _rng.choice(FIRST_NAMES), _rng.choice(LAST_NAMES)
        name = f"{fn} {ln}"
        if name in names:
            name = f"{fn} {ln}-{i}"
        names.add(name)
        club = CLUBS[i % len(CLUBS)]
        skill = float(np.clip(_rng.normal(0, 1), -2.5, 2.5))
        rows.append({"player_id": f"P{i:04d}", "player_name": name, "team": club, "skill": skill})
        i += 1
    return pd.DataFrame(rows)


_PLAYERS_CACHE = DATA_DIR / "_synthetic_players.pkl"
try:
    if _PLAYERS_CACHE.exists():
        PLAYERS = pd.read_pickle(_PLAYERS_CACHE)
    else:
        raise FileNotFoundError
except Exception as e:
    # Covers: no cache yet, corrupt file, or -- the common cross-machine case -- a cache
    # pickled under a different pandas/numpy version whose internal dtype representation
    # (e.g. StringDtype storage) this pandas can't deserialise. Any failure here just means
    # "rebuild from scratch," which costs time but never breaks the app.
    if not isinstance(e, FileNotFoundError):
        print(f"[data.py] Could not load cached players ({type(e).__name__}); rebuilding.")
    PLAYERS = _make_players()
    try:
        PLAYERS.to_pickle(_PLAYERS_CACHE)
    except Exception:
        pass  # caching is a pure optimisation; failing to write it must never crash startup


def _make_matches():
    rows, mid = [], 0
    for season in range(SEASON_MIN, SEASON_MAX + 1):
        rounds = 23
        for rnd in range(1, rounds + 1):
            shuffled = list(CLUBS)
            _rng.shuffle(shuffled)
            for a, b in zip(shuffled[0::2], shuffled[1::2]):
                home, away = a, b
                home_adv = 0.35
                margin_mu = (CLUB_STRENGTH[home] - CLUB_STRENGTH[away]) * 18 + home_adv * 6
                margin = int(_rng.normal(margin_mu, 22))
                home_score = max(30, 85 + margin // 2 + int(_rng.normal(0, 10)))
                away_score = max(30, home_score - margin)
                date = pd.Timestamp(year=season, month=4, day=1) + pd.Timedelta(days=(rnd - 1) * 7)
                rows.append({
                    "match_id": f"M{mid:06d}", "season": season, "round": rnd,
                    "match_date": date, "home_team": home, "away_team": away,
                    "home_score": home_score, "away_score": away_score,
                    "margin": home_score - away_score,
                    "winner": home if home_score >= away_score else away,
                    "venue_state": "VIC" if home in ("Collingwood Magpies", "Carlton Blues",
                                                       "Essendon Bombers", "Geelong Cats",
                                                       "Hawthorn Hawks", "Melbourne Demons",
                                                       "North Melbourne Kangaroos",
                                                       "Richmond Tigers", "St Kilda Saints",
                                                       "Western Bulldogs") else "OTHER",
                })
                mid += 1
    return pd.DataFrame(rows)


_MATCHES_CACHE = DATA_DIR / "_synthetic_matches.pkl"
try:
    if _MATCHES_CACHE.exists():
        MATCHES = pd.read_pickle(_MATCHES_CACHE)
    else:
        raise FileNotFoundError
except Exception as e:
    if not isinstance(e, FileNotFoundError):
        print(f"[data.py] Could not load cached matches ({type(e).__name__}); rebuilding.")
    MATCHES = _make_matches()
    try:
        MATCHES.to_pickle(_MATCHES_CACHE)
    except Exception:
        pass


def _make_player_games():
    rows = []
    m = MATCHES[MATCHES.season >= PSEASON_MIN]
    for _, match in m.iterrows():
        for side, team in (("home", match.home_team), ("away", match.away_team)):
            roster = PLAYERS[PLAYERS.team == team].sample(n=18, random_state=abs(hash((match.match_id, side))) % (2**32))
            for _, p in roster.iterrows():
                base = 18 + p.skill * 6
                disposals = max(2, int(_rng.normal(base, 5)))
                goals = max(0, int(_rng.normal(0.9 + p.skill * 0.6, 1.1)))
                fantasy_points = max(0, int(disposals * 3.2 + goals * 6 + _rng.normal(0, 8)))
                match_impact_score = round(float(np.clip(0.3 * p.skill + _rng.normal(0, 0.4), -3, 3)), 3)
                rows.append({
                    "match_id": match.match_id, "season": match.season, "round": match["round"],
                    "player_id": p.player_id, "player_name": p.player_name, "team": team,
                    "disposals": disposals, "goals": goals,
                    "fantasy_points": fantasy_points, "match_impact_score": match_impact_score,
                })
    return pd.DataFrame(rows)


_PG_CACHE = DATA_DIR / "_synthetic_player_games.pkl"
try:
    if _PG_CACHE.exists():
        PLAYER_GAMES = pd.read_pickle(_PG_CACHE)
    else:
        raise FileNotFoundError
except Exception as e:
    if not isinstance(e, FileNotFoundError):
        print(f"[data.py] Could not load cached player-games ({type(e).__name__}); rebuilding.")
    PLAYER_GAMES = _make_player_games()
    try:
        PLAYER_GAMES.to_pickle(_PG_CACHE)
    except Exception:
        pass

print(f"[data.py] synthetic dataset ready: {len(MATCHES):,} matches, "
      f"{len(PLAYER_GAMES):,} player-game rows, {PLAYERS.player_id.nunique()} players, "
      f"{len(CLUBS)} clubs.")

# --- production swap point ---------------------------------------------------
# def load_real_data():
#     matches = pd.read_parquet(".../match_features_v2.0.0.parquet")
#     player_games = pd.read_parquet(".../player_features_v2.0.0.parquet")
#     players = pd.read_parquet(".../players.parquet")
#     return matches, player_games, players
