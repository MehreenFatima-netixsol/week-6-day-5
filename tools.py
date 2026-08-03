"""
tools.py
--------
Structured, grounded lookups over the AFL dataset, plus fuzzy entity resolution.
Every lookup returns the same envelope shape so the grounding verifier (guardrails.py)
can confirm any number in a response traces back to a real tool call.
"""
from __future__ import annotations
import difflib
import re
from collections import defaultdict
import pandas as pd

from . import data as afl_data

KNOWN_TEAMS = afl_data.CLUBS
KNOWN_PLAYERS = afl_data.PLAYERS.player_name.tolist()
STAT_COLUMNS = afl_data.STAT_COLUMNS
SEASON_MIN, SEASON_MAX = afl_data.SEASON_MIN, afl_data.SEASON_MAX
PSEASON_MIN, PSEASON_MAX = afl_data.PSEASON_MIN, afl_data.PSEASON_MAX

STAT_ALIASES = {
    "disposal": "disposals", "disposals": "disposals", "touches": "disposals",
    "goal": "goals", "goals": "goals",
    "fantasy": "fantasy_points", "fantasy_points": "fantasy_points", "supercoach": "fantasy_points",
    "impact": "match_impact_score", "mis": "match_impact_score",
    "match_impact_score": "match_impact_score", "hangers": None,
}


class LookupError_(ValueError):
    """Raised when a lookup cannot be served. Message is safe to show a user."""


TEAM_ALIASES = {
    "cats": "Geelong Cats", "blues": "Carlton Blues", "magpies": "Collingwood Magpies",
    "pies": "Collingwood Magpies", "bombers": "Essendon Bombers", "dons": "Essendon Bombers",
    "dockers": "Fremantle Dockers", "freo": "Fremantle Dockers", "suns": "Gold Coast Suns",
    "giants": "Greater Western Sydney Giants", "gws": "Greater Western Sydney Giants",
    "hawks": "Hawthorn Hawks", "demons": "Melbourne Demons", "dees": "Melbourne Demons",
    "kangaroos": "North Melbourne Kangaroos", "roos": "North Melbourne Kangaroos",
    "north": "North Melbourne Kangaroos", "power": "Port Adelaide Power",
    "tigers": "Richmond Tigers", "saints": "St Kilda Saints", "swans": "Sydney Swans",
    "eagles": "West Coast Eagles", "bulldogs": "Western Bulldogs", "dogs": "Western Bulldogs",
    "doggies": "Western Bulldogs", "crows": "Adelaide Crows", "lions": "Brisbane Lions",
}
TEAM_ALIASES = {k: v for k, v in TEAM_ALIASES.items() if v in KNOWN_TEAMS}
_TEAM_LOWER = {t.lower(): t for t in KNOWN_TEAMS}
_PLAYER_LOWER = {}
for _n in KNOWN_PLAYERS:
    _PLAYER_LOWER.setdefault(_n.lower(), _n)
_SURNAME_INDEX = defaultdict(list)
for _n in KNOWN_PLAYERS:
    _SURNAME_INDEX[_n.split()[-1].lower()].append(_n)


def resolve_team(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise LookupError_("A team name is required.")
    key = re.sub(r"\s+", " ", name.strip().lower())
    key = re.sub(r"^(the)\s+", "", key)
    if key in _TEAM_LOWER:
        return _TEAM_LOWER[key]
    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]
    subs = [full for low, full in _TEAM_LOWER.items() if key in low or low in key]
    if len(set(subs)) == 1:
        return subs[0]
    close = difflib.get_close_matches(key, list(_TEAM_LOWER), n=3, cutoff=0.72)
    if len(close) == 1:
        return _TEAM_LOWER[close[0]]
    opts = sorted({*subs, *[_TEAM_LOWER[c] for c in close]})
    if opts:
        raise LookupError_(f"'{name}' matches several clubs: {', '.join(opts)}. Which one?")
    raise LookupError_(f"'{name}' is not a club in this dataset. Known clubs: {', '.join(KNOWN_TEAMS)}.")


def resolve_player(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise LookupError_("A player name is required.")
    key = re.sub(r"\s+", " ", name.strip().lower())
    if key in _PLAYER_LOWER:
        return _PLAYER_LOWER[key]
    if key in _SURNAME_INDEX and len(set(_SURNAME_INDEX[key])) == 1:
        return _SURNAME_INDEX[key][0]
    subs = sorted({full for low, full in _PLAYER_LOWER.items() if key in low})
    if len(subs) == 1:
        return subs[0]
    close = difflib.get_close_matches(key, list(_PLAYER_LOWER), n=4, cutoff=0.78)
    if len(close) == 1:
        return _PLAYER_LOWER[close[0]]
    opts = sorted({*subs, *[_PLAYER_LOWER[c] for c in close]})[:6]
    if opts:
        raise LookupError_(f"'{name}' could be several players: {', '.join(opts)}. Which one did you mean?")
    raise LookupError_(f"No player matching '{name}' appears in the dataset (coverage: {PSEASON_MIN}-{PSEASON_MAX}).")


def resolve_stat(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise LookupError_(f"A statistic is required. Available: {', '.join(sorted(STAT_COLUMNS))}.")
    key = name.strip().lower().replace(" ", "_")
    if key in STAT_ALIASES and STAT_ALIASES[key]:
        return STAT_ALIASES[key]
    if key in STAT_COLUMNS:
        return key
    close = difflib.get_close_matches(key, [k for k, v in STAT_ALIASES.items() if v], n=1, cutoff=0.75)
    if close:
        return STAT_ALIASES[close[0]]
    raise LookupError_(f"'{name}' is not a statistic in this dataset. Available: {', '.join(sorted(STAT_COLUMNS))}.")


def resolve_season(season) -> int:
    if season is None:
        raise LookupError_("A season (year) is required.")
    try:
        y = int(str(season).strip()[:4])
    except Exception:
        raise LookupError_(f"Could not read '{season}' as a season. Use a year like 2023.")
    if not (SEASON_MIN <= y <= SEASON_MAX):
        raise LookupError_(f"Season {y} is outside the dataset, which covers {SEASON_MIN}-{SEASON_MAX}.")
    return y


def _envelope(tool, query, facts, rows, summary, source, scanned):
    return {"ok": True, "tool": tool, "query": query, "facts": facts,
            "rows": rows, "summary": summary, "source": source, "rows_scanned": int(scanned)}


def _error_envelope(tool, query, message):
    return {"ok": False, "tool": tool, "query": query, "error": message}


# ------------------------------------------------------------------ lookups ---------
def q_head_to_head(team_a: str, team_b: str, from_season=None, to_season=None) -> dict:
    a, b = resolve_team(team_a), resolve_team(team_b)
    m = afl_data.MATCHES
    mask = ((m.home_team == a) & (m.away_team == b)) | ((m.home_team == b) & (m.away_team == a))
    sub = m[mask]
    if from_season:
        sub = sub[sub.season >= resolve_season(from_season)]
    if to_season:
        sub = sub[sub.season <= resolve_season(to_season)]
    if sub.empty:
        return _error_envelope("head_to_head", f"{a} vs {b}", f"No matches found between {a} and {b} in range.")
    wins_a = int((sub.winner == a).sum())
    wins_b = int((sub.winner == b).sum())
    facts = {"team_a": a, "team_b": b, "games": int(len(sub)), "wins_a": wins_a, "wins_b": wins_b}
    summary = f"{a} lead {wins_a}-{wins_b} over {b} across {len(sub)} meetings."
    return _envelope("head_to_head", f"{a} vs {b}", facts, [], summary, "synthetic_match_table", len(sub))


def q_team_season_stats(team: str, season) -> dict:
    t = resolve_team(team)
    y = resolve_season(season)
    m = afl_data.MATCHES
    sub = m[((m.home_team == t) | (m.away_team == t)) & (m.season == y)]
    if sub.empty:
        return _error_envelope("team_season_stats", f"{t} {y}", f"No {y} matches found for {t}.")
    wins = int((sub.winner == t).sum())
    facts = {"team": t, "season": y, "games": int(len(sub)), "wins": wins, "losses": int(len(sub) - wins)}
    summary = f"{t} played {len(sub)} games in {y}, winning {wins}."
    return _envelope("team_season_stats", f"{t} {y}", facts, [], summary, "synthetic_match_table", len(sub))


def q_player_stats(player: str, stat: str, season=None) -> dict:
    p = resolve_player(player)
    s = resolve_stat(stat)
    pg = afl_data.PLAYER_GAMES
    sub = pg[pg.player_name == p]
    if season:
        sub = sub[sub.season == resolve_season(season)]
    if sub.empty:
        return _error_envelope("player_stats", f"{p} {s}", f"No {s} data found for {p} in range.")
    facts = {"player": p, "stat": s, "games": int(len(sub)),
             "total": float(sub[s].sum()), "average": round(float(sub[s].mean()), 2)}
    summary = f"{p} averaged {facts['average']} {s} across {facts['games']} games."
    return _envelope("player_stats", f"{p} {s}", facts, [], summary, "synthetic_player_game_table", len(sub))


def q_last_round(team: str) -> dict:
    t = resolve_team(team)
    m = afl_data.MATCHES
    sub = m[(m.home_team == t) | (m.away_team == t)].sort_values("match_date")
    if sub.empty:
        return _error_envelope("last_round", t, f"No match history found for {t}.")
    last = sub.iloc[-1]
    opp = last.away_team if last.home_team == t else last.home_team
    result = "WIN" if last.winner == t else "LOSS"
    margin = last.margin if last.home_team == t else -last.margin
    pg = afl_data.PLAYER_GAMES
    game_players = pg[(pg.match_id == last.match_id) & (pg.team == t)]
    disp_leader = goal_leader = None
    if not game_players.empty:
        dl = game_players.loc[game_players.disposals.idxmax()]
        gl = game_players.loc[game_players.goals.idxmax()]
        disp_leader = f"{dl.player_name} ({int(dl.disposals)})"
        goal_leader = f"{gl.player_name} ({int(gl.goals)})"
    facts = {"team": t, "season": int(last.season), "round": int(last["round"]), "opponent": opp,
             "result": result, "margin": int(margin),
             "disposals_leader": disp_leader, "goals_leader": goal_leader}
    summary = (f"{t}'s last game (R{facts['round']}, {facts['season']}) was a {result} "
               f"(margin {margin:+d}) vs {opp}.")
    return _envelope("last_round", t, facts, [], summary, "synthetic_match_table", 1)


def q_ladder(season) -> dict:
    y = resolve_season(season)
    m = afl_data.MATCHES[afl_data.MATCHES.season == y]
    if m.empty:
        return _error_envelope("ladder", str(y), f"No {y} season data found.")
    records = defaultdict(lambda: {"wins": 0, "losses": 0, "pf": 0, "pa": 0})
    for _, r in m.iterrows():
        for team, is_home in ((r.home_team, True), (r.away_team, False)):
            rec = records[team]
            rec["pf"] += r.home_score if is_home else r.away_score
            rec["pa"] += r.away_score if is_home else r.home_score
            if r.winner == team:
                rec["wins"] += 1
            else:
                rec["losses"] += 1
    ladder = sorted(records.items(), key=lambda kv: (-kv[1]["wins"], -(kv[1]["pf"] - kv[1]["pa"])))
    rows = [{"rank": i + 1, "team": t, **v, "percentage": round(v["pf"] / max(v["pa"], 1) * 100, 1)}
            for i, (t, v) in enumerate(ladder)]
    facts = {"season": y, "top_team": rows[0]["team"], "top_wins": rows[0]["wins"]}
    summary = f"{y} ladder-leader: {rows[0]['team']} ({rows[0]['wins']} wins)."
    return _envelope("ladder", str(y), facts, rows[:8], summary, "synthetic_match_table", len(m))


AVAILABLE_TOOLS = {
    "head_to_head": q_head_to_head,
    "team_season_stats": q_team_season_stats,
    "player_stats": q_player_stats,
    "last_round": q_last_round,
    "ladder": q_ladder,
}
