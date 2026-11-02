"""
documents.py - Turn precomputed QB stats into text documents for retrieval (RAG).

Three kinds of documents are produced:
  * player       - one scouting summary per QB, with league ranks for key metrics
  * leaderboard  - top/bottom lists for a single metric ("who was best under pressure?")
  * methodology  - how the custom QB score is calculated

Each document is a dict: {"id": str, "text": str, "metadata": dict}.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_STATS_PATH = Path("data/qb_stats_2024.json")

# key -> (label, higher_is_better, format)
METRICS: dict[str, tuple[str, bool, str]] = {
    "score": ("Custom QB score", True, "{:.2f}"),
    "epa_play": ("EPA per play", True, "{:.2f}"),
    "pass_yards": ("Passing yards", True, "{:,.0f}"),
    "pass_touchdowns": ("Passing touchdowns", True, "{:.0f}"),
    "interceptions": ("Interceptions", False, "{:.0f}"),
    "completion_percentage_above_expectation": ("Completion % over expected (CPOE)", True, "{:+.1f}"),
    "passer_rating": ("Passer rating", True, "{:.1f}"),
    "yards_per_attempt": ("Yards per attempt", True, "{:.1f}"),
    "pressure_to_sack_percentage": ("Pressure-to-sack rate (%)", False, "{:.1f}%"),
    "turnover_worthy_throw_rate": ("Turnover-worthy throw rate (%)", False, "{:.1f}%"),
    "bad_throw_pct_average": ("Bad throw rate (%)", False, "{:.1f}%"),
    "avg_depth_of_target": ("Average depth of target (yards)", True, "{:.1f}"),
    "avg_time_to_throw": ("Average time to throw (seconds)", False, "{:.2f}s"),
    "rush_yards": ("Rushing yards", True, "{:,.0f}"),
    "rush_touchdowns": ("Rushing touchdowns", True, "{:.0f}"),
    "sacks_taken": ("Sacks taken", False, "{:.0f}"),
}

# Metrics whose league rank is called out in each player summary
RANKED_IN_SUMMARY = [
    "score",
    "epa_play",
    "completion_percentage_above_expectation",
    "pressure_to_sack_percentage",
    "turnover_worthy_throw_rate",
    "avg_depth_of_target",
]

METHODOLOGY_TEXT = (
    "How the custom QB score works (Adam Clark's 2024 QB Comparison formula). "
    "The score is a weighted sum of: yards per game above a 250-yard baseline (15%); "
    "touchdowns per pass or rush attempt (20%); a turnover component that compares "
    "turnover-worthy throws to actual interceptions relative to bad-throw rate, minus "
    "fumbles lost and interceptions (15%); aggressiveness times depth of target above "
    "7.5 yards (5%); EPA per play (25%, scaled by 30); time to throw relative to a "
    "2.5-second baseline (5%); and completion percentage over expected (10%). "
    "Pressure-to-sack rate divided by time to throw is subtracted (15%), as is 0.1 "
    "per sack taken. Higher scores are better. Sacks, interceptions and fumbles are "
    "season totals, so QBs who played fewer games are penalized less for them."
)


def load_stats(path: Path = DEFAULT_STATS_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def fmt(key: str, value: Any) -> str:
    if value is None:
        return "n/a"
    return METRICS[key][2].format(value)


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def league_ranks(players: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Return {metric: {player_name: rank}} where rank 1 is best."""
    ranks: dict[str, dict[str, int]] = {}
    for key, (_, higher_is_better, _) in METRICS.items():
        valid = [p for p in players if p.get(key) is not None]
        ordered = sorted(valid, key=lambda p: p[key], reverse=higher_is_better)
        ranks[key] = {p["name"]: i for i, p in enumerate(ordered, start=1)}
    return ranks


def player_document(player: dict[str, Any], ranks: dict[str, dict[str, int]], total: int, season: int) -> dict:
    name = player["name"]
    gp = player["games_played"]

    def ranked(key: str) -> str:
        r = ranks[key].get(name)
        return f"{fmt(key, player[key])} ({ordinal(r)} of {total})" if r else fmt(key, player[key])

    text = (
        f"{name} - {season} regular season QB scouting summary. "
        f"Custom QB score: {ranked('score')}. "
        f"Played {gp} games: {player['pass_attempts']} pass attempts, {fmt('pass_yards', player['pass_yards'])} "
        f"passing yards, {player['pass_touchdowns']} passing TDs, {player['interceptions']} interceptions, "
        f"{player['completion_percentage']}% completions, passer rating {fmt('passer_rating', player['passer_rating'])}, "
        f"{fmt('yards_per_attempt', player['yards_per_attempt'])} yards per attempt. "
        f"Efficiency: EPA per play {ranked('epa_play')}; CPOE {ranked('completion_percentage_above_expectation')}. "
        f"Under pressure: pressure-to-sack rate {ranked('pressure_to_sack_percentage')}, "
        f"{player['sacks_taken']:.0f} sacks taken, average time to throw {fmt('avg_time_to_throw', player['avg_time_to_throw'])}. "
        f"Ball security: {player['turnover_worthy_passes']:.0f} turnover-worthy throws, turnover-worthy throw rate "
        f"{ranked('turnover_worthy_throw_rate')}, bad throw rate {fmt('bad_throw_pct_average', player['bad_throw_pct_average'])}, "
        f"{player['fumbles_lost']:.0f} fumbles lost. "
        f"Style: average depth of target {ranked('avg_depth_of_target')}, aggressiveness "
        f"{player['aggressiveness']}% of throws into tight windows. "
        f"Rushing: {player['rush_attempts']} carries, {fmt('rush_yards', player['rush_yards'])} yards, "
        f"{player['rush_touchdowns']} rushing TDs."
    )
    return {
        "id": f"player:{name}",
        "text": text,
        "metadata": {"type": "player", "player": name, "season": season, "score": float(player["score"])},
    }


def leaderboard_document(key: str, players: list[dict[str, Any]], season: int, top_n: int = 10, bottom_n: int = 5) -> dict:
    label, higher_is_better, _ = METRICS[key]
    valid = [p for p in players if p.get(key) is not None]
    ordered = sorted(valid, key=lambda p: p[key], reverse=higher_is_better)
    direction = "higher is better" if higher_is_better else "lower is better"

    top = "; ".join(f"{i}. {p['name']} {fmt(key, p[key])}" for i, p in enumerate(ordered[:top_n], start=1))
    start = len(ordered) - bottom_n + 1
    bottom = "; ".join(
        f"{i}. {p['name']} {fmt(key, p[key])}" for i, p in enumerate(ordered[-bottom_n:], start=start)
    )
    text = (
        f"{season} QB leaderboard for {label} ({direction}), among {len(ordered)} qualifying QBs. "
        f"Best: {top}. Worst: {bottom}."
    )
    return {
        "id": f"leaderboard:{key}",
        "text": text,
        "metadata": {"type": "leaderboard", "metric": key, "season": season},
    }


def build_documents(stats: dict[str, Any]) -> list[dict]:
    players, season = stats["players"], stats["season"]
    ranks = league_ranks(players)
    docs = [player_document(p, ranks, len(players), season) for p in players]
    docs += [leaderboard_document(key, players, season) for key in METRICS]
    docs.append({
        "id": "methodology:score",
        "text": METHODOLOGY_TEXT,
        "metadata": {"type": "methodology", "season": season},
    })
    return docs


if __name__ == "__main__":
    documents = build_documents(load_stats())
    print(f"Built {len(documents)} documents. Examples:\n")
    for doc in (documents[0], next(d for d in documents if d["id"] == "leaderboard:pressure_to_sack_percentage")):
        print(f"[{doc['id']}]\n{doc['text']}\n")
