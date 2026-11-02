"""
tools.py - Exact-data tools the AI agent can call.

Retrieval finds relevant text, but numbers and comparisons should come from
the source data, not the model's memory. These tools read the precomputed
stats JSON and reuse the original scoring formula.
"""

from __future__ import annotations

import difflib
import json
from typing import Any, Optional

from documents import DEFAULT_STATS_PATH, METRICS, load_stats
from qb_stats import compare_qbs, score_breakdown

TOOL_SCHEMAS = [
    {
        "name": "search_scouting_docs",
        "description": (
            "Semantic search over 2024 QB scouting summaries, metric leaderboards, and the "
            "scoring methodology. Use for open-ended questions or to find context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "k": {"type": "integer", "description": "Number of results (1-8)", "default": 4},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_player_stats",
        "description": "Exact 2024 regular-season stats, custom score, overall rank, and per-component score breakdown for one QB.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "QB name, e.g. 'Joe Burrow' or 'Burrow'"}},
            "required": ["name"],
        },
    },
    {
        "name": "compare_qbs",
        "description": (
            "Compare two QBs with Adam Clark's weighted scoring formula. Returns both scores, "
            "the winner, the margin, side-by-side key stats, and a breakdown of which formula components drove the gap."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "qb_a": {"type": "string", "description": "First QB name"},
                "qb_b": {"type": "string", "description": "Second QB name"},
            },
            "required": ["qb_a", "qb_b"],
        },
    },
    {
        "name": "rank_qbs",
        "description": "Rank all qualifying QBs by one metric (best first, honoring whether higher or lower is better).",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "enum": list(METRICS.keys())},
                "top_n": {"type": "integer", "description": "How many to return (1-43)", "default": 5},
                "worst": {"type": "boolean", "description": "Return the worst instead of the best", "default": False},
            },
            "required": ["metric"],
        },
    },
]

KEY_STATS = [
    "score", "rank", "games_played", "pass_yards", "pass_touchdowns", "interceptions", "epa_play",
    "completion_percentage_above_expectation", "pressure_to_sack_percentage",
    "turnover_worthy_throw_rate", "avg_depth_of_target", "rush_yards",
]


class StatsStore:
    """Look up QBs in the precomputed stats file."""

    def __init__(self, stats: Optional[dict[str, Any]] = None):
        self.stats = stats if stats is not None else load_stats(DEFAULT_STATS_PATH)
        self.players = {p["name"]: p for p in self.stats["players"]}

    def find(self, name: str) -> Optional[dict[str, Any]]:
        """Match full names, last names, and small typos ('burow' -> 'Joe Burrow')."""
        query = name.strip().lower()
        by_lower = {n.lower(): n for n in self.players}
        if query in by_lower:
            return self.players[by_lower[query]]
        last_name_hits = [n for n in self.players if n.lower().split()[-1] == query]
        if len(last_name_hits) == 1:
            return self.players[last_name_hits[0]]
        close = difflib.get_close_matches(query, list(by_lower), n=1, cutoff=0.75)
        if close:
            return self.players[by_lower[close[0]]]
        close_last = difflib.get_close_matches(
            query, [n.lower().split()[-1] for n in self.players], n=1, cutoff=0.8
        )
        if close_last:
            matches = [n for n in self.players if n.lower().split()[-1] == close_last[0]]
            if len(matches) == 1:
                return self.players[matches[0]]
        return None

    def not_found(self, name: str) -> dict[str, Any]:
        suggestions = difflib.get_close_matches(name, list(self.players), n=3, cutoff=0.4)
        return {
            "error": f"No qualifying 2024 QB named '{name}' (min {self.stats.get('min_attempts', 100)} attempts).",
            "did_you_mean": suggestions,
        }

    def get_player_stats(self, name: str) -> dict[str, Any]:
        player = self.find(name)
        if player is None:
            return self.not_found(name)
        return {**player, "score_breakdown": score_breakdown(player)}

    def compare(self, qb_a: str, qb_b: str) -> dict[str, Any]:
        a, b = self.find(qb_a), self.find(qb_b)
        if a is None:
            return self.not_found(qb_a)
        if b is None:
            return self.not_found(qb_b)
        result = compare_qbs(a, b)
        result["key_stats"] = {p["name"]: {k: p.get(k) for k in KEY_STATS} for p in (a, b)}
        return result

    def rank(self, metric: str, top_n: int = 5, worst: bool = False) -> dict[str, Any]:
        if metric not in METRICS:
            return {"error": f"Unknown metric '{metric}'.", "valid_metrics": list(METRICS)}
        label, higher_is_better, _ = METRICS[metric]
        valid = [p for p in self.players.values() if p.get(metric) is not None]
        ordered = sorted(valid, key=lambda p: p[metric], reverse=higher_is_better)
        top_n = max(1, min(int(top_n), len(ordered)))
        positions = list(enumerate(ordered, start=1))
        chosen = positions[-top_n:][::-1] if worst else positions[:top_n]
        return {
            "metric": metric,
            "label": label,
            "direction": "higher is better" if higher_is_better else "lower is better",
            "showing": "worst" if worst else "best",
            "results": [{"rank": r, "name": p["name"], "value": p[metric]} for r, p in chosen],
        }


def execute_tool(name: str, args: dict[str, Any], store: StatsStore, retriever=None) -> dict[str, Any]:
    """Run one tool call and return a JSON-serializable result."""
    try:
        if name == "search_scouting_docs":
            if retriever is None:
                return {"error": "Search is unavailable."}
            k = max(1, min(int(args.get("k", 4)), 8))
            hits = retriever.retrieve(args["query"], k=k)
            return {"results": [{"id": h["id"], "text": h["text"]} for h in hits]}
        if name == "get_player_stats":
            return store.get_player_stats(args["name"])
        if name == "compare_qbs":
            return store.compare(args["qb_a"], args["qb_b"])
        if name == "rank_qbs":
            return store.rank(args["metric"], args.get("top_n", 5), args.get("worst", False))
        return {"error": f"Unknown tool '{name}'."}
    except (KeyError, ValueError, TypeError) as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}


def to_json(result: dict[str, Any]) -> str:
    return json.dumps(result, default=str)
