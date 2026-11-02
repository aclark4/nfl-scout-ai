"""Tests for documents.py and retriever.py.

Uses a tiny keyword-based embedding so tests run offline and fast
(no model download). Real retrieval uses Chroma's default model.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from documents import build_documents, league_ranks, ordinal  # noqa: E402


def player(name, **overrides):
    base = dict(
        name=name, rank=1, score=1.0, games_played=17, pass_attempts=500, pass_yards=4000,
        pass_touchdowns=30, interceptions=10, completion_percentage=65.0,
        completion_percentage_above_expectation=1.0, passer_rating=95.0, yards_per_attempt=8.0,
        pressure_to_sack_percentage=25.0, turnover_worthy_throw_rate=3.0, turnover_worthy_passes=15,
        bad_throw_pct_average=15.0, avg_depth_of_target=8.0, avg_time_to_throw=2.8, aggressiveness=15.0,
        rush_yards=200, rush_attempts=40, rush_touchdowns=2, sacks_taken=30, fumbles_lost=3, dakota=0.1,
    )
    base.update(overrides)
    return base


STATS = {
    "season": 2024,
    "players": [
        player("Lamar Jackson", score=6.86, epa_play=0.30, pressure_to_sack_percentage=21.5),
        player("Josh Allen", score=4.13, epa_play=0.32, pressure_to_sack_percentage=15.9),
        player("Will Levis", score=-13.09, epa_play=-0.24, pressure_to_sack_percentage=44.6),
    ],
}


def test_ordinal():
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 43)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "43rd"
    ]


def test_ranks_respect_direction():
    ranks = league_ranks(STATS["players"])
    assert ranks["epa_play"]["Josh Allen"] == 1  # higher is better
    assert ranks["pressure_to_sack_percentage"]["Josh Allen"] == 1  # lower is better
    assert ranks["pressure_to_sack_percentage"]["Will Levis"] == 3


def test_documents_cover_players_leaderboards_and_method():
    docs = build_documents(STATS)
    types = [d["metadata"]["type"] for d in docs]
    assert types.count("player") == 3
    assert types.count("methodology") == 1
    assert len({d["id"] for d in docs}) == len(docs)  # ids unique

    lamar = next(d for d in docs if d["id"] == "player:Lamar Jackson")
    assert "EPA per play 0.30 (2nd of 3)" in lamar["text"]
    assert "pressure-to-sack rate 21.5% (2nd of 3)" in lamar["text"]

    board = next(d for d in docs if d["id"] == "leaderboard:pressure_to_sack_percentage")
    assert board["text"].index("Josh Allen") < board["text"].index("Will Levis")


def test_documents_are_json_safe_metadata():
    for doc in build_documents(STATS):
        json.dumps(doc)
        assert all(isinstance(v, (str, int, float, bool)) for v in doc["metadata"].values())


chromadb = pytest.importorskip("chromadb")


class KeywordEmbedding(chromadb.EmbeddingFunction):
    """Deterministic bag-of-words hashing embedding for offline tests."""

    def __init__(self):
        pass

    def __call__(self, input):
        vectors = []
        for text in input:
            vec = [0.0] * 256
            for word in text.lower().replace(".", " ").replace(",", " ").split():
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1.0
            vectors.append(vec)
        return vectors

    @staticmethod
    def name():
        return "keyword-test"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return KeywordEmbedding()


def test_build_and_retrieve(tmp_path):
    from retriever import Retriever, build_index

    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(STATS))
    db = tmp_path / "chroma"
    ef = KeywordEmbedding()

    count = build_index(stats_path, db, embedding_function=ef)
    assert count == len(build_documents(STATS))

    retriever = Retriever(db, embedding_function=ef)
    hits = retriever.retrieve("Will Levis", k=3, where={"type": "player"})
    assert hits[0]["id"] == "player:Will Levis"
    assert len(retriever.retrieve("Will Levis", k=5)) == 5

    players_only = retriever.retrieve("pressure", k=3, where={"type": "player"})
    assert all(h["metadata"]["type"] == "player" for h in players_only)

    # rebuilding replaces rather than duplicates
    build_index(stats_path, db, embedding_function=ef)
    assert Retriever(db, embedding_function=ef).collection.count() == count
