"""Tests for tools.py, agent.py and app.py using a fake Claude client (no API calls)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import ScoutAgent  # noqa: E402
from tools import StatsStore, execute_tool  # noqa: E402


def player(name, score, **kw):
    base = dict(
        name=name, score=score, rank=0, games_played=17, pass_attempts=500, pass_yards=4000,
        pass_touchdowns=30, interceptions=10, completion_percentage=65.0,
        completion_percentage_above_expectation=1.0, passer_rating=95.0, yards_per_attempt=8.0,
        pressure_to_sack_percentage=25.0, turnover_worthy_throw_rate=3.0, turnover_worthy_passes=15,
        bad_throw_pct_average=15.0, avg_depth_of_target=8.0, avg_time_to_throw=2.8, aggressiveness=15.0,
        rush_yards=200, rush_attempts=40, rush_touchdowns=2, sacks_taken=30, fumbles_lost=3,
        epa_play=0.1, dakota=0.1,
    )
    base.update(kw)
    return base


STATS = {
    "season": 2024,
    "min_attempts": 100,
    "players": [
        player("Joe Burrow", -0.39, pass_yards=4918, pressure_to_sack_percentage=31.0),
        player("Josh Allen", 4.13, epa_play=0.32, pressure_to_sack_percentage=15.9),
        player("Will Levis", -13.09, pressure_to_sack_percentage=44.6),
        player("Joe Flacco", -2.01),
    ],
}


@pytest.fixture
def store():
    return StatsStore(STATS)


# ---------------------------------------------------------------- tools

def test_find_full_last_and_typo(store):
    assert store.find("joe burrow")["name"] == "Joe Burrow"
    assert store.find("Allen")["name"] == "Josh Allen"
    assert store.find("Burow")["name"] == "Joe Burrow"
    assert store.find("Tom Brady") is None


def test_unknown_player_gives_error_and_suggestions(store):
    result = store.get_player_stats("Joe Burro")  # close to two Joes; fuzzy match still finds Burrow
    assert result["name"] == "Joe Burrow"
    missing = store.get_player_stats("Tom Brady")
    assert "error" in missing and "did_you_mean" in missing


def test_compare_includes_winner_and_key_stats(store):
    result = execute_tool("compare_qbs", {"qb_a": "Burrow", "qb_b": "Josh Allen"}, store)
    assert result["winner"] in ("Joe Burrow", "Josh Allen")
    assert set(result["key_stats"]) == {"Joe Burrow", "Josh Allen"}


def test_rank_respects_direction(store):
    best = execute_tool("rank_qbs", {"metric": "pressure_to_sack_percentage", "top_n": 2}, store)
    assert [r["name"] for r in best["results"]] == ["Josh Allen", "Joe Flacco"]
    worst = execute_tool("rank_qbs", {"metric": "pressure_to_sack_percentage", "top_n": 1, "worst": True}, store)
    assert worst["results"][0] == {"rank": 4, "name": "Will Levis", "value": 44.6}


def test_bad_tool_inputs_return_errors(store):
    assert "error" in execute_tool("rank_qbs", {"metric": "made_up"}, store)
    assert "error" in execute_tool("compare_qbs", {"qb_a": "Burrow"}, store)
    assert "error" in execute_tool("nope", {}, store)


# ---------------------------------------------------------------- agent loop

def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(id_, name, input_):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


def response(blocks, stop_reason):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason,
                           usage=SimpleNamespace(input_tokens=100, output_tokens=20))


class FakeClient:
    """Returns scripted responses and records each request."""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.scripted.pop(0)


class FakeRetriever:
    def retrieve(self, question, k=5, where=None):
        return [{"id": "player:Joe Burrow", "text": "Joe Burrow summary", "metadata": {}, "distance": 0.1}][:k]


def test_agent_runs_tool_then_answers(store):
    client = FakeClient([
        response([tool_block("t1", "compare_qbs", {"qb_a": "Joe Burrow", "qb_b": "Josh Allen"})], "tool_use"),
        response([text_block("Josh Allen scores higher. [tool:compare_qbs]")], "end_turn"),
    ])
    agent = ScoutAgent(retriever=FakeRetriever(), store=store, client=client, model="test-model")
    result = agent.ask("Who's better, Burrow or Allen?")

    assert result["answer"].startswith("Josh Allen")
    assert [t["name"] for t in result["tools_used"]] == ["compare_qbs"]
    assert result["sources"] == ["player:Joe Burrow"]
    assert result["usage"] == {"input_tokens": 200, "output_tokens": 40}

    # The second request must carry the tool result back to Claude
    second = client.requests[1]["messages"]
    assert second[-1]["content"][0]["type"] == "tool_result"
    assert '"winner":' in second[-1]["content"][0]["content"]
    assert second[-1]["content"][0]["tool_use_id"] == "t1"
    # Retrieved context is in the first user message
    assert "[player:Joe Burrow]" in client.requests[0]["messages"][0]["content"]


def test_agent_without_tools_is_plain_rag(store):
    client = FakeClient([response([text_block("Answer from context.")], "end_turn")])
    agent = ScoutAgent(retriever=FakeRetriever(), store=store, client=client, use_tools=False)
    agent.ask("How did Burrow play?")
    assert "tools" not in client.requests[0]


def test_agent_stops_after_round_limit(store):
    looping = [response([tool_block(f"t{i}", "get_player_stats", {"name": "Burrow"})], "tool_use") for i in range(10)]
    client = FakeClient(looping)
    agent = ScoutAgent(retriever=FakeRetriever(), store=store, client=client)
    result = agent.ask("loop forever")
    assert len(client.requests) == 6  # 1 + MAX_TOOL_ROUNDS
    assert "couldn't produce an answer" in result["answer"]


# ---------------------------------------------------------------- API

def test_api_ask_health_and_players(store, tmp_path, monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import app as app_module

    monkeypatch.setattr(app_module, "LOG_PATH", tmp_path / "requests.jsonl")
    client = FakeClient([response([text_block("Burrow threw for 4,918 yards.")], "end_turn")])
    agent = ScoutAgent(retriever=FakeRetriever(), store=store, client=client, model="test-model")

    with TestClient(app_module.create_app(agent)) as api:
        assert api.get("/health").json() == {"status": "ok", "players": 4}
        assert "Joe Burrow" in api.get("/players").json()

        r = api.post("/ask", json={"question": "How many yards did Burrow throw?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "Burrow threw for 4,918 yards."
        assert body["model"] == "test-model"

        assert api.post("/ask", json={"question": "x"}).status_code == 422  # too short

    log_lines = (tmp_path / "requests.jsonl").read_text().strip().splitlines()
    assert len(log_lines) == 1
