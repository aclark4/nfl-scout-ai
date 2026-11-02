"""
app.py - FastAPI service for the NFL scouting agent.

Run locally:
    uvicorn app:app --reload
Then open http://localhost:8000/docs to try it.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

LOG_PATH = Path("logs/requests.jsonl")
logger = logging.getLogger("nfl-scout")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, examples=["Who's better, Joe Burrow or Josh Allen?"])


class ToolCall(BaseModel):
    name: str
    input: dict[str, Any]
    error: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    tools_used: list[ToolCall]
    model: str
    latency_ms: int
    usage: dict[str, int]


def log_request(question: str, result: Optional[dict[str, Any]], error: Optional[str] = None) -> None:
    """Append one structured line per request for later analysis."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "error": error,
    }
    if result:
        record.update(
            latency_ms=result["latency_ms"],
            tools=[t["name"] for t in result["tools_used"]],
            sources=result["sources"],
            usage=result["usage"],
            model=result["model"],
        )
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def create_app(agent=None) -> FastAPI:
    """Build the app. Pass an agent for tests; otherwise a real one is created at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if agent is not None:
            app.state.agent = agent
        else:
            load_dotenv()
            from agent import ScoutAgent
            from retriever import Retriever

            app.state.agent = ScoutAgent(retriever=Retriever())
            logger.info("Agent ready (model=%s)", app.state.agent.model)
        yield

    app = FastAPI(
        title="NFL Scout AI",
        description="RAG + tool-using Claude agent for 2024 NFL quarterback analysis.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "players": len(app.state.agent.store.players)}

    @app.get("/players")
    def players() -> list[str]:
        return sorted(app.state.agent.store.players)

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            result = app.state.agent.ask(req.question)
        except Exception as exc:
            logger.exception("ask failed")
            log_request(req.question, None, error=str(exc))
            raise HTTPException(status_code=502, detail="The model request failed. Check the server logs.")
        log_request(req.question, result)
        logger.info("ask ok in %d ms, tools=%s", (time.perf_counter() - start) * 1000,
                    [t["name"] for t in result["tools_used"]])
        return result

    return app


app = create_app()
