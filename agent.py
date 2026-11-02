"""
agent.py - Claude-powered NFL scouting agent (RAG + tool use).

Flow for each question:
  1. Retrieve the most relevant scouting documents from Chroma (RAG context).
  2. Send the question + context to Claude along with exact-data tools.
  3. Loop while Claude calls tools, feeding results back, until it answers.
  4. Return the answer plus sources, tools used, latency, and token usage.

Usage:
    python agent.py "Who's better, Joe Burrow or Josh Allen?"
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Optional

from tools import TOOL_SCHEMAS, StatsStore, execute_tool, to_json

DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
MAX_TOOL_ROUNDS = 5

BASE_RULES = """You are an NFL scouting assistant covering 2024 regular-season quarterbacks.

Rules:
- Answer only from the provided context{tool_clause}. Never use outside knowledge for stats.
- If the data does not cover something (other seasons, playoffs, non-QBs, QBs under 100 attempts), say so plainly.
- Keep answers concise: 2-5 sentences or a short list. Cite sources in brackets, e.g. [player:Joe Burrow]{tool_cite}."""

TOOL_RULES = """
- For any specific number, comparison, or ranking, call a tool (get_player_stats, compare_qbs, rank_qbs) rather than estimating.
- When asked who is "better", use compare_qbs. Explain the result with its "breakdown": name the 2-3 components with the largest "points" and who they favor, plus any notable edge for the other QB."""

NO_TOOL_RULES = """
- Use only numbers that appear in the context documents. If the context is not enough to answer, say what is missing."""

SYSTEM_PROMPT = BASE_RULES.format(tool_clause=" and tool results", tool_cite=" or [tool:compare_qbs]") + TOOL_RULES
SYSTEM_PROMPT_NO_TOOLS = BASE_RULES.format(tool_clause="", tool_cite="") + NO_TOOL_RULES


class ScoutAgent:
    def __init__(
        self,
        retriever=None,
        store: Optional[StatsStore] = None,
        client=None,
        model: str = DEFAULT_MODEL,
        context_k: int = 4,
        use_tools: bool = True,
    ):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self.client = client
        self.retriever = retriever
        self.store = store or StatsStore()
        self.model = model
        self.context_k = context_k
        self.use_tools = use_tools  # False = plain RAG baseline for evaluation

    def ask(self, question: str) -> dict[str, Any]:
        start = time.perf_counter()

        context_docs = self.retriever.retrieve(question, k=self.context_k) if self.retriever else []
        context = "\n\n".join(f"[{d['id']}]\n{d['text']}" for d in context_docs) or "(no documents retrieved)"
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"Context documents:\n{context}\n\nQuestion: {question}"}
        ]

        tools_used: list[dict[str, Any]] = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        system = SYSTEM_PROMPT if self.use_tools else SYSTEM_PROMPT_NO_TOOLS
        request = {"model": self.model, "max_tokens": 1024, "system": system}
        if self.use_tools:
            request["tools"] = TOOL_SCHEMAS

        for _ in range(MAX_TOOL_ROUNDS + 1):
            response = self.client.messages.create(messages=messages, **request)
            usage["input_tokens"] += response.usage.input_tokens
            usage["output_tokens"] += response.usage.output_tokens

            if response.stop_reason != "tool_use":
                break

            # Send Claude's blocks back unchanged (text, tool_use, and any thinking blocks,
            # which the API requires to be returned as-is during tool use).
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = execute_tool(block.name, block.input, self.store, self.retriever)
                tools_used.append({"name": block.name, "input": block.input, "error": result.get("error")})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": to_json(result)})
            messages.append({"role": "user", "content": results})
        # If the round limit is hit, the last response (a tool call) has no text and the fallback below applies

        answer = "".join(getattr(b, "text", "") for b in response.content if b.type == "text").strip()
        if not answer:
            answer = "Sorry, I couldn't produce an answer within the tool-call limit."

        return {
            "answer": answer,
            "sources": [d["id"] for d in context_docs],
            "tools_used": tools_used,
            "model": self.model,
            "latency_ms": round((time.perf_counter() - start) * 1000),
            "usage": usage,
        }


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python agent.py "your question"')
        sys.exit(1)
    from dotenv import load_dotenv

    from retriever import Retriever

    load_dotenv()
    agent = ScoutAgent(retriever=Retriever())
    result = agent.ask(" ".join(sys.argv[1:]))
    print(result["answer"])
    print(f"\nsources: {result['sources']}")
    print(f"tools: {[t['name'] for t in result['tools_used']]}")
    print(f"latency: {result['latency_ms']} ms, tokens: {result['usage']}")


if __name__ == "__main__":
    main()
