"""Smallest possible LLM-as-a-judge for the multi-agent provider.

Runs a handful of hardcoded user prompts through `MultiAgentProvider`, then
asks an OpenAI model whether each answer is correct and whether it was
routed to the right specialist. Traces still flow to Phoenix automatically
because provider.py imports tracing.py.

Run:
    uv run python eval.py
"""
from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from provider import MultiAgentProvider

load_dotenv()

CASES = [
    ("Show me my current card limits.", "Limit Agent"),
    ("Change my POS limit to 2500 euro permanently.", "Limit Agent"),
    ("Tell me a fun fact about ABN AMRO.", "FAQ Agent"),
    ("Where does the chat engineering team go for lunch?", "FAQ Agent"),
]

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")
JUDGE_PROMPT = """You are auditing a bank assistant with two specialists:
Limit Agent (card limit ops) and FAQ Agent (ABN AMRO facts).

User: {user}
Expected specialist: {expected}
Actual specialist: {actual}
Assistant answer: {answer}

Reply with JSON: {{"correct": true|false, "routing_ok": true|false, "why": "..."}}.
"""


async def run_one(provider: MultiAgentProvider, user: str) -> tuple[str, str]:
    answer, agent = "", "Supervisor"
    async for ev in provider.stream_turn(user):
        if ev["type"] == "delta":
            answer += ev["delta"]
        elif ev["type"] == "agent":
            agent = ev["agentName"]
        elif ev["type"] == "done" and ev.get("output"):
            answer = ev["output"]
            agent = ev.get("agentName", agent)
    return answer.strip(), agent


def judge(client: OpenAI, user: str, expected: str, actual: str, answer: str) -> dict:
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            user=user, expected=expected, actual=actual, answer=answer
        )}],
    )
    return json.loads(resp.choices[0].message.content)


async def main() -> None:
    provider = MultiAgentProvider()
    judge_client = OpenAI()

    correct = routed = 0
    for user, expected in CASES:
        answer, actual = await run_one(provider, user)
        verdict = judge(judge_client, user, expected, actual, answer)
        correct += int(verdict["correct"])
        routed += int(verdict["routing_ok"])
        mark = "OK" if verdict["correct"] and verdict["routing_ok"] else "FAIL"
        print(f"[{mark}] {user}\n    -> {actual}: {answer[:120]}\n    judge: {verdict['why']}\n")

    n = len(CASES)
    print(f"correctness: {correct}/{n}    routing: {routed}/{n}")


if __name__ == "__main__":
    asyncio.run(main())
