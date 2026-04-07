#!/usr/bin/env python3
from __future__ import annotations
import asyncio
import os
from uuid import uuid4
from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    SQLiteSession,
    Runner,
    TResponseInputItem,
    input_guardrail,
)
from dotenv import load_dotenv


INSTRUCTIONS = "You are a helpful assistant. Keep responses concise. do not use markdown formatting. Always answer in plain text."
SESSIONS_DB = "sessions.db"

from pydantic import BaseModel


class MathHomeworkOutput(BaseModel):
    is_math_homework: bool
    reasoning: str

guardrail_agent = Agent( 
    name="Guardrail check",
    instructions="Check if the user is asking you to do their math homework.",
    output_type=MathHomeworkOutput,
)


@input_guardrail
async def math_guardrail( 
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output, 
        tripwire_triggered=result.final_output.is_math_homework,
    )

async def main() -> None:
    load_dotenv()  # Load environment variables from .env file
    agent = Agent(
        name="Assistant",
        instructions=INSTRUCTIONS,
        model="gpt-4.1",
        input_guardrails=[math_guardrail]
    )
    session_id = f"minimal-{uuid4()}"
    session = SQLiteSession(session_id, SESSIONS_DB)

    print("Minimal Agent (type 'exit' to quit)\n")

    while True:
        user_message = input("You: ").strip()
        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            result = await Runner.run(agent, user_message, session=session)
            output = str(result.final_output or "").strip()
        except Exception as exc:
            print(f"Error: {exc}\n")
            continue

        print(f"\nAssistant: {output}\n")


if __name__ == "__main__":
    asyncio.run(main())
