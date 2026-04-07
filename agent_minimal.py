#!/usr/bin/env python3
from __future__ import annotations
import asyncio
import os
from uuid import uuid4
from agents import Agent, Runner, SQLiteSession
from dotenv import load_dotenv


INSTRUCTIONS = "You are a helpful assistant. Keep responses concise. do not use markdown formatting. Always answer in plain text."
SESSIONS_DB = "sessions.db"


async def main() -> None:
    load_dotenv()  # Load environment variables from .env file
    agent = Agent(
        name="Assistant",
        instructions=INSTRUCTIONS,
        model="gpt-4.1",
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
