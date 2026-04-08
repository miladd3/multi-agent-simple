#!/usr/bin/env python3
import asyncio
from provider import stream_turn


async def main() -> None:
    conversation_id: str | None = None

    print("Multi-Agent Card Limit System")
    print("Agents: Supervisor -> [Limit Agent, FAQ Agent]")
    print("Type 'exit' to quit.\n")

    while True:
        user_message = input("You: ").strip()
        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        current_agent = "Supervisor"
        final_output = ""
        trace_id = ""

        try:
            async for chunk in stream_turn(user_message, conversation_id):
                if chunk["type"] == "conversation":
                    conversation_id = chunk["conversationId"]
                elif chunk["type"] == "agent":
                    current_agent = chunk["agentName"]
                elif chunk["type"] == "delta":
                    print(chunk["delta"], end="", flush=True)
                elif chunk["type"] == "done":
                    current_agent = chunk["agentName"]
                    final_output = chunk["output"]
                    trace_id = chunk.get("traceId", "")
                elif chunk["type"] == "error":
                    raise RuntimeError(chunk["error"])
        except Exception as exc:
            print(f"Error: {exc}\n")
            continue

        if trace_id:
            print(f"\nTrace: https://platform.openai.com/traces/trace?trace_id={trace_id}")

        print(f"\n[{current_agent}]: {final_output}\n")


if __name__ == "__main__":
    asyncio.run(main())
