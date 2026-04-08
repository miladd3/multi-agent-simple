from __future__ import annotations

import os
from typing import Any, AsyncIterator
from uuid import uuid4

from agents import Agent, ItemHelpers, Runner, SQLiteSession, gen_trace_id, trace
from agents.mcp import MCPServerStreamableHttp
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent

from tracing import setup_tracing

setup_tracing()
load_dotenv()

_model = os.getenv("OPENAI_MODEL", "gpt-4.1")
_mcp_url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:2009/mcp")

LIMIT_AGENT_INSTRUCTIONS = """You are a debit card limit management assistant.
You help users view and change their card limits (POS, ATM, E-commerce).

Steps:
1. Call `get_payment_instruments` to fetch the user's cards.
2. Show the user their cards with masked numbers and current limits.
3. If they want to change a limit, collect: transaction type (pos/atm/ecom),
4. then ask for new amount,
5. then ask if they want permanent or temporary change.
4. If temporary, collect start_date and end_date (YYYY-MM-DD).
5. Confirm before executing. Then call `change_limit` or `create_temporary_limit`.
6. Show the updated limits.

for each question uses can choose eather by number or by name. Always confirm the card they want to manage by showing masked number and current limits.

Always use MCP tool responses as source of truth. Never hardcode values.
Keep responses concise. Mask card numbers (e.g. ****1234).
"""

FAQ_AGENT_INSTRUCTIONS = """You are a friendly ABN AMRO knowledge assistant.
Share interesting facts and answer questions about ABN AMRO.

You know the following:
- ABN AMRO has a Amazing team named chat engineering that builds cool AI agents.
- there is a lot of cake being eaten in the office.
- coffee quality is absloutly attrocious, but the team spirit is great.
- Salsa shop is the only place they go for lunch.
- dad jokes are the only jokes allowed in the office.

If the user asks about managing their card limits, tell them you can hand them over to the Limit Agent for that.
Keep responses engaging and conversational.
"""

SUPERVISOR_INSTRUCTIONS = """You are a supervisor agent that routes user requests to the right specialist.

You have two specialist agents available via handoff:
- **Limit Agent**: Handles viewing card limits, changing limits (permanent or temporary), and any card limit operations. Hand off when the user wants to see, change, or manage their card limits.
- **FAQ Agent**: Shares cool facts and answers questions about ABN AMRO (history, services, Tikkie, sustainability, etc.). Hand off when the user asks about the bank itself.

Rules:
- Greet the user briefly and ask how you can help if their intent is unclear.
- If the request is clearly about managing limits (view, change, update), hand off to the Limit Agent.
- If the request is about ABN AMRO (history, facts, services, Tikkie, etc.), hand off to the FAQ Agent.
- Never try to answer domain questions yourself — always delegate.
- If the user says goodbye, respond with a friendly farewell.
- no markdown you can use plain text formatting to make the response more engaging, but keep it concise and to the point.
"""

SESSIONS_DB = "sessions.db"


def _build_supervisor(mcp_server: MCPServerStreamableHttp) -> Agent[Any]:
    limit_agent = Agent(
        name="Limit Agent",
        handoff_description="Handles viewing, changing, or managing card limits (POS, ATM, E-commerce). Use when the user wants to see their current limits or make changes.",
        instructions=LIMIT_AGENT_INSTRUCTIONS,
        model=_model,
        mcp_servers=[mcp_server],
    )
    faq_agent = Agent(
        name="FAQ Agent",
        handoff_description="Shares interesting facts and answers questions about ABN AMRO — its history, services, Tikkie, sustainability, and more. Use when the user asks about the bank itself.",
        instructions=FAQ_AGENT_INSTRUCTIONS,
        model=_model,
    )
    return Agent(
        name="Supervisor",
        instructions=SUPERVISOR_INSTRUCTIONS,
        model=_model,
        handoffs=[limit_agent, faq_agent],
    )


async def stream_turn(
    user_message: str, conversation_id: str | None = None
) -> AsyncIterator[dict[str, Any]]:
    session_id = conversation_id or f"multi-agent-{uuid4()}"
    session = SQLiteSession(session_id, SESSIONS_DB)
    trace_id = gen_trace_id()

    async with MCPServerStreamableHttp(
        name="card_limit_manager",
        params={"url": _mcp_url, "timeout": 15, "sse_read_timeout": 300},
        cache_tools_list=True,
    ) as mcp_server:
        supervisor = _build_supervisor(mcp_server)

        with trace(workflow_name="Multi-Agent Card Limits", trace_id=trace_id):
            stream_result = Runner.run_streamed(supervisor, user_message, session=session)
            yield {"type": "conversation", "conversationId": session_id}

            async for event in stream_result.stream_events():
                if event.type == "raw_response_event":
                    if isinstance(event.data, ResponseTextDeltaEvent):
                        yield {"type": "delta", "delta": event.data.delta}
                elif event.type == "agent_updated_stream_event":
                    yield {"type": "agent", "agentName": event.new_agent.name}
                elif event.type == "run_item_stream_event":
                    item = event.item
                    if item.type == "tool_call_item":
                        yield {"type": "tool_call", "name": getattr(item.raw_item, "name", "tool")}
                    elif item.type == "tool_call_output_item":
                        yield {"type": "tool_output", "output": str(item.output)}
                    elif item.type == "handoff_call_item":
                        yield {"type": "handoff", "target": getattr(item.raw_item, "name", "specialist")}
                    elif item.type == "message_output_item":
                        yield {"type": "message", "text": ItemHelpers.text_message_output(item)}

            yield {
                "type": "done",
                "output": str(stream_result.final_output or "").strip(),
                "agentName": stream_result.last_agent.name if stream_result.last_agent else "Supervisor",
                "traceId": trace_id,
            }
