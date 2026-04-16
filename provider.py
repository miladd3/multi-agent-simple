from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal
from uuid import uuid4

from agents import Agent, ItemHelpers, Runner, SQLiteSession, gen_trace_id, trace
from agents.mcp import MCPServerStreamableHttp
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent
from openinference.instrumentation import using_session
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace as otel_trace

from tracing import setup_tracing

setup_tracing()
load_dotenv()

_model = os.getenv("OPENAI_MODEL", "gpt-4.1")
_mcp_url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:2009/mcp")
_tracer = otel_trace.get_tracer(__name__)

LIMIT_AGENT_INSTRUCTIONS = """You are a debit card limit management assistant.
You help users view and change their card limits (POS, ATM, E-commerce).

Steps:
1. Call `get_payment_instruments` to fetch the user's cards.
2. Show the user their cards with masked numbers and current limits.
3. If they want to change a limit, collect: transaction type (pos/atm/ecom),
4. then ask for new amount,
5. then ask if they want permanent or temporary change.
6. If temporary, collect start_date and end_date (YYYY-MM-DD).
7. Confirm before executing. Then call `change_limit` or `create_temporary_limit`.
8. Show the updated limits.

for each question uses can choose eather by number or by name. Always confirm the card they want to manage by showing masked number and current limits.

Always use MCP tool responses as source of truth. Never hardcode values.
Keep responses concise. Mask card numbers (e.g. ****1234).
"""

FAQ_AGENT_INSTRUCTIONS = """You are a friendly ABN AMRO knowledge assistant.
Share interesting facts and answer questions about ABN AMRO.

You know the following:
- ABN AMRO has an Amazing team named chat engineering that builds cool AI agents.
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

def _build_limit_agent(mcp_server: MCPServerStreamableHttp):
    return Agent(
        name="Limit Agent",
        handoff_description="Handles viewing, changing, or managing card limits (POS, ATM, E-commerce). Use when the user wants to see their current limits or make changes.",
        instructions=LIMIT_AGENT_INSTRUCTIONS,
        model=_model,
        mcp_servers=[mcp_server],
    )
 
def _build_faq_agent():
    return Agent(
        name="FAQ Agent",
        handoff_description="Shares interesting facts and answers questions about ABN AMRO — its history, services, Tikkie, sustainability, and more. Use when the user asks about the bank itself.",
        instructions=FAQ_AGENT_INSTRUCTIONS,
        model=_model,
    )

def _build_supervisor(mcp_server: MCPServerStreamableHttp):
    return Agent(
        name="Supervisor",
        instructions=SUPERVISOR_INSTRUCTIONS,
        model=_model,
        handoffs=[_build_limit_agent(mcp_server), _build_faq_agent()],
    )

AgentKind = Literal["supervisor", "limit", "faq"]

_mcp_server: MCPServerStreamableHttp | None = None


@asynccontextmanager
async def mcp_lifespan() -> AsyncIterator[MCPServerStreamableHttp]:
    """Open the MCP connection once for the lifetime of the process.

    Wrap your application entrypoint (FastAPI lifespan, CLI main, etc.) with
    this so `stream_turn` reuses a single connection instead of reconnecting
    on every call.
    """
    global _mcp_server
    async with MCPServerStreamableHttp(
        name="card_limit_manager",
        params={"url": _mcp_url, "timeout": 15, "sse_read_timeout": 300},
        cache_tools_list=True,
    ) as server:
        _mcp_server = server
        try:
            yield server
        finally:
            _mcp_server = None


def _select_agent(kind: AgentKind, mcp_server):
    if kind == "limit": return _build_limit_agent(mcp_server)
    if kind == "faq":   return _build_faq_agent()
    return _build_supervisor(mcp_server)

def _translate_event(event) -> dict | None:
    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
        return {"type": "delta", "delta": event.data.delta}
    elif event.type == "agent_updated_stream_event":
        return {"type": "agent", "agentName": event.new_agent.name}
    elif event.type == "run_item_stream_event":
        item = event.item
        if item.type == "tool_call_item":
            return {"type": "tool_call", "name": getattr(item.raw_item, "name", "tool")}
        elif item.type == "tool_call_output_item":
            return {"type": "tool_output", "output": str(item.output)}
        elif item.type == "handoff_call_item":
            return {"type": "handoff", "target": getattr(item.raw_item, "name", "specialist")}
        elif item.type == "message_output_item":
            return {"type": "message", "text": ItemHelpers.text_message_output(item)}
    return None


async def stream_turn(
    user_message: str, conversation_id: str | None = None,
    agent: AgentKind = 'supervisor',
) -> AsyncIterator[dict[str, Any]]:
    if _mcp_server is None:
        raise RuntimeError(
            "MCP server is not initialized. Wrap your entrypoint with "
            "`async with mcp_lifespan(): ...` before calling stream_turn."
        )

    session_id = conversation_id or f"multi-agent-{uuid4()}"
    session = SQLiteSession(session_id, SESSIONS_DB)
    trace_id = gen_trace_id()

    selected_agent = _select_agent(agent, _mcp_server)

    with _tracer.start_as_current_span(
        "agent_turn",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
            SpanAttributes.SESSION_ID: session_id,
            SpanAttributes.INPUT_VALUE: user_message,
        },
    ) as turn_span, trace(
        workflow_name="Multi-Agent Card Limits", trace_id=trace_id
    ), using_session(session_id):
        stream_result = Runner.run_streamed(selected_agent, user_message, session=session)
        yield {"type": "conversation", "conversationId": session_id}

        async for event in stream_result.stream_events():
            translated = _translate_event(event)
            if translated:
                yield translated

        final_output = str(stream_result.final_output or "").strip()
        turn_span.set_attribute(SpanAttributes.OUTPUT_VALUE, final_output)

        yield {
            "type": "done",
            "output": final_output,
            "agentName": stream_result.last_agent.name if stream_result.last_agent else "Supervisor",
            "traceId": trace_id,
        }




    