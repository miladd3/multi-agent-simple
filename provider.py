from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, AsyncIterator
from uuid import uuid4

from agents import Agent, Runner, SQLiteSession, gen_trace_id, trace
from agents.items import HandoffCallItem, ToolCallItem
from agents.mcp import MCPServerStreamableHttp
from agents.stream_events import AgentUpdatedStreamEvent, RawResponsesStreamEvent, RunItemStreamEvent
from dotenv import load_dotenv


LIMIT_AGENT_INSTRUCTIONS = """You are a debit card limit management assistant.
You help users view and change their card limits (POS, ATM, E-commerce).

Steps:
1. Call `get_payment_instruments` to fetch the user's cards.
2. Show the user their cards with masked numbers and current limits.
3. If they want to change a limit, collect: transaction type (pos/atm/ecom), new amount, permanent or temporary.
4. If temporary, collect start_date and end_date (YYYY-MM-DD).
5. Confirm before executing. Then call `change_limit` or `create_temporary_limit`.
6. Show the updated limits.

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
"""


@dataclass(slots=True)
class ProviderConfig:
    model: str
    mcp_server_url: str


class MultiAgentProvider:
    def __init__(self) -> None:
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
        self.config = ProviderConfig(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
            mcp_server_url=os.getenv("MCP_SERVER_URL", "http://127.0.0.1:2009/mcp"),
        )
        self._sessions: dict[str, SQLiteSession] = {}

    def _get_session(self, conversation_id: str | None) -> tuple[str, SQLiteSession]:
        session_id = conversation_id or f"multi-agent-{uuid4()}"
        session = self._sessions.get(session_id)
        if session is None:
            session = SQLiteSession(session_id=session_id)
            self._sessions[session_id] = session
        return session_id, session

    def _build_supervisor(self, mcp_server: MCPServerStreamableHttp) -> Agent[Any]:
        limit_agent = Agent(
            name="Limit Agent",
            handoff_description="Handles viewing, changing, or managing card limits (POS, ATM, E-commerce). Use when the user wants to see their current limits or make changes.",
            instructions=LIMIT_AGENT_INSTRUCTIONS,
            model=self.config.model,
            mcp_servers=[mcp_server],
        )

        faq_agent = Agent(
            name="FAQ Agent",
            handoff_description="Shares interesting facts and answers questions about ABN AMRO — its history, services, Tikkie, sustainability, and more. Use when the user asks about the bank itself.",
            instructions=FAQ_AGENT_INSTRUCTIONS,
            model=self.config.model,
        )

        return Agent(
            name="Supervisor",
            instructions=SUPERVISOR_INSTRUCTIONS,
            model=self.config.model,
            handoffs=[limit_agent, faq_agent],
        )

    async def stream_turn(
        self, user_message: str, conversation_id: str | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        session_id, session = self._get_session(conversation_id)
        trace_id = gen_trace_id()

        async with MCPServerStreamableHttp(
            name="card_limit_manager",
            params={"url": self.config.mcp_server_url, "timeout": 15, "sse_read_timeout": 300},
            require_approval="never",
            cache_tools_list=True,
        ) as mcp_server:
            supervisor = self._build_supervisor(mcp_server)

            with trace(workflow_name="Multi-Agent Card Limits", trace_id=trace_id):
                stream_result = Runner.run_streamed(supervisor, user_message, session=session)
                yield {"type": "conversation", "conversationId": session_id}

                async for event in stream_result.stream_events():
                    if isinstance(event, AgentUpdatedStreamEvent):
                        yield {"type": "agent", "agentName": event.new_agent.name}
                        continue

                    if not isinstance(event, RunItemStreamEvent | RawResponsesStreamEvent):
                        continue

                    if isinstance(event, RawResponsesStreamEvent):
                        delta = self._extract_text_delta(event.data)
                        if delta:
                            yield {"type": "delta", "delta": delta}
                        continue

                    if event.name in {"handoff_requested", "handoff_occured"}:
                        detail = self._describe_handoff(event.item)
                        yield {"type": "handoff", "detail": detail}
                        continue

                    if event.name == "tool_called":
                        detail = self._describe_tool_call(event.item)
                        yield {"type": "tool", "toolName": detail, "detail": detail}

                output = str(stream_result.final_output or "").strip()
                last_agent = stream_result.last_agent.name if stream_result.last_agent else "Supervisor"
                yield {
                    "type": "done",
                    "conversationId": session_id,
                    "output": output,
                    "agentName": last_agent,
                    "traceId": trace_id,
                }

    @staticmethod
    def _extract_text_delta(data: Any) -> str | None:
        event_type = getattr(data, "type", None)
        if event_type == "response.output_text.delta":
            return getattr(data, "delta", None)
        return None

    @staticmethod
    def _describe_handoff(item: Any) -> str:
        if isinstance(item, HandoffCallItem):
            target = getattr(item.raw_item, "name", None) or "specialist"
            return f"Supervisor initiated a handoff to {target}."
        return "A specialist handoff was requested."

    @staticmethod
    def _describe_tool_call(item: Any) -> str:
        if isinstance(item, ToolCallItem):
          name = getattr(item.raw_item, "name", None) or item.title or item.description or "tool"
          return f"{name} was called."
        return "A tool call was executed."
