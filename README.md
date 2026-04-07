# multi-agent-simple

A streaming multi-agent demo built with the OpenAI Agents SDK and FastAPI.

A **Supervisor** agent routes requests to one of two specialists:

- **Limit Agent** — views and changes debit card limits via the `limit-mcp` MCP server.
- **FAQ Agent** — answers fun questions about ABN AMRO.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- A running [limit-mcp](../limit-mcp) server (default: `http://127.0.0.1:2009/mcp`)
- `OPENAI_API_KEY` in a `.env` file

## Setup

```bash
uv sync
```

Create a `.env`:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1               # optional
MCP_SERVER_URL=http://127.0.0.1:2009/mcp   # optional
```

## Run

CLI chat:

```bash
uv run python agent.py
```

HTTP streaming API (FastAPI on `127.0.0.1:8000`):

```bash
uv run python server.py
```

Then POST to `/api/chat/stream`:

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "show my card limits"}'
```

Responses are NDJSON with `conversation`, `agent`, `delta`, `handoff`, `tool`, and `done` events.
