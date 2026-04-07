from __future__ import annotations

import json
from typing import AsyncIterator

import uvicorn
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.routing import Route

from provider import MultiAgentProvider


provider = MultiAgentProvider()


async def healthcheck(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


async def stream_chat(request: Request) -> StreamingResponse | JSONResponse:
    payload = await request.json()
    message = str(payload.get("message", "")).strip()
    conversation_id = payload.get("conversationId")

    if not message:
        return JSONResponse({"error": "Message is required."}, status_code=400)

    async def generate() -> AsyncIterator[str]:
        try:
            async for chunk in provider.stream_turn(message, conversation_id):
                yield json.dumps(chunk) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "error": str(exc)}) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def homepage(_: Request) -> PlainTextResponse:
    return PlainTextResponse("Multi-agent chat API is running.")


app = Starlette(
    debug=True,
    routes=[
        Route("/", homepage),
        Route("/health", healthcheck),
        Route("/api/chat/stream", stream_chat, methods=["POST"]),
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
