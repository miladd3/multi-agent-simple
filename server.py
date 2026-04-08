from __future__ import annotations

import json
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from provider import stream_turn


class StreamChatRequest(BaseModel):
    message: str
    conversationId: str | None = None


app = FastAPI(title="Multi-Agent Chat API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def healthcheck() -> dict[str, bool]:
    return {"ok": True}


@app.get("/", response_class=PlainTextResponse)
async def homepage() -> str:
    return "Multi-agent chat API is running."


@app.post("/api/chat/stream")
async def stream_chat(payload: StreamChatRequest) -> StreamingResponse:
    message = payload.message.strip()
    conversation_id = payload.conversationId

    if not message:
        raise HTTPException(status_code=400, detail="Message is required.")

    async def generate() -> AsyncIterator[str]:
        try:
            async for chunk in stream_turn(message, conversation_id):
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


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
