"""Phoenix tracing setup for the multi-agent provider.

Importing this module registers an OpenTelemetry tracer that ships traces to
a local Phoenix server (see ../pheonix/server.py) and instruments the OpenAI
Agents SDK so every agent run, handoff, tool call and LLM call shows up.

Configurable via env vars:
    PHOENIX_COLLECTOR_ENDPOINT  default http://localhost:6006
    PHOENIX_PROJECT_NAME        default multi-agent-simple
"""
from __future__ import annotations

import os

_INITIALIZED = False


def setup_tracing() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    project_name = os.getenv("PHOENIX_PROJECT_NAME", "multi-agent-simple")

    try:
        from phoenix.otel import register
        from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
    except ImportError as exc:  # pragma: no cover - optional dep
        print(
            "[tracing] Phoenix tracing disabled — missing package: "
            f"{exc.name}. Install with `uv add arize-phoenix-otel "
            "openinference-instrumentation-openai-agents`."
        )
        return

    tracer_provider = register(
        project_name=project_name,
        endpoint=os.environ["PHOENIX_COLLECTOR_ENDPOINT"] + "/v1/traces",
        auto_instrument=False,
        verbose=False,
    )
    OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
    _INITIALIZED = True
    print(
        f"[tracing] Phoenix tracing enabled -> "
        f"{os.environ['PHOENIX_COLLECTOR_ENDPOINT']} (project: {project_name})"
    )
