"""Local HTTP API for the BankGuard AI browser demo."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .compliance import extract_final_text
from .config import load_settings
from .orchestrator import BankGuardRuntime, build_bankguard_runtime

LOGGER = logging.getLogger(__name__)

SERVER_MAX_OUTPUT_TOKENS = 300
MAX_HISTORY_MESSAGES = 4


class HistoryMessage(BaseModel):
    """One compact conversation message accepted from the frontend."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4_000)

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, value: object) -> object:
        if isinstance(value, str):
            return " ".join(value.split())
        return value


class ChatRequest(BaseModel):
    """Validated token-efficient request produced by ``index.html``."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=500)
    history: list[HistoryMessage] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_MESSAGES,
    )
    max_output_tokens: int = Field(
        default=SERVER_MAX_OUTPUT_TOKENS,
        ge=1,
        le=SERVER_MAX_OUTPUT_TOKENS,
    )

    @field_validator("message", mode="before")
    @classmethod
    def normalize_message(cls, value: object) -> object:
        if isinstance(value, str):
            return " ".join(value.split())
        return value


class ChatResponse(BaseModel):
    """Stable response contract consumed by the single-file frontend."""

    response: str
    sources: list[str] = Field(default_factory=list)
    sql: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "BankGuard AI API"
    model: str
    max_output_tokens: int = SERVER_MAX_OUTPUT_TOKENS


app = FastAPI(
    title="BankGuard AI API",
    version="1.0.0",
    description="Adaptador HTTP local para el agente BankGuard AI.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8765",
        "http://localhost:8765",
        "null",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=600,
)

# A local model should process one browser request at a time. The frontend also
# disables its send button, while this server-side guard covers other clients.
_request_slot = asyncio.Semaphore(1)


def _source_labels(runtime: BankGuardRuntime) -> list[str]:
    """Expose only human-readable citations, never raw retrieved documents."""
    compliance = runtime.delegation_state.last_compliance
    if compliance is None:
        return []

    labels: list[str] = []
    for evidence in compliance.evidence:
        citation = evidence.citation.strip()
        title = (evidence.title or "").strip()
        label = (
            f"{title} — {citation}"
            if title and title.casefold() not in citation.casefold()
            else citation
        )
        if label and label not in labels:
            labels.append(label)
    return labels


def _run_chat_sync(request: ChatRequest) -> ChatResponse:
    """Build an isolated agent runtime and execute one top-level turn."""
    settings = replace(
        load_settings(),
        ollama_num_predict=request.max_output_tokens,
    )
    runtime = build_bankguard_runtime(settings)
    messages = [
        *(item.model_dump() for item in request.history),
        {"role": "user", "content": request.message},
    ]
    result = runtime.banking_orchestrator.invoke({"messages": messages})
    answer = extract_final_text(result).strip()
    if not answer:
        raise RuntimeError("The agent returned no final text")

    return ChatResponse(
        response=answer,
        sources=_source_labels(runtime),
        sql=None,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return readiness metadata without invoking the LLM."""
    settings = load_settings()
    return HealthResponse(model=settings.chat_model)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Answer one question through the real local BankGuard orchestrator."""
    try:
        async with _request_slot:
            return await asyncio.to_thread(_run_chat_sync, request)
    except Exception as exc:
        LOGGER.exception("BankGuard chat request failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail=(
                "No fue posible completar la consulta local. Verifica Ollama, "
                "el modelo configurado y la conexión de Pinecone."
            ),
        ) from exc
