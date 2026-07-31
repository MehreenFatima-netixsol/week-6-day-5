"""
api.py
------
FastAPI wrapper around the LangGraph AFL assistant.

Run:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /chat            {message, conversation_id}  -> response + prediction metadata
    GET  /health           liveness/readiness probe
    GET  /conversations/{conversation_id}   full turn history (debug/demo aid)
"""
from __future__ import annotations
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.graph import run_turn
from app.logging_utils import log_turn
from app import llm_client as _llm
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AFL Analyst API", version="1.0.0",
              description="Domain-scoped AFL chat + prediction assistant (LangGraph-backed).")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_llm_status = "checking..."


@app.on_event("startup")
def _startup_llm_probe():
    global _llm_status
    reachable = _llm.is_available()
    _llm_status = f"Netixsol endpoint {'REACHABLE (' + str(_llm._active_model) + ')' if reachable else 'unreachable -- using deterministic templates'}"
    print(f"[api.py] LLM backend: {_llm_status}")

# In-memory conversation store: conversation_id -> last AFLState.
# Production note: swap for Redis/DB so state survives process restarts and scales
# across workers; the state dict is already JSON-serialisable.
_CONVERSATIONS: Dict[str, dict] = {}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    intent: str
    needs_clarification: bool
    tools_called: list
    latency_ms: float
    prediction_metadata: Optional[Dict[str, Any]] = None


@app.get("/health")
def health():
    return {"status": "ok", "ts": time.time(), "llm_backend": _llm_status}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    conv_id = req.conversation_id or str(uuid.uuid4())
    prior_state = _CONVERSATIONS.get(conv_id)

    try:
        result = run_turn(req.message, prior_state=prior_state)
    except Exception as e:  # hardened outer boundary -- never let the API 500 silently
        raise HTTPException(status_code=500, detail=f"Internal error: {type(e).__name__}") from e

    _CONVERSATIONS[conv_id] = result
    log_turn(conv_id, result)

    prediction_metadata = None
    if result.get("intent") == "prediction" and result.get("tool_result", {}).get("model"):
        tr = result["tool_result"]
        prediction_metadata = {k: v for k, v in tr.items() if k not in ("feature_row",)}

    return ChatResponse(
        conversation_id=conv_id,
        response=result["final_response"],
        intent=result.get("intent", "unknown"),
        needs_clarification=result.get("needs_clarification", False),
        tools_called=result.get("tools_called", []),
        latency_ms=result.get("latency_ms", 0.0),
        prediction_metadata=prediction_metadata,
    )


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    state = _CONVERSATIONS.get(conversation_id)
    if not state:
        raise HTTPException(status_code=404, detail="Unknown conversation_id")
    return {"conversation_id": conversation_id, "history": state.get("conversation_history", [])}
