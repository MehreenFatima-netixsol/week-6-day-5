"""
logging_utils.py
-----------------
Structured, one-JSON-object-per-line logging -- the foundation for the monitoring
checklist in docs/monitoring_checklist.md.

Every chat turn logs: query, detected intent, tools called, latency, and token usage,
per the Task 3 requirement. Token usage provenance is explicit rather than silently
approximated:

  - If the Netixsol LLM was actually called during the turn (any rephrase/context-
    resolution call succeeded), the REAL prompt/completion/total token counts returned
    by the API are logged, with tokens_source="netixsol_api".
  - If the LLM was never called for that turn (unreachable, or the turn was handled
    entirely by deterministic templates -- e.g. a plain factual/off-topic answer needs
    no rephrase, or the LLM's output was rejected on a grounding check and the
    deterministic path was used instead), there is no real usage to report. In that case
    a rough proxy is logged instead (character count / 4, the standard order-of-magnitude
    heuristic for English text) with tokens_source="estimated_char_div4" so downstream
    readers of the log can tell the two cases apart and never mistake an estimate for a
    billed API figure.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "chat_turns.jsonl"
LOG_PATH.parent.mkdir(exist_ok=True)


def _approx_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def _token_usage_block(state: dict) -> dict:
    real = state.get("llm_token_usage")
    if real:
        return {
            "tokens_source": "netixsol_api",
            "prompt_tokens": real["prompt_tokens"],
            "completion_tokens": real["completion_tokens"],
            "total_tokens": real["total_tokens"],
            "llm_calls_this_turn": real["calls"],
            "llm_model": real.get("model"),
        }
    return {
        "tokens_source": "estimated_char_div4",
        "prompt_tokens": _approx_tokens(state.get("user_query")),
        "completion_tokens": _approx_tokens(state.get("final_response")),
        "total_tokens": _approx_tokens(state.get("user_query")) + _approx_tokens(state.get("final_response")),
        "llm_calls_this_turn": 0,
        "llm_model": None,
    }


def log_turn(conversation_id: str, state: dict) -> dict:
    record = {
        "ts": time.time(),
        "conversation_id": conversation_id,
        "query": state.get("user_query"),
        "intent": state.get("intent"),
        "guard_verdict": (state.get("guard") or {}).get("verdict"),
        "tools_called": state.get("tools_called", []),
        "validation_status": state.get("validation_status"),
        "tool_error": state.get("tool_error"),
        "latency_ms": state.get("latency_ms"),
        "needs_clarification": state.get("needs_clarification", False),
        **_token_usage_block(state),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def read_recent(n: int = 50) -> list:
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text().strip().splitlines()
    return [json.loads(l) for l in lines[-n:]]
