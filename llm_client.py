"""
llm_client.py
-------------
Thin wrapper around the Netixsol gateway (the same OpenAI-compatible endpoint used since
Week 5 / wired up in the Week 6 Day 2 notebook's `build_llm()`), used ONLY for natural-
language phrasing -- never for generating facts or numbers.

Design constraint (non-negotiable, per the AFL system prompt's grounding rule): the LLM
is called AFTER a deterministic tool result / template response already exists. It may
only rephrase that content. Every call site in graph.py verifies the LLM's output against
the same numeric-grounding check used elsewhere (guardrails.verify_grounding) and against
a few required-substring checks (e.g. the prediction disclaimer); if the LLM output fails
either check, or the call fails/times out/is unavailable, callers silently fall back to
the deterministic template. This means: with the LLM connected, replies are more natural
and handle conversational context; with it unreachable, the assistant is functionally
identical to the fully-deterministic build from before this endpoint was wired in.

Environment (matches the Day 2 notebook's Netixsol configuration):
    LLM_BASE_URL   default: https://llm.netixsol.com/v1
    LLM_API_KEY    default: the shared course/dev key used in the Day 2 notebook
    LLM_MODEL      default: "smart" (falls back to "coder" on failure)
"""
from __future__ import annotations
import os
import time
import logging

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

os.environ.setdefault("LLM_BASE_URL", "https://llm.netixsol.com/v1")
os.environ.setdefault("LLM_API_KEY", "sk-8y_p29OL8D-BldISQ4zJhQ")
os.environ.setdefault("LLM_MODEL", "smart")

FALLBACK_MODELS = ["smart", "coder"]
CALL_TIMEOUT_SECONDS = 6.0

try:
    from openai import OpenAI
    _HAVE_OPENAI = True
except Exception:
    _HAVE_OPENAI = False

_client = None
_active_model = None
_probed = False
_available = False


def _get_client():
    global _client
    if _client is None and _HAVE_OPENAI:
        _client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"],
                          timeout=CALL_TIMEOUT_SECONDS)
    return _client


def probe(force=False) -> bool:
    """Cheap one-time reachability probe, tried against each fallback model in turn.
    Cached so normal chat turns never pay a probe round-trip. Never raises."""
    global _probed, _available, _active_model
    if _probed and not force:
        return _available
    _probed = True
    if not _HAVE_OPENAI:
        _available = False
        return False
    client = _get_client()
    for model in [os.environ.get("LLM_MODEL", "smart")] + FALLBACK_MODELS:
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                max_tokens=5, temperature=0,
            )
            _active_model = model
            _available = True
            return True
        except Exception:
            continue
    _available = False
    return False


def is_available() -> bool:
    return probe()


_last_usage = None  # set after every successful completion; None if no call has succeeded yet


def get_last_usage() -> dict | None:
    """Real token counts (prompt/completion/total) from the most recent successful
    Netixsol call in this process, or None if the LLM hasn't been called successfully
    yet. Consumed by graph.py to attach real usage to a turn's log record."""
    return _last_usage


def rephrase(system_prompt: str, user_prompt: str, temperature: float = 0.3,
             max_tokens: int = 220) -> str | None:
    """Returns the model's text, or None on any failure (unreachable, timeout, quota,
    empty response). Callers must treat None as 'use the deterministic fallback' and
    must independently re-verify anything numeric in a successful response."""
    global _last_usage
    if not probe():
        return None
    client = _get_client()
    model = _active_model or os.environ.get("LLM_MODEL", "smart")
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=temperature, max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        if getattr(resp, "usage", None) is not None:
            _last_usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
                "model": model,
            }
        return text or None
    except Exception:
        return None
    finally:
        if time.time() - t0 > CALL_TIMEOUT_SECONDS:
            pass  # client-level timeout already bounds this; nothing further to do
