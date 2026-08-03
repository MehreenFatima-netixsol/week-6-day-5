"""
guardrails.py
-------------
Hardened v2 scope guard (from the Week 6 Day 2 evaluation cycle), jailbreak-pattern
detection, and a numeric grounding verifier that confirms every number in a response
traces back to a real tool call.
"""
from __future__ import annotations
import re
from .tools import KNOWN_TEAMS, TEAM_ALIASES

AFL_TERMS = {
    "afl", "vfl", "footy", "aussie rules", "australian rules", "australian football",
    "brownlow", "coleman", "norm smith", "grand final", "preliminary final", "elimination final",
    "qualifying final", "semi final", "finals", "ladder", "premiership", "flag", "minor premiership",
    "disposal", "disposals", "handball", "handballs", "kick", "kicks", "mark", "marks",
    "tackle", "tackles", "goal", "goals", "behind", "behinds", "ruck", "ruckman", "midfielder",
    "forward", "defender", "interchange", "centre bounce", "boundary throw", "50 metre penalty",
    "holding the ball", "deliberate out of bounds", "mcg", "marvel stadium", "adelaide oval",
    "optus stadium", "gabba", "scg", "kardinia", "gmhba", "round", "season", "margin",
    "supercoach", "sherrin", "guernsey", "banner", "siren", "quarter", "clearance", "inside 50",
}
AFL_TERMS |= {t.lower() for t in KNOWN_TEAMS}
AFL_TERMS |= {w.lower() for t in KNOWN_TEAMS for w in t.split() if len(w) > 4}
AFL_TERMS |= set(TEAM_ALIASES.keys())

OFF_TOPIC_TERMS = {
    "premier league", "la liga", "champions league", "world cup", "fifa", "messi", "ronaldo", "soccer",
    "nba", "nfl", "mlb", "nhl", "super bowl", "cricket", "test match", "ipl", "ashes", "wimbledon", "tennis",
    "formula 1", "f1", "nascar", "golf", "pga", "rugby league", "nrl", "state of origin", "ufc", "boxing",
    "olympics", "basketball", "baseball", "hockey",
    "recipe", "cook", "bake", "weather", "stock market", "crypto", "bitcoin", "python", "javascript",
    "code", "script", "program", "debug", "homework", "essay", "poem", "joke", "medical", "diagnos",
    "symptom", "lawyer", "legal advice", "invest", "election", "politic", "president", "prime minister",
    "movie", "netflix", "song", "lyrics", "mortgage", "insurance", "tax return", "landlord",
}

JAILBREAK_PATTERNS = [
    r"\bignore\s+(all\s+)?(your\s+|the\s+|previous\s+|prior\s+|above\s+)*(instruction|rule|prompt|direction)",
    r"\bdisregard\s+(all\s+|your\s+|the\s+|previous\s+|prior\s+)*(instruction|rule|prompt)",
    r"\bforget\s+(your|the|all|everything|previous|prior)\b",
    r"\byou\s+are\s+now\b", r"\byou'?re\s+now\b",
    r"\bpretend\s+(you'?re|you\s+are|to\s+be)\b",
    r"\bact\s+as\s+(a|an|if)\b", r"\brole\s*play\s+as\b",
    r"\bno\s+(restrictions|limits|rules|guardrails|filters)\b",
    r"\b(system|initial|original)\s+prompt\b",
    r"\b(system\s+messages?)\b",
    r"\b(print|repeat|reveal|show|output|recite|expose|leak|dump)\s+(me\s+)?(your|the)\s+"
    r"(prompt|instructions|rules|configuration|system\s+message|internal\s+code|source\s+code|"
    r"code|api|apis|dataset|datasets|langgraph|workflow|implementation)",
    r"\breveal\s+(your\s+|the\s+)?(hidden\s+prompts?|internal\s+code|apis?|datasets?|"
    r"langgraph\s+workflow|system\s+messages?|implementation\s+details?)",
    r"\blanggraph\s+workflow\b",
    r"\brepeat\s+the\s+instructions\b",
    r"\bword\s+for\s+word\b",
    r"\bdeveloper\s+mode\b", r"\bjailbreak\b", r"\bDAN\b",
    r"\bstop\s+being\b", r"\bdrop\s+(the\s+)?(act|character|persona)\b",
    r"\bnot\s+an?\s+afl\s+(bot|assistant|model)\b",
]
_JB_RE = [re.compile(p, re.I) for p in JAILBREAK_PATTERNS]

SPORT_GENERIC_TERMS = {"sport", "sports", "athlete", "athletes", "footballer", "footballers",
                        "football", "greatest of all time", "goat", "league", "grand final"}
ADJACENT_TERMS = {"aflw", "women's football", "sanfl", "wafl", "vfl women", "junior football",
                   "bet", "betting", "punt", "tips", "tipping", "wager", "bookie",
                   "multi", "parlay"}
WEAK_AFL_TERMS = {"round", "season", "margin", "quarter", "goal", "goals", "kick", "kicks",
                   "mark", "marks", "tackle", "tackles", "behind", "behinds", "forward",
                   "defender", "midfielder", "final", "finals", "ladder", "siren", "banner",
                   "interchange"}
ANAPHORA_RE = re.compile(
    r"\b(he|him|his|she|her|they|them|their|it|that|those|these|there|then)\b|"
    r"^(what about|how about|and |what if|compare|the round before|the game before)", re.I)


def has_afl_signal(text: str):
    low = " " + re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower()) + " "
    for t in AFL_TERMS - WEAK_AFL_TERMS:
        if f" {t} " in low:
            return True, t, "strong"
    for t in WEAK_AFL_TERMS:
        if f" {t} " in low:
            return True, t, "weak"
    return False, None, None


def scope_guard(text: str, context: str | None = None) -> dict:
    """Layered guard: jailbreak override check -> off-topic/AFL signal weighing -> ambiguity band."""
    raw = text or ""
    low = " " + re.sub(r"[^a-z0-9' ]+", " ", raw.lower()) + " "

    for rx in _JB_RE:
        if rx.search(raw):
            return {"verdict": "out_of_scope", "kind": "jailbreak",
                    "reason": f"instruction-override pattern matched: {rx.pattern[:40]}", "hits": []}

    off = [t for t in OFF_TOPIC_TERMS if f" {t} " in low]
    afl_hit, afl_term, strength = has_afl_signal(raw)
    adjacent = [t for t in ADJACENT_TERMS if f" {t} " in low]
    generic = [t for t in SPORT_GENERIC_TERMS if f" {t} " in low]

    if afl_hit and off and strength == "strong":
        return {"verdict": "ambiguous", "kind": "mixed",
                "reason": f"strong AFL signal ('{afl_term}') alongside off-topic term(s) {off[:2]}", "hits": off}
    if adjacent:
        return {"verdict": "ambiguous", "kind": "adjacent",
                "reason": f"AFL-adjacent topic: {adjacent[:2]}", "hits": adjacent}
    if off:
        return {"verdict": "out_of_scope", "kind": "off_topic",
                "reason": f"off-topic term(s) {off[:3]} with no AFL signal", "hits": off}
    if afl_hit:
        return {"verdict": "in_scope", "kind": "afl",
                "reason": f"{strength} AFL signal: '{afl_term}'", "hits": []}
    if generic:
        return {"verdict": "ambiguous", "kind": "generic_sport",
                "reason": f"generic sport language {generic[:2]} without an AFL anchor", "hits": generic}

    if context and ANAPHORA_RE.search(raw.strip()) and len(raw.split()) <= 25:
        ctx_hit, ctx_term, _ = has_afl_signal(context)
        if ctx_hit:
            return {"verdict": "in_scope", "kind": "continuation",
                    "reason": f"follow-up inheriting AFL context ('{ctx_term}')", "hits": []}

    return {"verdict": "out_of_scope", "kind": "no_signal",
            "reason": "no AFL entity, term or continuation context found", "hits": []}


# ------------------------------------------------------------- grounding verifier ---
_NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)")


def _walk_numbers(obj, pool):
    if isinstance(obj, dict):
        for v in obj.values():
            _walk_numbers(v, pool)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_numbers(v, pool)
    elif isinstance(obj, (int, float)):
        pool.add(float(obj))
    elif isinstance(obj, str):
        for m in _NUM_RE.findall(obj):
            try:
                pool.add(float(m))
            except ValueError:
                pass


def collect_grounded_numbers(tool_results: list, user_text: str = "") -> set:
    pool = set()
    for res in tool_results:
        _walk_numbers(res, pool)
    for m in _NUM_RE.findall(user_text or ""):
        try:
            pool.add(float(m))
        except ValueError:
            pass
    return pool


def verify_grounding(answer_text: str, tool_results: list, user_text: str = "") -> dict:
    """Every number in `answer_text` must appear in the pooled tool output (or the user's
    own question, e.g. a season they typed). Anything else is flagged as possibly fabricated."""
    grounded_pool = collect_grounded_numbers(tool_results, user_text)
    answer_nums = {float(m) for m in _NUM_RE.findall(answer_text or "")}
    ungrounded = sorted(answer_nums - grounded_pool)
    return {"grounded": len(ungrounded) == 0, "ungrounded": ungrounded,
            "answer_numbers": sorted(answer_nums), "pool_size": len(grounded_pool)}
