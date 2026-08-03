"""
streamlit_app.py
-----------------
Polished Streamlit chat UI for AFL Analyst. Talks to the FastAPI backend (api.py) over
HTTP, so it can run on a separate process/host from the API.

Run:
    uvicorn api:app --port 8000          # in one terminal
    streamlit run streamlit_app.py       # in another
"""
from __future__ import annotations
import requests
import streamlit as st

TURF = "#173B2E"
TURF_DARK = "#0E271F"
CHALK = "#F4F1E8"
SHERRIN = "#B5432C"
OCHRE = "#D9A441"

st.set_page_config(page_title="AFL Analyst", page_icon="🏉", layout="centered",
                    initial_sidebar_state="expanded")

st.markdown(f"""
<style>
.stApp {{
    background: radial-gradient(ellipse 140% 90% at 50% -10%, #1E4A39 0%, {TURF} 45%, {TURF_DARK} 100%);
}}
[data-testid="stSidebar"] {{
    background: {TURF_DARK};
}}
h1, h2, h3, p, span, label, div {{ color: {CHALK} !important; }}
[data-testid="stChatMessage"] {{
    background: rgba(244,241,232,0.06);
    border: 1px solid rgba(244,241,232,0.14);
    border-left: 3px solid {OCHRE};
    border-radius: 4px;
}}
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {{
    border-left-color: {SHERRIN};
}}
.stTextInput input, .stChatInputContainer textarea {{
    background: rgba(244,241,232,0.08) !important;
    color: {CHALK} !important;
}}
.pred-box {{
    margin-top: 8px; padding: 12px 14px;
    background: rgba(217,164,65,0.08); border: 1px dashed {OCHRE}; border-radius: 4px;
}}
.pred-label {{ font-family: monospace; font-size: 0.78rem; opacity: 0.85; margin-bottom: 4px; }}
.conf-track {{
    width: 100%; height: 22px; border-radius: 11px; overflow: hidden;
    display: flex; background: rgba(255,255,255,0.08); margin: 6px 0 4px;
}}
.conf-home {{ background: {OCHRE}; display:flex; align-items:center; justify-content:flex-start; padding-left:8px; }}
.conf-away {{ background: {SHERRIN}; display:flex; align-items:center; justify-content:flex-end; padding-right:8px; }}
.conf-text {{ font-family: monospace; font-size: 0.72rem; font-weight: 700; color: {TURF_DARK}; }}
.conf-text-away {{ color: {CHALK}; }}
.meta-row {{
    font-family: monospace; font-size: 0.7rem; opacity: 0.55; margin-top: 6px;
}}
.badge {{
    display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.68rem;
    font-family: monospace; margin-right: 6px; border: 1px solid rgba(244,241,232,0.25);
}}
.disclaimer {{
    font-size: 0.72rem; font-style: italic; opacity: 0.75; margin-top: 6px;
}}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------ session state
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, meta, pred}

# ------------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🏉 AFL Analyst")
    st.caption("LangGraph capstone // Streamlit demo")
    api_base = st.text_input("API base URL", value="http://localhost:8000")

    try:
        health = requests.get(f"{api_base}/health", timeout=2).json()
        st.success("API connected", icon="✅")
        st.caption(f"LLM backend: {health.get('llm_backend', 'unknown')}")
    except Exception:
        st.error("API unreachable -- start `uvicorn api:app --port 8000`", icon="⚠️")

    st.markdown("---")
    st.caption("Dataset coverage")
    st.caption("Matches: 1983–2025 · Player stats: 1999–2025")

    st.markdown("---")
    if st.button("🔄 New conversation"):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Try asking:")
    examples = [
        "How many teams are in the AFL?",
        "What were the Cats stats last round?",
        "Who will win Geelong vs Carlton on 2015-05-10?",
        "Who will get the most disposals for Fremantle?",
        "Show me the AFL ladder for 2015",
        "What's the capital of France?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["pending_input"] = ex

# --------------------------------------------------------------------------- main title
st.markdown("## AFL Analyst")
st.caption("Domain-scoped AFL chat + prediction assistant, live over the FastAPI backend.")

# ---------------------------------------------------------------------- confidence bar
def render_confidence_bar(meta: dict):
    """Two-sided confidence bar for match-winner predictions, or a single bar for
    top-player predictions."""
    if "home_win_probability_pct" in meta:
        home_pct = meta["home_win_probability_pct"]
        away_pct = meta["away_win_probability_pct"]
        home_team = meta.get("home_team", "Home")
        away_team = meta.get("away_team", "Away")
        st.markdown(f"""
<div class="pred-box">
  <div class="pred-label">{home_team} vs {away_team} &mdash; win probability</div>
  <div class="conf-track">
    <div class="conf-home" style="width:{home_pct}%;">
      <span class="conf-text">{home_pct:.1f}%</span>
    </div>
    <div class="conf-away" style="width:{away_pct}%;">
      <span class="conf-text conf-text-away">{away_pct:.1f}%</span>
    </div>
  </div>
  <div class="meta-row">
    <span class="badge">{home_team}</span> vs <span class="badge">{away_team}</span>
    &nbsp;·&nbsp; model: {meta.get('model', 'n/a')}
  </div>
  <div class="disclaimer">⚠️ {meta.get('disclaimer', 'This is a predicted probability, not a certainty.')}</div>
</div>
""", unsafe_allow_html=True)
    elif "ranking" in meta and meta["ranking"]:
        top = meta["ranking"][0]
        max_val = max(r["predicted_value"] for r in meta["ranking"]) or 1
        st.markdown(f'<div class="pred-box"><div class="pred-label">'
                    f'Predicted {meta.get("stat_type", "stat")} leader &mdash; {meta.get("team", "")}'
                    f'</div>', unsafe_allow_html=True)
        for r in meta["ranking"][:5]:
            pct = round(r["predicted_value"] / max_val * 100, 1)
            st.markdown(f"""
  <div style="margin:4px 0;">
    <div class="pred-label">{r['player_name']} &mdash; {r['predicted_value']}</div>
    <div class="conf-track" style="height:14px;">
      <div class="conf-home" style="width:{pct}%;"></div>
    </div>
  </div>
""", unsafe_allow_html=True)
        st.markdown(f'<div class="meta-row">model: {meta.get("model", "n/a")}</div>'
                    f'<div class="disclaimer">⚠️ {meta.get("disclaimer", "This is a predicted estimate, not a certainty.")}</div></div>',
                    unsafe_allow_html=True)


def render_meta(meta_dict: dict):
    badges = f'<span class="badge">intent: {meta_dict.get("intent")}</span>' \
             f'<span class="badge">latency: {meta_dict.get("latency_ms")}ms</span>'
    if meta_dict.get("tools_called"):
        badges += f'<span class="badge">tools: {", ".join(meta_dict["tools_called"])}</span>'
    if meta_dict.get("needs_clarification"):
        badges += '<span class="badge">awaiting clarification</span>'
    st.markdown(f'<div class="meta-row">{badges}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- chat log
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("pred"):
            render_confidence_bar(msg["pred"])
        if msg.get("meta"):
            render_meta(msg["meta"])

# ---------------------------------------------------------------------------- input
pending = st.session_state.pop("pending_input", None)
user_input = st.chat_input("Ask AFL Analyst something...") or pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{api_base}/chat",
                    json={"message": user_input, "conversation_id": st.session_state.conversation_id},
                    timeout=15,
                ).json()
                st.session_state.conversation_id = resp["conversation_id"]
                st.write(resp["response"])
                pred_meta = resp.get("prediction_metadata")
                if pred_meta:
                    render_confidence_bar(pred_meta)
                meta = {"intent": resp.get("intent"), "latency_ms": resp.get("latency_ms"),
                        "tools_called": resp.get("tools_called"),
                        "needs_clarification": resp.get("needs_clarification")}
                render_meta(meta)
                st.session_state.messages.append({
                    "role": "assistant", "content": resp["response"],
                    "pred": pred_meta, "meta": meta,
                })
            except Exception as e:
                err = f"Could not reach the AFL Analyst API at {api_base}: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
