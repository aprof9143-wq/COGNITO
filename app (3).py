"""
NeuroSymbolic Verifier — Streamlit App (Real-Time Engine)
All LLM calls use OpenAI GPT (openai SDK) — gpt-5-mini-2025-08-07.
Real-time upgrades: streaming draft, parallel research + rule parsing,
batched audit, pure-math LTN (no TensorFlow), shared client cache.
"""

import streamlit as st
import sys
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroSymbolic Verifier",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0c0e14 !important;
    color: #e8e4dc !important;
}
.stApp { background: #0c0e14; }

.nsv-header {
    text-align: center;
    padding: 3rem 0 1.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 2rem;
}
.nsv-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    font-weight: 400;
    letter-spacing: -0.02em;
    color: #f0ebe0;
    line-height: 1;
    margin-bottom: 0.35rem;
}
.nsv-logo span { color: #c8a96e; font-style: italic; }
.nsv-subtitle {
    font-size: 0.8rem;
    font-weight: 300;
    color: rgba(232,228,220,0.4);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.section-label {
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #c8a96e;
    font-weight: 600;
    margin-bottom: 0.55rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(200,169,110,0.18);
}

.glass-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.4rem;
    margin-bottom: 1.1rem;
    backdrop-filter: blur(12px);
}

.rules-container { display: flex; flex-direction: column; gap: 0.45rem; }
.rule-chip {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    background: rgba(200,169,110,0.07);
    border: 1px solid rgba(200,169,110,0.18);
    border-radius: 10px;
    padding: 0.5rem 0.85rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #e8e4dc;
    animation: fadeSlide 0.25s ease forwards;
}
.rule-chip .rule-num {
    background: rgba(200,169,110,0.22);
    color: #c8a96e;
    border-radius: 5px;
    padding: 0.1rem 0.38rem;
    font-size: 0.7rem;
    font-weight: 700;
    min-width: 22px;
    text-align: center;
    flex-shrink: 0;
}
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(-5px); }
    to   { opacity: 1; transform: translateY(0); }
}

.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 12px !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}
.stTextArea textarea:focus {
    border-color: rgba(200,169,110,0.45) !important;
    box-shadow: 0 0 0 3px rgba(200,169,110,0.07) !important;
}

.stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #e8e4dc !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.84rem !important;
}
.stTextInput input:focus {
    border-color: rgba(200,169,110,0.45) !important;
    box-shadow: 0 0 0 3px rgba(200,169,110,0.07) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #c8a96e 0%, #a8854a 100%) !important;
    border: none !important;
    border-radius: 11px !important;
    color: #0c0e14 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 1.5rem !important;
    width: 100%;
    box-shadow: 0 4px 18px rgba(200,169,110,0.22);
    transition: opacity 0.18s, transform 0.13s !important;
}
.stButton > button:hover  { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

.score-big {
    font-family: 'DM Serif Display', serif;
    font-size: 4rem;
    line-height: 1;
    font-weight: 400;
    margin-bottom: 0.25rem;
}
.score-pass { color: #6dcea8; }
.score-warn { color: #e8c06d; }
.score-fail { color: #e8736d; }
.score-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(232,228,220,0.38);
}

.audit-card {
    border-radius: 13px;
    border: 1px solid;
    padding: 1rem 1.2rem;
    margin-bottom: 0.65rem;
    transition: transform 0.13s;
}
.audit-card:hover { transform: translateX(3px); }
.audit-card.pass { background: rgba(109,206,168,0.04); border-color: rgba(109,206,168,0.18); }
.audit-card.fail { background: rgba(232,115,109,0.05); border-color: rgba(232,115,109,0.22); }
.ac-header { display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.5rem; }
.ac-badge {
    font-size: 0.7rem; font-weight: 700;
    padding: 0.18rem 0.5rem; border-radius: 6px;
    letter-spacing: 0.06em;
    font-family: 'DM Mono', monospace;
    flex-shrink: 0;
}
.audit-card.pass .ac-badge { background: rgba(109,206,168,0.18); color: #6dcea8; }
.audit-card.fail .ac-badge { background: rgba(232,115,109,0.18); color: #e8736d; }
.ac-rule { font-family: 'DM Mono', monospace; font-size: 0.8rem; color: #e8e4dc; flex: 1; }
.audit-meta {
    display: grid; grid-template-columns: repeat(3,1fr);
    gap: 0.45rem; margin-top: 0.5rem;
}
.audit-meta-item {
    background: rgba(255,255,255,0.025);
    border-radius: 7px; padding: 0.35rem 0.55rem;
}
.amk { font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(232,228,220,0.32); margin-bottom: 0.12rem; }
.amv { font-family: 'DM Mono', monospace; font-size: 0.78rem; color: #e8e4dc; }
.audit-explanation {
    font-size: 0.8rem; color: rgba(232,228,220,0.55);
    margin-top: 0.55rem; line-height: 1.55;
    border-top: 1px solid rgba(255,255,255,0.05); padding-top: 0.5rem;
}
.domain-warn {
    font-size: 0.76rem; color: #e8c06d; margin-top: 0.35rem;
    background: rgba(232,192,109,0.07); padding: 0.3rem 0.55rem;
    border-radius: 6px; border-left: 3px solid #e8c06d;
}

.gen-output {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 13px;
    padding: 1.3rem 1.5rem;
    font-size: 0.86rem; line-height: 1.78;
    color: #d0cbbf;
    white-space: pre-wrap; word-break: break-word;
    font-family: 'DM Sans', sans-serif;
    max-height: 440px; overflow-y: auto;
}
.gen-output::-webkit-scrollbar { width: 4px; }
.gen-output::-webkit-scrollbar-thumb { background: rgba(200,169,110,0.22); border-radius: 4px; }

.ref-pill {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: rgba(200,169,110,0.07);
    border: 1px solid rgba(200,169,110,0.16);
    border-radius: 20px; padding: 0.28rem 0.7rem;
    font-size: 0.74rem; color: #c8a96e;
    text-decoration: none; margin-right: 0.35rem; margin-bottom: 0.35rem;
}

.pipe-flow {
    display: flex; align-items: center; justify-content: center;
    gap: 0; padding: 1rem 0; flex-wrap: wrap;
}
.pipe-node { display: flex; flex-direction: column; align-items: center; gap: 0.28rem; padding: 0 0.4rem; }
.pipe-node .pn-icon {
    width: 40px; height: 40px; border-radius: 11px;
    border: 1px solid rgba(255,255,255,0.09);
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; background: rgba(255,255,255,0.035);
}
.pipe-node .pn-label {
    font-size: 0.58rem; text-transform: uppercase;
    letter-spacing: 0.09em; color: rgba(232,228,220,0.35);
    text-align: center; max-width: 56px;
}
.pipe-arrow { color: rgba(200,169,110,0.35); font-size: 0.9rem; padding: 0 0.2rem; }

hr { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 1.3rem 0; }

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 13px; padding: 0.9rem 1.1rem;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.9rem !important; color: #e8e4dc;
}
[data-testid="stMetricLabel"] {
    font-size: 0.68rem !important; text-transform: uppercase;
    letter-spacing: 0.1em; color: rgba(232,228,220,0.38);
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.07); gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: rgba(232,228,220,0.42) !important;
    border-bottom: 2px solid transparent !important;
    font-size: 0.83rem; padding: 0.55rem 1.1rem;
    font-family: 'DM Sans', sans-serif;
}
.stTabs [aria-selected="true"] {
    color: #c8a96e !important;
    border-bottom-color: #c8a96e !important;
    background: transparent !important;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, #c8a96e, #6dcea8) !important;
    border-radius: 4px;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(200,169,110,0.18); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nsv-header">
  <div class="nsv-logo">Neuro<span>Symbolic</span> Verifier</div>
  <div class="nsv-subtitle">Real-Time · Parallel · Streaming · LTN · GPT · Vector Memory</div>
</div>
""", unsafe_allow_html=True)

# ── Pipeline diagram ──────────────────────────────────────────────────────────
st.markdown("""
<div class="glass-panel" style="margin-bottom:1.8rem;">
  <div class="pipe-flow">
    <div class="pipe-node"><div class="pn-icon">📝</div><div class="pn-label">Rules &amp; Prompt</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">🌐</div><div class="pn-label">M4 Research</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">🧩</div><div class="pn-label">M2 Parser</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">💾</div><div class="pn-label">M3 Memory</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">✍️</div><div class="pn-label">Draft Gen</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">🔍</div><div class="pn-label">M2 Audit</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">⚖️</div><div class="pn-label">M1 LTN</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">📊</div><div class="pn-label">Verdict</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for _k, _v in [("rules", []), ("results", None), ("input_counter", 0)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Module path: ensure app's own directory is on sys.path ───────────────────
_app_dir = os.path.dirname(os.path.abspath(__file__))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

# ── Resolve API key: st.secrets (Cloud) → os.getenv (.env) → empty ───────────
# We do this ONCE here and store in session_state so it survives reruns.
# Never pass value= to a password widget — it locks the field on rerun.
def _resolve_api_key() -> str:
    """Read key from Streamlit secrets first, then env, then return empty."""
    # Streamlit Cloud secrets
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            return key.strip()
    except Exception:
        pass
    # Local .env / environment variable
    key = os.getenv("OPENAI_API_KEY", "")
    return key.strip()

if "resolved_api_key" not in st.session_state:
    st.session_state.resolved_api_key = _resolve_api_key()

# ════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 1.15], gap="large")

with col_left:

    # ── API Key ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">🔑 OpenAI API Key</div>', unsafe_allow_html=True)

    # Show a pre-filled indicator if key came from secrets/env, but still
    # allow the user to paste their own. We do NOT pass value= here —
    # that's what caused the key to be silently ignored on reruns.
    _key_hint = "Auto-loaded from secrets ✓" if st.session_state.resolved_api_key else "sk-…"
    api_key_input = st.text_input(
        "api_key", label_visibility="collapsed",
        type="password", placeholder=_key_hint,
        key="api_key_field",
    )
    # Prefer what the user typed; fall back to resolved secret/env key
    api_key = api_key_input.strip() if api_key_input.strip() else st.session_state.resolved_api_key

    # Show a small status line so the user knows whether the key was found
    if st.session_state.resolved_api_key and not api_key_input.strip():
        st.markdown(
            '<p style="font-size:0.72rem;color:rgba(109,206,168,0.7);margin-top:-0.3rem;">'
            '🔒 Key loaded from environment / Streamlit secrets</p>',
            unsafe_allow_html=True,
        )
    elif not api_key:
        st.markdown(
            '<p style="font-size:0.72rem;color:rgba(232,115,109,0.7);margin-top:-0.3rem;">'
            '⚠ No key found — paste your <code>sk-</code> key above</p>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Mode ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">⚡ Pipeline Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "mode", label_visibility="collapsed",
        options=["🔬 Full Pipeline", "📐 Rules + Audit Only", "🌐 Research + Generate"],
        horizontal=True, key="pipeline_mode",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Generation Prompt ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">💬 Generation Prompt</div>', unsafe_allow_html=True)
    user_prompt = st.text_area(
        "prompt", label_visibility="collapsed",
        placeholder="e.g. Write a weekly study plan to improve SAT Math from 600 to 750 in 8 weeks.",
        height=105, key="user_prompt",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Existing Draft (optional) ─────────────────────────────────────────────
    st.markdown('<div class="section-label">📄 Existing Draft (optional)</div>', unsafe_allow_html=True)
    existing_draft = st.text_area(
        "draft", label_visibility="collapsed",
        placeholder="Paste an existing draft to audit — or leave blank to generate from the prompt.",
        height=115, key="existing_draft",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Constraint Rules (chip input) ────────────────────────────────────────
    st.markdown('<div class="section-label">📏 Constraint Rules</div>', unsafe_allow_html=True)

    # Changing the key forces Streamlit to re-render the textarea blank.
    # This avoids the StreamlitAPIException from writing to a live widget key.
    textarea_key = f"rule_input_{st.session_state.input_counter}"

    rule_text = st.text_area(
        "rules_raw", label_visibility="collapsed",
        placeholder=(
            "Type rules — one per line — then press Add:\n"
            "  • Study sessions must be ≤ 2 hours each\n"
            "  • Weekly practice tests ≥ 2\n"
            "  • Total weekly study hours ≤ 14"
        ),
        height=125, key=textarea_key,
    )

    btn_add, btn_clear = st.columns([1, 1])
    with btn_add:
        if st.button("＋ Add Rule(s)", key="add_rule_btn"):
            raw = st.session_state.get(textarea_key, "").strip()
            if raw:
                for line in raw.split("\n"):
                    line = line.strip().lstrip("-•*›▸").strip()
                    if line and line not in st.session_state.rules:
                        st.session_state.rules.append(line)
            st.session_state.input_counter += 1
            st.rerun()

    with btn_clear:
        if st.button("✕ Clear All", key="clear_rules_btn"):
            st.session_state.rules = []
            st.rerun()

    if st.session_state.rules:
        chips_html = '<div class="rules-container" style="margin-top:0.75rem;">'
        for i, r in enumerate(st.session_state.rules):
            escaped = r.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            chips_html += f'<div class="rule-chip"><span class="rule-num">R{i+1}</span><span style="flex:1">{escaped}</span></div>'
        chips_html += '</div>'
        st.markdown(chips_html, unsafe_allow_html=True)

        with st.expander("🗑 Remove individual rules"):
            for i, r in enumerate(list(st.session_state.rules)):
                label = r[:60] + ("…" if len(r) > 60 else "")
                if st.button(f"Remove R{i+1}: {label}", key=f"del_rule_{i}"):
                    st.session_state.rules.pop(i)
                    st.rerun()
    else:
        st.markdown("""
        <div style="text-align:center;color:rgba(232,228,220,0.22);font-size:0.8rem;
                    padding:1rem;border:1px dashed rgba(255,255,255,0.06);
                    border-radius:11px;margin-top:0.5rem;">
            No rules yet — type above and click Add
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("⚡ Run Pipeline", key="run_pipeline_btn")


# ── RIGHT COLUMN ──────────────────────────────────────────────────────────────
with col_right:

    if not run_btn and st.session_state.results is None:
        st.markdown("""
        <div class="glass-panel" style="text-align:center;padding:3rem 2rem;">
            <div style="font-size:2.8rem;margin-bottom:0.9rem;opacity:0.35;">⚖️</div>
            <div style="font-family:'DM Serif Display',serif;font-size:1.35rem;
                        color:rgba(232,228,220,0.45);margin-bottom:0.45rem;">
                Awaiting verification
            </div>
            <div style="font-size:0.8rem;color:rgba(232,228,220,0.22);line-height:1.65;">
                Configure your prompt, rules, and API key<br>
                then press <strong style="color:rgba(200,169,110,0.45)">Run Pipeline</strong> to begin.
            </div>
        </div>""", unsafe_allow_html=True)

    # ── PIPELINE EXECUTION ────────────────────────────────────────────────────
    if run_btn:

        if not api_key.strip():
            st.error("⚠️  Please enter your OpenAI API key (starts with `sk-`).")
            st.stop()
        if not user_prompt.strip() and not existing_draft.strip():
            st.error("⚠️  Please enter a prompt or paste an existing draft.")
            st.stop()

        # ── Import openai SDK ─────────────────────────────────────────────────
        try:
            import openai as _openai
        except ImportError:
            st.error(
                "**`openai` package is not installed.**\n\n"
                "Make sure `requirements.txt` contains `openai>=1.0.0` and is committed "
                "to your repo root. Then go to **Manage app → Reboot app** on Streamlit Cloud."
            )
            st.stop()

        # ── Import pipeline modules ───────────────────────────────────────────
        try:
            import m2_llm_parser as m2
        except ImportError as e:
            st.error(f"Cannot import `m2_llm_parser`: {e}\n\nMake sure it is in the same folder as `app.py`.")
            st.stop()

        try:
            import m3_vector_db as m3
            has_m3 = True
        except ImportError:
            has_m3 = False
            st.warning("⚠️  `chromadb` / `m3_vector_db` unavailable — semantic memory step skipped.")

        try:
            import m4_agentic_router as m4
            has_m4 = True
        except ImportError:
            has_m4 = False
            st.warning("⚠️  `m4_agentic_router` unavailable — research step skipped.")

        try:
            import m1_ltn_core as m1
            has_ltn = True
        except ImportError:
            has_ltn = False
            st.warning("⚠️  m1_ltn_core unavailable — LTN scoring skipped.")

        # ── Progress ──────────────────────────────────────────────────────────
        prog   = st.progress(0, text="Initialising…")
        status = st.empty()
        results = {}

        # ── PARALLEL PHASE: Research (M4) + Rule Parsing (M2) run together ────
        source_results   = []
        structured_rules = []
        memory_context   = []

        do_research = has_m4 and mode in ["🔬 Full Pipeline", "🌐 Research + Generate"]
        do_rules    = bool(st.session_state.rules)

        prog.progress(10, text="⚡ Research & rule parsing in parallel…")

        def _run_research():
            return m4.research_all_sources(user_prompt, api_key=api_key)

        def _run_rule_parse():
            return m2.parse_rules_parallel(st.session_state.rules, api_key)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {}
            if do_research:
                futures["research"] = pool.submit(_run_research)
            if do_rules:
                futures["rules"] = pool.submit(_run_rule_parse)

            if "research" in futures:
                status.info("Module 4 — Searching Wikipedia & DuckDuckGo (concurrent)…")
            if "rules" in futures:
                status.info(f"Module 2 — Parsing {len(st.session_state.rules)} rule(s) in parallel…")

            if "research" in futures:
                try:
                    source_results = futures["research"].result()
                    results["sources"] = source_results
                except Exception as e:
                    st.warning(f"M4 research failed (non-fatal): {e}")

            if "rules" in futures:
                try:
                    structured_rules = futures["rules"].result()
                    results["structured_rules"] = structured_rules
                except Exception as e:
                    st.error(f"Rule parsing failed: {e}")
                    st.stop()

        status.empty()
        prog.progress(35, text="💾 Semantic memory…")

        # STEP 3 — ChromaDB (M3)
        if has_m3 and mode in ["🔬 Full Pipeline", "🌐 Research + Generate"]:
            status.info("Module 3 — ChromaDB batch ingestion…")
            try:
                collection = m3.setup_memory()
                for src in source_results:
                    m3.store_knowledge(collection, f"src_{src.get('source_name','x')}", src["context"])
                if structured_rules:
                    m3.store_all_rules(collection, structured_rules)
                query          = user_prompt or existing_draft
                memory_context = m3.retrieve_context(collection, query, n_results=4)
                results["memory_context"] = memory_context
            except Exception as e:
                st.warning(f"ChromaDB step failed (non-fatal): {e}")
            status.empty()

        # STEP 4 — Generate draft with STREAMING
        draft_text = existing_draft.strip()
        if not draft_text and user_prompt.strip():
            prog.progress(50, text="✍️ Streaming draft…")
            status.info("Generating content with GPT (streaming)…")
            try:
                client = _openai.OpenAI(api_key=api_key.strip())

                ctx_parts = [
                    f"[{s.get('source_name','Source')}] {s['context']}"
                    for s in source_results
                ]
                if memory_context:
                    ctx_parts.append("Relevant constraints:\n" + "\n".join(memory_context))

                rule_block = ""
                if structured_rules:
                    rule_block = "\n\nYou MUST strictly obey these constraints:\n" + "\n".join(
                        f"  • {r.get('display', r.get('original', ''))}"
                        for r in structured_rules
                    )

                full_prompt = (
                    ("CONTEXT:\n" + "\n\n".join(ctx_parts) + "\n\n" if ctx_parts else "")
                    + f"TASK: {user_prompt}"
                    + rule_block
                )

                # Streaming: show tokens as they arrive
                stream_placeholder = st.empty()
                stream_box = st.empty()
                stream_placeholder.info("⚡ Streaming response…")
                collected = []

                stream = client.chat.completions.create(
                    model  = "gpt-5-mini-2025-08-07",
                    max_completion_tokens = 16000,
                    stream = True,
                    messages = [{"role": "user", "content": full_prompt}],
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        collected.append(delta)
                        live_text = "".join(collected)
                        safe = live_text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                        stream_box.markdown(
                            f'<div class="gen-output" style="max-height:320px">{safe}</div>',
                            unsafe_allow_html=True,
                        )

                stream_placeholder.empty()
                draft_text = "".join(collected)

                if not draft_text.strip():
                    st.error("GPT returned an empty response. Try rephrasing your prompt.")
                    st.stop()

                results["draft"] = draft_text
            except st.StopException:
                raise
            except Exception as e:
                st.error(f"Draft generation failed: {e}")
                st.stop()
            status.empty()
        else:
            results["draft"] = draft_text

        # STEP 5 — Batched Audit (M2) — single API call for all rules
        audit_results = []
        if structured_rules and draft_text and mode in ["🔬 Full Pipeline", "📐 Rules + Audit Only"]:
            prog.progress(75, text="🔍 Batched constraint audit…")
            status.info(f"Module 2 — Auditing {len(structured_rules)} rule(s) in one call…")
            try:
                audit_results = m2.structured_audit(draft_text, structured_rules, api_key)
                results["audit"] = audit_results
            except Exception as e:
                st.error(f"Audit failed: {e}")
                st.stop()
            status.empty()

        # STEP 6 — LTN (M1) — pure-math, instant
        if has_ltn and audit_results:
            prog.progress(92, text="⚖️ LTN verification…")
            try:
                ltn_score, violations = m1.verify_and_report(audit_results)
                results["ltn_score"]  = ltn_score
                results["violations"] = violations
            except Exception as e:
                st.warning(f"LTN scoring failed (non-fatal): {e}")

        prog.progress(100, text="✅ Done!")
        time.sleep(0.35)
        prog.empty()
        status.empty()

        st.session_state.results = results
        st.rerun()

    # ── RENDER RESULTS ────────────────────────────────────────────────────────
    if st.session_state.results:
        res  = st.session_state.results
        tabs = st.tabs(["📊 Verdict", "📄 Draft", "🔍 Audit Detail", "🌐 Sources", "🗂 Raw JSON"])

        # Tab 1 — Verdict
        with tabs[0]:
            audit  = res.get("audit", [])
            score  = res.get("ltn_score")
            viols  = res.get("violations", [])
            passed = sum(1 for r in audit if r.get("satisfies"))
            failed = len(audit) - passed

            if score is not None:
                sc = "score-pass" if score >= 0.8 else ("score-warn" if score >= 0.5 else "score-fail")
                vt = "PASS ✓"    if score >= 0.8 else ("MARGINAL ⚠" if score >= 0.5 else "FAIL ✗")
                vc = "#6dcea8"   if score >= 0.8 else ("#e8c06d"    if score >= 0.5 else "#e8736d")
                st.markdown(f"""
                <div class="glass-panel" style="text-align:center;padding:1.8rem;">
                    <div class="score-big {sc}">{score:.3f}</div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.88rem;
                                color:{vc};font-weight:600;letter-spacing:0.1em;margin-top:0.25rem;">
                        {vt}
                    </div>
                    <div class="score-label" style="margin-top:0.35rem;">LTN Universal Verification Score</div>
                </div>""", unsafe_allow_html=True)

            if audit:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rules Checked", len(audit))
                c2.metric("Passed ✅", passed)
                c3.metric("Failed ❌", failed)
                c4.metric("Pass Rate", f"{int(100*passed/len(audit))}%")
            elif res.get("draft"):
                st.success("Draft generated successfully — no rules audited.")
            else:
                st.info("No audit results. Add rules and re-run.")

            if viols:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">⚠ Violations</div>', unsafe_allow_html=True)
                for v in viols:
                    st.markdown(f"""
                    <div class="audit-card fail">
                        <div class="ac-header">
                            <span class="ac-badge">FAIL</span>
                            <span class="ac-rule">{v.get('rule_display','')}</span>
                        </div>
                        <div class="audit-explanation">{v.get('explanation','')}</div>
                    </div>""", unsafe_allow_html=True)

        # Tab 2 — Draft
        with tabs[1]:
            draft = res.get("draft", "")
            if draft:
                st.markdown('<div class="section-label">✍ Generated / Audited Content</div>',
                            unsafe_allow_html=True)
                safe_draft = draft.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                st.markdown(f'<div class="gen-output">{safe_draft}</div>', unsafe_allow_html=True)
                st.download_button("⬇ Download Draft", data=draft,
                                   file_name="draft_output.txt", mime="text/plain", key="dl_draft")
            else:
                st.info("No draft available.")

        # Tab 3 — Audit Detail
        with tabs[2]:
            audit = res.get("audit", [])
            if audit:
                st.markdown('<div class="section-label">🔍 Per-Rule Audit Results</div>',
                            unsafe_allow_html=True)
                for r in audit:
                    s      = "pass" if r.get("satisfies") else "fail"
                    badge  = "PASS ✅" if r.get("satisfies") else "FAIL ❌"
                    sym    = ('<span style="font-size:0.68rem;color:rgba(200,169,110,0.55)">[symbolic]</span>'
                              if r.get("symbolic_check_used") else "")
                    dw     = r.get("domain_warning", "")
                    rid    = r.get("rule_id", "")
                    ridlbl = (rid + 1) if isinstance(rid, int) else rid
                    dw_html = f'<div class="domain-warn">{dw}</div>' if dw else ""
                    st.markdown(f"""
                    <div class="audit-card {s}">
                        <div class="ac-header">
                            <span class="ac-badge">{badge}</span>
                            <span class="ac-rule">R{ridlbl} — {r.get('rule_display','')}</span>
                            {sym}
                        </div>
                        <div class="audit-meta">
                            <div class="audit-meta-item">
                                <div class="amk">Extracted</div>
                                <div class="amv">{r.get('extracted_value_raw','N/A')}</div>
                            </div>
                            <div class="audit-meta-item">
                                <div class="amk">Scope</div>
                                <div class="amv">{r.get('scope','—').upper()}</div>
                            </div>
                            <div class="audit-meta-item">
                                <div class="amk">P → C Confidence</div>
                                <div class="amv">{r.get('premise_confidence',1.0):.2f} → {r.get('conclusion_confidence',0):.2f}</div>
                            </div>
                        </div>
                        {dw_html}
                        <div class="audit-explanation">{r.get('explanation','No explanation.')}</div>
                    </div>""", unsafe_allow_html=True)

                with st.expander("🗂 Structured Rule Constraints (JSON)"):
                    for sr in res.get("structured_rules", []):
                        st.json(sr)
            else:
                st.info("No audit results. Add rules and run the pipeline.")

        # Tab 4 — Sources
        with tabs[3]:
            sources = res.get("sources", [])
            mem_ctx = res.get("memory_context", [])
            if sources:
                st.markdown('<div class="section-label">🌐 Research Sources</div>',
                            unsafe_allow_html=True)
                for src in sources:
                    with st.expander(f"📖 {src.get('source_name','Source')} — {src.get('title','')}"):
                        st.write(src.get("context", ""))
                        ref = src.get("reference", "")
                        if ref and ref != "None":
                            st.markdown(
                                f'<a class="ref-pill" href="{ref}" target="_blank">🔗 {ref[:65]}…</a>',
                                unsafe_allow_html=True)
            else:
                st.info("No research sources used in this run.")

            if mem_ctx:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">💾 Memory Context Retrieved</div>',
                            unsafe_allow_html=True)
                for i, ctx in enumerate(mem_ctx):
                    st.markdown(f"""
                    <div class="rule-chip" style="margin-bottom:0.38rem;">
                        <span class="rule-num">M{i+1}</span>
                        <span style="flex:1;color:rgba(232,228,220,0.65);font-size:0.78rem;">{ctx}</span>
                    </div>""", unsafe_allow_html=True)

        # Tab 5 — Raw JSON
        with tabs[4]:
            st.markdown('<div class="section-label">🗂 Full Pipeline Output</div>',
                        unsafe_allow_html=True)
            display = {k: v for k, v in res.items() if k != "draft"}
            display["draft_preview"] = (res.get("draft","")[:600] + "…") if res.get("draft") else ""
            st.json(display)
            st.download_button(
                "⬇ Download Full JSON",
                data=json.dumps(res, indent=2, default=str),
                file_name="pipeline_results.json",
                mime="application/json",
                key="dl_json",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset Results", key="reset_results"):
            st.session_state.results = None
            st.rerun()
