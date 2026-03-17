"""
NeuroSymbolic Verifier — Streamlit App (Real-Time Engine)
OpenAI GPT gpt-5-mini-2025-08-07 · Qdrant vector memory · Second Brain UI
"""

import streamlit as st
import sys, os, json, time, uuid
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(
    page_title="NeuroSymbolic Verifier",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:#0c0e14!important;color:#e8e4dc!important;}
.stApp{background:#0c0e14;}

/* ── Header ── */
.nsv-header{text-align:center;padding:2.4rem 0 1.4rem;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:1.8rem;}
.nsv-logo{font-family:'DM Serif Display',serif;font-size:2.4rem;font-weight:400;letter-spacing:-0.02em;color:#f0ebe0;line-height:1;margin-bottom:0.3rem;}
.nsv-logo span{color:#c8a96e;font-style:italic;}
.nsv-subtitle{font-size:0.75rem;font-weight:300;color:rgba(232,228,220,0.38);letter-spacing:0.12em;text-transform:uppercase;}

/* ── Section label ── */
.section-label{font-size:0.66rem;letter-spacing:0.14em;text-transform:uppercase;color:#c8a96e;font-weight:600;margin-bottom:0.5rem;display:flex;align-items:center;gap:0.5rem;}
.section-label::after{content:'';flex:1;height:1px;background:rgba(200,169,110,0.18);}

/* ── Glass panel ── */
.glass-panel{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:16px;padding:1.3rem;margin-bottom:1rem;backdrop-filter:blur(12px);}

/* ── Rule chips ── */
.rules-container{display:flex;flex-direction:column;gap:0.4rem;}
.rule-chip{display:flex;align-items:center;gap:0.6rem;background:rgba(200,169,110,0.07);border:1px solid rgba(200,169,110,0.18);border-radius:10px;padding:0.45rem 0.8rem;font-family:'DM Mono',monospace;font-size:0.78rem;color:#e8e4dc;animation:fadeSlide 0.22s ease forwards;}
.rule-chip .rule-num{background:rgba(200,169,110,0.22);color:#c8a96e;border-radius:5px;padding:0.08rem 0.35rem;font-size:0.68rem;font-weight:700;min-width:20px;text-align:center;flex-shrink:0;}
@keyframes fadeSlide{from{opacity:0;transform:translateY(-4px);}to{opacity:1;transform:translateY(0);}}

/* ── Inputs ── */
.stTextArea textarea{background:rgba(255,255,255,0.04)!important;border:1px solid rgba(255,255,255,0.09)!important;border-radius:12px!important;color:#e8e4dc!important;font-family:'DM Sans',sans-serif!important;font-size:0.87rem!important;}
.stTextArea textarea:focus{border-color:rgba(200,169,110,0.45)!important;box-shadow:0 0 0 3px rgba(200,169,110,0.07)!important;}
.stTextInput input{background:rgba(255,255,255,0.04)!important;border:1px solid rgba(255,255,255,0.09)!important;border-radius:10px!important;color:#e8e4dc!important;font-family:'DM Mono',monospace!important;font-size:0.83rem!important;}
.stTextInput input:focus{border-color:rgba(200,169,110,0.45)!important;box-shadow:0 0 0 3px rgba(200,169,110,0.07)!important;}

/* ── Buttons ── */
.stButton>button{background:linear-gradient(135deg,#c8a96e 0%,#a8854a 100%)!important;border:none!important;border-radius:11px!important;color:#0c0e14!important;font-family:'DM Sans',sans-serif!important;font-weight:600!important;font-size:0.87rem!important;padding:0.55rem 1.4rem!important;width:100%;box-shadow:0 4px 18px rgba(200,169,110,0.2);transition:opacity 0.18s,transform 0.13s!important;}
.stButton>button:hover{opacity:0.88!important;transform:translateY(-1px)!important;}
.stButton>button:active{transform:translateY(0)!important;}

/* ── Score ── */
.score-big{font-family:'DM Serif Display',serif;font-size:3.8rem;line-height:1;font-weight:400;margin-bottom:0.2rem;}
.score-pass{color:#6dcea8;}.score-warn{color:#e8c06d;}.score-fail{color:#e8736d;}
.score-label{font-size:0.68rem;text-transform:uppercase;letter-spacing:0.12em;color:rgba(232,228,220,0.35);}

/* ── Audit cards (redesigned) ── */
.audit-row{display:grid;grid-template-columns:auto 1fr;gap:0;margin-bottom:0.75rem;border-radius:14px;overflow:hidden;border:1px solid rgba(255,255,255,0.07);}
.audit-sidebar{width:6px;flex-shrink:0;}
.audit-sidebar.pass{background:#6dcea8;}
.audit-sidebar.fail{background:#e8736d;}
.audit-body{padding:0.85rem 1.1rem;background:rgba(255,255,255,0.025);}
.audit-top{display:flex;align-items:flex-start;gap:0.7rem;margin-bottom:0.65rem;}
.audit-badge{font-size:0.66rem;font-weight:700;padding:0.15rem 0.45rem;border-radius:5px;letter-spacing:0.07em;font-family:'DM Mono',monospace;flex-shrink:0;margin-top:1px;}
.audit-badge.pass{background:rgba(109,206,168,0.15);color:#6dcea8;}
.audit-badge.fail{background:rgba(232,115,109,0.15);color:#e8736d;}
.audit-rule-text{font-family:'DM Mono',monospace;font-size:0.79rem;color:#e8e4dc;line-height:1.4;flex:1;}
.audit-sym-tag{font-size:0.62rem;color:rgba(200,169,110,0.6);background:rgba(200,169,110,0.08);border-radius:4px;padding:0.1rem 0.3rem;white-space:nowrap;align-self:flex-start;margin-top:2px;}
.audit-pills{display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:0.55rem;}
.audit-pill{display:inline-flex;flex-direction:column;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:8px;padding:0.3rem 0.55rem;min-width:80px;}
.ap-label{font-size:0.58rem;text-transform:uppercase;letter-spacing:0.09em;color:rgba(232,228,220,0.3);margin-bottom:0.1rem;}
.ap-value{font-family:'DM Mono',monospace;font-size:0.75rem;color:#d8d4ca;}
.audit-explanation{font-size:0.79rem;color:rgba(232,228,220,0.5);line-height:1.55;padding-top:0.45rem;border-top:1px solid rgba(255,255,255,0.05);}
.audit-domain-warn{font-size:0.74rem;color:#e8c06d;background:rgba(232,192,109,0.07);padding:0.25rem 0.5rem;border-radius:5px;border-left:3px solid #e8c06d;margin-bottom:0.45rem;}
.confidence-bar-wrap{display:flex;align-items:center;gap:0.5rem;margin-top:0.35rem;}
.confidence-bar-bg{flex:1;height:4px;background:rgba(255,255,255,0.07);border-radius:2px;overflow:hidden;}
.confidence-bar-fill{height:100%;border-radius:2px;}
.cb-label{font-size:0.62rem;font-family:'DM Mono',monospace;color:rgba(232,228,220,0.35);white-space:nowrap;}

/* ── Second Brain ── */
.brain-header{display:flex;align-items:center;gap:0.75rem;margin-bottom:1.2rem;}
.brain-title{font-family:'DM Serif Display',serif;font-size:1.3rem;color:#f0ebe0;}
.brain-subtitle{font-size:0.72rem;color:rgba(232,228,220,0.4);text-transform:uppercase;letter-spacing:0.1em;}
.brain-stat{display:flex;flex-direction:column;align-items:center;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:0.85rem 1.2rem;text-align:center;}
.brain-stat-num{font-family:'DM Serif Display',serif;font-size:2rem;color:#c8a96e;line-height:1;}
.brain-stat-label{font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;color:rgba(232,228,220,0.35);margin-top:0.2rem;}
.sym-node{background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.55rem;transition:transform 0.12s,border-color 0.12s;}
.sym-node:hover{transform:translateX(4px);border-color:rgba(200,169,110,0.25);}
.sym-node-top{display:flex;align-items:center;gap:0.6rem;margin-bottom:0.45rem;}
.sym-type-badge{font-size:0.62rem;font-weight:700;padding:0.12rem 0.4rem;border-radius:5px;font-family:'DM Mono',monospace;letter-spacing:0.06em;flex-shrink:0;}
.sym-type-constraint{background:rgba(200,169,110,0.14);color:#c8a96e;}
.sym-type-observation{background:rgba(109,206,168,0.12);color:#6dcea8;}
.sym-type-source{background:rgba(138,110,200,0.14);color:#a88ecc;}
.sym-type-audit-pass{background:rgba(109,206,168,0.12);color:#6dcea8;}
.sym-type-audit-fail{background:rgba(232,115,109,0.14);color:#e8736d;}
.sym-node-text{font-family:'DM Mono',monospace;font-size:0.77rem;color:#d8d4ca;flex:1;line-height:1.35;}
.sym-meta-row{display:flex;flex-wrap:wrap;gap:0.35rem;}
.sym-meta-chip{font-size:0.62rem;font-family:'DM Mono',monospace;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);border-radius:5px;padding:0.1rem 0.35rem;color:rgba(232,228,220,0.45);}
.sym-meta-chip.highlight{background:rgba(200,169,110,0.1);border-color:rgba(200,169,110,0.2);color:#c8a96e;}
.sym-ts{font-size:0.6rem;color:rgba(232,228,220,0.22);font-family:'DM Mono',monospace;margin-left:auto;white-space:nowrap;}
.vector-viz{font-family:'DM Mono',monospace;font-size:0.58rem;color:rgba(200,169,110,0.45);letter-spacing:0.03em;line-height:1.6;word-break:break-all;margin-top:0.3rem;}
.collection-tab{display:inline-flex;align-items:center;gap:0.35rem;padding:0.35rem 0.8rem;border-radius:20px;font-size:0.73rem;font-weight:600;cursor:pointer;transition:all 0.15s;}

/* ── Gen output ── */
.gen-output{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:13px;padding:1.2rem 1.4rem;font-size:0.85rem;line-height:1.78;color:#d0cbbf;white-space:pre-wrap;word-break:break-word;font-family:'DM Sans',sans-serif;max-height:420px;overflow-y:auto;}
.gen-output::-webkit-scrollbar{width:4px;}.gen-output::-webkit-scrollbar-thumb{background:rgba(200,169,110,0.22);border-radius:4px;}

/* ── Sources ── */
.ref-pill{display:inline-flex;align-items:center;gap:0.3rem;background:rgba(200,169,110,0.07);border:1px solid rgba(200,169,110,0.16);border-radius:20px;padding:0.25rem 0.65rem;font-size:0.72rem;color:#c8a96e;text-decoration:none;margin-right:0.3rem;margin-bottom:0.3rem;}

/* ── Pipeline flow ── */
.pipe-flow{display:flex;align-items:center;justify-content:center;gap:0;padding:0.9rem 0;flex-wrap:wrap;}
.pipe-node{display:flex;flex-direction:column;align-items:center;gap:0.25rem;padding:0 0.35rem;}
.pipe-node .pn-icon{width:38px;height:38px;border-radius:10px;border:1px solid rgba(255,255,255,0.09);display:flex;align-items:center;justify-content:center;font-size:0.95rem;background:rgba(255,255,255,0.03);}
.pipe-node .pn-label{font-size:0.56rem;text-transform:uppercase;letter-spacing:0.09em;color:rgba(232,228,220,0.32);text-align:center;max-width:54px;}
.pipe-arrow{color:rgba(200,169,110,0.32);font-size:0.85rem;padding:0 0.15rem;}

/* ── Misc ── */
hr{border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1.2rem 0;}
[data-testid="metric-container"]{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:13px;padding:0.85rem 1rem;}
[data-testid="stMetricValue"]{font-family:'DM Serif Display',serif!important;font-size:1.8rem!important;color:#e8e4dc;}
[data-testid="stMetricLabel"]{font-size:0.66rem!important;text-transform:uppercase;letter-spacing:0.1em;color:rgba(232,228,220,0.35);}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid rgba(255,255,255,0.07);gap:0;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:rgba(232,228,220,0.4)!important;border-bottom:2px solid transparent!important;font-size:0.82rem;padding:0.5rem 1rem;font-family:'DM Sans',sans-serif;}
.stTabs [aria-selected="true"]{color:#c8a96e!important;border-bottom-color:#c8a96e!important;background:transparent!important;}
.stProgress>div>div>div{background:linear-gradient(90deg,#c8a96e,#6dcea8)!important;border-radius:4px;}
::-webkit-scrollbar{width:5px;height:5px;}::-webkit-scrollbar-track{background:transparent;}::-webkit-scrollbar-thumb{background:rgba(200,169,110,0.18);border-radius:4px;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="nsv-header">
  <div class="nsv-logo">Neuro<span>Symbolic</span> Verifier</div>
  <div class="nsv-subtitle">Real-Time · Parallel · Streaming · LTN · GPT · Qdrant Second Brain</div>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="glass-panel" style="margin-bottom:1.6rem;">
  <div class="pipe-flow">
    <div class="pipe-node"><div class="pn-icon">📝</div><div class="pn-label">Rules &amp; Prompt</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">🌐</div><div class="pn-label">M4 Research</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">🧩</div><div class="pn-label">M2 Parser</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">🧠</div><div class="pn-label">Qdrant Brain</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">✍️</div><div class="pn-label">Draft Gen</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">🔍</div><div class="pn-label">M2 Audit</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">⚖️</div><div class="pn-label">M1 LTN</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="pn-icon">📊</div><div class="pn-label">Verdict</div></div>
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for _k, _v in [("rules", []), ("results", None), ("input_counter", 0),
               ("qdrant_client", None), ("brain_records", {})]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

_app_dir = os.path.dirname(os.path.abspath(__file__))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

# ── API key resolution ────────────────────────────────────────────────────────
def _resolve(env_key: str) -> str:
    try:
        v = st.secrets.get(env_key, "")
        if v: return v.strip()
    except Exception: pass
    return os.getenv(env_key, "").strip()

if "resolved_api_key" not in st.session_state:
    st.session_state.resolved_api_key = _resolve("OPENAI_API_KEY")
if "resolved_qdrant_url" not in st.session_state:
    st.session_state.resolved_qdrant_url = _resolve("QDRANT_URL")
if "resolved_qdrant_key" not in st.session_state:
    st.session_state.resolved_qdrant_key = _resolve("QDRANT_API_KEY")

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT — Left column (controls) / Right column (results)
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:

    # ── OpenAI API Key ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">🔑 OpenAI API Key</div>', unsafe_allow_html=True)
    _hint = "Auto-loaded ✓" if st.session_state.resolved_api_key else "sk-…"
    api_key_input = st.text_input("api_key", label_visibility="collapsed",
                                   type="password", placeholder=_hint, key="api_key_field")
    api_key = api_key_input.strip() or st.session_state.resolved_api_key
    if st.session_state.resolved_api_key and not api_key_input.strip():
        st.markdown('<p style="font-size:0.7rem;color:rgba(109,206,168,0.7);margin-top:-0.25rem;">🔒 Loaded from environment</p>', unsafe_allow_html=True)
    elif not api_key:
        st.markdown('<p style="font-size:0.7rem;color:rgba(232,115,109,0.7);margin-top:-0.25rem;">⚠ No key found</p>', unsafe_allow_html=True)

    # ── Qdrant Config ─────────────────────────────────────────────────────────
    with st.expander("🧠 Qdrant Config (optional — defaults to in-memory)", expanded=False):
        st.markdown('<p style="font-size:0.73rem;color:rgba(232,228,220,0.45);margin-bottom:0.5rem;">Leave blank to use fast in-memory Qdrant. Set a URL for persistent cloud storage.</p>', unsafe_allow_html=True)
        qdrant_url_input = st.text_input("Qdrant URL", placeholder="https://…qdrant.io:6333 or blank for in-memory",
                                          key="qdrant_url_field", label_visibility="visible")
        qdrant_key_input = st.text_input("Qdrant API Key", type="password",
                                          placeholder="your-qdrant-api-key or blank",
                                          key="qdrant_key_field", label_visibility="visible")
    qdrant_url = qdrant_url_input.strip() or st.session_state.resolved_qdrant_url or None
    qdrant_key = qdrant_key_input.strip() or st.session_state.resolved_qdrant_key or None

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pipeline Mode ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">⚡ Pipeline Mode</div>', unsafe_allow_html=True)
    mode = st.radio("mode", label_visibility="collapsed",
                    options=["🔬 Full Pipeline", "📐 Rules + Audit Only", "🌐 Research + Generate"],
                    horizontal=True, key="pipeline_mode")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Generation Prompt ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">💬 Generation Prompt</div>', unsafe_allow_html=True)
    user_prompt = st.text_area("prompt", label_visibility="collapsed",
                                placeholder="e.g. Write a weekly study plan to improve SAT Math from 600 to 750 in 8 weeks.",
                                height=100, key="user_prompt")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Existing Draft ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">📄 Existing Draft (optional)</div>', unsafe_allow_html=True)
    existing_draft = st.text_area("draft", label_visibility="collapsed",
                                   placeholder="Paste an existing draft to audit — or leave blank to generate.",
                                   height=110, key="existing_draft")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Constraint Rules ──────────────────────────────────────────────────────
    st.markdown('<div class="section-label">📏 Constraint Rules</div>', unsafe_allow_html=True)
    textarea_key = f"rule_input_{st.session_state.input_counter}"
    st.text_area("rules_raw", label_visibility="collapsed",
                 placeholder="One rule per line:\n  • Study sessions ≤ 2 hours each\n  • Weekly tests ≥ 2\n  • Total weekly hours ≤ 14",
                 height=120, key=textarea_key)

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
        chips = '<div class="rules-container" style="margin-top:0.7rem;">'
        for i, r in enumerate(st.session_state.rules):
            e = r.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            chips += f'<div class="rule-chip"><span class="rule-num">R{i+1}</span><span style="flex:1">{e}</span></div>'
        chips += '</div>'
        st.markdown(chips, unsafe_allow_html=True)
        with st.expander("🗑 Remove individual rules"):
            for i, r in enumerate(list(st.session_state.rules)):
                lbl = r[:55] + ("…" if len(r) > 55 else "")
                if st.button(f"Remove R{i+1}: {lbl}", key=f"del_{i}"):
                    st.session_state.rules.pop(i); st.rerun()
    else:
        st.markdown('<div style="text-align:center;color:rgba(232,228,220,0.2);font-size:0.78rem;padding:0.9rem;border:1px dashed rgba(255,255,255,0.06);border-radius:10px;margin-top:0.4rem;">No rules yet — type above and click Add</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("⚡ Run Pipeline", key="run_pipeline_btn")


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN
# ══════════════════════════════════════════════════════════════════════════════
with col_right:

    if not run_btn and st.session_state.results is None:
        st.markdown("""
        <div class="glass-panel" style="text-align:center;padding:2.8rem 2rem;">
            <div style="font-size:2.6rem;margin-bottom:0.8rem;opacity:0.3;">⚖️</div>
            <div style="font-family:'DM Serif Display',serif;font-size:1.3rem;color:rgba(232,228,220,0.4);margin-bottom:0.4rem;">Awaiting verification</div>
            <div style="font-size:0.78rem;color:rgba(232,228,220,0.2);line-height:1.65;">Configure prompt, rules &amp; API key<br>then press <strong style="color:rgba(200,169,110,0.4)">Run Pipeline</strong></div>
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PIPELINE EXECUTION
    # ══════════════════════════════════════════════════════════════════════════
    if run_btn:
        if not api_key.strip():
            st.error("⚠️ Please enter your OpenAI API key."); st.stop()
        if not user_prompt.strip() and not existing_draft.strip():
            st.error("⚠️ Please enter a prompt or paste an existing draft."); st.stop()

        # ── Import modules ────────────────────────────────────────────────────
        try:
            import openai as _openai
        except ImportError:
            st.error("`openai` not installed. Add `openai>=1.0.0` to requirements.txt"); st.stop()

        try:
            import m2_llm_parser as m2
        except ImportError as e:
            st.error(f"Cannot import m2_llm_parser: {e}"); st.stop()

        try:
            import m3_vector_db as m3
            has_m3 = True
        except ImportError:
            has_m3 = False
            st.warning("⚠️ qdrant-client unavailable — memory step skipped.")

        try:
            import m4_agentic_router as m4
            has_m4 = True
        except ImportError:
            has_m4 = False

        try:
            import m1_ltn_core as m1
            has_ltn = True
        except ImportError:
            has_ltn = False

        prog   = st.progress(0, text="Initialising…")
        status = st.empty()
        results = {}
        run_id  = str(uuid.uuid4())[:8]

        # ── Snapshot session_state before threads ─────────────────────────────
        rules_snapshot = list(st.session_state.rules)

        # ── Init Qdrant ───────────────────────────────────────────────────────
        qdrant_client = None
        if has_m3:
            try:
                qdrant_client = m3.setup_memory(
                    url=qdrant_url,
                    openai_api_key=api_key,
                    qdrant_api_key=qdrant_key,
                )
                st.session_state.qdrant_client = qdrant_client
            except Exception as e:
                st.warning(f"Qdrant init failed (non-fatal): {e}")
                has_m3 = False

        # ── PARALLEL: Research + Rule Parsing ────────────────────────────────
        source_results   = []
        structured_rules = []
        memory_context   = []

        do_research = has_m4 and mode in ["🔬 Full Pipeline", "🌐 Research + Generate"]
        do_rules    = bool(rules_snapshot)

        prog.progress(8, text="⚡ Research & rule parsing in parallel…")

        def _run_research():
            return m4.research_all_sources(user_prompt, api_key=api_key)

        def _run_rule_parse():
            return m2.parse_rules_parallel(rules_snapshot, api_key)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {}
            if do_research: futures["research"] = pool.submit(_run_research)
            if do_rules:    futures["rules"]    = pool.submit(_run_rule_parse)

            msgs = []
            if "research" in futures: msgs.append("M4 searching Wikipedia & DuckDuckGo")
            if "rules"    in futures: msgs.append(f"M2 parsing {len(rules_snapshot)} rule(s)")
            if msgs: status.info(" · ".join(msgs) + "…")

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
                    st.error(f"Rule parsing failed: {e}"); st.stop()

        status.empty()
        prog.progress(32, text="🧠 Storing in Qdrant…")

        # ── M3: Store in Qdrant ───────────────────────────────────────────────
        brain_records = {"rules": [], "sources": [], "audit": []}
        if has_m3 and qdrant_client:
            status.info("Qdrant — batch-embedding & storing symbolic references…")
            try:
                if source_results:
                    for src in source_results:
                        m3.store_source(qdrant_client, src, api_key)
                    brain_records["sources"] = m3.get_all_records(qdrant_client, "sources")

                if structured_rules:
                    m3.store_all_rules(qdrant_client, structured_rules, api_key)
                    brain_records["rules"] = m3.get_all_records(qdrant_client, "rules")

                query          = user_prompt or existing_draft
                memory_context = m3.retrieve_context(qdrant_client, query, api_key, n_results=4)
                results["memory_context"] = memory_context
            except Exception as e:
                st.warning(f"Qdrant step failed (non-fatal): {e}")
            status.empty()

        # ── STEP 4: Streaming draft ───────────────────────────────────────────
        draft_text = existing_draft.strip()
        if not draft_text and user_prompt.strip():
            prog.progress(48, text="✍️ Streaming draft…")
            status.info("Generating with GPT (streaming)…")
            try:
                client = _openai.OpenAI(api_key=api_key.strip())

                ctx_parts = [f"[{s.get('source_name','Source')}] {s['context']}"
                             for s in source_results]
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
                    + f"TASK: {user_prompt}" + rule_block
                )

                stream_hdr = st.empty()
                stream_box = st.empty()
                stream_hdr.info("⚡ Streaming response…")
                collected  = []

                stream = client.chat.completions.create(
                    model="gpt-5-mini-2025-08-07",
                    max_completion_tokens=16000,
                    stream=True,
                    messages=[{"role": "user", "content": full_prompt}],
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        collected.append(delta)
                        live = "".join(collected)
                        safe = live.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                        stream_box.markdown(
                            f'<div class="gen-output" style="max-height:300px">{safe}</div>',
                            unsafe_allow_html=True)

                stream_hdr.empty()
                draft_text = "".join(collected)
                if not draft_text.strip():
                    st.error("GPT returned an empty response. Try rephrasing your prompt."); st.stop()
                results["draft"] = draft_text
            except st.StopException:
                raise
            except Exception as e:
                st.error(f"Draft generation failed: {e}"); st.stop()
            status.empty()
        else:
            results["draft"] = draft_text

        # ── STEP 5: Batched audit ─────────────────────────────────────────────
        audit_results = []
        if structured_rules and draft_text and mode in ["🔬 Full Pipeline", "📐 Rules + Audit Only"]:
            prog.progress(72, text="🔍 Batched constraint audit…")
            status.info(f"M2 — auditing {len(structured_rules)} rule(s) in one call…")
            try:
                audit_results = m2.structured_audit(draft_text, structured_rules, api_key)
                results["audit"] = audit_results
            except Exception as e:
                st.error(f"Audit failed: {e}"); st.stop()
            status.empty()

            # Store audit results in Qdrant
            if has_m3 and qdrant_client and audit_results:
                try:
                    for ar in audit_results:
                        m3.store_audit_result(qdrant_client, ar, api_key, run_id=run_id)
                    brain_records["audit"] = m3.get_all_records(qdrant_client, "audit")
                except Exception as e:
                    pass  # non-fatal

        # ── STEP 6: LTN ──────────────────────────────────────────────────────
        if has_ltn and audit_results:
            prog.progress(90, text="⚖️ LTN verification…")
            try:
                ltn_score, violations = m1.verify_and_report(audit_results)
                results["ltn_score"]  = ltn_score
                results["violations"] = violations
            except Exception as e:
                st.warning(f"LTN scoring failed (non-fatal): {e}")

        prog.progress(100, text="✅ Done!")
        time.sleep(0.3)
        prog.empty(); status.empty()

        results["brain_records"] = brain_records
        results["run_id"]        = run_id
        st.session_state.results = results
        st.session_state.brain_records = brain_records
        st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # RESULTS TABS
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.results:
        res  = st.session_state.results
        tabs = st.tabs(["📊 Verdict", "📄 Draft", "🔍 Audit Detail",
                        "🧠 Second Brain", "🌐 Sources", "🗂 Raw JSON"])

        # ── Tab 1: Verdict ────────────────────────────────────────────────────
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
                <div class="glass-panel" style="text-align:center;padding:1.6rem;">
                    <div class="score-big {sc}">{score:.3f}</div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.85rem;color:{vc};
                                font-weight:600;letter-spacing:0.1em;margin-top:0.2rem;">{vt}</div>
                    <div class="score-label" style="margin-top:0.3rem;">LTN Universal Verification Score</div>
                </div>""", unsafe_allow_html=True)

            if audit:
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Checked",   len(audit))
                c2.metric("Passed ✅", passed)
                c3.metric("Failed ❌", failed)
                c4.metric("Pass Rate", f"{int(100*passed/len(audit))}%")
            elif res.get("draft"):
                st.success("Draft generated — no rules audited.")
            else:
                st.info("No audit results. Add rules and re-run.")

            if viols:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">⚠ Violations</div>', unsafe_allow_html=True)
                for v in viols:
                    e_rule = v.get('rule_display','').replace("&","&amp;").replace("<","&lt;")
                    e_expl = v.get('explanation','').replace("&","&amp;").replace("<","&lt;")
                    st.markdown(f"""
                    <div class="audit-row">
                        <div class="audit-sidebar fail"></div>
                        <div class="audit-body">
                            <div class="audit-top">
                                <span class="audit-badge fail">FAIL</span>
                                <span class="audit-rule-text">{e_rule}</span>
                            </div>
                            <div class="audit-explanation">{e_expl}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

        # ── Tab 2: Draft ──────────────────────────────────────────────────────
        with tabs[1]:
            draft = res.get("draft", "")
            if draft:
                st.markdown('<div class="section-label">✍ Generated Content</div>', unsafe_allow_html=True)
                safe = draft.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                st.markdown(f'<div class="gen-output">{safe}</div>', unsafe_allow_html=True)
                st.download_button("⬇ Download Draft", data=draft,
                                   file_name="draft_output.txt", mime="text/plain", key="dl_draft")
            else:
                st.info("No draft available.")

        # ── Tab 3: Audit Detail (redesigned) ─────────────────────────────────
        with tabs[2]:
            audit = res.get("audit", [])
            if not audit:
                st.info("No audit results. Add rules and run the pipeline.")
            else:
                st.markdown('<div class="section-label">🔍 Per-Rule Audit Results</div>', unsafe_allow_html=True)

                # Summary bar at top
                pass_pct = int(100 * passed / len(audit)) if audit else 0
                bar_html = f"""
                <div style="margin-bottom:1.2rem;">
                  <div style="display:flex;justify-content:space-between;
                               font-size:0.68rem;font-family:'DM Mono',monospace;
                               color:rgba(232,228,220,0.4);margin-bottom:0.35rem;">
                    <span>{passed} passed</span><span>{failed} failed</span>
                  </div>
                  <div style="height:6px;background:rgba(255,255,255,0.07);border-radius:3px;overflow:hidden;">
                    <div style="height:100%;width:{pass_pct}%;
                                background:linear-gradient(90deg,#6dcea8,#4ab898);border-radius:3px;"></div>
                  </div>
                </div>"""
                st.markdown(bar_html, unsafe_allow_html=True)

                for r in audit:
                    s        = "pass" if r.get("satisfies") else "fail"
                    badge_lbl= "PASS" if r.get("satisfies") else "FAIL"
                    rid      = r.get("rule_id", "")
                    ridlbl   = (rid + 1) if isinstance(rid, int) else rid
                    sym_tag  = '<span class="audit-sym-tag">symbolic ✓</span>' if r.get("symbolic_check_used") else ""
                    dw       = r.get("domain_warning","")
                    dw_html  = f'<div class="audit-domain-warn">⚠ {dw}</div>' if dw else ""

                    # Confidence bar
                    p_conf   = r.get("premise_confidence", 1.0)
                    c_conf   = r.get("conclusion_confidence", 0.0)
                    c_color  = "#6dcea8" if c_conf >= 0.5 else "#e8736d"
                    conf_html= f"""
                    <div class="confidence-bar-wrap">
                        <span class="cb-label">P→C</span>
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill"
                                 style="width:{int(p_conf*100)}%;background:#c8a96e;"></div>
                        </div>
                        <span class="cb-label">{p_conf:.2f}</span>
                        <span class="cb-label" style="opacity:0.4;">→</span>
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill"
                                 style="width:{int(c_conf*100)}%;background:{c_color};"></div>
                        </div>
                        <span class="cb-label" style="color:{c_color}">{c_conf:.2f}</span>
                    </div>"""

                    # Safe-encode dynamic values
                    rule_disp = r.get("rule_display","").replace("&","&amp;").replace("<","&lt;")
                    extr_raw  = str(r.get("extracted_value_raw","N/A")).replace("<","&lt;")
                    extr_num  = r.get("extracted_value_num")
                    extr_num_s= f"{extr_num:.4g}" if extr_num is not None else "—"
                    unit_note = r.get("unit_conversion_note","—").replace("<","&lt;")
                    scope_val = r.get("scope","—").upper()
                    expl      = r.get("explanation","No explanation.").replace("<","&lt;")

                    st.markdown(f"""
                    <div class="audit-row">
                        <div class="audit-sidebar {s}"></div>
                        <div class="audit-body">
                            <div class="audit-top">
                                <span class="audit-badge {s}">{badge_lbl}</span>
                                <span class="audit-rule-text">R{ridlbl} — {rule_disp}</span>
                                {sym_tag}
                            </div>
                            {dw_html}
                            <div class="audit-pills">
                                <div class="audit-pill">
                                    <span class="ap-label">Extracted</span>
                                    <span class="ap-value">{extr_raw}</span>
                                </div>
                                <div class="audit-pill">
                                    <span class="ap-label">Numeric</span>
                                    <span class="ap-value">{extr_num_s}</span>
                                </div>
                                <div class="audit-pill">
                                    <span class="ap-label">Scope</span>
                                    <span class="ap-value">{scope_val}</span>
                                </div>
                                <div class="audit-pill">
                                    <span class="ap-label">Unit Conv.</span>
                                    <span class="ap-value">{unit_note}</span>
                                </div>
                            </div>
                            {conf_html}
                            <div class="audit-explanation">{expl}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                with st.expander("🗂 Structured Rule JSON"):
                    for sr in res.get("structured_rules", []):
                        st.json(sr)

        # ── Tab 4: Second Brain ───────────────────────────────────────────────
        with tabs[3]:
            brain = res.get("brain_records", st.session_state.brain_records or {})

            st.markdown("""
            <div class="brain-header">
                <div>
                    <div class="brain-title">🧠 Second Brain</div>
                    <div class="brain-subtitle">Qdrant symbolic memory — live view of stored references</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Stats row
            rules_recs  = brain.get("rules",   [])
            source_recs = brain.get("sources", [])
            audit_recs  = brain.get("audit",   [])
            total       = len(rules_recs) + len(source_recs) + len(audit_recs)

            sc1,sc2,sc3,sc4 = st.columns(4)
            with sc1:
                st.markdown(f'<div class="brain-stat"><div class="brain-stat-num">{total}</div><div class="brain-stat-label">Total Nodes</div></div>', unsafe_allow_html=True)
            with sc2:
                st.markdown(f'<div class="brain-stat"><div class="brain-stat-num">{len(rules_recs)}</div><div class="brain-stat-label">Rule Nodes</div></div>', unsafe_allow_html=True)
            with sc3:
                st.markdown(f'<div class="brain-stat"><div class="brain-stat-num">{len(source_recs)}</div><div class="brain-stat-label">Source Nodes</div></div>', unsafe_allow_html=True)
            with sc4:
                st.markdown(f'<div class="brain-stat"><div class="brain-stat-num">{len(audit_recs)}</div><div class="brain-stat-label">Audit Nodes</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if total == 0:
                st.markdown('<div style="text-align:center;color:rgba(232,228,220,0.2);padding:2rem;font-size:0.82rem;">No symbolic references stored yet — run the pipeline first.</div>', unsafe_allow_html=True)
            else:
                brain_tab = st.radio("View collection", ["Rules", "Sources", "Audit"],
                                     horizontal=True, key="brain_tab_select",
                                     label_visibility="collapsed")

                if brain_tab == "Rules":
                    records = rules_recs
                elif brain_tab == "Sources":
                    records = source_recs
                else:
                    records = audit_recs

                if not records:
                    st.info(f"No {brain_tab.lower()} records stored in this run.")
                else:
                    st.markdown(f'<div class="section-label">📦 {brain_tab} — {len(records)} symbolic node(s)</div>', unsafe_allow_html=True)

                    for rec in records:
                        rtype      = rec.get("record_type", "")
                        text       = rec.get("text", "")[:120]
                        stored_at  = rec.get("stored_at", "")[:19].replace("T", " ")

                        # Pick badge type
                        if rtype == "rule":
                            nature    = rec.get("rule_nature", "constraint")
                            badge_cls = f"sym-type-{nature}"
                            badge_lbl = nature.upper()
                        elif rtype == "source":
                            badge_cls = "sym-type-source"
                            badge_lbl = "SOURCE"
                        elif rtype == "audit":
                            ok = rec.get("satisfies", False)
                            badge_cls = "sym-type-audit-pass" if ok else "sym-type-audit-fail"
                            badge_lbl = "AUDIT PASS" if ok else "AUDIT FAIL"
                        else:
                            badge_cls = "sym-type-constraint"
                            badge_lbl = rtype.upper()

                        # Build meta chips
                        chips_html = ""
                        if rtype == "rule":
                            op     = rec.get("operator","")
                            thresh = rec.get("threshold")
                            unit   = rec.get("unit","")
                            scope  = rec.get("scope","")
                            ctype  = rec.get("constraint_type","")
                            var    = rec.get("variable","")
                            if var:    chips_html += f'<span class="sym-meta-chip highlight">{var}</span>'
                            if ctype:  chips_html += f'<span class="sym-meta-chip">{ctype}</span>'
                            if op:     chips_html += f'<span class="sym-meta-chip highlight">{op}</span>'
                            if thresh is not None: chips_html += f'<span class="sym-meta-chip highlight">{thresh}{" "+unit if unit else ""}</span>'
                            if scope:  chips_html += f'<span class="sym-meta-chip">scope:{scope}</span>'
                        elif rtype == "source":
                            src_name = rec.get("source_name","")
                            title    = rec.get("title","")[:40]
                            if src_name: chips_html += f'<span class="sym-meta-chip highlight">{src_name}</span>'
                            if title:    chips_html += f'<span class="sym-meta-chip">{title}</span>'
                        elif rtype == "audit":
                            p_c  = rec.get("premise_confidence", 1.0)
                            c_c  = rec.get("conclusion_confidence", 0.0)
                            scope= rec.get("scope","")
                            extr = rec.get("extracted_value_raw","")[:30]
                            sym  = rec.get("symbolic_check_used", False)
                            chips_html += f'<span class="sym-meta-chip highlight">P:{p_c:.2f}→C:{c_c:.2f}</span>'
                            if scope: chips_html += f'<span class="sym-meta-chip">scope:{scope}</span>'
                            if extr:  chips_html += f'<span class="sym-meta-chip">{extr}</span>'
                            if sym:   chips_html += f'<span class="sym-meta-chip highlight">symbolic✓</span>'

                        # Vector preview (simulated snippet — actual vectors not returned by scroll)
                        vec_preview = "· · · [1536-dim cosine embedding stored in Qdrant] · · ·"

                        safe_text = text.replace("&","&amp;").replace("<","&lt;")
                        st.markdown(f"""
                        <div class="sym-node">
                            <div class="sym-node-top">
                                <span class="sym-type-badge {badge_cls}">{badge_lbl}</span>
                                <span class="sym-node-text">{safe_text}</span>
                                <span class="sym-ts">{stored_at}</span>
                            </div>
                            <div class="sym-meta-row">{chips_html}</div>
                            <div class="vector-viz">{vec_preview}</div>
                        </div>""", unsafe_allow_html=True)

                # Memory retrieval context
                mem_ctx = res.get("memory_context", [])
                if mem_ctx:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-label">🔎 Retrieved for This Run</div>', unsafe_allow_html=True)
                    for i, ctx in enumerate(mem_ctx):
                        safe_ctx = ctx.replace("&","&amp;").replace("<","&lt;")
                        st.markdown(f"""
                        <div class="rule-chip" style="margin-bottom:0.35rem;">
                            <span class="rule-num">M{i+1}</span>
                            <span style="flex:1;color:rgba(232,228,220,0.6);font-size:0.76rem;">{safe_ctx[:150]}</span>
                        </div>""", unsafe_allow_html=True)

        # ── Tab 5: Sources ────────────────────────────────────────────────────
        with tabs[4]:
            sources = res.get("sources", [])
            if sources:
                st.markdown('<div class="section-label">🌐 Research Sources</div>', unsafe_allow_html=True)
                for src in sources:
                    with st.expander(f"📖 {src.get('source_name','Source')} — {src.get('title','')}"):
                        st.write(src.get("context",""))
                        ref = src.get("reference","")
                        if ref and ref != "None":
                            st.markdown(f'<a class="ref-pill" href="{ref}" target="_blank">🔗 {ref[:65]}…</a>', unsafe_allow_html=True)
            else:
                st.info("No research sources used in this run.")

        # ── Tab 6: Raw JSON ───────────────────────────────────────────────────
        with tabs[5]:
            st.markdown('<div class="section-label">🗂 Full Pipeline Output</div>', unsafe_allow_html=True)
            display = {k: v for k, v in res.items() if k not in ("draft", "brain_records")}
            display["draft_preview"] = (res.get("draft","")[:500]+"…") if res.get("draft") else ""
            st.json(display)
            st.download_button("⬇ Download JSON",
                               data=json.dumps(res, indent=2, default=str),
                               file_name="pipeline_results.json",
                               mime="application/json", key="dl_json")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset", key="reset_results"):
            st.session_state.results = None
            st.session_state.brain_records = {}
            st.rerun()
