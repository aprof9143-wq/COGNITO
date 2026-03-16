"""
NeuroSymbolic Verifier — Streamlit App
A beautiful UI for the LTN + LLM constraint verification pipeline.
"""
import streamlit as st
import sys
import os
import json
import time
import textwrap

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroSymbolic Verifier",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Globals ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0c0e14 !important;
    color: #e8e4dc !important;
}

.stApp {
    background: #0c0e14;
}

/* ── Header ── */
.nsv-header {
    text-align: center;
    padding: 3.5rem 0 2rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 2.5rem;
}
.nsv-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    font-weight: 400;
    letter-spacing: -0.02em;
    color: #f0ebe0;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.nsv-logo span {
    color: #c8a96e;
    font-style: italic;
}
.nsv-subtitle {
    font-size: 0.875rem;
    font-weight: 300;
    color: rgba(232,228,220,0.45);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #c8a96e;
    font-weight: 600;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(200,169,110,0.2);
}

/* ── Glass panels ── */
.glass-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(12px);
}

/* ── Rule chip row ── */
.rules-container {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}
.rule-chip {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    background: rgba(200,169,110,0.08);
    border: 1px solid rgba(200,169,110,0.2);
    border-radius: 10px;
    padding: 0.55rem 0.9rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #e8e4dc;
    animation: fadeSlide 0.3s ease forwards;
}
.rule-chip .rule-num {
    background: rgba(200,169,110,0.25);
    color: #c8a96e;
    border-radius: 5px;
    padding: 0.1rem 0.4rem;
    font-size: 0.72rem;
    font-weight: 600;
    min-width: 22px;
    text-align: center;
}
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(-6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Mode selector ── */
.stRadio > div {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
}
.stRadio label {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    padding: 0.5rem 1rem !important;
    cursor: pointer;
    font-size: 0.85rem;
    transition: all 0.2s;
    color: #e8e4dc !important;
}
.stRadio label:has(input:checked) {
    background: rgba(200,169,110,0.15) !important;
    border-color: rgba(200,169,110,0.5) !important;
    color: #c8a96e !important;
}

/* ── Text areas ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    resize: vertical;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus {
    border-color: rgba(200,169,110,0.5) !important;
    box-shadow: 0 0 0 3px rgba(200,169,110,0.08) !important;
}

/* ── Text input (API key etc.) ── */
.stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8e4dc !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}
.stTextInput input:focus {
    border-color: rgba(200,169,110,0.5) !important;
    box-shadow: 0 0 0 3px rgba(200,169,110,0.08) !important;
}

/* ── Run button ── */
.stButton > button {
    background: linear-gradient(135deg, #c8a96e 0%, #a8854a 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #0c0e14 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 2rem !important;
    letter-spacing: 0.03em;
    transition: opacity 0.2s, transform 0.15s !important;
    width: 100%;
    box-shadow: 0 4px 20px rgba(200,169,110,0.25);
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Score gauge ── */
.score-ring-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1.5rem 0;
}
.score-big {
    font-family: 'DM Serif Display', serif;
    font-size: 4rem;
    line-height: 1;
    font-weight: 400;
    margin-bottom: 0.3rem;
}
.score-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(232,228,220,0.4);
}
.score-pass { color: #6dcea8; }
.score-warn { color: #e8c06d; }
.score-fail { color: #e8736d; }

/* ── Audit result cards ── */
.audit-card {
    border-radius: 14px;
    border: 1px solid;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.75rem;
    transition: transform 0.15s;
}
.audit-card:hover {
    transform: translateX(3px);
}
.audit-card.pass {
    background: rgba(109,206,168,0.05);
    border-color: rgba(109,206,168,0.2);
}
.audit-card.fail {
    background: rgba(232,115,109,0.06);
    border-color: rgba(232,115,109,0.25);
}
.audit-card .ac-header {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.55rem;
}
.audit-card .ac-badge {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    letter-spacing: 0.06em;
    font-family: 'DM Mono', monospace;
    flex-shrink: 0;
}
.audit-card.pass .ac-badge {
    background: rgba(109,206,168,0.2);
    color: #6dcea8;
}
.audit-card.fail .ac-badge {
    background: rgba(232,115,109,0.2);
    color: #e8736d;
}
.audit-card .ac-rule {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #e8e4dc;
    flex: 1;
}
.audit-meta {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
    margin-top: 0.6rem;
}
.audit-meta-item {
    background: rgba(255,255,255,0.03);
    border-radius: 8px;
    padding: 0.4rem 0.6rem;
}
.audit-meta-item .amk {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(232,228,220,0.35);
    margin-bottom: 0.15rem;
}
.audit-meta-item .amv {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #e8e4dc;
}
.audit-explanation {
    font-size: 0.82rem;
    color: rgba(232,228,220,0.6);
    margin-top: 0.6rem;
    line-height: 1.55;
    border-top: 1px solid rgba(255,255,255,0.05);
    padding-top: 0.55rem;
}
.domain-warn {
    font-size: 0.78rem;
    color: #e8c06d;
    margin-top: 0.4rem;
    background: rgba(232,192,109,0.08);
    padding: 0.35rem 0.6rem;
    border-radius: 6px;
    border-left: 3px solid #e8c06d;
}

/* ── Generated output box ── */
.gen-output {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    font-size: 0.88rem;
    line-height: 1.75;
    color: #d8d3c8;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: 'DM Sans', sans-serif;
    max-height: 420px;
    overflow-y: auto;
}
.gen-output::-webkit-scrollbar { width: 4px; }
.gen-output::-webkit-scrollbar-track { background: transparent; }
.gen-output::-webkit-scrollbar-thumb { background: rgba(200,169,110,0.25); border-radius: 4px; }

/* ── Source reference pills ── */
.ref-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(200,169,110,0.08);
    border: 1px solid rgba(200,169,110,0.18);
    border-radius: 20px;
    padding: 0.3rem 0.75rem;
    font-size: 0.76rem;
    color: #c8a96e;
    text-decoration: none;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}

/* ── Pipeline flow diagram ── */
.pipe-flow {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 1.2rem 0;
    flex-wrap: wrap;
}
.pipe-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.3rem;
    padding: 0 0.5rem;
}
.pipe-node .pn-icon {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.1);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    background: rgba(255,255,255,0.04);
}
.pipe-node .pn-label {
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(232,228,220,0.4);
    text-align: center;
    max-width: 60px;
}
.pipe-arrow {
    color: rgba(200,169,110,0.4);
    font-size: 1rem;
    padding: 0 0.25rem;
}
.pipe-node.active .pn-icon {
    border-color: rgba(200,169,110,0.5);
    background: rgba(200,169,110,0.12);
}
.pipe-node.active .pn-label { color: #c8a96e; }

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 1.5rem 0;
}

/* ── Selectbox ── */
.stSelectbox select,
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8e4dc !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #c8a96e, #6dcea8) !important;
    border-radius: 4px;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    color: #e8e4dc !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: rgba(232,228,220,0.45) !important;
    border-bottom: 2px solid transparent !important;
    font-size: 0.85rem;
    padding: 0.6rem 1.2rem;
    font-family: 'DM Sans', sans-serif;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    color: #c8a96e !important;
    border-bottom-color: #c8a96e !important;
    background: transparent !important;
}

/* ── Info / warning / success boxes ── */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
    font-size: 0.85rem;
}

/* ── Scrollbar global ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(200,169,110,0.2); border-radius: 4px; }

/* ── Metric ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem !important;
    color: #e8e4dc;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(232,228,220,0.4);
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #c8a96e !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nsv-header">
  <div class="nsv-logo">Neuro<span>Symbolic</span> Verifier</div>
  <div class="nsv-subtitle">LTN · LLM · Vector Memory · Agentic Research</div>
</div>
""", unsafe_allow_html=True)

# ── Pipeline flow ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="glass-panel" style="margin-bottom:2rem;">
  <div class="pipe-flow">
    <div class="pipe-node">
      <div class="pn-icon">📝</div>
      <div class="pn-label">Rules &amp; Prompt</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node">
      <div class="pn-icon">🌐</div>
      <div class="pn-label">M4 Research</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node">
      <div class="pn-icon">🧩</div>
      <div class="pn-label">M2 Parser</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node">
      <div class="pn-icon">💾</div>
      <div class="pn-label">M3 Memory</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node">
      <div class="pn-icon">✍️</div>
      <div class="pn-label">Draft Gen</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node">
      <div class="pn-icon">🔍</div>
      <div class="pn-label">M2 Audit</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node">
      <div class="pn-icon">⚖️</div>
      <div class="pn-label">M1 LTN</div>
    </div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node">
      <div class="pn-icon">📊</div>
      <div class="pn-label">Verdict</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "rules" not in st.session_state:
    st.session_state.rules = []
if "rule_input" not in st.session_state:
    st.session_state.rule_input = ""
if "results" not in st.session_state:
    st.session_state.results = None

# ── Layout: 2 columns ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.15], gap="large")

# ════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — Inputs
# ════════════════════════════════════════════════════════════════════════════
with col_left:

    # ── API Key ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">🔑 Gemini API Key</div>', unsafe_allow_html=True)
    api_key = st.text_input(
        label="api_key",
        label_visibility="collapsed",
        type="password",
        placeholder="AIza...",
        key="api_key_input"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Mode ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">⚡ Pipeline Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "mode",
        label_visibility="collapsed",
        options=["🔬 Full Pipeline", "📐 Rules + Audit Only", "🌐 Research + Generate"],
        horizontal=True,
        key="pipeline_mode"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── User Prompt ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">💬 Generation Prompt</div>', unsafe_allow_html=True)
    user_prompt = st.text_area(
        label="prompt",
        label_visibility="collapsed",
        placeholder="e.g. Write a weekly study plan to improve SAT Math scores from 600 to 750 in 8 weeks.",
        height=110,
        key="user_prompt"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pre-existing Draft (optional) ────────────────────────────────────────
    st.markdown('<div class="section-label">📄 Existing Draft (optional)</div>', unsafe_allow_html=True)
    existing_draft = st.text_area(
        label="draft",
        label_visibility="collapsed",
        placeholder="Paste an existing draft here to audit it directly — or leave blank to generate one from the prompt.",
        height=120,
        key="existing_draft"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Rule Input — Claude-style ─────────────────────────────────────────────
    st.markdown('<div class="section-label">📏 Constraint Rules</div>', unsafe_allow_html=True)

    def _add_rule_from_input():
        raw = st.session_state.get("rule_text_input", "").strip()
        if raw:
            for line in raw.split("\n"):
                line = line.strip().lstrip("-•*").strip()
                if line and line not in st.session_state.rules:
                    st.session_state.rules.append(line)
            st.session_state.rule_text_input = ""

    rule_input_val = st.text_area(
        label="rules_input",
        label_visibility="collapsed",
        placeholder=(
            "Type a rule and press Add — one per line, or multiple at once:\n"
            "  • Study sessions must be ≤ 2 hours each\n"
            "  • Weekly practice tests ≥ 2\n"
            "  • Total weekly hours ≤ 14"
        ),
        height=130,
        key="rule_text_input"
    )

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("＋ Add Rule(s)", key="add_rule_btn"):
            _add_rule_from_input()
            st.rerun()
    with btn_col2:
        if st.button("✕ Clear All Rules", key="clear_rules_btn"):
            st.session_state.rules = []
            st.rerun()

    # Display added rules as chips
    if st.session_state.rules:
        chips_html = '<div class="rules-container" style="margin-top:0.8rem;">'
        for i, r in enumerate(st.session_state.rules):
            chips_html += f'''
            <div class="rule-chip">
                <span class="rule-num">R{i+1}</span>
                <span style="flex:1">{r}</span>
            </div>'''
        chips_html += '</div>'
        st.markdown(chips_html, unsafe_allow_html=True)

        # Per-rule delete
        with st.expander("🗑 Remove individual rules"):
            for i, r in enumerate(list(st.session_state.rules)):
                if st.button(f"Remove: {r[:55]}{'…' if len(r)>55 else ''}", key=f"del_rule_{i}"):
                    st.session_state.rules.pop(i)
                    st.rerun()
    else:
        st.markdown("""
        <div style="text-align:center;color:rgba(232,228,220,0.25);
                    font-size:0.82rem;padding:1.2rem;border:1px dashed rgba(255,255,255,0.07);
                    border-radius:12px;margin-top:0.5rem;">
            No rules added yet — type above and click Add
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Run button ────────────────────────────────────────────────────────────
    run_btn = st.button("⚡ Run Pipeline", key="run_pipeline_btn")


# ════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Results (or idle placeholder)
# ════════════════════════════════════════════════════════════════════════════
with col_right:

    if not run_btn and st.session_state.results is None:
        # ── Idle state ────────────────────────────────────────────────────────
        st.markdown("""
        <div class="glass-panel" style="text-align:center;padding:3rem 2rem;">
            <div style="font-size:3rem;margin-bottom:1rem;opacity:0.4;">⚖️</div>
            <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;
                        color:rgba(232,228,220,0.5);margin-bottom:0.5rem;">
                Awaiting verification
            </div>
            <div style="font-size:0.82rem;color:rgba(232,228,220,0.25);line-height:1.6;">
                Configure your prompt, rules, and API key<br>
                then press <strong style="color:rgba(200,169,110,0.5)">Run Pipeline</strong> to begin.
            </div>
        </div>
        """, unsafe_allow_html=True)

    if run_btn:
        # ── Validation ────────────────────────────────────────────────────────
        if not api_key:
            st.error("Please enter your Gemini API key.")
            st.stop()
        if not user_prompt.strip() and not existing_draft.strip():
            st.error("Please enter a prompt or paste an existing draft.")
            st.stop()

        # ── Import modules ────────────────────────────────────────────────────
        # Add upload directory to sys.path so modules can be imported
        uploads_path = "/mnt/user-data/uploads"
        if uploads_path not in sys.path:
            sys.path.insert(0, uploads_path)

        try:
            import m2_llm_parser as m2
            import m3_vector_db as m3
        except ImportError as e:
            st.error(f"Could not import pipeline modules: {e}")
            st.stop()

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

        # ── LTN availability warning ──────────────────────────────────────────
        if not has_ltn:
            st.warning("⚠️  LTN (ltn / tensorflow) not installed — skipping M1 scoring. Install with `pip install ltn tensorflow`.")

        # ── Progress placeholders ─────────────────────────────────────────────
        prog_bar   = st.progress(0, text="Initialising…")
        status_box = st.empty()

        results = {}

        # ════════════════════════════════════════════════════════════
        # STEP 1 — Research (M4)
        # ════════════════════════════════════════════════════════════
        source_results = []
        if has_m4 and mode in ["🔬 Full Pipeline", "🌐 Research + Generate"]:
            prog_bar.progress(10, text="🌐 Researching topic…")
            status_box.info("Module 4 — Searching Wikipedia & DuckDuckGo for domain context…")
            try:
                source_results = m4.research_all_sources(user_prompt, api_key=api_key)
                results["sources"] = source_results
                status_box.empty()
            except Exception as e:
                st.warning(f"M4 research failed (non-fatal): {e}")
                source_results = []

        # ════════════════════════════════════════════════════════════
        # STEP 2 — Parse rules (M2)
        # ════════════════════════════════════════════════════════════
        structured_rules = []
        rules_to_parse = st.session_state.rules

        if rules_to_parse:
            prog_bar.progress(25, text="🧩 Parsing rules into structured constraints…")
            status_box.info(f"Module 2 — Parsing {len(rules_to_parse)} rule(s) into formal constraints…")
            try:
                for r in rules_to_parse:
                    parsed = m2.parse_rule_to_constraint(r, api_key)
                    structured_rules.append(parsed)
                results["structured_rules"] = structured_rules
                status_box.empty()
            except Exception as e:
                st.error(f"Rule parsing failed: {e}")
                st.stop()
        else:
            st.info("No rules provided — running generation + audit without constraints.")

        # ════════════════════════════════════════════════════════════
        # STEP 3 — Store in ChromaDB (M3)
        # ════════════════════════════════════════════════════════════
        collection     = None
        stored_ids     = []
        memory_context = []

        if mode in ["🔬 Full Pipeline", "🌐 Research + Generate"]:
            prog_bar.progress(38, text="💾 Loading semantic memory…")
            status_box.info("Module 3 — Storing rules & context in ChromaDB…")
            try:
                collection = m3.setup_memory()

                # Store source context
                for src in source_results:
                    m3.store_knowledge(collection, f"source_{src.get('source_name','x')}", src["context"])

                # Store structured rules
                if structured_rules:
                    stored_ids = m3.store_all_rules(collection, structured_rules)

                # Retrieve relevant context
                query = user_prompt or existing_draft
                memory_context = m3.retrieve_context(collection, query, n_results=4)
                results["memory_context"] = memory_context
                status_box.empty()
            except Exception as e:
                st.warning(f"ChromaDB unavailable (non-fatal): {e}")

        # ════════════════════════════════════════════════════════════
        # STEP 4 — Generate draft (M2 / Gemini)
        # ════════════════════════════════════════════════════════════
        draft_text = existing_draft.strip()

        if not draft_text and user_prompt.strip():
            prog_bar.progress(52, text="✍️ Generating draft…")
            status_box.info("Generating content with Gemini…")
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                gen_model = genai.GenerativeModel("gemini-2.5-flash")

                # Build system context
                ctx_parts = []
                if source_results:
                    for src in source_results:
                        ctx_parts.append(
                            f"[{src.get('source_name','Source')}] {src['context']}"
                        )
                if memory_context:
                    ctx_parts.append("Relevant constraints:\n" + "\n".join(memory_context))

                rule_list = ""
                if structured_rules:
                    rule_list = "\n\nYou MUST strictly obey these constraints:\n" + "\n".join(
                        f"  • {r.get('display', r.get('original',''))}" for r in structured_rules
                    )

                full_prompt = (
                    ("CONTEXT:\n" + "\n\n".join(ctx_parts) + "\n\n" if ctx_parts else "")
                    + f"TASK: {user_prompt}"
                    + rule_list
                )

                response   = gen_model.generate_content(full_prompt)
                draft_text = response.text
                results["draft"] = draft_text
                status_box.empty()
            except Exception as e:
                st.error(f"Draft generation failed: {e}")
                st.stop()
        else:
            results["draft"] = draft_text

        # ════════════════════════════════════════════════════════════
        # STEP 5 — Audit (M2)
        # ════════════════════════════════════════════════════════════
        audit_results = []

        if structured_rules and draft_text and mode in ["🔬 Full Pipeline", "📐 Rules + Audit Only"]:
            prog_bar.progress(70, text="🔍 Auditing constraints…")
            status_box.info(f"Module 2 — Auditing {len(structured_rules)} rule(s) against draft…")
            try:
                audit_results = m2.structured_audit(draft_text, structured_rules, api_key)
                results["audit"] = audit_results
                status_box.empty()
            except Exception as e:
                st.error(f"Audit failed: {e}")
                st.stop()

        # ════════════════════════════════════════════════════════════
        # STEP 6 — LTN verification (M1)
        # ════════════════════════════════════════════════════════════
        ltn_score  = None
        violations = []

        if has_ltn and audit_results:
            prog_bar.progress(88, text="⚖️ Running LTN verification…")
            status_box.info("Module 1 — Logic Tensor Network scoring…")
            try:
                ltn_score, violations = m1.verify_and_report(audit_results)
                results["ltn_score"]  = ltn_score
                results["violations"] = violations
                status_box.empty()
            except Exception as e:
                st.warning(f"LTN scoring failed (non-fatal): {e}")

        prog_bar.progress(100, text="✅ Done!")
        time.sleep(0.4)
        prog_bar.empty()
        status_box.empty()

        st.session_state.results = results

    # ── Render stored results ─────────────────────────────────────────────────
    if st.session_state.results:
        res = st.session_state.results
        tabs = st.tabs(["📊 Verdict", "📄 Draft", "🔍 Audit Detail", "🌐 Sources", "🗂 Raw JSON"])

        # ── TAB 1: Verdict ────────────────────────────────────────────────────
        with tabs[0]:
            audit  = res.get("audit", [])
            score  = res.get("ltn_score")
            viols  = res.get("violations", [])
            passed = sum(1 for r in audit if r.get("satisfies"))
            failed = len(audit) - passed

            if score is not None:
                score_class = "score-pass" if score >= 0.8 else ("score-warn" if score >= 0.5 else "score-fail")
                verdict_text = "PASS ✓" if score >= 0.8 else ("MARGINAL ⚠" if score >= 0.5 else "FAIL ✗")
                st.markdown(f"""
                <div class="glass-panel" style="text-align:center;padding:2rem;">
                    <div class="score-big {score_class}">{score:.3f}</div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.9rem;
                                color:{'#6dcea8' if score>=0.8 else ('#e8c06d' if score>=0.5 else '#e8736d')};
                                font-weight:600;letter-spacing:0.1em;margin-top:0.3rem;">
                        {verdict_text}
                    </div>
                    <div class="score-label" style="margin-top:0.4rem;">LTN Universal Verification Score</div>
                </div>
                """, unsafe_allow_html=True)

            # Metrics row
            if audit:
                m1c, m2c, m3c, m4c = st.columns(4)
                with m1c:
                    st.metric("Rules Checked", len(audit))
                with m2c:
                    st.metric("Passed ✅", passed)
                with m3c:
                    st.metric("Failed ❌", failed)
                with m4c:
                    pct = int(100 * passed / len(audit)) if audit else 0
                    st.metric("Pass Rate", f"{pct}%")

            elif not audit and res.get("draft"):
                st.success("Draft generated successfully. No rules were audited.")
            else:
                st.info("No audit results — add rules and run again.")

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
                    </div>
                    """, unsafe_allow_html=True)

        # ── TAB 2: Draft ──────────────────────────────────────────────────────
        with tabs[1]:
            draft = res.get("draft", "")
            if draft:
                st.markdown('<div class="section-label">✍ Generated / Audited Content</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="gen-output">{draft}</div>', unsafe_allow_html=True)
                st.download_button(
                    "⬇ Download Draft",
                    data=draft,
                    file_name="draft_output.txt",
                    mime="text/plain",
                    key="dl_draft"
                )
            else:
                st.info("No draft available.")

        # ── TAB 3: Audit Detail ───────────────────────────────────────────────
        with tabs[2]:
            audit = res.get("audit", [])
            if audit:
                st.markdown('<div class="section-label">🔍 Per-Rule Audit Results</div>',
                            unsafe_allow_html=True)
                for r in audit:
                    status   = "pass" if r.get("satisfies") else "fail"
                    badge    = "PASS ✅" if r.get("satisfies") else "FAIL ❌"
                    sym_tag  = ' <span style="font-size:0.7rem;color:rgba(200,169,110,0.6)">[symbolic]</span>' \
                               if r.get("symbolic_check_used") else ""
                    dw       = r.get("domain_warning", "")
                    expl     = r.get("explanation", "No explanation.")
                    ext_raw  = r.get("extracted_value_raw", "N/A")
                    scope    = r.get("scope", "—")
                    rule_id  = r.get("rule_id", "")

                    dw_html = f'<div class="domain-warn">{dw}</div>' if dw else ""
                    st.markdown(f"""
                    <div class="audit-card {status}">
                        <div class="ac-header">
                            <span class="ac-badge">{badge}</span>
                            <span class="ac-rule">R{rule_id+1 if isinstance(rule_id,int) else rule_id} — {r.get('rule_display','')}</span>
                            {sym_tag}
                        </div>
                        <div class="audit-meta">
                            <div class="audit-meta-item">
                                <div class="amk">Extracted</div>
                                <div class="amv">{ext_raw}</div>
                            </div>
                            <div class="audit-meta-item">
                                <div class="amk">Scope</div>
                                <div class="amv">{scope.upper()}</div>
                            </div>
                            <div class="audit-meta-item">
                                <div class="amk">Premise → Conclusion</div>
                                <div class="amv">{r.get('premise_confidence',1.0):.2f} → {r.get('conclusion_confidence',0):.2f}</div>
                            </div>
                        </div>
                        {dw_html}
                        <div class="audit-explanation">{expl}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Show structured rule details in expander
                with st.expander("🗂 Structured Rule Constraints"):
                    for sr in res.get("structured_rules", []):
                        st.json(sr)
            else:
                st.info("No audit results. Add rules and run the pipeline.")

        # ── TAB 4: Sources ────────────────────────────────────────────────────
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
                                f'<a class="ref-pill" href="{ref}" target="_blank">🔗 {ref[:60]}…</a>',
                                unsafe_allow_html=True
                            )
            else:
                st.info("No research sources used in this run.")

            if mem_ctx:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">💾 Retrieved Memory Context</div>',
                            unsafe_allow_html=True)
                for i, ctx in enumerate(mem_ctx):
                    st.markdown(f"""
                    <div class="rule-chip" style="margin-bottom:0.4rem;">
                        <span class="rule-num">M{i+1}</span>
                        <span style="flex:1;color:rgba(232,228,220,0.7);font-size:0.8rem;">{ctx}</span>
                    </div>""", unsafe_allow_html=True)

        # ── TAB 5: Raw JSON ───────────────────────────────────────────────────
        with tabs[4]:
            st.markdown('<div class="section-label">🗂 Full Pipeline Output (JSON)</div>',
                        unsafe_allow_html=True)
            # Truncate draft for readability
            display_res = {k: v for k, v in res.items() if k != "draft"}
            display_res["draft_preview"] = (res.get("draft", "")[:500] + "…") if res.get("draft") else ""
            st.json(display_res)
            full_json = json.dumps(res, indent=2, default=str)
            st.download_button(
                "⬇ Download Full JSON",
                data=full_json,
                file_name="pipeline_results.json",
                mime="application/json",
                key="dl_json"
            )

        # ── Clear results ─────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset Results", key="reset_results"):
            st.session_state.results = None
            st.rerun()
