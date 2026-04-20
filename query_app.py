import re
import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from multiagent import build_agent_system




def safe_md(text: str) -> str:
    """Escape bare dollar signs so Streamlit doesn't render them as LaTeX."""
    return re.sub(r'\$(?=[\d\s(])', r'\\$', text)


def revision_controls(section_key: str, section_label: str, result_idx: int = -1):
    """
    Render a compact revision input + button for a given section.
    On submit, stores the pending revision in session state and triggers a rerun.
    """
    fb_key  = f"fb_{section_key}"
    btn_key = f"btn_revise_{section_key}"

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">Revise this section</div>',
        unsafe_allow_html=True,
    )
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        feedback = st.text_input(
            label="feedback",
            label_visibility="collapsed",
            placeholder=f"Describe what to change in the {section_label}...",
            key=fb_key,
        )
    with col_btn:
        pressed = st.button("Revise", key=btn_key, use_container_width=True)

    if pressed and feedback.strip():
        st.session_state["pending_revision"] = {
            "section":  section_key,
            "feedback": feedback.strip(),
        }
        st.rerun()


load_dotenv()

st.set_page_config(
    page_title="Portfolio AI",
    page_icon="📊",
    layout="wide"
)

st.write("APP STARTED")
st.write("MONGO_URI_USER:", os.getenv("MONGO_URI_USER") is not None)
st.write("MONGO_URI_ADMIN:", os.getenv("MONGO_URI_ADMIN") is not None)
st.write("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY") is not None)
st.write("CLAUDE_API_KEY:", os.getenv("CLAUDE_API_KEY") is not None)

# -------------------------
# Top Navigation Bar
# -------------------------

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Jost:wght@300;400;500&display=swap" rel="stylesheet">

    <style>
        header[data-testid="stHeader"] { display: none !important; }

        .navbar {
            display: flex;
            align-items: center;
            background-color: #f5f0e8;
            padding: 14px 32px;
            margin: -60px -4rem 32px -4rem;
            border-bottom: 1px solid #ddd5c4;
            gap: 32px;
        }
        .navbar-brand {
            font-family: 'Cormorant Garamond', serif;
            font-weight: 600;
            font-size: 17px;
            color: #2c2c2c;
            margin-right: auto;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .navbar a {
            font-family: 'Cormorant Garamond', serif;
            color: #7a6e60;
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            transition: color 0.2s;
        }
        .navbar a:hover { color: #2c2c2c; }
    </style>

    <div class="navbar">
        <span class="navbar-brand">Navigation</span>
        <a href="https://honte-search-app.streamlit.app/" target="_blank">Search</a>
    </div>
""", unsafe_allow_html=True)

# ============================================================
# Custom Styling
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=Jost:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Jost', sans-serif;
        background-color: #ffffff;
        color: #2c2c2c;
    }

    .main { background-color: #ffffff; }

    h1, h2, h3 {
        font-family: 'Cormorant Garamond', serif !important;
        color: #1a1a1a !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
    }

    p, .stMarkdown {
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
        color: #4a4a4a !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
    }

    /* Login inputs */
    .stTextInput label {
        font-family: 'Jost', sans-serif !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #7a6e60 !important;
    }

    .stTextInput input {
        background-color: #faf8f5 !important;
        color: #2c2c2c !important;
        border: 1px solid #ddd5c4 !important;
        border-radius: 2px !important;
        font-family: 'Jost', sans-serif !important;
        font-size: 15px !important;
        font-weight: 300 !important;
        box-shadow: none !important;
    }

    .stTextInput input:focus {
        border-color: #b8a99a !important;
        box-shadow: 0 0 0 1px #b8a99a !important;
    }

    /* Label above text area */
    .stTextArea label {
        font-family: 'Jost', sans-serif !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #7a6e60 !important;
    }

    /* Text area */
    .stTextArea textarea {
        background-color: #faf8f5 !important;
        color: #2c2c2c !important;
        border: 1px solid #ddd5c4 !important;
        border-radius: 2px !important;
        font-family: 'Jost', sans-serif !important;
        font-size: 15px !important;
        font-weight: 300 !important;
        box-shadow: none !important;
    }

    .stTextArea textarea:focus {
        border-color: #b8a99a !important;
        box-shadow: 0 0 0 1px #b8a99a !important;
    }

    /* Button */
    .stButton > button {
        background-color: #2c2c2c !important;
        color: #f5f0e8 !important;
        font-family: 'Jost', sans-serif !important;
        font-weight: 400 !important;
        font-size: 13px !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        border: none !important;
        border-radius: 1px !important;
        padding: 0.55rem 2.2rem !important;
        transition: background-color 0.2s !important;
    }

    .stButton > button p {
        color: #f5f0e8 !important;
    }

    .stButton > button:disabled {
        background-color: #2c2c2c !important;
        color: #f5f0e8 !important;
        opacity: 0.6 !important;
    }

    .stButton > button:hover { background-color: #4a4a4a !important; }

    /* Divider */
    hr {
        border: none !important;
        border-top: 1px solid #e8e2d9 !important;
        margin: 2rem 0 !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-family: 'Jost', sans-serif !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: #7a6e60 !important;
        background-color: #faf8f5 !important;
        border: 1px solid #e8e2d9 !important;
        border-radius: 1px !important;
    }

    .streamlit-expanderContent {
        background-color: #faf8f5 !important;
        border: 1px solid #e8e2d9 !important;
        border-top: none !important;
        padding: 1.5rem !important;
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
        font-size: 15px !important;
        line-height: 1.8 !important;
        color: #2c2c2c !important;
    }

    /* Newsletter output block */
    .newsletter-block {
        background-color: #faf8f5;
        border-left: 2px solid #c9a87a;
        padding: 2rem 2.5rem;
        margin-top: 1rem;
        font-size: 16px;
        line-height: 1.9;
        color: #2c2c2c;
        font-family: 'Jost', sans-serif;
        font-weight: 300;
    }

    /* Subheader label */
    .section-label {
        font-family: 'Jost', sans-serif;
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #9a8f82;
        margin-bottom: 0.5rem;
    }

    /* Warning / Alert */
    .stAlert {
        background-color: #faf8f5 !important;
        border: 1px solid #ddd5c4 !important;
        color: #7a6e60 !important;
        border-radius: 1px !important;
    }

    /* Spinner */
    .stSpinner > div { border-top-color: #c9a87a !important; }

    /* Session history item */
    .history-label {
        font-family: 'Jost', sans-serif;
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #9a8f82;
        margin-top: 1.5rem;
        margin-bottom: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Login
# ============================================================

from auth_helper import verify_login, is_authenticated

# ── Auth gate ──────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = None

if not is_authenticated(st.session_state):
    st.title("Portfolio Newsletter Generator")
    st.divider()

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = verify_login(username.strip(), password)
        if user:
            st.session_state.authenticated = True
            st.session_state.username = user["username"]
            st.session_state.role = user["role"]
            st.rerun()
        else:
            st.error("Incorrect username or password.")
    st.stop()

# ============================================================
# Orchestrator — shared across all users, created once
# Uses read-only MongoDB credentials via MONGO_URI in .env
# ============================================================

# @st.cache_resource
# def get_orchestrator():
#     return build_agent_system()

# orchestrator = get_orchestrator()

@st.cache_resource
def get_orchestrator():
    try:
        return build_agent_system()
    except Exception as e:
        st.error(f"Failed to initialize agent system: {e}")
        st.stop()

orchestrator = get_orchestrator()

# ============================================================
# Per-user session history
# Lost on tab close — nothing written to MongoDB
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# Header
# ============================================================

col1, col2 = st.columns([6, 1])
with col1:
    st.title("Portfolio Newsletter Generator")
    st.markdown("Generate portfolio commentary using internal research databases.")
with col2:
    st.markdown("<div style='padding-top: 1.8rem;'></div>", unsafe_allow_html=True)
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.history = []
        st.rerun()

# ============================================================
# Input
# ============================================================

query = st.text_area(
    "Enter Request",
    height=160,
    placeholder="Example: Draft a weekly performance summary..."
)

generate = st.button("Generate")

# ============================================================
# Execution — read only, no writes to MongoDB
# ============================================================

if generate and query.strip():
    with st.spinner("Running multi-agent analysis..."):
        result = asyncio.run(
            orchestrator.run_parallel(query)
        )

    # Save to session history (RAM only — private to this user)
    st.session_state.history.append({
        "query":   query,
        "result":  result,
    })

elif generate:
    st.warning("Please enter a request.")

# ============================================================
# Results — show most recent first
# ============================================================

# ============================================================
# Revision execution — runs before display so the updated
# result is shown immediately in the same render pass.
# ============================================================
if st.session_state.get("pending_revision") and st.session_state.history:
    rev = st.session_state.pop("pending_revision")
    current = st.session_state.history[-1]["result"]
    label_map = {
        "newsletter":  "Newsletter",
        "risk":        "Risk Analysis",
        "performance": "Portfolio Performance",
        "market":      "Market Context",
    }
    label = label_map.get(rev["section"], rev["section"].title())
    with st.spinner(f"Revising {label}..."):
        updated = asyncio.run(
            orchestrator.revise_section(rev["section"], rev["feedback"], current)
        )
    st.session_state.history[-1]["result"] = updated
    st.rerun()

if st.session_state.history:
    latest = st.session_state.history[-1]
    revisions = latest["result"].get("revisions", [])

    st.divider()

    # Revision badge — shows how many revisions have been applied
    if revisions:
        rev_summary = ", ".join(
            f"{r['section']} ×{sum(1 for x in revisions if x['section'] == r['section'])}"
            for r in {r['section']: r for r in revisions}.values()
        )
        st.caption(f"✏ {len(revisions)} revision(s) applied — {rev_summary}")

    st.subheader("Generated Newsletter")
    st.markdown(safe_md(latest["result"]["newsletter"]["newsletter"]))
    revision_controls("newsletter", "Newsletter")

    with st.expander("View Market Context Analysis"):
        st.markdown(safe_md(latest["result"]["market"]["analysis"]))
        revision_controls("market", "Market Context")

    with st.expander("View Portfolio Performance Analysis"):
        st.markdown(safe_md(latest["result"]["performance"]["analysis"]))
        revision_controls("performance", "Portfolio Performance")

    with st.expander("View Risk Analysis"):
        risk = latest["result"]["risk"]
        metrics = risk.get("metrics", "")
        if metrics:
            st.markdown(
                '<div class="section-label">Portfolio Metrics</div>',
                unsafe_allow_html=True,
            )
            st.code(metrics, language=None)
            st.divider()
        st.markdown(
            '<div class="section-label">Analysis</div>',
            unsafe_allow_html=True,
        )
        st.markdown(safe_md(risk["analysis"]))
        revision_controls("risk", "Risk Analysis")

    with st.expander("View Weekly Market Data Analysis"):
        st.markdown(safe_md(latest["result"]["weekly"]["analysis"]))

    # Debug: critique logs — hidden unless admin or debug mode
    if st.session_state.get("role") == "admin":
        with st.expander("Debug: Self-Critique Logs", expanded=False):
            for agent_key, label in [("performance", "Portfolio Performance"), ("newsletter", "Newsletter")]:
                logs = latest["result"][agent_key].get("critique_log", [])
                if logs:
                    st.markdown(f"**{label}** — {len(logs)} critique round(s)")
                    for i, log in enumerate(logs):
                        st.markdown(f"Round {i+1}: `{log[:200]}{'...' if len(log) > 200 else ''}`")
                else:
                    st.markdown(f"**{label}** — no critique log")

    # ── Session history — previous queries this session ────
    if len(st.session_state.history) > 1:
        st.divider()
        st.markdown(
            '<div class="history-label">Previous Queries This Session</div>',
            unsafe_allow_html=True
        )
        for item in reversed(st.session_state.history[:-1]):
            with st.expander(item["query"][:80]):
                st.markdown(safe_md(item["result"]["newsletter"]["newsletter"]))

                with st.expander("Market Context"):
                    st.markdown(safe_md(item["result"]["market"]["analysis"]))
                with st.expander("Portfolio Performance"):
                    st.markdown(safe_md(item["result"]["performance"]["analysis"]))
                with st.expander("Risk Analysis"):
                    _risk = item["result"]["risk"]
                    _metrics = _risk.get("metrics", "")
                    if _metrics:
                        st.code(_metrics, language=None)
                        st.divider()
                    st.markdown(safe_md(_risk["analysis"]))
                with st.expander("Weekly Market Data"):
                    st.markdown(safe_md(item["result"]["weekly"]["analysis"]))