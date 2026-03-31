import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from multiagent import build_agent_system

load_dotenv()

st.set_page_config(
    page_title="Portfolio AI",
    page_icon="📊",
    layout="wide"
)

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

@st.cache_resource
def get_orchestrator():
    return build_agent_system()

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

if st.session_state.history:
    latest = st.session_state.history[-1]

    st.divider()

    st.subheader("Generated Newsletter")
    st.markdown(latest["result"]["newsletter"]["newsletter"])

    with st.expander("View Market Context Analysis"):
        st.markdown(latest["result"]["market"]["analysis"])

    with st.expander("View Portfolio Performance Analysis"):
        st.markdown(latest["result"]["performance"]["analysis"])

    with st.expander("View Risk Analysis"):
        st.markdown(latest["result"]["risk"]["analysis"])

    with st.expander("View Weekly Market Data Analysis"):
        st.markdown(latest["result"]["weekly"]["analysis"])

    # ── Session history — previous queries this session ────
    if len(st.session_state.history) > 1:
        st.divider()
        st.markdown(
            '<div class="history-label">Previous Queries This Session</div>',
            unsafe_allow_html=True
        )
        for item in reversed(st.session_state.history[:-1]):
            with st.expander(item["query"][:80]):
                st.markdown(item["result"]["newsletter"]["newsletter"])

                with st.expander("Market Context"):
                    st.markdown(item["result"]["market"]["analysis"])
                with st.expander("Portfolio Performance"):
                    st.markdown(item["result"]["performance"]["analysis"])
                with st.expander("Risk Analysis"):
                    st.markdown(item["result"]["risk"]["analysis"])
                with st.expander("Weekly Market Data"):
                    st.markdown(item["result"]["weekly"]["analysis"])