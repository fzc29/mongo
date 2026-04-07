import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
)
from langchain_core.documents import Document
from user_management import create_user, delete_user, list_users, change_role

# ── All data management functions from build_index ─────────
from build import (
    delete_by_source,
    deduplicate,
    list_sources,
    get_collection,
    embedding,
    COLLECTIONS,
)

from auth_helper import verify_login, is_authenticated, is_admin

load_dotenv()

st.set_page_config(
    page_title="RAG Admin",
    page_icon=None,
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
        <a href="https://honte-pnl-query.streamlit.app/" target="_blank">PnL Query</a>
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

    p, .stMarkdown p {
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
        color: #4a4a4a !important;
        font-size: 15px !important;
    }

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

    .stFileUploader label {
        font-family: 'Jost', sans-serif !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #7a6e60 !important;
    }

    .stFileUploader > div {
        background-color: #faf8f5 !important;
        border: 1px dashed #ddd5c4 !important;
        border-radius: 2px !important;
    }

    .stSelectbox label {
        font-family: 'Jost', sans-serif !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #7a6e60 !important;
    }

    .stSelectbox > div > div {
        background-color: #faf8f5 !important;
        border: 1px solid #ddd5c4 !important;
        border-radius: 2px !important;
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
        color: #2c2c2c !important;
    }

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

    .stButton > button p { color: #f5f0e8 !important; }
    .stButton > button:hover { background-color: #4a4a4a !important; }

    .stSuccess {
        background-color: #f5f5f0 !important;
        border: 1px solid #c9a87a !important;
        border-left: 3px solid #c9a87a !important;
        color: #2c2c2c !important;
        border-radius: 1px !important;
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
    }

    .stInfo {
        background-color: #faf8f5 !important;
        border: 1px solid #ddd5c4 !important;
        border-left: 3px solid #b8a99a !important;
        color: #2c2c2c !important;
        border-radius: 1px !important;
        font-family: 'Jost', sans-serif !important;
        font-weight: 300 !important;
    }

    .stWarning {
        background-color: #faf8f5 !important;
        border: 1px solid #ddd5c4 !important;
        color: #7a6e60 !important;
        border-radius: 1px !important;
    }

    .stError {
        background-color: #fdf5f5 !important;
        border: 1px solid #e8c4c4 !important;
        color: #8a4a4a !important;
        border-radius: 1px !important;
    }

    .stSpinner > div { border-top-color: #c9a87a !important; }

    hr {
        border: none !important;
        border-top: 1px solid #e8e2d9 !important;
        margin: 2rem 0 !important;
    }

    .stats-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid #e8e2d9;
        font-family: 'Jost', sans-serif;
        font-size: 14px;
        font-weight: 300;
        color: #2c2c2c;
    }
    .stats-label {
        font-weight: 500;
        letter-spacing: 0.04em;
        color: #4a4a4a;
    }
    .stats-count {
        font-family: 'Cormorant Garamond', serif;
        font-size: 18px;
        font-weight: 600;
        color: #c9a87a;
    }
    .source-item {
        font-family: 'Jost', sans-serif;
        font-size: 13px;
        font-weight: 300;
        color: #4a4a4a;
        padding: 0.4rem 0;
        border-bottom: 1px solid #f0ebe3;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Auth — admin role only
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = None

if not is_authenticated(st.session_state):
    st.title("Admin View")
    st.divider()

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = verify_login(username.strip(), password)
        if user and user["role"] == "admin":
            st.session_state.authenticated = True
            st.session_state.username = user["username"]
            st.session_state.role = user["role"]
            st.rerun()
        elif user:
            st.error("You do not have admin permissions.")
        else:
            st.error("Incorrect username or password.")
    st.stop()

# ============================================================
# Header
# ============================================================

col1, col2 = st.columns([6, 1])
with col1:
    st.title("Admin Panel")
    st.markdown(f"Logged in as **{st.session_state.username}**")
with col2:
    st.markdown("<div style='padding-top: 1.8rem;'></div>", unsafe_allow_html=True)
    if st.button("Logout"):
        for key in ["authenticated", "username", "role"]:
            st.session_state[key] = None
        st.session_state.authenticated = False
        st.rerun()

st.divider()

# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Upload & Index",
    "Delete Documents",
    "Deduplicate",
    "Browse Sources",
    "Database Stats",
    "Verify Embeddings",
    "User Management",
])


# ============================================================
# TAB 1 — Upload & Index
# ============================================================

with tab1:
    st.subheader("Upload & Index Documents")
    st.markdown("Documents indexed here are immediately available to all users.")

    if "upload_category" not in st.session_state:
        st.session_state.upload_category = None
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    store_type = st.selectbox(
        "Select Knowledge Base",
        list(COLLECTIONS.keys()),
        key="upload_collection",
    )

    if st.session_state.upload_category != store_type:
        st.session_state.upload_category = store_type
        st.session_state.uploader_key += 1 

    uploaded_files = st.file_uploader(
        "Upload Doc (PDF, Markdown, CSV)",
        type=["pdf", "md", "csv"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    collection_name, index_name = COLLECTIONS[store_type]
    st.info(f"Files will be added to **{collection_name}**.")

    upload_mode = st.radio(
        "Upload mode",
        ["Add new (skip if exists)", "Reindex (replace existing)"],
        help="Reindex deletes existing chunks for this file first, then re-ingests.",
        key="upload_mode",
    )

    if uploaded_files and st.button("Index Document", key="btn_upload"):
        all_chunks = []
        errors = []

        with st.spinner("Processing and embedding..."):
            for uploaded_file in uploaded_files:
                suffix = Path(uploaded_file.name).suffix

                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_path = Path(tmp.name)

                try:
                    # If reindex mode, remove existing chunks first
                    if "Reindex" in upload_mode:
                        deleted = delete_by_source(
                            uploaded_file.name, collection_name
                        )
                        if deleted:
                            st.write(
                                f"Removed {deleted} existing chunks "
                                f"for `{uploaded_file.name}`"
                            )

                    # Load file
                    ext = temp_path.suffix.lower()
                    if ext == ".pdf":
                        docs = PyPDFLoader(str(temp_path)).load()
                    elif ext == ".md":
                        text = temp_path.read_text(encoding="utf-8")
                        docs = [Document(page_content=text, metadata={"source": uploaded_file.name})]
                    elif ext == ".csv":
                        docs = CSVLoader(str(temp_path)).load()
                    else:
                        raise ValueError(f"Unsupported file type: {ext}")

                    # Tag with original filename and uploader
                    for doc in docs:
                        doc.metadata = doc.metadata or {} 
                        doc.metadata["source"] = uploaded_file.name           # overwrite temp path
                        doc.metadata["original_filename"] = uploaded_file.name
                        doc.metadata["uploaded_by"] = st.session_state.username
                        doc.metadata["collection"] = store_type

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=200
                    )
                    chunks = splitter.split_documents(docs)
                    all_chunks.extend(chunks)
                    st.write(f"`{uploaded_file.name}` — {len(chunks)} chunks")

                except Exception as e:
                    errors.append(f"{uploaded_file.name}: {str(e)}")
                    st.warning(f"Error: `{uploaded_file.name}` — {str(e)}")

                finally:
                    temp_path.unlink(missing_ok=True)

            if all_chunks:
                MongoDBAtlasVectorSearch.from_documents(
                    documents=all_chunks,
                    embedding=embedding,
                    collection=get_collection(collection_name),
                    index_name=index_name,
                )
                st.info(f"Written to **{collection_name}** in MongoDB.")

        if all_chunks:
            st.success(
                f"{len(all_chunks)} chunks indexed into `{store_type}`. "
                f"All users can access this data immediately."
            )
        if errors:
            st.error(f"Failed: {', '.join(errors)}")


# ============================================================
# TAB 2 — Delete Documents
# ============================================================

with tab2:
    st.subheader("Delete Documents")
    st.markdown("Remove all chunks belonging to a specific source file.")

    del_collection = st.selectbox(
        "Knowledge Base",
        list(COLLECTIONS.keys()),
        key="del_collection",
    )
    del_col_name, _ = COLLECTIONS[del_collection]
    
    if st.button("Load Sources", key="btn_load_del_sources"):
        with st.spinner("Loading sources..."):
            sources = list_sources(del_col_name, silent=True)
            source_filenames = sorted(set(Path(s).name for s in sources if s))
            st.session_state["del_sources"] = source_filenames

    source_filenames = st.session_state.get("del_sources", [])

    if source_filenames:
        selected_source = st.selectbox(
            "Select file to delete",
            source_filenames,
            key="del_source",
        )

        dry_run_del = st.checkbox(
            "Preview only (dry run)", value=True, key="del_dryrun"
        )

        if st.button("Delete Selected File", key="btn_delete"):
            with st.spinner("Processing..."):
                count = delete_by_source(
                    selected_source,
                    del_col_name,
                    dry_run=dry_run_del,
                )
            if dry_run_del:
                st.info(
                    f"Preview: {count} chunks would be deleted "
                    f"for `{selected_source}`. Uncheck preview to delete."
                )
            else:
                st.success(
                    f"Deleted {count} chunks for `{selected_source}` "
                    f"from `{del_col_name}`."
                )
    else:
        st.info("No source files found in this collection.")


# ============================================================
# TAB 3 — Deduplicate
# ============================================================

with tab3:
    st.subheader("Find & Remove Duplicates")
    st.markdown(
        "Scans for chunks with identical text content. "
        "Keeps the first occurrence and removes the rest."
    )

    dedup_scope = st.radio(
        "Scope",
        ["All collections", "Single collection"],
        key="dedup_scope",
    )

    if dedup_scope == "Single collection":
        dedup_cat = st.selectbox(
            "Collection",
            list(COLLECTIONS.keys()),
            key="dedup_collection",
        )
        dedup_targets = [COLLECTIONS[dedup_cat][0]]
    else:
        dedup_targets = [v[0] for v in COLLECTIONS.values()]

    dry_run_dedup = st.checkbox(
        "Preview only (dry run)", value=True, key="dedup_dryrun"
    )

    if st.button("Run Deduplication", key="btn_dedup"):
        total_removed = 0
        with st.spinner("Scanning..."):
            for col_name in dedup_targets:
                removed = deduplicate(col_name, dry_run=dry_run_dedup)
                total_removed += removed
                if dry_run_dedup:
                    st.info(f"`{col_name}`: {removed} duplicates found.")
                else:
                    st.success(f"`{col_name}`: {removed} duplicates removed.")

        if dry_run_dedup:
            st.info(
                f"Total duplicates found across all targets: {total_removed}. "
                f"Uncheck preview to remove."
            )
        else:
            st.success(f"Deduplication complete. Total removed: {total_removed}.")


# ============================================================
# TAB 4 — Browse Sources
# ============================================================

with tab4:
    st.subheader("Browse Indexed Sources")
    st.markdown("See every source file currently indexed in a collection.")

    browse_cat = st.selectbox(
        "Collection",
        list(COLLECTIONS.keys()),
        key="browse_collection",
    )
    browse_col_name, _ = COLLECTIONS[browse_cat]

    if st.button("Load Sources", key="btn_sources"):
        with st.spinner("Loading..."):
            sources = list_sources(browse_col_name, silent=True)
            filenames = sorted(set(Path(s).name for s in sources if s))

        if filenames:
            st.markdown(
                f"**{len(filenames)} files indexed in `{browse_col_name}`:**"
            )
            for fname in filenames:
                st.markdown(
                    f'<div class="source-item">📄 {fname}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No sources found in this collection.")


# ============================================================
# TAB 5 — Database Stats
# ============================================================

with tab5:
    st.subheader("Database Status")
    st.markdown("Live document counts across all collections.")

    if st.button("Refresh Stats", key="btn_stats"):
        with st.spinner("Fetching..."):
            for label, (col_name, _) in COLLECTIONS.items():
                try:
                    count = get_collection(col_name).count_documents({})
                    st.markdown(
                        f'<div class="stats-row">'
                        f'<span class="stats-label">{col_name}</span>'
                        f'<span class="stats-count">{count:,} documents</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.warning(f"Could not fetch count for {col_name}: {e}")


# ============================================================
# TAB 6 — Verify Embeddings
# ============================================================

with tab6:
    st.subheader("Verify Embeddings")
    st.markdown(
        "Checks a sample of documents in each collection to confirm "
        "embedding dimensions are correct and consistent."
    )

    verify_scope = st.radio(
        "Scope",
        ["All collections", "Single collection"],
        key="verify_scope",
    )

    if verify_scope == "Single collection":
        verify_cat = st.selectbox(
            "Collection",
            list(COLLECTIONS.keys()),
            key="verify_collection",
        )
        verify_targets = {verify_cat: COLLECTIONS[verify_cat][0]}
    else:
        verify_targets = {k: v[0] for k, v in COLLECTIONS.items()}

    if st.button("Run Verification", key="btn_verify"):
        provider = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
        expected_dim = 3072 if provider == "openai" else 768

        with st.spinner("Checking..."):
            for cat, col_name in verify_targets.items():
                collection = get_collection(col_name)
                samples = list(collection.find(
                    {"embedding": {"$exists": True}},
                    {"embedding": 1, "source": 1, "metadata": 1},
                ).limit(3))

                if not samples:
                    st.warning(f"`{col_name}`: no embeddings found.")
                    continue

                dims = set(len(s["embedding"]) for s in samples)
                dim = list(dims)[0]

                if len(dims) == 1 and dim == expected_dim:
                    st.success(
                        f"`{col_name}`: {dim}d embeddings — correct for {provider}."
                    )
                elif len(dims) > 1:
                    st.error(
                        f"`{col_name}`: inconsistent dimensions {dims}. "
                        f"Consider re-ingesting this collection."
                    )
                else:
                    st.error(
                        f"`{col_name}`: {dim}d found, expected {expected_dim}d "
                        f"for {provider}."
                    )

# ============================================================
# TAB 7 — User Management
# ============================================================
with tab7:
    st.subheader("User Management")

    # ── Create user ────────────────────────────────────────
    st.markdown("**Add New User**")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_username = st.text_input("Username", key="new_username")
    with col2:
        new_password = st.text_input("Password", type="password", key="new_password")
    with col3:
        new_role = st.selectbox("Role", ["guest", "admin"], key="new_role")

    if st.button("Create User", key="btn_create_user"):
        if new_username and new_password:
            try:
                create_user(new_username, new_password, new_role)
                st.success(f"Created {new_role} user: {new_username}")
            except Exception as e:
                st.error(f"Failed: {e}")
        else:
            st.warning("Username and password are required.")

    st.divider()

    # ── List + delete users ────────────────────────────────
    st.markdown("**Current Users**")

    if st.button("Refresh Users", key="btn_refresh_users"):
        client = MongoClient(os.getenv("MONGO_URI_ADMIN"))
        users = list(
            client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["users"]
            .find({}, {"username": 1, "role": 1, "created_at": 1})
        )

        for u in users:
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.markdown(
                    f'<div class="source-item">{u["username"]}</div>',
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f'<div class="source-item">{u["role"]}</div>',
                    unsafe_allow_html=True
                )
            with col3:
                # Prevent admin from deleting themselves
                if u["username"] != st.session_state.username:
                    if st.button("Remove", key=f"del_user_{u['username']}"):
                        delete_user(u["username"])
                        st.success(f"Removed {u['username']}")
                        st.rerun()