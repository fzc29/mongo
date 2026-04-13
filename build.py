"""
build.py -> MongoDB ingestion and management pipeline.

Functions:
    ingest_folder()           — load all docs from a folder into a collection
    ingest_file()             — load a single file into a collection
    delete_by_source()        — remove all chunks from a specific source file
    delete_collection()       — wipe an entire collection
    deduplicate()             — find and remove duplicate chunks
    list_sources()            — show all source files in a collection
    collection_stats()        — document counts and storage info
    verify_embeddings()       — confirm embedding dimensions are correct
    reindex_source()          — delete + re-ingest a source file (update flow)

    ingest_pnl_structured()   — parse a PnL file into structured row documents (pnl_table)
    delete_pnl_period()       — delete all rows for a PnL reporting period
    list_pnl_periods()        — list all PnL periods in pnl_table
"""

import os
import re
import csv
import hashlib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
)
from pymongo import MongoClient

load_dotenv()

# ============================================================
# Embedding Setup
# ============================================================

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()

if EMBEDDING_PROVIDER == "openai":
    from langchain_openai import OpenAIEmbeddings
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
elif EMBEDDING_PROVIDER == "gemini":
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
else:
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}")

# ============================================================
# MongoDB Setup
# ============================================================

def get_client() -> MongoClient:
    return MongoClient(os.getenv("MONGO_URI_ADMIN"))

def get_collection(collection_name: str):
    client = get_client()
    return client[os.getenv("MONGO_DB_NAME", "portfolio_rag")][collection_name]

# ============================================================
# Collections + Index Name Map
# Add new categories here as needed — nothing else to change
# ============================================================

COLLECTIONS = {
    "context":            ("context_vectors",   "vector_index"),
    "pnl":                ("pnl_vectors",        "vector_index"),
    "newsletter":         ("newsletter_vectors", "vector_index"),
    "weekly_market_data": ("weekly_vectors",     "vector_index"),
}

BASE_DIR = Path(__file__).resolve().parent

FOLDER_MAP = {
    "context":            BASE_DIR / "data" / "context",
    "pnl":                BASE_DIR / "data" / "pnl",
    "newsletter":         BASE_DIR / "data" / "newsletters",
    "weekly_market_data": BASE_DIR / "data" / "weekly_market_data",
}

# ============================================================
# Excel Serial Date Conversion
# ============================================================

# Excel stores dates as days since Dec 30, 1899 (accounts for the
# intentional Feb 29 1900 bug). When CSVs are exported with columns
# formatted as General/Number instead of Date, the raw integer leaks
# into the file instead of a human-readable date string.
#
# Valid range: 40000–60000 covers roughly 2009–2064. We use this as
# a safe heuristic — financial notional values are far larger, and
# index/ratio values have decimals, so 5-digit integers in this band
# are almost certainly Excel dates in a PnL context.

_EXCEL_EPOCH = datetime(1899, 12, 30)
# Exclude matches preceded or followed by a digit or decimal point.
# This prevents false positives on decimal values like 51850.69071 or 26120.51579,
# where the integer part or fractional part falls in the Excel date serial range.
_EXCEL_DATE_RE = re.compile(r"(?<![.\d])(4[0-9]{4}|5[0-9]{4})(?![.\d])")


def _excel_serial_to_date(serial: int) -> str:
    """Convert an Excel date serial number to an ISO date string (YYYY-MM-DD)."""
    return (_EXCEL_EPOCH + timedelta(days=serial)).strftime("%Y-%m-%d")


def convert_excel_dates(text: str) -> str:
    """
    Replace Excel serial date integers in document text with ISO date strings.
    Only replaces standalone 5-digit integers in the range 40000–59999.
    Safe to call on any text — non-date numbers are left untouched.
    """
    def replace_match(m: re.Match) -> str:
        return _excel_serial_to_date(int(m.group(0)))

    return _EXCEL_DATE_RE.sub(replace_match, text)


# ============================================================
# File Loading
# ============================================================

def load_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    elif ext == ".md":
        text = path.read_text(encoding="utf-8")
        return [Document(page_content=text, metadata={"source": path.name})]
    elif ext == ".csv":
        return CSVLoader(str(path)).load()
    elif ext == ".txt":
        return TextLoader(str(path)).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_folder(folder: Path) -> list[Document]:
    docs = []
    supported = {".pdf", ".md", ".csv", ".txt"}
    files = [f for f in folder.rglob("*") if f.suffix.lower() in supported]

    if not files:
        print(f"  No supported files found in {folder}")
        return []

    for file in files:
        try:
            loaded = load_file(file)
            docs.extend(loaded)
            print(f"  Loaded: {file.name} ({len(loaded)} pages)")
        except Exception as e:
            print(f"  Failed: {file.name} — {e}")

    return docs


def chunk_documents(docs: list[Document], chunk_size=1000, chunk_overlap=200, fix_dates=False) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    if fix_dates:
        for chunk in chunks:
            chunk.page_content = convert_excel_dates(chunk.page_content)
    return chunks

# ============================================================
# 1. INGEST — Folder
# ============================================================

def ingest_folder(category: str, chunk_size=1000, chunk_overlap=200):
    """
    Load all documents from a category's folder and ingest into MongoDB.
    Skips files already present in the collection (by source filename).

    Usage:
        ingest_folder("context")
        ingest_folder("pnl")
    """
    if category not in COLLECTIONS:
        raise ValueError(f"Unknown category '{category}'. Options: {list(COLLECTIONS.keys())}")

    collection_name, index_name = COLLECTIONS[category]
    folder = FOLDER_MAP[category]

    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    print(f"\nIngesting folder: {folder.name} → {collection_name}")

    # Get list of sources already in the collection to avoid duplicates
    existing = set(list_sources(collection_name, silent=True))
    print(f"  Already indexed: {len(existing)} source files")

    docs = load_folder(folder)
    if not docs:
        return

    # Filter out docs from already-indexed sources
    new_docs = [d for d in docs if d.metadata.get("source", "") not in existing]

    if not new_docs:
        print("  All files already indexed. Nothing to add.")
        return

    print(f"  New documents to index: {len(new_docs)}")

    if category == "context":
        # Larger chunks for prose documents; add period metadata + source prefix
        chunks = chunk_documents(new_docs, chunk_size=2500, chunk_overlap=400)
        for chunk in chunks:
            src_name = Path(chunk.metadata.get("source", "")).name
            period = _extract_context_period(src_name)
            if period:
                chunk.metadata["report_period"] = period
            if src_name:
                chunk.page_content = f"[{src_name}]\n{chunk.page_content}"
    else:
        chunks = chunk_documents(new_docs, chunk_size, chunk_overlap, fix_dates=(category == "pnl"))

    print(f"  Chunks to embed: {len(chunks)}")

    MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=embedding,
        collection=get_collection(collection_name),
        index_name=index_name,
    )

    print(f"  Done → {collection_name} (+{len(chunks)} chunks)")


# ============================================================
# 2. INGEST — Single File
# ============================================================

def ingest_file(file_path: str | Path, category: str, chunk_size=1000, chunk_overlap=200):
    """
    Ingest a single file into a specific collection.

    Usage:
        ingest_file("data/context/report.pdf", "context")
        ingest_file("/absolute/path/to/file.csv", "pnl")
    """
    if category not in COLLECTIONS:
        raise ValueError(f"Unknown category '{category}'. Options: {list(COLLECTIONS.keys())}")

    collection_name, index_name = COLLECTIONS[category]
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"\nIngesting file: {path.name} → {collection_name}")

    # Check if already indexed
    existing = list_sources(collection_name, silent=True)
    if str(path) in existing or path.name in existing:
        print(f"  Already indexed. Use reindex_source() to update it.")
        return

    docs = load_file(path)
    if not docs:
        print("  No content loaded.")
        return

    if category == "context":
        chunks = chunk_documents(docs, chunk_size=2500, chunk_overlap=400)
        for chunk in chunks:
            src_name = Path(chunk.metadata.get("source", path.name)).name
            period = _extract_context_period(src_name)
            if period:
                chunk.metadata["report_period"] = period
            chunk.page_content = f"[{src_name}]\n{chunk.page_content}"
    else:
        chunks = chunk_documents(docs, chunk_size, chunk_overlap, fix_dates=(category == "pnl"))

    print(f"  {len(chunks)} chunks to embed...")

    MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=embedding,
        collection=get_collection(collection_name),
        index_name=index_name,
    )

    print(f"  Done → {collection_name} (+{len(chunks)} chunks)")


# ============================================================
# 3. DELETE — By Source File
# ============================================================

def delete_by_source(source_filename: str, collection_name: str, dry_run=False):
    """
    Delete all chunks that came from a specific source file.

    The source_filename can be:
      - Just the filename:        "report_oct2025.pdf"
      - The full path as stored:  "/Users/.../data/context/report_oct2025.pdf"

    dry_run=True shows what would be deleted without deleting anything.

    Usage:
        delete_by_source("report_oct2025.pdf", "context_vectors")
        delete_by_source("report_oct2025.pdf", "context_vectors", dry_run=True)
    """
    collection = get_collection(collection_name)

    # Search by both exact match and filename-only match
    query = {
        "$or": [
            {"source":                    source_filename},
            {"metadata.source":           source_filename},
            {"source":                    {"$regex": source_filename, "$options": "i"}},
            {"metadata.source":           {"$regex": source_filename, "$options": "i"}},
            {"metadata.original_filename": source_filename},
        ]
    }

    count = collection.count_documents(query)

    if count == 0:
        print(f"  No documents found matching '{source_filename}' in {collection_name}")
        return 0

    print(f"  Found {count} chunks matching '{source_filename}' in {collection_name}")

    if dry_run:
        print(f"  DRY RUN — nothing deleted. Remove dry_run=True to delete.")
        return count

    result = collection.delete_many(query)
    print(f"  Deleted {result.deleted_count} chunks from {collection_name}")
    return result.deleted_count


# ============================================================
# 4. DELETE — Entire Collection
# ============================================================

def delete_collection(collection_name: str, confirm=False):
    """
    Wipe all documents from a collection.
    Requires confirm=True as a safety check.

    Usage:
        delete_collection("context_vectors", confirm=True)
    """
    if not confirm:
        print(f"  Safety check: pass confirm=True to wipe {collection_name}")
        return

    collection = get_collection(collection_name)
    count_before = collection.count_documents({})
    collection.delete_many({})
    print(f"  Deleted all {count_before} documents from {collection_name}")


# ============================================================
# 5. DEDUPLICATE — Find and Remove Duplicate Chunks
# ============================================================

def deduplicate(collection_name: str, dry_run=False):
    """
    Find and remove duplicate chunks based on text content hash.
    Keeps the first occurrence, removes subsequent duplicates.

    dry_run=True reports duplicates without deleting.

    Usage:
        deduplicate("context_vectors")
        deduplicate("context_vectors", dry_run=True)
    """
    collection = get_collection(collection_name)
    total = collection.count_documents({})
    print(f"\nScanning {collection_name} for duplicates ({total} documents)...")

    seen_hashes = {}     # hash → _id of first occurrence
    duplicate_ids = []   # _ids to delete

    # Stream documents in batches to avoid loading all into memory at once
    cursor = collection.find({}, {"_id": 1, "text": 1}, batch_size=500)

    for doc in cursor:
        text = doc.get("text", "")
        # Hash the text content — same text = duplicate
        text_hash = hashlib.md5(text.strip().encode()).hexdigest()

        if text_hash in seen_hashes:
            duplicate_ids.append(doc["_id"])
        else:
            seen_hashes[text_hash] = doc["_id"]

    if not duplicate_ids:
        print(f"  No duplicates found in {collection_name}")
        return 0

    print(f"  Found {len(duplicate_ids)} duplicate chunks")

    if dry_run:
        print(f"  DRY RUN — nothing deleted. Remove dry_run=True to delete.")
        return len(duplicate_ids)

    # Delete in batches of 500
    deleted = 0
    batch_size = 500
    for i in range(0, len(duplicate_ids), batch_size):
        batch = duplicate_ids[i:i + batch_size]
        result = collection.delete_many({"_id": {"$in": batch}})
        deleted += result.deleted_count

    print(f"  Removed {deleted} duplicates. {total - deleted} unique chunks remain.")
    return deleted


# ============================================================
# 6. LIST SOURCES — See What Files Are Indexed
# ============================================================

def list_sources(collection_name: str, silent=False) -> list[str]:
    """
    List all unique source files currently indexed in a collection.

    Usage:
        list_sources("context_vectors")
        sources = list_sources("pnl_vectors")
    """
    collection = get_collection(collection_name)

    # Try both flat and nested metadata structures
    sources = set()

    for field in ["source", "metadata.source", "metadata.original_filename"]:
        values = collection.distinct(field)
        sources.update([v for v in values if v])

    sources = {s for s in sources if not s.startswith("/tmp")}

    sources = sorted(sources)

    if not silent:
        print(f"\nSources in {collection_name} ({len(sources)} files):")
        for s in sources:
            # Show just the filename, not full path
            print(f"  {Path(s).name}")

    return sources


# ============================================================
# 7. COLLECTION STATS — Overview of All Collections
# ============================================================

def collection_stats():
    """
    Print document counts and storage info for all collections.

    Usage:
        collection_stats()
    """
    client = get_client()
    db = client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]

    print(f"\n{'='*55}")
    print(f"  Database: {db.name}")
    print(f"{'='*55}")
    print(f"  {'Collection':<30} {'Documents':>10} {'Size':>10}")
    print(f"  {'-'*50}")

    total_docs = 0
    for col_name in sorted(db.list_collection_names()):
        # Skip MongoDB internal collections
        if col_name in ("system.views",):
            continue
        count = db[col_name].count_documents({})
        stats = db.command("collStats", col_name)
        size_mb = stats.get("size", 0) / (1024 * 1024)
        print(f"  {col_name:<30} {count:>10,} {size_mb:>9.1f}M")
        total_docs += count

    print(f"  {'-'*50}")
    print(f"  {'TOTAL':<30} {total_docs:>10,}")
    print(f"{'='*55}\n")


# ============================================================
# 8. VERIFY EMBEDDINGS — Confirm Dimensions Are Correct
# ============================================================

def verify_embeddings(collection_name: str):
    """
    Check a sample of documents to confirm embedding dimensions
    are consistent and match the expected model output.

    Usage:
        verify_embeddings("context_vectors")
    """
    collection = get_collection(collection_name)
    samples = list(collection.find(
        {"embedding": {"$exists": True}},
        {"embedding": 1, "text": 1, "source": 1, "metadata": 1}
    ).limit(5))

    if not samples:
        print(f"  No documents with embeddings found in {collection_name}")
        return

    print(f"\nEmbedding check for {collection_name}:")
    dims = set()
    for s in samples:
        emb = s.get("embedding", [])
        dims.add(len(emb))
        source = s.get("source") or s.get("metadata", {}).get("source", "unknown")
        print(f"  dim={len(emb)}  source={Path(str(source)).name}")

    if len(dims) == 1:
        dim = list(dims)[0]
        expected = 3072 if EMBEDDING_PROVIDER == "openai" else 768
        if dim == expected:
            print(f"  All embeddings are {dim}d — correct for {EMBEDDING_PROVIDER}")
        else:
            print(f"  WARNING: embeddings are {dim}d but expected {expected}d for {EMBEDDING_PROVIDER}")
    else:
        print(f"  WARNING: inconsistent dimensions found: {dims}")
        print(f"  This may cause search errors — consider re-ingesting this collection.")


# ============================================================
# 9. REINDEX SOURCE — Delete + Re-ingest a File (Update Flow)
# ============================================================

def reindex_source(file_path: str | Path, category: str):
    """
    Delete all existing chunks from a source file then re-ingest it.
    Use this when a document has been updated and you want the new version.

    Usage:
        reindex_source("data/context/report_oct2025.pdf", "context")
    """
    path = Path(file_path)
    collection_name, _ = COLLECTIONS[category]

    print(f"\nReindexing: {path.name}")

    # Step 1 — delete existing chunks
    deleted = delete_by_source(path.name, collection_name)
    if deleted == 0:
        # Also try with full path
        deleted = delete_by_source(str(path), collection_name)

    # Step 2 — re-ingest
    ingest_file(file_path, category)
    print(f"  Reindex complete.")


# ============================================================
# 10. STRUCTURED PnL — pnl_table (replaces vector ingestion)
# ============================================================

# Month abbreviation → zero-padded number
_MONTH_ABBR = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}

# Full + abbreviated month names for context filename parsing
_CONTEXT_MONTH_MAP = {
    "january": "01",  "jan": "01",
    "february": "02", "feb": "02",
    "march": "03",    "mar": "03",
    "april": "04",    "apr": "04",
    "may": "05",
    "june": "06",     "jun": "06",
    "july": "07",     "jul": "07",
    "august": "08",   "aug": "08",
    "september": "09","sep": "09",
    "october": "10",  "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "dec": "12",
}


def _extract_context_period(filename: str) -> str | None:
    """
    Extract a YYYY-MM period string from a context document filename.
    Handles the formats found in practice:
      - YYYYMMDD embedded:   GLOBAL_20260227_0042.pdf       → 2026-02
      - Month YYYY explicit: Commodity Market Feb 2026.pdf  → 2026-02
      - MMDDYY 6-digit:      Macro Commentary 120125.pdf    → 2025-12
      - Month name only:     February 13 Soft CPI.pdf       → 2026-02 (inferred)
                             August 1 Payroll.pdf            → 2025-08 (inferred)
    Returns None if no period can be determined.
    """
    stem = Path(filename).stem
    s = stem.lower()

    # 1. YYYYMMDD embedded (e.g. GLOBAL_20260227)
    m = re.search(r'(20\d{2})(0[1-9]|1[0-2])\d{2}', stem)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    # 2. "Month ... YYYY" or "YYYY ... Month" (explicit year)
    for name, num in _CONTEXT_MONTH_MAP.items():
        m = re.search(rf'\b{name}\b.*?\b(20\d{{2}})\b', s)
        if m:
            return f"{m.group(1)}-{num}"
        m = re.search(rf'\b(20\d{{2}})\b.*?\b{name}\b', s)
        if m:
            return f"{m.group(1)}-{num}"

    # 3. MMDDYY 6-digit date (e.g. 120125 = Dec 01 2025)
    m = re.search(r'\b(0[1-9]|1[0-2])(\d{2})(2[0-9])\b', stem)
    if m:
        return f"20{m.group(3)}-{m.group(1)}"

    # 4. Month name only — infer year from month
    #    Jun–Dec files without a year are 2025 uploads; Jan–May are 2026
    for name, num in _CONTEXT_MONTH_MAP.items():
        if re.search(rf'\b{name}\b', s):
            inferred = "2025" if int(num) >= 6 else "2026"
            return f"{inferred}-{num}"

    return None


def _extract_report_period(filename: str) -> str:
    """
    Auto-detect a YYYY-MM period string from a PnL filename.
    e.g. PNL_FEB_2026.md → "2026-02", PNL_OCT.md → "2025-10"
    """
    stem = Path(filename).stem.lower()
    year_match = re.search(r'\b(20\d{2})\b', stem)
    year = year_match.group(1) if year_match else str(datetime.now(timezone.utc).year)
    for abbr, num in _MONTH_ABBR.items():
        if abbr in stem:
            return f"{year}-{num}"
    return f"{year}-??"


def _normalize_col(name: str) -> str:
    """Lowercase, strip BOM/special chars, collapse to snake_case."""
    name = name.replace("ï»¿", "").replace("\ufeff", "").strip()
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip("_")


def _parse_pnl_markdown(path: Path) -> list[dict]:
    """Parse a PnL markdown table into a list of row dicts."""
    text = convert_excel_dates(path.read_text(encoding="utf-8"))
    rows = []
    header = None

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue

        cells = [c.strip() for c in line.strip("|").split("|")]

        # Skip separator rows (| --- | --- |)
        if all(re.match(r'^-+:?$|^$', c.replace(" ", "")) for c in cells):
            continue

        if header is None:
            header = [_normalize_col(c) for c in cells]
            continue

        # Skip totals/AUM summary row
        if cells and re.search(r'\baum\b', cells[0], re.I):
            continue

        if header:
            row = {header[i]: (cells[i] if i < len(cells) else "") for i in range(len(header))}
            if any(v.strip() for v in row.values()):
                rows.append(row)

    return rows


def _parse_pnl_csv(path: Path) -> list[dict]:
    """Parse a PnL CSV file into a list of row dicts."""
    import csv as _csv
    rows = []
    with open(path, encoding="utf-8-sig") as f:  # utf-8-sig strips BOM
        reader = _csv.DictReader(f)
        for row in reader:
            clean = {_normalize_col(k): convert_excel_dates(str(v).strip())
                     for k, v in row.items() if k}
            if re.search(r'\baum\b', " ".join(clean.keys()), re.I):
                continue
            if any(v.strip() for v in clean.values()):
                rows.append(clean)
    return rows


def ingest_pnl_structured(
    file_path: str | Path,
    report_period: str = None,
    source_name: str = None,
    uploaded_by: str = "system",
) -> int:
    """
    Parse a PnL .md or .csv file and insert each row as a structured
    document into the pnl_table collection with a report_period field.

    This replaces vector-based PnL ingestion — queries are exact by period,
    not semantic similarity.

    report_period is auto-detected from the filename if not provided (YYYY-MM).
    source_name overrides the filename used for period detection and metadata
    (useful when file_path is a temp path from Streamlit uploads).

    Usage:
        ingest_pnl_structured("data/pnl/PNL_FEB_2026.md")
        ingest_pnl_structured("/tmp/xyz.md", source_name="PNL_MAR_2026.md")
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    display_name = source_name or path.name
    if report_period is None:
        report_period = _extract_report_period(display_name)

    print(f"\nIngesting PnL: {display_name} → period={report_period}")

    col = get_collection("pnl_table")
    existing = col.count_documents({"report_period": report_period})
    if existing > 0:
        print(f"  Period {report_period} already has {existing} rows. "
              f"Call delete_pnl_period('{report_period}') first to replace.")
        return 0

    ext = path.suffix.lower()
    if ext == ".md":
        rows = _parse_pnl_markdown(path)
    elif ext == ".csv":
        rows = _parse_pnl_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .md or .csv")

    if not rows:
        print("  No rows parsed from file.")
        return 0

    now = datetime.now(timezone.utc)
    for row in rows:
        row["report_period"] = report_period
        row["source_file"] = display_name
        row["uploaded_by"] = uploaded_by
        row["uploaded_at"] = now

    col.insert_many(rows)
    print(f"  Inserted {len(rows)} rows into pnl_table for {report_period}")
    return len(rows)


def delete_pnl_period(report_period: str, dry_run: bool = False) -> int:
    """
    Delete all rows for a specific PnL reporting period from pnl_table.

    Usage:
        delete_pnl_period("2026-02")
        delete_pnl_period("2026-02", dry_run=True)
    """
    col = get_collection("pnl_table")
    count = col.count_documents({"report_period": report_period})
    if count == 0:
        print(f"  No rows found for period {report_period}")
        return 0
    print(f"  Found {count} rows for period {report_period}")
    if dry_run:
        print("  DRY RUN — nothing deleted.")
        return count
    col.delete_many({"report_period": report_period})
    print(f"  Deleted {count} rows.")
    return count


def backfill_report_periods(collection_name: str = "context_vectors") -> int:
    """
    Add report_period metadata to existing documents that don't have it yet.
    Derives the period from the stored source filename — no re-embedding needed.
    Run this once after upgrading to the period-aware ingestion pipeline.

    Usage:
        backfill_report_periods()
        backfill_report_periods("context_vectors")
    """
    col = get_collection(collection_name)
    total = col.count_documents({})
    missing = col.count_documents({"report_period": {"$exists": False}})
    print(f"\nBackfilling report_period in {collection_name}")
    print(f"  Total docs: {total}  |  Missing period: {missing}")

    if missing == 0:
        print("  All documents already have report_period. Nothing to do.")
        return 0

    updated = 0
    skipped = 0
    cursor = col.find(
        {"report_period": {"$exists": False}},
        {"_id": 1, "source": 1, "metadata": 1},
        batch_size=500,
    )
    for doc in cursor:
        src = (
            doc.get("source")
            or (doc.get("metadata") or {}).get("source")
            or ""
        )
        src_name = Path(str(src)).name
        period = _extract_context_period(src_name)
        if period:
            col.update_one({"_id": doc["_id"]}, {"$set": {"report_period": period}})
            updated += 1
        else:
            skipped += 1

    print(f"  Updated: {updated}  |  Skipped (no period detectable): {skipped}")
    return updated


def list_pnl_periods() -> list[str]:
    """
    List all PnL reporting periods currently in pnl_table.

    Usage:
        list_pnl_periods()
    """
    col = get_collection("pnl_table")
    periods = sorted(col.distinct("report_period"))
    print(f"\nPnL periods in pnl_table ({len(periods)}):")
    for p in periods:
        count = col.count_documents({"report_period": p})
        src = col.find_one({"report_period": p}, {"source_file": 1})
        src_name = (src or {}).get("source_file", "?")
        print(f"  {p}  —  {count} positions  (from {src_name})")
    return periods


# # ============================================================
# # Main — run all folders on direct execution
# # ============================================================

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="MongoDB index management")
#     parser.add_argument("--action", choices=[
#         "ingest_all",
#         "ingest",
#         "delete",
#         "reindex",
#         "deduplicate",
#         "stats",
#         "sources",
#         "verify",
#     ], default="stats")
#     parser.add_argument("--category",  help="Category: context, pnl, newsletter, weekly_market_data")
#     parser.add_argument("--file",      help="File path for ingest/delete/reindex actions")
#     parser.add_argument("--dry-run",   action="store_true", help="Preview without making changes")
#     args = parser.parse_args()

#     if args.action == "ingest_all":
#         for cat in COLLECTIONS:
#             ingest_folder(cat)

#     elif args.action == "ingest":
#         if args.file:
#             ingest_file(args.file, args.category)
#         elif args.category:
#             ingest_folder(args.category)
#         else:
#             print("Provide --category or --file")

#     elif args.action == "delete":
#         if not args.file or not args.category:
#             print("Provide --file and --category")
#         else:
#             col_name, _ = COLLECTIONS[args.category]
#             delete_by_source(args.file, col_name, dry_run=args.dry_run)

#     elif args.action == "reindex":
#         if not args.file or not args.category:
#             print("Provide --file and --category")
#         else:
#             reindex_source(args.file, args.category)

#     elif args.action == "deduplicate":
#         if args.category:
#             col_name, _ = COLLECTIONS[args.category]
#             deduplicate(col_name, dry_run=args.dry_run)
#         else:
#             for cat, (col_name, _) in COLLECTIONS.items():
#                 deduplicate(col_name, dry_run=args.dry_run)

#     elif args.action == "stats":
#         collection_stats()

#     elif args.action == "sources":
#         if not args.category:
#             for cat, (col_name, _) in COLLECTIONS.items():
#                 list_sources(col_name)
#         else:
#             col_name, _ = COLLECTIONS[args.category]
#             list_sources(col_name)

#     elif args.action == "verify":
#         if args.category:
#             col_name, _ = COLLECTIONS[args.category]
#             verify_embeddings(col_name)
#         else:
#             for cat, (col_name, _) in COLLECTIONS.items():
#                 verify_embeddings(col_name)