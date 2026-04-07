"""
build.py -> MongoDB ingestion and management pipeline.

Functions:
    ingest_folder()       — load all docs from a folder into a collection
    ingest_file()         — load a single file into a collection
    delete_by_source()    — remove all chunks from a specific source file
    delete_collection()   — wipe an entire collection
    deduplicate()         — find and remove duplicate chunks
    list_sources()        — show all source files in a collection
    collection_stats()    — document counts and storage info
    verify_embeddings()   — confirm embedding dimensions are correct
    reindex_source()      — delete + re-ingest a source file (update flow)
"""

import os
import csv
import hashlib
from pathlib import Path
from datetime import datetime, timezone
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
    "context":            ("context_vectors",   "vector_index_context"),
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


def chunk_documents(docs: list[Document], chunk_size=1000, chunk_overlap=200) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)

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

    chunks = chunk_documents(new_docs, chunk_size, chunk_overlap)
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

    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
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