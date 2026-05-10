"""
track_genai/src/ingest.py

Uploads PDF files from track_genai/data/ to Cloud Storage and maintains
a local JSON index of ingested documents.

Usage:
    # Ingest all PDFs in the data folder
    python track_genai/src/ingest.py

    # Ingest a specific file
    python track_genai/src/ingest.py --file path/to/doc.pdf

    # List what has been ingested so far
    python track_genai/src/ingest.py --list

    # Clear the index and re-ingest everything
    python track_genai/src/ingest.py --reset
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import storage  # type: ignore[attr-defined]

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
BUCKET_NAME = os.getenv("GCS_BUCKET")
GCS_PREFIX = "track_genai/data"

# Paths relative to project root
DATA_DIR = Path("track_genai/data")
INDEX_FILE = Path("track_genai/documents.json")


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------


def load_index() -> dict:
    """Load the local document index, or return an empty one."""
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text())
    return {"documents": []}


def save_index(index: dict) -> None:
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index, indent=2))


def find_in_index(index: dict, filename: str) -> dict | None:
    """Return the index entry for a filename, or None."""
    return next(
        (d for d in index["documents"] if d["filename"] == filename),
        None,
    )


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


def get_storage_client() -> storage.Client:
    return storage.Client(project=PROJECT_ID)


def upload_file(local_path: Path, overwrite: bool = False) -> dict:
    """
    Upload a single PDF to Cloud Storage.

    Returns a document record dict ready to be stored in the index.
    Raises FileExistsError if the file is already in GCS and overwrite=False.
    """
    client = get_storage_client()
    bucket = client.bucket(BUCKET_NAME)
    blob_name = f"{GCS_PREFIX}/{local_path.name}"
    blob = bucket.blob(blob_name)

    if blob.exists() and not overwrite:
        raise FileExistsError(
            f"gs://{BUCKET_NAME}/{blob_name} already exists. Use --overwrite to replace it."
        )

    print(f"  Uploading {local_path.name} → gs://{BUCKET_NAME}/{blob_name} ...")
    blob.upload_from_filename(str(local_path), content_type="application/pdf")

    record = {
        "filename": local_path.name,
        "gcs_uri": f"gs://{BUCKET_NAME}/{blob_name}",
        "size_bytes": local_path.stat().st_size,
        "ingested_at": datetime.now(UTC).isoformat(),
    }
    print(f"  ✓ Done — {record['size_bytes'] / 1024:.1f} KB")  # type: ignore[operator]
    return record


# ---------------------------------------------------------------------------
# High-level actions
# ---------------------------------------------------------------------------


def ingest_file(path: Path, overwrite: bool = False) -> None:
    """Ingest a single PDF and update the index."""
    if not path.exists():
        print(f"[error] File not found: {path}")
        sys.exit(1)
    if path.suffix.lower() != ".pdf":
        print(f"[error] Only PDF files are supported: {path}")
        sys.exit(1)

    index = load_index()

    existing = find_in_index(index, path.name)
    if existing and not overwrite:
        print(f"  ~ {path.name} is already in the index. Use --overwrite to replace.")
        return

    record = upload_file(path, overwrite=overwrite)

    if existing:
        index["documents"] = [
            record if d["filename"] == path.name else d for d in index["documents"]
        ]
    else:
        index["documents"].append(record)

    save_index(index)
    print(f"  ✓ Index updated — {len(index['documents'])} document(s) total")


def ingest_all(overwrite: bool = False) -> None:
    """Ingest every PDF found in the data directory."""
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"[info] No PDFs found in {DATA_DIR}. Add some and retry.")
        return

    print(f"\n[ingest] Found {len(pdfs)} PDF(s) in {DATA_DIR}\n")
    for pdf in pdfs:
        ingest_file(pdf, overwrite=overwrite)

    print("\n[ingest] All done.")


def list_documents() -> None:
    """Print a summary of ingested documents."""
    index = load_index()
    docs = index["documents"]

    if not docs:
        print("[info] No documents ingested yet.")
        return

    print(f"\n{'#':<4} {'Filename':<40} {'Size':>10}  {'Ingested at'}")
    print("-" * 75)
    for i, doc in enumerate(docs, 1):
        size_kb = doc["size_bytes"] / 1024  # type: ignore[operator]
        print(f"{i:<4} {doc['filename']:<40} {size_kb:>8.1f}KB  {doc['ingested_at'][:19]}")
    print()


def reset_index() -> None:
    """Delete the local index (does NOT delete files from GCS)."""
    if INDEX_FILE.exists():
        INDEX_FILE.unlink()
        print("[reset] Index cleared. GCS files are untouched.")
    else:
        print("[reset] No index to clear.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Cloud Storage")
    parser.add_argument("--file", type=Path, help="Ingest a specific PDF file")
    parser.add_argument("--list", action="store_true", help="List ingested documents")
    parser.add_argument("--reset", action="store_true", help="Clear the local index")
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-upload even if already ingested"
    )
    args = parser.parse_args()

    if args.list:
        list_documents()
    elif args.reset:
        reset_index()
    elif args.file:
        ingest_file(args.file, overwrite=args.overwrite)
    else:
        ingest_all(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
