"""
track_genai/src/gemini_client.py

Handles all communication with the Gemini API.
Reads documents directly from Cloud Storage by URI — no chunking or
embeddings needed. Gemini processes the PDFs natively.

Usage (standalone test):
    python track_genai/src/gemini_client.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION", "us-central1")
MODEL = "gemini-2.5-flash"
INDEX_FILE = Path("track_genai/documents.json")

SYSTEM_PROMPT = """You are a helpful research assistant. You answer questions
based strictly on the documents provided. When you use information from a
document, mention the document name and, where possible, the section or page.
If the answer is not found in the documents, say so clearly — do not invent
information. Be concise but complete."""


# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------


def get_client() -> genai.Client:
    """Return an authenticated Gemini client via Vertex AI."""
    if not PROJECT_ID:
        raise OSError("GCP_PROJECT_ID is not set in .env")
    return genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------


def load_index() -> list[dict]:
    """Return the list of ingested document records."""
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"No index found at {INDEX_FILE}. Run ingest.py first to upload your documents."
        )
    data = json.loads(INDEX_FILE.read_text())
    return data.get("documents", [])


def get_gcs_uris() -> list[str]:
    """Return all GCS URIs from the index."""
    return [doc["gcs_uri"] for doc in load_index()]


# ---------------------------------------------------------------------------
# Core Q&A function
# ---------------------------------------------------------------------------


def ask(question: str, history: list[dict] | None = None) -> str:
    """
    Ask a question about the ingested documents.

    Args:
        question: The user's question as a plain string.
        history:  Optional list of previous turns for multi-turn conversation.
                  Each entry is {"role": "user"|"model", "text": "..."}.

    Returns:
        The model's answer as a plain string.
    """
    client = get_client()
    uris = get_gcs_uris()

    if not uris:
        return "No documents have been ingested yet. Run ingest.py first."

    # Build the document parts — Gemini reads PDFs directly from GCS
    doc_parts = [types.Part.from_uri(file_uri=uri, mime_type="application/pdf") for uri in uris]

    # Build the conversation contents
    contents = []

    # Add previous turns if this is a multi-turn conversation
    if history:
        for turn in history:
            contents.append(
                types.Content(
                    role=turn["role"],
                    parts=[types.Part.from_text(text=str(turn["text"]))],
                )
            )

    # Final user turn: documents + question
    contents.append(
        types.Content(
            role="user",
            parts=doc_parts + [types.Part.from_text(text=question)],
        )
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,  # low temperature = more factual, less creative
        max_output_tokens=2048,
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,  # type: ignore[arg-type]
        config=config,
    )

    return response.text or ""


# ---------------------------------------------------------------------------
# Usage metadata helper (useful for cost monitoring)
# ---------------------------------------------------------------------------


def ask_with_metadata(question: str, history: list[dict] | None = None) -> dict:
    """
    Like ask(), but also returns token usage metadata.

    Returns:
        {
            "answer": str,
            "input_tokens": int,
            "output_tokens": int,
            "model": str,
        }
    """
    client = get_client()
    uris = get_gcs_uris()

    doc_parts = [types.Part.from_uri(file_uri=uri, mime_type="application/pdf") for uri in uris]

    contents = []
    if history:
        for turn in history:
            contents.append(
                types.Content(
                    role=turn["role"],
                    parts=[types.Part.from_text(text=turn["text"])],
                )
            )

    contents.append(
        types.Content(
            role="user",
            parts=doc_parts + [types.Part.from_text(text=question)],
        )
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
        max_output_tokens=2048,
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,  # type: ignore[arg-type]
        config=config,
    )

    usage = response.usage_metadata
    return {
        "answer": response.text or "",
        "input_tokens": usage.prompt_token_count if usage else 0,
        "output_tokens": usage.candidates_token_count if usage else 0,
        "model": MODEL,
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[test] Loading index...")
    docs = load_index()
    print(f"[test] {len(docs)} document(s) found:")
    for d in docs:
        print(f"       - {d['filename']}  ({d['gcs_uri']})")

    question = (
        "What problem does this paper solve, and what is the proposed solution "
        "in one short paragraph?"
    )
    print(f"\n[test] Question: {question}\n")

    result = ask_with_metadata(question)
    print(f"Answer:\n{result['answer']}")
    print(f"\nTokens used — input: {result['input_tokens']}, output: {result['output_tokens']}")
    estimated_cost = (
        result["input_tokens"] / 1_000_000 * 0.15 + result["output_tokens"] / 1_000_000 * 0.60
    )
    print(f"Estimated cost: ${estimated_cost:.5f}")
