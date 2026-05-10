"""
infra/auth.py

Central auth helper — import this at the top of any src/ file instead of
repeating the boilerplate everywhere.

Usage:
    from infra.auth import init_vertex

    client = init_vertex()   # returns the aiplatform module, ready to use
"""

import os

from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION", "europe-west1")
BUCKET = os.getenv("GCS_BUCKET")


def init_vertex():
    """
    Initialise the Vertex AI SDK with project + region from .env.
    Credentials are picked up automatically from Application Default Credentials
    (set by: gcloud auth application-default login).

    Returns the aiplatform module so callers can do:
        aip = init_vertex()
        aip.Model.list()
    """
    try:
        from google.cloud import aiplatform
    except ImportError as err:
        raise ImportError(
            "google-cloud-aiplatform is not installed.\nRun: pip install google-cloud-aiplatform"
        ) from err

    if not PROJECT_ID:
        raise OSError("GCP_PROJECT_ID is not set. Fill in your .env file.")

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET}" if BUCKET else None,
    )

    print(f"[auth] Vertex AI initialised — project={PROJECT_ID}, region={REGION}")
    return aiplatform


def get_gemini_client():
    """
    Return a Gemini client using the new google-genai SDK (replaces vertexai).
    """
    from google import genai

    if not PROJECT_ID:
        raise OSError("GCP_PROJECT_ID is not set.")

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=REGION,
    )
    return client


if __name__ == "__main__":
    client = get_gemini_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say hello in one sentence.",
    )

    print(f"[auth] Gemini test: {response.text.strip()}")
