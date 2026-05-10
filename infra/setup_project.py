"""
infra/setup_project.py

Run once to bootstrap your GCP project for Vertex AI:
  - Enables required APIs
  - Creates a Cloud Storage bucket
  - Prints a summary of what was set up

Usage:
    python infra/setup_project.py
"""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION", "europe-west1")
BUCKET = os.getenv("GCS_BUCKET")

REQUIRED_APIS = [
    "aiplatform.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
]


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def validate_env() -> None:
    missing = [v for v in ("GCP_PROJECT_ID", "GCS_BUCKET") if not os.getenv(v)]
    if missing:
        print(f"[error] Missing environment variables: {', '.join(missing)}")
        print("        Fill in your .env file before running this script.")
        sys.exit(1)


def enable_apis() -> None:
    assert PROJECT_ID is not None
    print("\n[1/3] Enabling APIs...")
    for api in REQUIRED_APIS:
        result = run(
            ["gcloud", "services", "enable", api, "--project", PROJECT_ID],
            check=False,
        )
        if result.returncode == 0:
            print(f"      ✓ {api}")
        else:
            print(f"      ✗ {api} — {result.stderr.strip()}")
            sys.exit(1)


def create_bucket() -> None:
    assert PROJECT_ID is not None
    assert BUCKET is not None
    print("\n[2/3] Creating Cloud Storage bucket...")

    # Check if bucket already exists
    check = run(
        ["gcloud", "storage", "buckets", "describe", f"gs://{BUCKET}"],
        check=False,
    )
    if check.returncode == 0:
        print(f"      ~ gs://{BUCKET} already exists, skipping.")
        return

    result = run(
        [
            "gcloud",
            "storage",
            "buckets",
            "create",
            f"gs://{BUCKET}",
            "--project",
            PROJECT_ID,
            "--location",
            REGION,
            "--uniform-bucket-level-access",
        ],
        check=False,
    )

    if result.returncode == 0:
        print(f"      ✓ gs://{BUCKET} created in {REGION}")
    else:
        print(f"      ✗ Could not create bucket: {result.stderr.strip()}")
        sys.exit(1)


def create_bucket_folders() -> None:
    """Create logical 'folders' inside the bucket by uploading empty placeholder files."""
    assert BUCKET is not None
    folders = [
        "track_genai/data/",
        "track_automl/data/raw/",
        "track_automl/data/processed/",
        "models/",
    ]
    placeholder = Path("/tmp/.keep")
    placeholder.write_text("")

    for folder in folders:
        run(
            [
                "gcloud",
                "storage",
                "cp",
                str(placeholder),
                f"gs://{BUCKET}/{folder}.keep",
            ],
            check=False,
        )

    print(f"      ✓ Folder structure created inside gs://{BUCKET}")


def print_summary() -> None:
    assert PROJECT_ID is not None
    assert BUCKET is not None
    print("\n[3/3] Summary")
    print(f"      Project  : {PROJECT_ID}")
    print(f"      Region   : {REGION}")
    print(f"      Bucket   : gs://{BUCKET}")
    print(f"      APIs     : {', '.join(a.split('.')[0] for a in REQUIRED_APIS)}")
    print("\n  All done. You can now run the notebooks or src scripts.\n")


if __name__ == "__main__":
    validate_env()
    enable_apis()
    create_bucket()
    create_bucket_folders()
    print_summary()
