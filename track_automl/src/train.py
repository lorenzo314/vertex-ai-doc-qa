"""
track_automl/src/train.py

Preprocesses the Telco Customer Churn dataset, uploads it to Cloud Storage,
creates a Vertex AI tabular dataset, and kicks off an AutoML training job.

Usage:
    # Full pipeline: preprocess → upload → create dataset → train
    python track_automl/src/train.py

    # Preprocess and upload only (no training job)
    python track_automl/src/train.py --upload-only

    # Check the status of a running training job
    python track_automl/src/train.py --status JOB_RESOURCE_NAME
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import aiplatform, storage  # type: ignore[attr-defined]

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION", "us-central1")
BUCKET = os.getenv("GCS_BUCKET")

RAW_CSV = Path("track_automl/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_CSV = Path("track_automl/data/processed/telco_churn_clean.csv")
GCS_PREFIX = "track_automl/data"
TARGET_COLUMN = "Churn"
MODEL_DISPLAY_NAME = "telco-churn-automl"
DATASET_DISPLAY_NAME = "telco-churn"

# AutoML training budget — 1 node hour is enough for this dataset and keeps costs low.
# At ~$3.46/node hour this job costs roughly $3-4.
TRAINING_BUDGET_NODE_HOURS = 1


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess(raw_path: Path, output_path: Path) -> None:
    """
    Clean the raw CSV and write a processed version:
    - Drop customerID (unique identifier, not a feature)
    - Fill blank TotalCharges with 0
    - Convert TotalCharges to float
    """
    try:
        import pandas as pd
    except ImportError as err:
        raise ImportError("pandas is required. Run: pip install pandas") from err

    print(f"[preprocess] Reading {raw_path} ...")
    df = pd.read_csv(raw_path)
    print(f"[preprocess] Shape: {df.shape}")

    # Drop identifier column
    df = df.drop(columns=["customerID"])

    # Fix TotalCharges — blank strings for customers with tenure=0
    df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Verify target column
    print(f"[preprocess] Churn distribution:\n{df[TARGET_COLUMN].value_counts()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[preprocess] ✓ Saved to {output_path} ({len(df)} rows, {len(df.columns)} columns)")


# ---------------------------------------------------------------------------
# Upload to GCS
# ---------------------------------------------------------------------------


def upload_to_gcs(local_path: Path) -> str:
    """
    Upload the processed CSV to Cloud Storage.
    Returns the GCS URI.
    """
    assert BUCKET is not None
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)
    blob_name = f"{GCS_PREFIX}/{local_path.name}"
    blob = bucket.blob(blob_name)

    print(f"[upload] Uploading {local_path.name} → gs://{BUCKET}/{blob_name} ...")
    blob.upload_from_filename(str(local_path), content_type="text/csv")

    gcs_uri = f"gs://{BUCKET}/{blob_name}"
    print(f"[upload] ✓ {gcs_uri}")
    return gcs_uri


# ---------------------------------------------------------------------------
# Vertex AI dataset
# ---------------------------------------------------------------------------


def create_dataset(gcs_uri: str) -> aiplatform.TabularDataset:
    """
    Create a Vertex AI tabular dataset from the GCS CSV.
    If a dataset with the same display name already exists, returns the first match.
    """
    assert PROJECT_ID is not None

    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Check if dataset already exists
    existing = aiplatform.TabularDataset.list(
        filter=f'display_name="{DATASET_DISPLAY_NAME}"',
    )
    if existing:
        print(f"[dataset] ~ Dataset '{DATASET_DISPLAY_NAME}' already exists, reusing.")
        return existing[0]  # type: ignore[return-value]

    print(f"[dataset] Creating Vertex AI dataset '{DATASET_DISPLAY_NAME}' ...")
    dataset = aiplatform.TabularDataset.create(
        display_name=DATASET_DISPLAY_NAME,
        gcs_source=gcs_uri,
    )
    print(f"[dataset] ✓ Created: {dataset.resource_name}")
    return dataset


# ---------------------------------------------------------------------------
# AutoML training job
# ---------------------------------------------------------------------------


def start_training(dataset: aiplatform.TabularDataset) -> aiplatform.AutoMLTabularTrainingJob:
    """
    Start an AutoML tabular training job.

    The job runs asynchronously — this function returns immediately after
    submission. Use --status to check progress, or monitor in the GCP console
    under Vertex AI → Training.

    Budget: 1 node hour (~$3-4). Sufficient for this dataset size.
    """
    assert PROJECT_ID is not None

    aiplatform.init(project=PROJECT_ID, location=REGION)

    print(f"[train] Starting AutoML training job '{MODEL_DISPLAY_NAME}' ...")
    print(f"[train] Budget: {TRAINING_BUDGET_NODE_HOURS} node hour(s)")
    print("[train] This will take 1-2 hours. Monitor progress at:")
    print(f"        https://console.cloud.google.com/vertex-ai/training?project={PROJECT_ID}")

    job = aiplatform.AutoMLTabularTrainingJob(
        display_name=MODEL_DISPLAY_NAME,
        optimization_prediction_type="classification",
        optimization_objective="maximize-au-roc",
    )

    # run() with sync=False submits the job and returns immediately
    job.run(
        dataset=dataset,
        target_column=TARGET_COLUMN,
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1,
        budget_milli_node_hours=TRAINING_BUDGET_NODE_HOURS * 1000,
        model_display_name=MODEL_DISPLAY_NAME,
        disable_early_stopping=False,
        sync=False,
    )

    print("\n[train] ✓ Job submitted.")
    try:
        resource_name = job.resource_name
        print(f"[train] Job resource name: {resource_name}")
        print(f"\n  Save this for later:\n  {resource_name}\n")
    except (AttributeError, RuntimeError):
        print("[train] Resource name not yet available (job submitted asynchronously).")
        print(
            f"[train] Monitor: https://console.cloud.google.com/vertex-ai/training?project={PROJECT_ID}"
        )
    return job


# ---------------------------------------------------------------------------
# Status check
# ---------------------------------------------------------------------------


def check_status(job_resource_name: str) -> None:
    """Print the current status of a training job by resource name."""
    assert PROJECT_ID is not None
    aiplatform.init(project=PROJECT_ID, location=REGION)

    job = aiplatform.AutoMLTabularTrainingJob.get(job_resource_name)

    print(f"[status] Job    : {job.display_name}")
    print(f"[status] State  : {job.state.name}")  # type: ignore[union-attr]
    print(f"[status] Created: {job.create_time}")

    if hasattr(job, "end_time") and job.end_time is not None:
        print(f"[status] Ended  : {job.end_time}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def validate_env() -> None:
    missing = [v for v in ("GCP_PROJECT_ID", "GCS_BUCKET") if not os.getenv(v)]
    if missing:
        print(f"[error] Missing environment variables: {', '.join(missing)}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Telco churn AutoML model on Vertex AI")
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Preprocess and upload to GCS, but do not start a training job",
    )
    parser.add_argument(
        "--status",
        metavar="JOB_RESOURCE_NAME",
        help="Check the status of an existing training job",
    )
    args = parser.parse_args()

    validate_env()

    if args.status:
        check_status(args.status)
        return

    if not RAW_CSV.exists():
        print(f"[error] Raw CSV not found: {RAW_CSV}")
        print(
            "        Download it from https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
        )
        print(f"        and place it at {RAW_CSV}")
        sys.exit(1)

    # Step 1 — preprocess
    preprocess(RAW_CSV, PROCESSED_CSV)

    # Step 2 — upload to GCS
    gcs_uri = upload_to_gcs(PROCESSED_CSV)

    if args.upload_only:
        print("\n[done] Upload complete. Run without --upload-only to start training.")
        return

    # Step 3 — create Vertex AI dataset
    dataset = create_dataset(gcs_uri)

    # Step 4 — start AutoML training job
    start_training(dataset)


if __name__ == "__main__":
    main()
