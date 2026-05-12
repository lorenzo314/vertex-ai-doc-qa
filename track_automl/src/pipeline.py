"""
track_automl/src/pipeline.py

Orchestrates the full AutoML workflow as a Vertex AI Pipeline:
    preprocess → upload → create dataset → train → evaluate

Each step is a reusable pipeline component. The pipeline is compiled to a
YAML file and submitted to Vertex AI Pipelines for execution.

Usage:
    # Compile only (no run) — safe to do anytime
    python track_automl/src/pipeline.py --compile-only

    # Compile and run the full pipeline on Vertex AI
    python track_automl/src/pipeline.py

    # Check status of a running pipeline
    python track_automl/src/pipeline.py --status PIPELINE_JOB_NAME

Requirements:
    pip install kfp google-cloud-pipeline-components
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION", "us-central1")
BUCKET = os.getenv("GCS_BUCKET")

PIPELINE_NAME = "telco-churn-pipeline"
PIPELINE_FILE = Path("track_automl/telco_churn_pipeline.yaml")
TARGET_COLUMN = "Churn"
MODEL_DISPLAY_NAME = "telco-churn-automl-pipeline"
DATASET_DISPLAY_NAME = "telco-churn-pipeline"
TRAINING_BUDGET_NODE_HOURS = 1


# ---------------------------------------------------------------------------
# Validate environment
# ---------------------------------------------------------------------------


def validate_env() -> None:
    missing = [v for v in ("GCP_PROJECT_ID", "GCS_BUCKET") if not os.getenv(v)]
    if missing:
        print(f"[error] Missing environment variables: {', '.join(missing)}")
        sys.exit(1)
    try:
        import kfp  # noqa: F401
        from google_cloud_pipeline_components.v1.automl.training_job import (  # noqa: F401
            AutoMLTabularTrainingJobRunOp,
        )
    except ImportError:
        print("[error] Required packages not installed.")
        print("        Run: pip install kfp google-cloud-pipeline-components")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------


def build_pipeline():  # type: ignore[return]
    from google_cloud_pipeline_components.v1.automl.training_job import (
        AutoMLTabularTrainingJobRunOp,
    )
    from google_cloud_pipeline_components.v1.dataset import TabularDatasetCreateOp
    from kfp.dsl import Metrics, Output, component, pipeline

    # ── Step 1: Preprocess and upload ───────────────────────────────────────
    # Returns a plain string (GCS URI) so TabularDatasetCreateOp can consume it.

    @component(
        base_image="python:3.12-slim",
        packages_to_install=["pandas==2.2.2", "google-cloud-storage==2.18.0"],
    )
    def preprocess_and_upload(
        project: str,
        bucket: str,
        gcs_prefix: str,
    ) -> str:
        """
        Download the raw CSV from GCS, clean it, re-upload the processed
        version, and return its GCS URI as a plain string.
        """
        from pathlib import Path

        import pandas as pd
        from google.cloud import storage  # type: ignore[attr-defined]

        raw_path = Path("/tmp/raw.csv")
        clean_path = Path("/tmp/telco_churn_clean.csv")

        client = storage.Client(project=project)
        bucket_obj = client.bucket(bucket)

        # Download raw CSV
        blob = bucket_obj.blob(f"{gcs_prefix}/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        blob.download_to_filename(str(raw_path))

        # Clean
        df = pd.read_csv(raw_path)
        df = df.drop(columns=["customerID"])
        df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
        df.to_csv(clean_path, index=False)

        # Upload cleaned CSV
        dest_blob = bucket_obj.blob(f"{gcs_prefix}/processed/telco_churn_clean.csv")
        dest_blob.upload_from_filename(str(clean_path), content_type="text/csv")

        gcs_uri = f"gs://{bucket}/{gcs_prefix}/processed/telco_churn_clean.csv"
        print(f"[preprocess] ✓ Uploaded to {gcs_uri}")
        return gcs_uri

    # ── Step 4: Evaluate ────────────────────────────────────────────────────

    @component(
        base_image="python:3.12-slim",
        packages_to_install=["google-cloud-aiplatform==1.90.0"],
    )
    def evaluate_model(
        project: str,
        region: str,
        model_display_name: str,
        metrics: Output[Metrics],
    ) -> None:
        """Retrieve and log model evaluation metrics to the pipeline UI."""
        from google.cloud import aiplatform

        aiplatform.init(project=project, location=region)

        models = aiplatform.Model.list(
            filter=f'display_name="{model_display_name}"',
            order_by="create_time desc",
        )
        if not models:
            print(f"[evaluate] Model '{model_display_name}' not found.")
            return

        evaluations = list(models[0].list_model_evaluations())
        if not evaluations:
            print("[evaluate] No evaluations found.")
            return

        m = dict(evaluations[0].metrics)
        auc_roc = float(m.get("auRoc", 0.0))
        auc_prc = float(m.get("auPrc", 0.0))
        log_loss = float(m.get("logLoss", 0.0))

        metrics.log_metric("auc_roc", auc_roc)
        metrics.log_metric("auc_prc", auc_prc)
        metrics.log_metric("log_loss", log_loss)

        print(f"[evaluate] AUC-ROC : {auc_roc:.4f}")
        print(f"[evaluate] AUC-PRC : {auc_prc:.4f}")
        print(f"[evaluate] Log loss: {log_loss:.4f}")

    # ── Pipeline ────────────────────────────────────────────────────────────

    @pipeline(
        name=PIPELINE_NAME,
        description=("Telco churn classification: preprocess → dataset → train → evaluate"),
    )
    def telco_churn_pipeline(
        project: str = PROJECT_ID or "",
        region: str = REGION,
        bucket: str = BUCKET or "",
        gcs_prefix: str = "track_automl/data",
        target_column: str = TARGET_COLUMN,
        model_display_name: str = MODEL_DISPLAY_NAME,
        dataset_display_name: str = DATASET_DISPLAY_NAME,
        budget_milli_node_hours: int = TRAINING_BUDGET_NODE_HOURS * 1000,
    ) -> None:

        # Step 1 — preprocess and upload; returns a plain string URI
        preprocess_task = preprocess_and_upload(
            project=project,
            bucket=bucket,
            gcs_prefix=gcs_prefix,
        )

        # Step 2 — create Vertex AI tabular dataset from the URI string
        dataset_task = TabularDatasetCreateOp(
            project=project,
            display_name=dataset_display_name,
            gcs_source=preprocess_task.output,  # plain string output
        ).after(preprocess_task)

        # Step 3 — AutoML training
        training_task = AutoMLTabularTrainingJobRunOp(
            project=project,
            display_name=model_display_name,
            optimization_prediction_type="classification",
            optimization_objective="maximize-au-roc",
            dataset=dataset_task.outputs["dataset"],
            target_column=target_column,
            training_fraction_split=0.8,
            validation_fraction_split=0.1,
            test_fraction_split=0.1,
            budget_milli_node_hours=budget_milli_node_hours,
        ).after(dataset_task)

        # Step 4 — evaluate
        evaluate_model(
            project=project,
            region=region,
            model_display_name=model_display_name,
        ).after(training_task)

    return telco_churn_pipeline


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------


def compile_pipeline() -> None:
    import kfp

    print("[compile] Building pipeline definition ...")
    pipeline_func = build_pipeline()

    print(f"[compile] Compiling to {PIPELINE_FILE} ...")
    PIPELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=str(PIPELINE_FILE),
    )
    print(f"[compile] ✓ Pipeline compiled to {PIPELINE_FILE}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_pipeline() -> None:
    assert PROJECT_ID is not None
    assert BUCKET is not None

    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT_ID, location=REGION)

    if not PIPELINE_FILE.exists():
        compile_pipeline()

    pipeline_root = f"gs://{BUCKET}/pipeline_root/{PIPELINE_NAME}"
    print(f"[run] Submitting pipeline '{PIPELINE_NAME}' ...")
    print(f"[run] Pipeline root: {pipeline_root}")

    job = aiplatform.PipelineJob(
        display_name=PIPELINE_NAME,
        template_path=str(PIPELINE_FILE),
        pipeline_root=pipeline_root,
        enable_caching=True,
    )
    job.submit()

    print("\n[run] ✓ Pipeline submitted.")
    print("[run] Monitor at:")
    print(f"      https://console.cloud.google.com/vertex-ai/pipelines?project={PROJECT_ID}")


# ---------------------------------------------------------------------------
# Status check
# ---------------------------------------------------------------------------


def check_status(job_name: str) -> None:
    assert PROJECT_ID is not None
    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT_ID, location=REGION)
    job = aiplatform.PipelineJob.get(job_name)

    if hasattr(job, "display_name") and job.display_name is not None:
        print(f"[status] Name   : {job.display_name}")

    if hasattr(job, "state_name") and job.state_name is not None:
        assert job.state is not None
        print(f"[status] State  : {job.state.name}")

    if hasattr(job, "create_time") and job.create_time is not None:
        print(f"[status] Created: {job.create_time}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile and run the Telco churn Vertex AI Pipeline"
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile the pipeline to YAML without running it",
    )
    parser.add_argument(
        "--status",
        metavar="JOB_NAME",
        help="Check the status of a running pipeline job",
    )
    args = parser.parse_args()

    validate_env()

    if args.status:
        check_status(args.status)
    elif args.compile_only:
        compile_pipeline()
    else:
        compile_pipeline()
        run_pipeline()


if __name__ == "__main__":
    main()
