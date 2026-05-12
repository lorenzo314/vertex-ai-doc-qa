"""
track_automl/src/predict.py

Runs predictions against a deployed Vertex AI AutoML model endpoint.

Two modes:
- Single prediction: pass customer features as CLI arguments or use the
  built-in example customer profiles.
- Batch prediction: pass a CSV file of customers to score.

Usage:
    # Run a single prediction using a built-in example profile
    python track_automl/src/predict.py --example high-risk

    # Run a single prediction using a built-in example profile
    python track_automl/src/predict.py --example low-risk

    # Batch predict from a CSV file
    python track_automl/src/predict.py --batch path/to/customers.csv

    # List available deployed endpoints
    python track_automl/src/predict.py --list-endpoints

NOTE: The model must be deployed to an endpoint before predictions can be made.
      Deploy from the GCP console: Vertex AI → Model Registry → telco-churn-automl
      → Deploy to endpoint, or use the deploy() function below.
"""

import argparse
import os
import sys
from typing import Any

from dotenv import load_dotenv
from google.cloud import aiplatform  # type: ignore[attr-defined]

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION", "us-central1")
MODEL_DISPLAY_NAME = "telco-churn-automl"
ENDPOINT_DISPLAY_NAME = "telco-churn-automl_endpoint"
# ---------------------------------------------------------------------------
# Example customer profiles for quick testing
# ---------------------------------------------------------------------------

EXAMPLE_PROFILES: dict[str, dict[str, Any]] = {
    "high-risk": {
        "description": "Month-to-month contract, high charges, no security services",
        "features": {
            "gender": "Female",
            "SeniorCitizen": "0",
            "Partner": "No",
            "Dependents": "No",
            "tenure": "2",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": "85.0",
            "TotalCharges": "170.0",
        },
    },
    "low-risk": {
        "description": "Two-year contract, long tenure, bundled services",
        "features": {
            "gender": "Male",
            "SeniorCitizen": "0",
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": "60",
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": "65.0",
            "TotalCharges": "3900.0",
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def init_vertex() -> None:
    assert PROJECT_ID is not None
    aiplatform.init(project=PROJECT_ID, location=REGION)


def get_endpoint(display_name: str) -> aiplatform.Endpoint:
    """Return the first endpoint matching display_name."""
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{display_name}"',
        order_by="create_time desc",
    )
    if not endpoints:
        print(f"[error] No endpoint found with display name '{display_name}'.")
        print("        Deploy your model first:")
        print("        GCP Console → Vertex AI → Model Registry → telco-churn-automl")
        print("        → Deploy to endpoint")
        sys.exit(1)
    return endpoints[0]


def list_endpoints() -> None:
    """Print all available endpoints in the project."""
    init_vertex()
    endpoints = aiplatform.Endpoint.list(order_by="create_time desc")
    if not endpoints:
        print("[info] No endpoints found.")
        return
    print(f"\n{'Display name':<35} {'Resource name'}")
    print("-" * 80)
    for ep in endpoints:
        print(f"{ep.display_name:<35} {ep.resource_name}")
    print()


# ---------------------------------------------------------------------------
# Deployment helper (optional — can also deploy via console)
# ---------------------------------------------------------------------------


def deploy_model() -> aiplatform.Endpoint:
    """
    Deploy the trained model to a new endpoint.

    Uses the smallest machine type (n1-standard-2) to minimise costs.
    Note: a deployed endpoint incurs hourly charges (~$0.10/hr) even when
    idle. Undeploy when not in use.
    """
    assert PROJECT_ID is not None
    init_vertex()

    models = aiplatform.Model.list(
        filter=f'display_name="{MODEL_DISPLAY_NAME}"',
        order_by="create_time desc",
    )
    if not models:
        print(f"[error] Model '{MODEL_DISPLAY_NAME}' not found.")
        sys.exit(1)

    model = models[0]
    print(f"[deploy] Deploying {model.display_name} to endpoint '{ENDPOINT_DISPLAY_NAME}' ...")
    print("[deploy] This takes ~10 minutes.")

    endpoint = model.deploy(
        deployed_model_display_name=MODEL_DISPLAY_NAME,
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1,
        sync=True,
    )

    print(f"[deploy] ✓ Endpoint ready: {endpoint.resource_name}")
    print("\n  ⚠️  Remember to undeploy when done to avoid ongoing charges:")
    print("  python track_automl/src/predict.py --undeploy\n")
    return endpoint


def undeploy_model() -> None:
    """Undeploy all models from the endpoint and delete it."""
    init_vertex()
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"',
    )
    if not endpoints:
        print(f"[info] No endpoint named '{ENDPOINT_DISPLAY_NAME}' found.")
        return
    for endpoint in endpoints:
        print(f"[undeploy] Undeploying all models from {endpoint.display_name} ...")
        endpoint.undeploy_all()
        endpoint.delete()
        print("[undeploy] ✓ Endpoint deleted.")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_single(features: dict, description: str = "") -> dict:
    """
    Run a single prediction against the deployed endpoint.

    Args:
        features:    Dict of feature name → value matching the training schema.
        description: Optional description for display purposes.

    Returns:
        Dict with prediction result and confidence scores.
    """
    init_vertex()
    endpoint = get_endpoint(ENDPOINT_DISPLAY_NAME)

    if description:
        print(f"\n[predict] Profile: {description}")
    print(f"[predict] Sending request to {endpoint.display_name} ...")

    response = endpoint.predict(instances=[features])

    predictions = response.predictions[0]

    # AutoML tabular returns classes and scores
    classes = predictions.get("classes", [])
    scores = predictions.get("scores", [])

    result = dict(zip(classes, scores, strict=False))

    churn_prob = float(result.get("Yes", result.get("1", 0.0)) or 0.0)
    no_churn_prob = float(result.get("No", result.get("0", 0.0)) or 0.0)

    prediction = "Yes" if churn_prob > 0.5 else "No"

    print("\n── Prediction result ───────────────────────────────────")
    print(f"  Churn prediction : {prediction}")
    print(f"  Churn probability: {churn_prob:.1%}")
    print(f"  Retain probability: {no_churn_prob:.1%}")
    bar_len = int(churn_prob * 40)
    bar = "█" * bar_len + "░" * (40 - bar_len)
    print(f"  Risk             : [{bar}] {churn_prob:.1%}")
    print("────────────────────────────────────────────────────────\n")

    return {
        "prediction": prediction,
        "churn_probability": churn_prob,
        "retain_probability": no_churn_prob,
        "raw_scores": result,
    }


def predict_batch(csv_path: str, output_path: str | None = None) -> None:
    """
    Run batch predictions from a CSV file using Vertex AI Batch Prediction.
    The CSV must have the same columns as the training data (without Churn).

    Args:
        csv_path:    Path to the input CSV file.
        output_path: Optional GCS URI for output (defaults to bucket/predictions/).
    """
    bucket = os.getenv("GCS_BUCKET")
    assert bucket is not None, "GCS_BUCKET is not set in .env"
    init_vertex()

    models = aiplatform.Model.list(
        filter=f'display_name="{MODEL_DISPLAY_NAME}"',
        order_by="create_time desc",
    )
    if not models:
        print(f"[error] Model '{MODEL_DISPLAY_NAME}' not found.")
        sys.exit(1)

    model = models[0]

    # Upload input CSV to GCS
    from google.cloud import storage as gcs  # type: ignore[attr-defined]

    client = gcs.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket)
    blob_name = f"track_automl/predictions/input/{os.path.basename(csv_path)}"
    bucket.blob(blob_name).upload_from_filename(csv_path, content_type="text/csv")
    gcs_input = f"gs://{bucket}/{blob_name}"
    gcs_output = output_path or f"gs://{bucket}/track_automl/predictions/output/"

    print(f"[batch] Input : {gcs_input}")
    print(f"[batch] Output: {gcs_output}")
    print("[batch] Submitting batch prediction job ...")

    batch_job = model.batch_predict(
        job_display_name="telco-churn-batch",
        gcs_source=gcs_input,
        gcs_destination_prefix=gcs_output,
        instances_format="csv",
        predictions_format="jsonl",
        sync=False,
    )

    print(f"[batch] ✓ Job submitted: {batch_job.display_name}")
    print(f"[batch] Results will appear at: {gcs_output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def validate_env() -> None:
    if not PROJECT_ID:
        print("[error] GCP_PROJECT_ID is not set in .env")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run predictions against a deployed Vertex AI AutoML churn model"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--example",
        choices=list(EXAMPLE_PROFILES.keys()),
        help="Run a prediction using a built-in example profile",
    )
    group.add_argument(
        "--batch",
        metavar="CSV_PATH",
        help="Run batch predictions from a CSV file",
    )
    group.add_argument(
        "--list-endpoints",
        action="store_true",
        help="List available deployed endpoints",
    )
    group.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy the trained model to an endpoint",
    )
    group.add_argument(
        "--undeploy",
        action="store_true",
        help="Undeploy the model and delete the endpoint",
    )
    args = parser.parse_args()

    validate_env()

    if args.list_endpoints:
        list_endpoints()
    elif args.deploy:
        deploy_model()
    elif args.undeploy:
        undeploy_model()
    elif args.example:
        profile = EXAMPLE_PROFILES[args.example]
        predict_single(profile["features"], description=profile["description"])
    elif args.batch:
        predict_batch(args.batch)
    else:
        # Default: run both example profiles to show contrast
        for profile_name, profile_data in EXAMPLE_PROFILES.items():
            predict_single(
                features=profile_data["features"],
                description=f"{profile_name} — {profile_data['description']}",
            )


if __name__ == "__main__":
    main()
