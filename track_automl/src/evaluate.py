"""
track_automl/src/evaluate.py

Retrieves evaluation metrics and feature importance for a trained
Vertex AI AutoML model.

Usage:
    # Evaluate the most recently trained model
    python track_automl/src/evaluate.py

    # Evaluate a specific model by display name
    python track_automl/src/evaluate.py --model-name telco-churn-automl

    # Export a summary to a JSON file
    python track_automl/src/evaluate.py --export evaluation_summary.json
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime

from dotenv import load_dotenv
from google.cloud import aiplatform  # type: ignore[attr-defined]

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION", "us-central1")
MODEL_DISPLAY_NAME = "telco-churn-automl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def init_vertex() -> None:
    assert PROJECT_ID is not None
    aiplatform.init(project=PROJECT_ID, location=REGION)


def get_model(display_name: str) -> aiplatform.Model:
    """Return the most recently created model matching display_name."""
    models = aiplatform.Model.list(
        filter=f'display_name="{display_name}"',
        order_by="create_time desc",
    )
    if not models:
        print(f"[error] No model found with display name '{display_name}'.")
        print("        Has the training job finished? Check with:")
        print("        python track_automl/src/train.py --status JOB_RESOURCE_NAME")
        sys.exit(1)

    model = models[0]
    print(f"[model] Found: {model.display_name}")
    print(f"        Resource: {model.resource_name}")
    print(f"        Created : {model.create_time}")
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def print_classification_metrics(evaluation: aiplatform.ModelEvaluation) -> dict:
    """Print and return classification metrics from a model evaluation."""
    metrics = dict(evaluation.metrics)

    print("\n── Classification metrics ──────────────────────────────")

    # Core metrics
    core = {
        "auRoc": "AUC-ROC      ",
        "auPrc": "AUC-PRC      ",
        "logLoss": "Log loss     ",
    }
    for key, label in core.items():
        if key in metrics:
            print(f"  {label}: {metrics[key]:.4f}")

    # Threshold-dependent metrics (at default 0.5 threshold)
    if "confidenceMetrics" in metrics:
        # find the entry closest to 0.5 threshold
        cm = metrics["confidenceMetrics"]
        closest = min(cm, key=lambda x: abs(x.get("confidenceThreshold", 0) - 0.5))
        threshold = closest.get("confidenceThreshold", "?")
        print(f"\n  At threshold {threshold:.2f}:")
        for metric in ("precision", "recall", "f1Score", "falsePositiveRate"):
            if metric in closest:
                print(f"    {metric:<22}: {closest[metric]:.4f}")

    print("────────────────────────────────────────────────────────")
    return metrics


def print_feature_importance(evaluation: aiplatform.ModelEvaluation) -> list[dict]:
    """Print feature importance from evaluation metrics."""
    try:
        fi = dict(evaluation.metrics).get("featureImportance")
        if not fi:
            print("\n[info] No feature importance data available yet.")
            return []

        sorted_features = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)

        print("\n── Feature importance ──────────────────────────────────")
        max_val = abs(sorted_features[0][1]) if sorted_features else 1
        for feature, importance in sorted_features:
            bar_len = int(abs(importance) / max_val * 30)
            bar = "█" * bar_len
            print(f"  {feature:<25} {bar:<30} {importance:+.4f}")
        print("────────────────────────────────────────────────────────")

        return [{"feature": f, "importance": v} for f, v in sorted_features]

    except Exception as e:
        print(f"\n[info] Could not retrieve feature importance: {e}")
        return []


# ---------------------------------------------------------------------------
# Main evaluation flow
# ---------------------------------------------------------------------------


def evaluate(model_display_name: str, export_path: str | None = None) -> dict:
    """
    Pull evaluation metrics and feature importance for a trained model.

    Args:
        model_display_name: Display name of the model in Vertex AI.
        export_path:        Optional path to write a JSON summary.

    Returns:
        A dict with metrics and feature importance.
    """
    init_vertex()
    model = get_model(model_display_name)

    # Get model evaluations
    evaluations = model.list_model_evaluations()
    evaluations = list(evaluations)

    if not evaluations:
        print("[error] No evaluations found. The model may still be training.")
        sys.exit(1)

    evaluation = evaluations[0]
    print(f"\n[eval] Evaluation slice: {evaluation.display_name or 'overall'}")

    metrics = print_classification_metrics(evaluation)
    feature_importance = print_feature_importance(evaluation)

    summary = {
        "model_display_name": model_display_name,
        "model_resource_name": model.resource_name,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "metrics": {
            "auc_roc": metrics.get("auRoc"),
            "auc_prc": metrics.get("auPrc"),
            "log_loss": metrics.get("logLoss"),
        },
        "feature_importance": feature_importance,
    }

    if export_path:
        with open(export_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[export] Summary written to {export_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def validate_env() -> None:
    if not PROJECT_ID:
        print("[error] GCP_PROJECT_ID is not set in .env")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Vertex AI AutoML model")
    parser.add_argument(
        "--model-name",
        default=MODEL_DISPLAY_NAME,
        help=f"Model display name (default: {MODEL_DISPLAY_NAME})",
    )
    parser.add_argument(
        "--export",
        metavar="PATH",
        help="Export evaluation summary to a JSON file",
    )
    args = parser.parse_args()

    validate_env()
    evaluate(args.model_name, export_path=args.export)


if __name__ == "__main__":
    main()
