"""Export training metrics to reports/evaluation.md."""

from __future__ import annotations

import json
import os
import sys

import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.paths import get_models_dir

REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "evaluation.md")
METRICS_PATH = os.path.join(PROJECT_ROOT, "reports", "metrics.json")


def collect_metrics() -> dict:
    models_dir = get_models_dir()
    metrics: dict = {"tasks": {}}

    cls_path = os.path.join(models_dir, "classification_model.pkl")
    if os.path.exists(cls_path):
        bundle = joblib.load(cls_path)
        metrics["tasks"]["priority_classification"] = {
            "model": bundle.get("model_name", "unknown"),
            "accuracy": round(float(bundle.get("accuracy", 0)), 4),
            "target": ">= 0.80",
        }

    reg_path = os.path.join(models_dir, "regression_model.pkl")
    if os.path.exists(reg_path):
        bundle = joblib.load(reg_path)
        metrics["tasks"]["resolution_regression"] = {
            "model": bundle.get("model_name", "unknown"),
            "r2": round(float(bundle.get("r2", 0)), 4),
            "target": ">= 0.70",
        }

    sat_path = os.path.join(models_dir, "satisfaction_model.pkl")
    if os.path.exists(sat_path):
        bundle = joblib.load(sat_path)
        metrics["tasks"]["satisfaction_classification"] = {
            "model": bundle.get("model_name", "unknown"),
            "accuracy": round(float(bundle.get("accuracy", 0)), 4),
            "target": ">= 0.75",
        }

    metrics["methodology"] = {
        "split": "80/20 train-test",
        "sample_size": "up to 30,000 rows per task",
        "labels": "engineered via src/label_engineering.py (see README caveats)",
    }
    return metrics


def write_reports() -> None:
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    metrics = collect_metrics()
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    lines = [
        "# Support Ops Intelligence — Evaluation Report",
        "",
        "## Methodology",
        "- 80/20 stratified train-test split (classification tasks)",
        "- Up to 30,000 row sample per task for laptop-friendly training",
        "- Labels engineered from business heuristics — see README caveats",
        "",
        "## Metrics",
        "",
    ]
    for task, values in metrics.get("tasks", {}).items():
        lines.append(f"### {task.replace('_', ' ').title()}")
        for key, val in values.items():
            lines.append(f"- **{key}:** {val}")
        lines.append("")

    lines.extend(
        [
            "## Limitations",
            "- Metrics reflect pipeline validity on engineered labels, not human-annotated ground truth.",
            "- Regenerate: `python src/train_models.py && python scripts/export_evaluation.py`",
            "",
        ]
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    write_reports()
