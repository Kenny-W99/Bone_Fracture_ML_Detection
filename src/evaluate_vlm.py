"""
Evaluate VLM Baseline Predictions for FracAtlas
===============================================

This script computes the same metrics used by the CNN experiments for a VLM
baseline predictions CSV:
    - accuracy
    - precision
    - recall/sensitivity
    - specificity
    - F1-score
    - ROC-AUC, if fracture_probability is available
    - confusion matrix counts

Example:
    python src/evaluate_vlm.py \
        --predictions_csv outputs/vlm/openai_simple_predictions.csv \
        --output_dir outputs/vlm/openai_simple_eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def plot_confusion_matrix(y_true, y_pred, save_path: Path, title_suffix: str = "VLM Baseline") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    colors = np.array([
        [0.85, 0.95, 0.85],
        [1.00, 0.86, 0.70],
        [1.00, 0.70, 0.70],
        [0.75, 0.90, 0.75],
    ]).reshape(2, 2, 3)
    ax.imshow(colors, interpolation="nearest")

    labels_text = [
        [f"TN\n{tn}\n(correct normal)", f"FP\n{fp}\n(false alarm)"],
        [f"FN\n{fn}\n(MISSED FRACTURE)", f"TP\n{tp}\n(caught fracture)"],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels_text[i][j], ha="center", va="center", fontsize=11, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: No Fracture", "Predicted: Fracture"])
    ax.set_yticklabels(["Actual: No Fracture", "Actual: Fracture"])
    ax.set_title(f"Confusion Matrix — {title_suffix}")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path: Path, title_suffix: str = "VLM Baseline") -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6.5, 5.6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--", label="Random (AUC = 0.5)")
    plt.fill_between(fpr, tpr, alpha=0.12)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title(f"ROC Curve — {title_suffix}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def evaluate(args: argparse.Namespace) -> None:
    predictions_csv = Path(args.predictions_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions_csv)
    required = {"true_label", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    # Drop rows where API failed or JSON parsing failed.
    valid = df.dropna(subset=["true_label", "prediction"]).copy()
    valid["true_label"] = valid["true_label"].astype(int)
    valid["prediction"] = valid["prediction"].astype(int)

    if len(valid) == 0:
        raise ValueError("No valid predictions to evaluate.")

    y_true = valid["true_label"].to_numpy()
    y_pred = valid["prediction"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    roc_auc = np.nan
    if "fracture_probability" in valid.columns and valid["fracture_probability"].notna().any():
        y_prob = valid["fracture_probability"].astype(float).clip(0, 1).to_numpy()
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            plot_roc_curve(y_true, y_prob, figures_dir / "roc_curve.png", title_suffix=args.title)
        except ValueError:
            roc_auc = np.nan
    else:
        y_prob = y_pred.astype(float)

    metrics = {
        "model": args.model_name or infer_model_name(df),
        "prompt_mode": infer_prompt_mode(df),
        "n_total_rows": int(len(df)),
        "n_valid_predictions": int(len(valid)),
        "n_failed_predictions": int(len(df) - len(valid)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    pd.DataFrame([metrics]).to_csv(output_dir / "test_metrics.csv", index=False)
    valid.to_csv(output_dir / "valid_predictions.csv", index=False)
    plot_confusion_matrix(y_true, y_pred, figures_dir / "confusion_matrix.png", title_suffix=args.title)

    print("=" * 72)
    print("  VLM BASELINE EVALUATION")
    print("=" * 72)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:24s}: {v:.4f}")
        else:
            print(f"{k:24s}: {v}")
    print("=" * 72)
    print(f"Saved metrics          : {output_dir / 'test_metrics.csv'}")
    print(f"Saved confusion matrix : {figures_dir / 'confusion_matrix.png'}")
    if not np.isnan(metrics["roc_auc"]):
        print(f"Saved ROC curve        : {figures_dir / 'roc_curve.png'}")


def infer_model_name(df: pd.DataFrame) -> str:
    if "provider" in df.columns and "model" in df.columns:
        providers = sorted(set(df["provider"].dropna().astype(str)))
        models = sorted(set(df["model"].dropna().astype(str)))
        if providers and models:
            return f"{providers[0]}:{models[0]}"
    return "vlm_baseline"


def infer_prompt_mode(df: pd.DataFrame) -> str:
    if "prompt_mode" in df.columns:
        modes = sorted(set(df["prompt_mode"].dropna().astype(str)))
        if modes:
            return modes[0]
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a VLM predictions CSV against FracAtlas labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--title", default="VLM Baseline")
    parser.add_argument("--model_name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
