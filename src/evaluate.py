"""
FracAtlas — Evaluation Script
===============================

Loads a trained checkpoint and evaluates it on the test set.

Outputs:
  • Classification report (accuracy, precision, recall, specificity, F1, AUC)
  • Confusion matrix figure
  • ROC curve figure
  • Per-image predictions CSV

Why False Negatives Matter in Fracture Detection:
-------------------------------------------------
In medical imaging, a FALSE NEGATIVE means the model says "no fracture" when
there IS a fracture. This is far more dangerous than a false positive because:

  • A missed fracture → no treatment → bone heals incorrectly → chronic pain,
    disability, or need for surgery later.
  • A false positive → unnecessary follow-up imaging → minor inconvenience and
    cost, but NO patient harm.

Therefore, we prioritize RECALL (sensitivity) — the fraction of actual fractures
that the model correctly identifies. A model with 95% recall and 80% precision
is MUCH safer than one with 80% recall and 95% precision.

This is why we also report SPECIFICITY (true negative rate) separately — to
understand the trade-off between catching all fractures and avoiding false alarms.

Usage:
------
    python src/evaluate.py \\
        --checkpoint outputs/best_model.pth \\
        --test_csv data/FracAtlas/test.csv \\
        --image_dir data/FracAtlas/images \\
        --output_dir outputs

Author : FracAtlas Project
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Our project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import FracAtlasDataset, get_val_transforms, create_dataloader
from src.models import get_model


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """
    Run inference on the entire test set.

    Returns
    -------
    tuple
        (all_labels, all_preds, all_probs, all_filenames)
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in dataloader:
        images = images.to(device)

        outputs = model(images).squeeze(1)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# =============================================================================
# CONFUSION MATRIX PLOT
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Plot and save a confusion matrix with annotations.

    The matrix shows:
      ┌─────────────────┬──────────────────┐
      │  True Negative   │  False Positive  │
      │  (correct normal)│  (false alarm)   │
      ├─────────────────┼──────────────────┤
      │  False Negative  │  True Positive   │
      │  (MISSED FRAC!)  │  (caught frac)   │
      └─────────────────┴──────────────────┘

    False Negatives (bottom-left) are the MOST DANGEROUS in medical imaging.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(7, 6))

    # Color-coded: red for dangerous errors (FN), orange for FP
    colors = np.array([
        [0.85, 0.95, 0.85],  # TN — light green (good)
        [1.0, 0.85, 0.7],    # FP — light orange (minor error)
        [1.0, 0.7, 0.7],     # FN — light red (DANGEROUS)
        [0.75, 0.9, 0.75],   # TP — green (good)
    ]).reshape(2, 2, 3)

    ax.imshow(colors, interpolation="nearest")

    # Annotate cells
    labels_text = [
        [f"TN\n{tn}\n(correct normal)", f"FP\n{fp}\n(false alarm)"],
        [f"FN\n{fn}\n(MISSED FRACTURE!)", f"TP\n{tp}\n(caught fracture)"],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels_text[i][j], ha="center", va="center",
                    fontsize=12, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: No Fracture", "Predicted: Fracture"])
    ax.set_yticklabels(["Actual: No Fracture", "Actual: Fracture"])
    ax.set_title("Confusion Matrix — Fracture Detection\n"
                 "(Red = dangerous missed fractures)", fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Confusion matrix saved: {save_path}")


# =============================================================================
# ROC CURVE PLOT
# =============================================================================

def plot_roc_curve(y_true, y_prob, save_path):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    The ROC curve shows the trade-off between:
      • True Positive Rate (Sensitivity/Recall) on the y-axis
      • False Positive Rate (1 - Specificity) on the x-axis

    A perfect model hugs the top-left corner (AUC = 1.0).
    A random model follows the diagonal (AUC = 0.5).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--",
             label="Random (AUC = 0.5)")
    plt.fill_between(fpr, tpr, alpha=0.1, color="darkorange")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("ROC Curve — Fracture Detection")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 ROC curve saved: {save_path}")


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate(args):
    """
    Main evaluation function.

    Steps:
      1. Load checkpoint and reconstruct model
      2. Load test dataset
      3. Run inference
      4. Compute and print all metrics
      5. Save confusion matrix, ROC curve, and predictions CSV
    """
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n{'='*60}")
    print(f"  FRACATLAS TEST EVALUATION")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    # -------------------------------------------------------------------------
    # Load checkpoint
    # -------------------------------------------------------------------------
    print(f"\n  Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Get model name from checkpoint (or from args)
    model_name = checkpoint.get("model_name", args.model)
    print(f"  Model architecture: {model_name}")
    print(f"  Trained for {checkpoint.get('epoch', '?')} epochs")
    print(f"  Best val F1: {checkpoint.get('val_f1', '?')}")

    # -------------------------------------------------------------------------
    # Reconstruct model and load weights
    # -------------------------------------------------------------------------
    model = get_model(model_name, pretrained=False)  # Don't need ImageNet weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # Load test data
    # -------------------------------------------------------------------------
    print(f"\n  Loading test data: {args.test_csv}")
    test_dataset = FracAtlasDataset(
        csv_path=args.test_csv,
        img_dir=args.image_dir,
        transform=get_val_transforms(),
    )
    test_loader = create_dataloader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # -------------------------------------------------------------------------
    # Run inference
    # -------------------------------------------------------------------------
    print(f"  Running inference on {len(test_dataset)} images...")
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)

    # -------------------------------------------------------------------------
    # Compute metrics
    # -------------------------------------------------------------------------
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc_score = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_score = 0.0

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc_score,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_samples": len(y_true),
    }

    # Print results
    print(f"\n{'='*60}")
    print(f"  TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"  {'Metric':<25s} {'Value':>10s}")
    print(f"  {'-'*25} {'-'*10}")
    print(f"  {'Accuracy':<25s} {metrics['accuracy']:>10.4f}")
    print(f"  {'Precision':<25s} {metrics['precision']:>10.4f}")
    print(f"  {'Recall (Sensitivity)':<25s} {metrics['recall_sensitivity']:>10.4f}")
    print(f"  {'Specificity':<25s} {metrics['specificity']:>10.4f}")
    print(f"  {'F1-Score':<25s} {metrics['f1_score']:>10.4f}")
    print(f"  {'ROC-AUC':<25s} {metrics['roc_auc']:>10.4f}")
    print(f"  {'-'*25} {'-'*10}")
    print(f"  {'True Positives (TP)':<25s} {metrics['true_positives']:>10d}")
    print(f"  {'False Positives (FP)':<25s} {metrics['false_positives']:>10d}")
    print(f"  {'True Negatives (TN)':<25s} {metrics['true_negatives']:>10d}")
    print(f"  {'False Negatives (FN)':<25s} {metrics['false_negatives']:>10d}")
    print(f"  {'Total Samples':<25s} {metrics['total_samples']:>10d}")

    # -------------------------------------------------------------------------
    # Clinical interpretation
    # -------------------------------------------------------------------------
    print(f"\n  📋 CLINICAL INTERPRETATION:")
    print(f"     • Recall = {metrics['recall_sensitivity']:.1%} of fractures were correctly detected.")
    print(f"     • {metrics['false_negatives']} fractures were MISSED (false negatives).")
    print(f"     • {metrics['false_positives']} normal images were incorrectly flagged (false positives).")
    if metrics["false_negatives"] > 0:
        print(f"     ⚠️  Each missed fracture is a potential patient safety issue!")
        print(f"        Consider: lowering the threshold, using pos_weight, or ensembling.")

    # -------------------------------------------------------------------------
    # Save confusion matrix figure
    # -------------------------------------------------------------------------
    cm_path = str(figures_dir / "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, cm_path)

    # -------------------------------------------------------------------------
    # Save ROC curve figure
    # -------------------------------------------------------------------------
    roc_path = str(figures_dir / "roc_curve.png")
    plot_roc_curve(y_true, y_prob, roc_path)

    # -------------------------------------------------------------------------
    # Save predictions CSV
    # -------------------------------------------------------------------------
    predictions_df = pd.DataFrame({
        "filename": test_dataset.filenames,
        "true_label": y_true.astype(int),
        "predicted_label": y_pred.astype(int),
        "predicted_probability": y_prob.round(4),
        "correct": (y_true == y_pred).astype(int),
    })
    predictions_path = output_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"\n  📄 Predictions saved: {predictions_path}")

    # Save metrics summary as JSON-like CSV
    metrics_summary_path = output_dir / "test_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_summary_path, index=False)
    print(f"  📄 Metrics summary: {metrics_summary_path}")

    print(f"\n{'='*60}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*60}\n")


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained FracAtlas model on the test set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/evaluate.py \\
      --checkpoint outputs/best_model.pth \\
      --test_csv data/FracAtlas/test.csv \\
      --image_dir data/FracAtlas/images \\
      --output_dir outputs

  # Specify model explicitly (if not saved in checkpoint):
  python src/evaluate.py \\
      --checkpoint outputs/best_model.pth \\
      --model densenet121 \\
      --test_csv data/test.csv \\
      --image_dir data/images
        """,
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the trained model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--model", type=str, default="resnet50",
        choices=["custom_cnn", "mobilenet_v2", "resnet50", "densenet121"],
        help="Model architecture (default: resnet50). Overridden by checkpoint if available.",
    )
    parser.add_argument(
        "--test_csv", type=str, required=True,
        help="Path to test CSV file",
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Path to image directory",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory for output figures and predictions (default: outputs)",
    )

    return parser.parse_args()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
