"""
Append one evaluated VLM result to the existing CNN results summary CSV.

Example:
    python src/append_vlm_to_summary.py \
        --cnn_summary outputs/results_summary.csv \
        --vlm_metrics outputs/vlm/openai_simple_eval/test_metrics.csv \
        --output_csv outputs/results_summary_with_vlm.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cnn_summary", required=True)
    parser.add_argument("--vlm_metrics", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model_label", default=None)
    args = parser.parse_args()

    cnn = pd.read_csv(args.cnn_summary)
    vlm = pd.read_csv(args.vlm_metrics).iloc[0]

    model_label = args.model_label or f"{vlm.get('model', 'vlm')} ({vlm.get('prompt_mode', 'prompt')})"
    row = {
        "model": model_label,
        "epochs": "zero-shot",
        "accuracy": vlm["accuracy"],
        "precision": vlm["precision"],
        "recall": vlm["recall_sensitivity"],
        "specificity": vlm["specificity"],
        "f1_score": vlm["f1_score"],
        "roc_auc": vlm["roc_auc"],
        "tp": int(vlm["true_positives"]),
        "fp": int(vlm["false_positives"]),
        "tn": int(vlm["true_negatives"]),
        "fn": int(vlm["false_negatives"]),
    }

    out = pd.concat([cnn, pd.DataFrame([row])], ignore_index=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Saved combined summary to {output_csv}")


if __name__ == "__main__":
    main()
