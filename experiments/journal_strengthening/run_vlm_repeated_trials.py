"""Guarded VLM repeated prompt trials.

Paid/API VLM calls run only when RUN_VLM=1. API keys are read only from provider
environment variables used by the provider SDKs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import EXP_DIR, RESULTS_DIR, ROOT, TABLES_DIR, compute_metrics, ensure_dirs, format_float, save_latex_table, write_text


PROMPT_MODES = ["simple", "conservative", "sensitive"]
TRIALS = [1, 2, 3]


def evaluate_predictions(path: Path, prompt_mode: str, trial_id: int, model: str) -> dict:
    df = pd.read_csv(path).dropna(subset=["true_label", "prediction"]).copy()
    y_true = df["true_label"].astype(int).to_numpy()
    y_prob = df["fracture_probability"].astype(float).clip(0, 1).to_numpy()
    metrics = compute_metrics(y_true, y_prob, 0.5)
    return {
        "model": model,
        "prompt_mode": prompt_mode,
        "trial_id": trial_id,
        "n_valid_predictions": int(len(df)),
        **metrics,
    }


def main() -> None:
    ensure_dirs()
    model = os.environ.get("VLM_MODEL", "gpt-4o-mini")
    if os.environ.get("RUN_VLM") != "1":
        write_text(
            EXP_DIR / "vlm_repeated_trials_summary.md",
            "\n".join(
                [
                    "# VLM Repeated Trials Summary",
                    "",
                    "Repeated VLM trials were not run because `RUN_VLM=1` was not set.",
                    "",
                    "Run later with:",
                    "",
                    "```bash",
                    "RUN_VLM=1 VLM_MODEL=gpt-4o-mini python experiments/journal_strengthening/run_vlm_repeated_trials.py",
                    "```",
                ]
            ),
        )
        print("RUN_VLM is not set; wrote instructions only.")
        return

    pred_paths = []
    metric_rows = []
    for mode in PROMPT_MODES:
        for trial_id in TRIALS:
            out_csv = RESULTS_DIR / f"vlm_{mode}_trial_{trial_id}_predictions.csv"
            cmd = [
                sys.executable,
                "src/vlm_baseline.py",
                "--provider",
                "openai",
                "--model",
                model,
                "--test_csv",
                "data/FracAtlas/test.csv",
                "--image_dir",
                "data/FracAtlas/images_vlm_clean",
                "--prompt_mode",
                mode,
                "--output_csv",
                str(out_csv),
                "--seed",
                str(2026 + trial_id),
                "--resume",
            ]
            print(" ".join(cmd))
            subprocess.run(cmd, cwd=ROOT, check=True)
            pred_paths.append(out_csv)
            metric_rows.append(evaluate_predictions(out_csv, mode, trial_id, model))

    predictions = []
    for path in pred_paths:
        df = pd.read_csv(path)
        parts = path.stem.split("_")
        df["trial_id"] = int(parts[-2] if parts[-1] == "predictions" else parts[-1])
        if "image_id" not in df.columns and "filename" in df.columns:
            df["image_id"] = df["filename"]
        if "predicted_label" not in df.columns and "prediction" in df.columns:
            df["predicted_label"] = df["prediction"]
        if "confidence" not in df.columns and "fracture_probability" in df.columns:
            df["confidence"] = df["fracture_probability"]
        predictions.append(df)
    pd.concat(predictions, ignore_index=True).to_csv(RESULTS_DIR / "vlm_repeated_trial_predictions.csv", index=False)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(RESULTS_DIR / "vlm_repeated_trial_metrics.csv", index=False)
    metric_cols = ["accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "roc_auc"]
    agg = metrics_df.groupby(["model", "prompt_mode"])[metric_cols].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(RESULTS_DIR / "vlm_repeated_trial_summary.csv", index=False)

    table = agg.copy()
    for metric in metric_cols:
        table[metric] = table.apply(lambda r: f"{format_float(r[f'{metric}_mean'])} ± {format_float(r[f'{metric}_std'])}", axis=1)
    save_latex_table(table[["model", "prompt_mode", *metric_cols]], TABLES_DIR / "vlm_repeated_trials_table.tex", "VLM repeated-trial stability.", "tab:vlm_repeated_trials")
    write_text(EXP_DIR / "vlm_repeated_trials_summary.md", "# VLM Repeated Trials Summary\n\nRepeated VLM trials were run and aggregated in `results/vlm_repeated_trial_summary.csv`.")


if __name__ == "__main__":
    main()
