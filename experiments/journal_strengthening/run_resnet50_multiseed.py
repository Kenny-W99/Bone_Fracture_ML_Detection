"""Guarded ResNet50 multi-seed training stability experiment.

This script does nothing expensive unless RUN_MULTISEED=1 is set.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from _common import EXP_DIR, RESULTS_DIR, ROOT, TABLES_DIR, ensure_dirs, format_float, save_latex_table, write_text


SEEDS = [42, 123, 2026]


def run_cmd(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ensure_dirs()
    base = EXP_DIR / "multiseed"
    base.mkdir(parents=True, exist_ok=True)

    if os.environ.get("RUN_MULTISEED") != "1":
        write_text(
            EXP_DIR / "resnet50_multiseed_summary.md",
            "\n".join(
                [
                    "# ResNet50 Multi-Seed Summary",
                    "",
                    "Multi-seed training was not run because `RUN_MULTISEED=1` was not set.",
                    "",
                    "Run later with:",
                    "",
                    "```bash",
                    "RUN_MULTISEED=1 python experiments/journal_strengthening/run_resnet50_multiseed.py",
                    "```",
                ]
            ),
        )
        print("RUN_MULTISEED is not set; wrote instructions only.")
        return

    rows = []
    for seed in SEEDS:
        out_dir = base / f"resnet50_seed_{seed}"
        train_cmd = [
            sys.executable,
            "src/train.py",
            "--model",
            "resnet50",
            "--train_csv",
            "data/FracAtlas/train.csv",
            "--val_csv",
            "data/FracAtlas/val.csv",
            "--test_csv",
            "data/FracAtlas/test.csv",
            "--image_dir",
            "data/FracAtlas/images",
            "--epochs",
            os.environ.get("MULTISEED_EPOCHS", "30"),
            "--batch_size",
            os.environ.get("MULTISEED_BATCH_SIZE", "32"),
            "--lr",
            "1e-4",
            "--use_pos_weight",
            "--seed",
            str(seed),
            "--output_dir",
            str(out_dir),
        ]
        run_cmd(train_cmd)
        eval_dir = out_dir / "evaluation"
        eval_cmd = [
            sys.executable,
            "src/evaluate.py",
            "--checkpoint",
            str(out_dir / "best_model.pth"),
            "--model",
            "resnet50",
            "--test_csv",
            "data/FracAtlas/test.csv",
            "--image_dir",
            "data/FracAtlas/images",
            "--output_dir",
            str(eval_dir),
        ]
        run_cmd(eval_cmd)
        metrics_path = eval_dir / "test_metrics.csv"
        metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
        metrics["seed"] = seed
        rows.append(metrics)
        with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "resnet50_multiseed_results.csv", index=False)
    metrics = ["accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "roc_auc"]
    summary_rows = []
    for metric in metrics:
        summary_rows.append({"metric": metric, "mean": df[metric].mean(), "std": df[metric].std(ddof=1)})
    summary = pd.DataFrame(summary_rows)
    table = summary.copy()
    table["mean ± std"] = table.apply(lambda r: f"{format_float(r['mean'])} ± {format_float(r['std'])}", axis=1)
    save_latex_table(table[["metric", "mean ± std"]], TABLES_DIR / "resnet50_multiseed_table.tex", "ResNet50 multi-seed stability.", "tab:resnet50_multiseed")
    write_text(EXP_DIR / "resnet50_multiseed_summary.md", "# ResNet50 Multi-Seed Summary\n\nMulti-seed training was run and aggregated in `results/resnet50_multiseed_results.csv`.")
    print("ResNet50 multi-seed experiment complete.")


if __name__ == "__main__":
    main()
