"""Bootstrap 95% confidence intervals for CNN test-set metrics."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import (
    MODEL_CONFIGS,
    RESULTS_DIR,
    ROOT,
    TABLES_DIR,
    compute_metrics,
    ensure_dirs,
    format_float,
    load_predictions,
    save_latex_table,
    write_text,
)


METRICS = ["accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "roc_auc"]


def bootstrap_one(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, n_bootstrap: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rows = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        rows.append(compute_metrics(y_true[idx], y_prob[idx], threshold))
    boot = pd.DataFrame(rows)
    out = {}
    point = compute_metrics(y_true, y_prob, threshold)
    for metric in METRICS:
        values = boot[metric].dropna().to_numpy()
        out[f"{metric}_point"] = point[metric]
        if len(values):
            out[f"{metric}_ci_low"] = float(np.percentile(values, 2.5))
            out[f"{metric}_ci_high"] = float(np.percentile(values, 97.5))
            out[f"{metric}_n_boot_valid"] = int(len(values))
        else:
            out[f"{metric}_ci_low"] = np.nan
            out[f"{metric}_ci_high"] = np.nan
            out[f"{metric}_n_boot_valid"] = 0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    ensure_dirs()
    threshold_path = RESULTS_DIR / "threshold_tuning_results.csv"
    threshold_df = pd.read_csv(threshold_path) if threshold_path.exists() else pd.DataFrame()
    rows = []
    notes = []

    for model_key, cfg in MODEL_CONFIGS.items():
        try:
            test = load_predictions(model_key, "test")
        except FileNotFoundError as exc:
            notes.append(f"- `{model_key}` skipped: {exc}")
            continue
        y_true = test["true_label"].to_numpy(dtype=int)
        y_prob = test["predicted_probability"].to_numpy(dtype=float)

        if len(threshold_df):
            model_thresholds = threshold_df[threshold_df["model_name"] == model_key][["threshold_rule", "threshold"]]
        else:
            model_thresholds = pd.DataFrame([{"threshold_rule": "fixed_0_50", "threshold": 0.5}])

        for trow in model_thresholds.itertuples(index=False):
            stats = bootstrap_one(y_true, y_prob, float(trow.threshold), args.n_bootstrap, args.seed)
            row = {
                "model_name": model_key,
                "display_name": cfg["display_name"],
                "threshold_rule": trow.threshold_rule,
                "threshold": float(trow.threshold),
                "n_test": int(len(test)),
                "n_bootstrap_requested": args.n_bootstrap,
                "seed": args.seed,
            }
            row.update(stats)
            rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "bootstrap_ci_results.csv", index=False)

    table = results[results["threshold_rule"].isin(["fixed_0_50", "max_val_f1"])].copy()
    for metric in METRICS:
        table[metric] = table.apply(
            lambda r: f"{format_float(r[f'{metric}_point'])} [{format_float(r[f'{metric}_ci_low'])}, {format_float(r[f'{metric}_ci_high'])}]",
            axis=1,
        )
    table = table[["display_name", "threshold_rule", "threshold", *METRICS]]
    table["threshold"] = table["threshold"].map(format_float)
    save_latex_table(
        table,
        TABLES_DIR / "bootstrap_ci_table.tex",
        "Bootstrap 95\\% confidence intervals for test-set metrics.",
        "tab:bootstrap_ci",
    )

    method_text = (
        "For uncertainty estimation, 95% confidence intervals were computed by nonparametric bootstrap resampling of the held-out test set. "
        f"We sampled test cases with replacement for {args.n_bootstrap} bootstrap replicates using random seed {args.seed}. "
        "For each replicate, accuracy, precision, sensitivity, specificity, F1-score, and ROC-AUC were recomputed. "
        "Intervals were defined by the 2.5th and 97.5th percentiles of the bootstrap distribution. "
        "Bootstrap samples containing only one class were excluded for ROC-AUC only."
    )
    summary = "\n".join(
        [
            "# Bootstrap CI Summary",
            "",
            method_text,
            "",
            "## Notes",
            *(notes or ["- All available normalized test prediction CSVs were processed."]),
        ]
    )
    write_text(ROOT / "experiments" / "journal_strengthening" / "bootstrap_ci_summary.md", summary)
    print("Bootstrap CI complete.")


if __name__ == "__main__":
    main()
