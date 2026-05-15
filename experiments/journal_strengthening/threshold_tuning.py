"""Validation-only threshold tuning for CNN prediction CSVs."""

from __future__ import annotations

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


THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def choose_threshold(curves: pd.DataFrame, rule: str) -> tuple[float, str]:
    if rule == "fixed_0_50":
        return 0.50, ""
    if rule == "max_val_f1":
        row = curves.sort_values(["val_f1_score", "threshold"], ascending=[False, True]).iloc[0]
        return float(row["threshold"]), ""
    if rule == "max_youden_j":
        tmp = curves.assign(youden_j=curves["val_recall_sensitivity"] + curves["val_specificity"] - 1)
        row = tmp.sort_values(["youden_j", "threshold"], ascending=[False, True]).iloc[0]
        return float(row["threshold"]), ""
    if rule == "high_sensitivity_precision_constrained":
        eligible = curves[curves["val_recall_sensitivity"] >= 0.90]
        if len(eligible):
            row = eligible.sort_values(["val_precision", "threshold"], ascending=[False, True]).iloc[0]
            return float(row["threshold"]), ""
        row = curves.sort_values(["val_recall_sensitivity", "threshold"], ascending=[False, True]).iloc[0]
        return float(row["threshold"]), "fallback: no validation threshold reached recall >= 0.90; selected highest validation recall."
    raise ValueError(rule)


def main() -> None:
    ensure_dirs()
    curve_rows = []
    result_rows = []
    fallback_notes = []

    for model_key, cfg in MODEL_CONFIGS.items():
        try:
            val = load_predictions(model_key, "val")
            test = load_predictions(model_key, "test")
        except FileNotFoundError as exc:
            fallback_notes.append(f"- `{model_key}` skipped: {exc}")
            continue

        yv, pv = val["true_label"].to_numpy(), val["predicted_probability"].to_numpy()
        yt, pt = test["true_label"].to_numpy(), test["predicted_probability"].to_numpy()
        model_curves = []
        for threshold in THRESHOLDS:
            metrics = compute_metrics(yv, pv, float(threshold))
            row = {
                "model_name": model_key,
                "display_name": cfg["display_name"],
                "threshold": float(threshold),
            }
            row.update({f"val_{k}": v for k, v in metrics.items()})
            curve_rows.append(row)
            model_curves.append(row)

        curves_df = pd.DataFrame(model_curves)
        for rule in [
            "fixed_0_50",
            "max_val_f1",
            "max_youden_j",
            "high_sensitivity_precision_constrained",
        ]:
            threshold, note = choose_threshold(curves_df, rule)
            test_metrics = compute_metrics(yt, pt, threshold)
            val_metrics = compute_metrics(yv, pv, threshold)
            result = {
                "model_name": model_key,
                "display_name": cfg["display_name"],
                "threshold_rule": rule,
                "threshold": threshold,
                "selection_note": note,
            }
            result.update(test_metrics)
            result.update({f"val_{k}": v for k, v in val_metrics.items() if k not in ["TP", "FP", "TN", "FN"]})
            result_rows.append(result)
            if note:
                fallback_notes.append(f"- `{cfg['display_name']}` high-sensitivity rule used fallback at threshold {threshold:.2f}: {note}")

    curves = pd.DataFrame(curve_rows)
    results = pd.DataFrame(result_rows)
    curves.to_csv(RESULTS_DIR / "threshold_curves_by_model.csv", index=False)
    results.to_csv(RESULTS_DIR / "threshold_tuning_results.csv", index=False)

    table = results.copy()
    table = table[
        [
            "display_name",
            "threshold_rule",
            "threshold",
            "accuracy",
            "precision",
            "recall_sensitivity",
            "specificity",
            "f1_score",
            "roc_auc",
            "TP",
            "FP",
            "TN",
            "FN",
        ]
    ]
    for col in ["threshold", "accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "roc_auc"]:
        table[col] = table[col].map(format_float)
    save_latex_table(
        table,
        TABLES_DIR / "threshold_tuning_table.tex",
        "Test-set performance under validation-selected decision thresholds.",
        "tab:threshold_tuning",
    )

    if len(results):
        paper_rows = results[results["threshold_rule"] == "max_val_f1"].copy()
        chosen_lines = [
            f"- {r.display_name}: report `max_val_f1` threshold {r.threshold:.2f} as the primary tuned-threshold result; keep fixed 0.50 as a comparator."
            for r in paper_rows.itertuples()
        ]
        selected_lines = [
            f"- {r.display_name} / {r.threshold_rule}: threshold {r.threshold:.2f}; test F1 {r.f1_score:.4f}, recall {r.recall_sensitivity:.4f}, specificity {r.specificity:.4f}."
            for r in results.itertuples()
        ]
    else:
        chosen_lines = ["- No thresholds selected because normalized prediction CSVs were unavailable."]
        selected_lines = []

    summary = "\n".join(
        [
            "# Threshold Tuning Summary",
            "",
            "Decision thresholds were selected using validation predictions only and then evaluated once on the held-out test set.",
            "",
            "## Selected Thresholds",
            *selected_lines,
            "",
            "## High-Sensitivity Rule Notes",
            *(fallback_notes or ["- No high-sensitivity fallback was needed."]),
            "",
            "## Recommended Paper Reporting",
            *chosen_lines,
            "",
            "Validation-based threshold tuning is stronger than using a fixed 0.50 cutoff because neural-network probabilities are often not calibrated to a universal operating point. Selecting the threshold on validation data makes the sensitivity-specificity trade-off explicit while preserving the test set for unbiased final evaluation.",
        ]
    )
    write_text(ROOT / "experiments" / "journal_strengthening" / "threshold_tuning_summary.md", summary)
    print("Threshold tuning complete.")


if __name__ == "__main__":
    main()
