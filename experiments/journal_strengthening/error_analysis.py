"""False-positive/false-negative and subgroup analysis for ResNet50."""

from __future__ import annotations

import pandas as pd

from _common import ERROR_DIR, EXP_DIR, RESULTS_DIR, ROOT, compute_metrics, ensure_dirs, load_predictions, write_text


SUBGROUP_COLUMNS = ["hand", "leg", "hip", "shoulder", "mixed", "hardware", "multiscan", "frontal", "lateral", "oblique"]


def main() -> None:
    ensure_dirs()
    pred = load_predictions("resnet50", "test")
    threshold = 0.5
    threshold_path = RESULTS_DIR / "threshold_tuning_results.csv"
    if threshold_path.exists():
        tdf = pd.read_csv(threshold_path)
        chosen = tdf[(tdf["model_name"] == "resnet50") & (tdf["threshold_rule"] == "max_val_f1")]
        if len(chosen):
            threshold = float(chosen.iloc[0]["threshold"])

    pred["predicted_label"] = (pred["predicted_probability"] >= threshold).astype(int)
    pred["error_type"] = pred.apply(
        lambda r: "TP" if r.true_label == 1 and r.predicted_label == 1 else
        "FP" if r.true_label == 0 and r.predicted_label == 1 else
        "TN" if r.true_label == 0 and r.predicted_label == 0 else "FN",
        axis=1,
    )

    mapping = {
        "FN": "resnet50_false_negatives.csv",
        "FP": "resnet50_false_positives.csv",
        "TP": "resnet50_true_positives.csv",
        "TN": "resnet50_true_negatives.csv",
    }
    for err, filename in mapping.items():
        pred[pred["error_type"] == err].to_csv(ERROR_DIR / filename, index=False)

    metadata_path = ROOT / "data" / "FracAtlas" / "test.csv"
    subgroup_note = ""
    if metadata_path.exists():
        meta = pd.read_csv(metadata_path)
        merged = pred.merge(meta, on="image_id", how="left")
        rows = []
        for col in SUBGROUP_COLUMNS:
            if col not in merged.columns:
                continue
            sub = merged[pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int) == 1]
            if len(sub) < 5:
                continue
            m = compute_metrics(sub["true_label"], sub["predicted_probability"], threshold)
            rows.append({"subgroup": col, "n": len(sub), **m})
        if rows:
            pd.DataFrame(rows).to_csv(ERROR_DIR / "resnet50_subgroup_metrics.csv", index=False)
            subgroup_note = "- Subgroup metrics were computed using FracAtlas test metadata columns."
        else:
            subgroup_note = "- Metadata exists, but no supported subgroup columns had enough samples for subgroup metrics."
    else:
        subgroup_note = "- Test metadata was not found, so subgroup analysis was not possible."

    counts = pred["error_type"].value_counts().to_dict()
    summary = "\n".join(
        [
            "# Error Analysis Summary",
            "",
            f"ResNet50 error analysis used threshold `{threshold:.2f}`. This is the validation-selected max-F1 threshold when available; otherwise 0.50.",
            "",
            f"- False negatives: {counts.get('FN', 0)}",
            f"- False positives: {counts.get('FP', 0)}",
            f"- True positives: {counts.get('TP', 0)}",
            f"- True negatives: {counts.get('TN', 0)}",
            subgroup_note,
            "",
            "Recommended Grad-CAM examples for the discussion should be sampled from the saved false-negative and false-positive CSVs, prioritizing high-confidence errors and clinically interpretable metadata groups where available.",
        ]
    )
    write_text(EXP_DIR / "error_analysis_summary.md", summary)
    print("Error analysis complete.")


if __name__ == "__main__":
    main()
