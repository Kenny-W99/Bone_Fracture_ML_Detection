"""Create the final journal-strengthening report from generated artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from _common import EXP_DIR, RESULTS_DIR, ROOT, rel, write_text


def file_line(path: Path) -> str:
    return f"- `{rel(path)}`: {'present' if path.exists() and path.stat().st_size > 0 else 'not generated'}"


def main() -> None:
    threshold_path = RESULTS_DIR / "threshold_tuning_results.csv"
    ci_path = RESULTS_DIR / "bootstrap_ci_results.csv"
    threshold_note = "No threshold results were available."
    if threshold_path.exists():
        t = pd.read_csv(threshold_path)
        res = t[(t["model_name"] == "resnet50") & (t["threshold_rule"] == "max_val_f1")]
        if len(res):
            r = res.iloc[0]
            threshold_note = (
                f"Newly computed ResNet50 validation-selected threshold was {r['threshold']:.2f}; "
                f"test F1={r['f1_score']:.4f}, sensitivity={r['recall_sensitivity']:.4f}, specificity={r['specificity']:.4f}."
            )

    ci_note = "No bootstrap CI results were available."
    if ci_path.exists():
        b = pd.read_csv(ci_path)
        res = b[(b["model_name"] == "resnet50") & (b["threshold_rule"] == "max_val_f1")]
        if len(res):
            r = res.iloc[0]
            ci_note = (
                f"Newly computed ResNet50 tuned-threshold F1 95% CI was "
                f"{r['f1_score_point']:.4f} [{r['f1_score_ci_low']:.4f}, {r['f1_score_ci_high']:.4f}]."
            )

    outputs = [
        EXP_DIR / "00_repo_audit.md",
        EXP_DIR / "prediction_generation_notes.md",
        RESULTS_DIR / "threshold_tuning_results.csv",
        RESULTS_DIR / "threshold_curves_by_model.csv",
        RESULTS_DIR / "bootstrap_ci_results.csv",
        EXP_DIR / "error_analysis_summary.md",
        EXP_DIR / "figure_generation_notes.md",
        EXP_DIR / "resnet50_multiseed_summary.md",
        EXP_DIR / "vlm_repeated_trials_summary.md",
        EXP_DIR / "external_validation_summary.md",
    ]

    report = "\n".join(
        [
            "# Journal Strengthening Report",
            "",
            "## 1. What Was Implemented",
            "",
            "A reproducible strengthening package was added under `experiments/journal_strengthening/`. It includes repository audit, CNN prediction normalization/generation, validation-based threshold tuning, bootstrap confidence intervals, guarded multi-seed training, guarded VLM repeated trials, external-validation scaffold, ResNet50 error/subgroup analysis, publication figure generation, and a safe runner.",
            "",
            "## 2. What Was Successfully Run",
            "",
            "Safe analyses run by the wrapper are recorded in `logs/run_all_strengthening.log` when the wrapper is used. Generated CSVs and summaries listed below indicate completed steps.",
            "",
            "## 3. What Could Not Be Run and Why",
            "",
            "- Multi-seed training is not run unless `RUN_MULTISEED=1` is set.",
            "- Repeated VLM API calls are not run unless `RUN_VLM=1` is set and the provider API key is available in the environment.",
            "- External validation is not run unless an external dataset exists and label mapping is validated.",
            "",
            "## 4. New Result Files Generated",
            "",
            *[file_line(p) for p in outputs],
            "",
            "## 5. Commands That Were Run",
            "",
            "Primary intended command:",
            "",
            "```bash",
            "bash experiments/journal_strengthening/run_all_strengthening.sh",
            "```",
            "",
            "Optional commands:",
            "",
            "```bash",
            "RUN_MULTISEED=1 bash experiments/journal_strengthening/run_all_strengthening.sh",
            "RUN_VLM=1 VLM_MODEL=gpt-4o-mini bash experiments/journal_strengthening/run_all_strengthening.sh",
            "```",
            "",
            "## 6. Recommended Tables to Add",
            "",
            "- `tables/threshold_tuning_table.tex`",
            "- `tables/bootstrap_ci_table.tex`",
            "- Add `tables/resnet50_multiseed_table.tex`, `tables/vlm_repeated_trials_table.tex`, and `tables/external_validation_table.tex` only after those experiments are actually run.",
            "",
            "## 7. Recommended Figures to Add",
            "",
            "- `figures/resnet50_threshold_tuning_curve.png`",
            "- `figures/bootstrap_ci_f1_point_range.png`",
            "- Add multi-seed, repeated-VLM, and external-validation figures only when real outputs exist.",
            "",
            "## 8. Suggested Methods Text",
            "",
            "Decision thresholds for CNNs were tuned using validation-set predictions only. Thresholds from 0.01 to 0.99 were evaluated, and selected thresholds were then applied once to the held-out test set. Test-set uncertainty was estimated using nonparametric bootstrap resampling with replacement across test images, using 2,000 replicates and the 2.5th and 97.5th percentiles as 95% confidence intervals.",
            "",
            "## 9. Suggested Results Text",
            "",
            threshold_note,
            ci_note,
            "",
            "## 10. Suggested Discussion Text",
            "",
            "Validation-selected thresholds provide a more transparent operating-point analysis than a fixed 0.50 cutoff, especially in a medical-imaging screening context where sensitivity and specificity trade-offs must be stated explicitly. These results should be interpreted as research-only evidence on FracAtlas rather than clinical diagnostic performance.",
            "",
            "## 11. Suggested Limitations Text",
            "",
            "The experiments remain limited by the size and source distribution of FracAtlas. External validation, repeated VLM trials, and multi-seed retraining should be reported only after they are completed with real data and documented compute/API settings. The models are not intended for clinical use.",
            "",
            "## 12. Suggested Updated Abstract Paragraph",
            "",
            "We further assessed operating-point selection and statistical uncertainty by tuning CNN decision thresholds on the validation set and estimating 95% confidence intervals on the held-out test set using bootstrap resampling. These additions quantify performance variability and make the sensitivity-specificity trade-off explicit without using test data for model selection.",
            "",
            "## 13. Suggested Updated Title",
            "",
            "Transfer Learning and Vision-Language Model Baselines for Research-Only Bone Fracture Classification in FracAtlas X-rays",
            "",
            "## 14. Recommended Journal-Submission Strategy",
            "",
            "Lead with the validated CNN results, threshold tuning, bootstrap intervals, and error analysis. Present the VLM baseline as exploratory, not as a clinical comparator. Treat external validation and multi-seed stability as high-priority additions before submission if time and data availability permit.",
            "",
            "## Result Provenance",
            "",
            "Threshold tuning and bootstrap intervals generated by this package are newly computed from available prediction CSVs. Existing CNN test predictions and VLM baseline summaries are pre-existing project outputs unless explicitly regenerated by the scripts. No metrics are fabricated.",
        ]
    )
    write_text(EXP_DIR / "JOURNAL_STRENGTHENING_REPORT.md", report)
    print("Final report written.")


if __name__ == "__main__":
    main()
