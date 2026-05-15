"""Generate publication-friendly figures from available strengthening outputs."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _common import FIGURES_DIR, RESULTS_DIR, ROOT, ensure_dirs, write_text


def save_fig(name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_dirs()
    notes = ["# Figure Generation Notes", ""]

    curves_path = RESULTS_DIR / "threshold_curves_by_model.csv"
    if curves_path.exists():
        curves = pd.read_csv(curves_path)
        res = curves[curves["model_name"] == "resnet50"]
        if len(res):
            plt.figure(figsize=(6.5, 4.2))
            plt.plot(res["threshold"], res["val_recall_sensitivity"], label="Sensitivity")
            plt.plot(res["threshold"], res["val_specificity"], label="Specificity")
            plt.plot(res["threshold"], res["val_f1_score"], label="F1-score")
            plt.xlabel("Decision threshold")
            plt.ylabel("Validation metric")
            plt.ylim(0, 1.02)
            plt.legend(frameon=False)
            plt.grid(alpha=0.25)
            save_fig("resnet50_threshold_tuning_curve")
            notes.append("- Generated ResNet50 threshold tuning curve.")
        else:
            notes.append("- ResNet50 threshold curve not available.")
    else:
        notes.append("- Threshold curves file not found.")

    ci_path = RESULTS_DIR / "bootstrap_ci_results.csv"
    if ci_path.exists():
        ci = pd.read_csv(ci_path)
        ci = ci[ci["threshold_rule"] == "max_val_f1"] if "threshold_rule" in ci else ci
        if len(ci):
            y = range(len(ci))
            plt.figure(figsize=(6.7, 4.5))
            x = ci["f1_score_point"]
            xerr = [x - ci["f1_score_ci_low"], ci["f1_score_ci_high"] - x]
            plt.errorbar(x, y, xerr=xerr, fmt="o", capsize=4)
            plt.yticks(list(y), ci["display_name"])
            plt.xlabel("F1-score with bootstrap 95% CI")
            plt.xlim(0, 1)
            plt.grid(axis="x", alpha=0.25)
            save_fig("bootstrap_ci_f1_point_range")
            notes.append("- Generated bootstrap CI point-range plot.")
        else:
            notes.append("- Bootstrap CI data was empty.")
    else:
        notes.append("- Bootstrap CI results file not found.")

    ms_path = RESULTS_DIR / "resnet50_multiseed_results.csv"
    if ms_path.exists() and ms_path.stat().st_size > 0:
        ms = pd.read_csv(ms_path)
        if len(ms):
            means = ms[["accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "roc_auc"]].mean()
            stds = ms[means.index].std(ddof=1)
            plt.figure(figsize=(7.2, 4.4))
            plt.bar(means.index, means.values, yerr=stds.values, capsize=4)
            plt.xticks(rotation=30, ha="right")
            plt.ylabel("Mean ± SD")
            plt.ylim(0, 1)
            save_fig("resnet50_multiseed_summary")
            notes.append("- Generated ResNet50 multi-seed summary.")
    else:
        notes.append("- ResNet50 multi-seed results not found; figure skipped.")

    vlm_path = RESULTS_DIR / "vlm_repeated_trial_summary.csv"
    if vlm_path.exists() and vlm_path.stat().st_size > 0:
        vlm = pd.read_csv(vlm_path)
        if len(vlm):
            plt.figure(figsize=(6.7, 4.2))
            plt.bar(vlm["prompt_mode"], vlm["f1_score_mean"], yerr=vlm["f1_score_std"], capsize=4)
            plt.ylabel("F1-score mean ± SD")
            plt.ylim(0, 1)
            save_fig("vlm_repeated_trials_f1")
            notes.append("- Generated VLM repeated-trials summary.")
    else:
        notes.append("- VLM repeated-trial results not found; figure skipped.")

    ext_path = RESULTS_DIR / "external_validation_results.csv"
    if ext_path.exists() and ext_path.stat().st_size > 5:
        ext = pd.read_csv(ext_path)
        if len(ext):
            plt.figure(figsize=(5.8, 4))
            plt.bar(ext["dataset"], ext["f1_score"])
            plt.ylabel("F1-score")
            plt.ylim(0, 1)
            save_fig("external_validation_comparison")
            notes.append("- Generated external validation comparison.")
    else:
        notes.append("- External validation results not found; figure skipped.")

    write_text(ROOT / "experiments" / "journal_strengthening" / "figure_generation_notes.md", "\n".join(notes))
    print("Figure generation complete.")


if __name__ == "__main__":
    main()
