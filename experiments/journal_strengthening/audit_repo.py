"""Write repository audit markdown for the journal-strengthening package."""

from __future__ import annotations

from pathlib import Path

from _common import MODEL_CONFIGS, ROOT, SPLIT_CSVS, IMAGE_DIR, VLM_IMAGE_DIR, rel, write_text


def exists_line(path: Path) -> str:
    return f"- `{rel(path)}`: {'present' if path.exists() else 'missing'}"


def main() -> None:
    scripts = [
        "src/train.py",
        "src/evaluate.py",
        "src/dataset.py",
        "src/models.py",
        "src/vlm_baseline.py",
        "src/evaluate_vlm.py",
        "src/gradcam.py",
        "src/split_dataset.py",
        "scripts/make_clean_vlm_images.py",
    ]
    result_patterns = [
        "outputs/results_summary.csv",
        "outputs/final_figures/results_summary.csv",
        "outputs/vlm/vlm_fulltest613_clean_summary.csv",
    ]
    lines = [
        "# Repository Audit",
        "",
        "## Repository Structure Summary",
        "",
        "The repository contains a PyTorch CNN pipeline under `src/`, FracAtlas data and split CSVs under `data/FracAtlas/`, saved model outputs under `outputs/`, VLM baseline utilities, and paper figures/summary artifacts.",
        "",
        "## Existing Scripts Found",
        "",
        *[exists_line(ROOT / s) for s in scripts],
        "",
        "## Existing Result Files Found",
        "",
        *[exists_line(ROOT / p) for p in result_patterns],
    ]
    for key, cfg in MODEL_CONFIGS.items():
        lines.append(exists_line(ROOT / cfg["existing_test_predictions"]))
    for p in sorted((ROOT / "outputs" / "vlm").glob("*test_metrics.csv")):
        lines.append(f"- `{rel(p)}`: present")

    lines.extend(["", "## Existing Checkpoints Found", ""])
    for key, cfg in MODEL_CONFIGS.items():
        lines.append(exists_line(ROOT / cfg["checkpoint"]))

    lines.extend(["", "## Train/Validation/Test Split Files", ""])
    for split, path in SPLIT_CSVS.items():
        lines.append(exists_line(path))

    lines.extend(["", "## Expected Data Paths", ""])
    lines.append(exists_line(IMAGE_DIR))
    lines.append(exists_line(VLM_IMAGE_DIR))
    lines.append(exists_line(ROOT / "data" / "FracAtlas" / "dataset.csv"))
    lines.append(exists_line(ROOT / "data" / "MURA"))
    lines.append(exists_line(ROOT / "data" / "GRAZPEDWRI-DX"))
    lines.append(exists_line(ROOT / "data" / "external"))

    lines.extend(
        [
            "",
            "## What Can Be Run Immediately",
            "",
            "- CNN validation/test prediction generation can be run from existing checkpoints and split CSVs.",
            "- Threshold tuning can run after validation predictions are generated.",
            "- Bootstrap confidence intervals can run after normalized test predictions are available.",
            "- ResNet50 error/subgroup analysis can run using FracAtlas test metadata.",
            "- Paper figure generation can run for analyses with available CSV outputs.",
            "",
            "## What Cannot Be Run Without Additional Inputs",
            "",
            "- Multi-seed ResNet50 retraining is intentionally guarded by `RUN_MULTISEED=1` because it is expensive.",
            "- Repeated VLM trials are intentionally guarded by `RUN_VLM=1` and require a provider API key in the environment.",
            "- External validation cannot produce metrics unless an external dataset and validated fracture/no-fracture label mapping are provided.",
            "",
            "## Recommended Next Commands",
            "",
            "```bash",
            "bash experiments/journal_strengthening/run_all_strengthening.sh",
            "RUN_MULTISEED=1 bash experiments/journal_strengthening/run_all_strengthening.sh",
            "RUN_VLM=1 VLM_MODEL=gpt-4o-mini bash experiments/journal_strengthening/run_all_strengthening.sh",
            "```",
        ]
    )
    write_text(ROOT / "experiments" / "journal_strengthening" / "00_repo_audit.md", "\n".join(lines))
    print("Repository audit written.")


if __name__ == "__main__":
    main()
