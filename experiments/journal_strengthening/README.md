# Journal Strengthening Package

This folder contains reproducible add-on experiments for strengthening the FracAtlas bone-fracture classification paper. All outputs are written inside `experiments/journal_strengthening/`.

## Safe Analysis

Run the non-expensive pipeline:

```bash
bash experiments/journal_strengthening/run_all_strengthening.sh
```

This audits the repo, normalizes or generates CNN prediction CSVs, tunes thresholds, computes bootstrap confidence intervals, runs ResNet50 error analysis, makes figures, and writes the final report.

## Multi-Seed Training

Multi-seed ResNet50 training is expensive and is skipped by default. Run it explicitly:

```bash
RUN_MULTISEED=1 bash experiments/journal_strengthening/run_all_strengthening.sh
```

Outputs are saved under `experiments/journal_strengthening/multiseed/resnet50_seed_*`.

## VLM Repeated Trials

Paid VLM calls are skipped by default. The API key must be available through the provider environment variable; it is never stored in this repo.

```bash
RUN_VLM=1 VLM_MODEL=gpt-4o-mini bash experiments/journal_strengthening/run_all_strengthening.sh
```

## External Datasets

Place external validation data under one of:

- `data/MURA/`
- `data/GRAZPEDWRI-DX/`
- `data/external/`

Include a labels CSV such as `test.csv`, `labels.csv`, `metadata.csv`, or `dataset.csv`. The safest schema is `image_id` or `image_path` plus binary `fractured`, where 1 means fracture and 0 means no-fracture. If labels cannot be mapped cleanly, the scaffold stops rather than producing misleading metrics.

## Outputs

- Normalized CNN predictions: `predictions/`
- Result CSVs: `results/`
- LaTeX tables: `tables/`
- Paper figures: `figures/`
- Error-analysis CSVs: `error_analysis/`
- Run logs: `logs/`
- Narrative report: `JOURNAL_STRENGTHENING_REPORT.md`

## Paper Use

Use `tables/threshold_tuning_table.tex` and `tables/bootstrap_ci_table.tex` first. Add the multi-seed, VLM repeated-trial, and external-validation tables only after those experiments are run with real data.

## Troubleshooting

- Missing validation predictions: run `python experiments/journal_strengthening/generate_cnn_predictions.py --force`.
- Missing checkpoints: restore the relevant `outputs/*/best_model.pth` file or rerun training.
- Missing VLM key: export the provider API key before setting `RUN_VLM=1`.
- Undefined bootstrap ROC-AUC values can occur in bootstrap samples containing only one class; those samples are skipped for ROC-AUC only.
