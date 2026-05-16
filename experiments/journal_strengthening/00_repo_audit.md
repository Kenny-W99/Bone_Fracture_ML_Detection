# Repository Audit

## Repository Structure Summary

The repository contains a PyTorch CNN pipeline under `src/`, FracAtlas data and split CSVs under `data/FracAtlas/`, saved model outputs under `outputs/`, VLM baseline utilities, and paper figures/summary artifacts.

## Existing Scripts Found

- `src/train.py`: present
- `src/evaluate.py`: present
- `src/dataset.py`: present
- `src/models.py`: present
- `src/vlm_baseline.py`: present
- `src/evaluate_vlm.py`: present
- `src/gradcam.py`: present
- `src/split_dataset.py`: present
- `scripts/make_clean_vlm_images.py`: present

## Existing Result Files Found

- `outputs/results_summary.csv`: present
- `outputs/final_figures/results_summary.csv`: present
- `experiments/journal_strengthening/results/vlm_fulltest613_clean_summary.csv`: present
- `outputs/resnet50_5ep/evaluation/predictions.csv`: present
- `outputs/mobilenet_v2_5ep/evaluation/predictions.csv`: present
- `outputs/densenet121_5ep/evaluation/predictions.csv`: present
- `outputs/custom_cnn_10ep/evaluation/predictions.csv`: present

## Existing Checkpoints Found

- `outputs/resnet50_5ep/best_model.pth`: present
- `outputs/mobilenet_v2_5ep/best_model.pth`: present
- `outputs/densenet121_5ep/best_model.pth`: present
- `outputs/custom_cnn_10ep/best_model.pth`: present

## Train/Validation/Test Split Files

- `data/FracAtlas/train.csv`: present
- `data/FracAtlas/val.csv`: present
- `data/FracAtlas/test.csv`: present

## Expected Data Paths

- `data/FracAtlas/images`: present
- `data/FracAtlas/images_vlm_clean`: present
- `data/FracAtlas/dataset.csv`: present
- `data/MURA`: missing
- `data/GRAZPEDWRI-DX`: missing
- `data/external`: missing

## What Can Be Run Immediately

- CNN validation/test prediction generation can be run from existing checkpoints and split CSVs.
- Threshold tuning can run after validation predictions are generated.
- Bootstrap confidence intervals can run after normalized test predictions are available.
- ResNet50 error/subgroup analysis can run using FracAtlas test metadata.
- Paper figure generation can run for analyses with available CSV outputs.

## What Cannot Be Run Without Additional Inputs

- Multi-seed ResNet50 retraining is intentionally guarded by `RUN_MULTISEED=1` because it is expensive.
- Repeated VLM trials are intentionally guarded by `RUN_VLM=1` and require a provider API key in the environment.
- External validation cannot produce metrics unless an external dataset and validated fracture/no-fracture label mapping are provided.

## Recommended Next Commands

```bash
bash experiments/journal_strengthening/run_all_strengthening.sh
RUN_MULTISEED=1 bash experiments/journal_strengthening/run_all_strengthening.sh
RUN_VLM=1 VLM_MODEL=gpt-4o-mini bash experiments/journal_strengthening/run_all_strengthening.sh
```
