#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

EXP_DIR="experiments/journal_strengthening"
LOG_DIR="$EXP_DIR/logs"
mkdir -p "$EXP_DIR"/{predictions,results,tables,figures,error_analysis,multiseed,logs}
LOG_FILE="$LOG_DIR/run_all_strengthening.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting journal strengthening pipeline"

echo "[1/7] Repository audit"
python "$EXP_DIR/audit_repo.py"

echo "[2/7] Generate or locate CNN predictions"
python "$EXP_DIR/generate_cnn_predictions.py"

echo "[3/7] Threshold tuning"
python "$EXP_DIR/threshold_tuning.py"

echo "[4/7] Bootstrap confidence intervals"
python "$EXP_DIR/bootstrap_ci.py"

echo "[5/7] Error analysis"
python "$EXP_DIR/error_analysis.py"

if [[ "${RUN_MULTISEED:-0}" == "1" ]]; then
  echo "[optional] RUN_MULTISEED=1 detected; running ResNet50 multi-seed experiment"
  python "$EXP_DIR/run_resnet50_multiseed.py"
else
  echo "[optional] Multi-seed skipped; set RUN_MULTISEED=1 to run"
  python "$EXP_DIR/run_resnet50_multiseed.py"
fi

if [[ "${RUN_VLM:-0}" == "1" ]]; then
  echo "[optional] RUN_VLM=1 detected; running VLM repeated trials"
  python "$EXP_DIR/run_vlm_repeated_trials.py"
else
  echo "[optional] VLM repeated trials skipped; set RUN_VLM=1 to run"
  python "$EXP_DIR/run_vlm_repeated_trials.py"
fi

if [[ -d data/MURA || -d data/GRAZPEDWRI-DX || -d data/external ]]; then
  echo "[optional] External dataset directory detected; checking mapping scaffold"
  python "$EXP_DIR/external_validation.py"
else
  echo "[optional] External validation skipped; no external data directory found"
  python "$EXP_DIR/external_validation.py"
fi

echo "[6/7] Paper figures"
python "$EXP_DIR/make_paper_figures.py"

echo "[7/7] Final report"
python "$EXP_DIR/make_final_report.py"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Journal strengthening pipeline complete"
