"""Shared helpers for journal-strengthening analyses.

All helpers are intentionally file-based and conservative: they never invent
metrics when inputs are missing, and they keep generated artifacts inside this
experiment folder.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "journal_strengthening"
PRED_DIR = EXP_DIR / "predictions"
RESULTS_DIR = EXP_DIR / "results"
TABLES_DIR = EXP_DIR / "tables"
FIGURES_DIR = EXP_DIR / "figures"
ERROR_DIR = EXP_DIR / "error_analysis"
LOG_DIR = EXP_DIR / "logs"

MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "resnet50": {
        "display_name": "ResNet50",
        "architecture": "resnet50",
        "checkpoint": "outputs/resnet50_5ep/best_model.pth",
        "existing_test_predictions": "outputs/resnet50_5ep/evaluation/predictions.csv",
    },
    "mobilenetv2": {
        "display_name": "MobileNetV2",
        "architecture": "mobilenet_v2",
        "checkpoint": "outputs/mobilenet_v2_5ep/best_model.pth",
        "existing_test_predictions": "outputs/mobilenet_v2_5ep/evaluation/predictions.csv",
    },
    "densenet121": {
        "display_name": "DenseNet121",
        "architecture": "densenet121",
        "checkpoint": "outputs/densenet121_5ep/best_model.pth",
        "existing_test_predictions": "outputs/densenet121_5ep/evaluation/predictions.csv",
    },
    "custom_cnn": {
        "display_name": "Custom CNN",
        "architecture": "custom_cnn",
        "checkpoint": "outputs/custom_cnn_10ep/best_model.pth",
        "existing_test_predictions": "outputs/custom_cnn_10ep/evaluation/predictions.csv",
    },
}

SPLIT_CSVS = {
    "train": ROOT / "data" / "FracAtlas" / "train.csv",
    "val": ROOT / "data" / "FracAtlas" / "val.csv",
    "test": ROOT / "data" / "FracAtlas" / "test.csv",
}
IMAGE_DIR = ROOT / "data" / "FracAtlas" / "images"
VLM_IMAGE_DIR = ROOT / "data" / "FracAtlas" / "images_vlm_clean"


def ensure_dirs() -> None:
    for path in [PRED_DIR, RESULTS_DIR, TABLES_DIR, FIGURES_DIR, ERROR_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def normalized_prediction_path(model_key: str, split: str) -> Path:
    return PRED_DIR / f"{model_key}_{split}_predictions.csv"


def normalize_prediction_df(df: pd.DataFrame, model_key: str, split: str) -> pd.DataFrame:
    """Return standard columns for threshold/bootstrap scripts."""
    out = df.copy()
    if "image_id" not in out.columns:
        if "filename" in out.columns:
            out["image_id"] = out["filename"]
        elif "image_path" in out.columns:
            out["image_id"] = out["image_path"].astype(str).map(lambda x: Path(x).name)
        else:
            raise ValueError(f"No image id/path column in columns: {list(out.columns)}")
    if "image_path" not in out.columns:
        out["image_path"] = out["image_id"].astype(str).map(lambda x: f"data/FracAtlas/images/{x}")
    if "true_label" not in out.columns:
        if "fractured" in out.columns:
            out["true_label"] = out["fractured"]
        else:
            raise ValueError(f"No true_label/fractured column in columns: {list(out.columns)}")
    if "predicted_probability" not in out.columns:
        for candidate in ["probability", "fracture_probability", "score"]:
            if candidate in out.columns:
                out["predicted_probability"] = out[candidate]
                break
    if "predicted_probability" not in out.columns:
        raise ValueError(f"No predicted probability column in columns: {list(out.columns)}")
    out["true_label"] = out["true_label"].astype(int)
    out["predicted_probability"] = pd.to_numeric(out["predicted_probability"], errors="coerce")
    out["predicted_label_at_0_5"] = (out["predicted_probability"] >= 0.5).astype(int)
    out["model_name"] = model_key
    out["split"] = split
    cols = [
        "image_id",
        "image_path",
        "true_label",
        "predicted_probability",
        "predicted_label_at_0_5",
        "model_name",
        "split",
    ]
    return out[cols]


def load_predictions(model_key: str, split: str) -> pd.DataFrame:
    path = normalized_prediction_path(model_key, split)
    if not path.exists():
        raise FileNotFoundError(f"Missing normalized prediction CSV: {rel(path)}")
    df = pd.read_csv(path)
    return normalize_prediction_df(df, model_key, split)


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return np.nan


def compute_metrics(y_true: Iterable[int], y_prob: Iterable[float], threshold: float) -> Dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_prob_arr = np.asarray(list(y_prob), dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true_arr, y_prob_arr),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def format_float(x: object, digits: int = 4) -> str:
    try:
        val = float(x)
    except Exception:
        return ""
    if math.isnan(val):
        return "NA"
    return f"{val:.{digits}f}"


def save_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=True, caption=caption, label=label))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def add_repo_to_path() -> None:
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
