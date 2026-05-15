"""Generate normalized validation/test CNN prediction CSVs.

This script reuses saved project checkpoints and the deterministic validation
transform from `src.dataset`. Existing test prediction CSVs are copied into the
normalized experiment format when available, while missing validation CSVs are
generated from checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import (
    IMAGE_DIR,
    MODEL_CONFIGS,
    PRED_DIR,
    ROOT,
    SPLIT_CSVS,
    add_repo_to_path,
    ensure_dirs,
    normalize_prediction_df,
    normalized_prediction_path,
    rel,
    write_text,
)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def infer_checkpoint(model_key: str, split: str, batch_size: int, num_workers: int) -> pd.DataFrame:
    add_repo_to_path()
    from src.dataset import FracAtlasDataset, create_dataloader, get_val_transforms
    from src.models import get_model

    cfg = MODEL_CONFIGS[model_key]
    checkpoint_path = ROOT / cfg["checkpoint"]
    split_csv = SPLIT_CSVS[split]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {rel(checkpoint_path)}")
    if not split_csv.exists():
        raise FileNotFoundError(f"Missing split CSV: {rel(split_csv)}")
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Missing image directory: {rel(IMAGE_DIR)}")

    device = select_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint.get("model_name", cfg["architecture"])
    model = get_model(model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = FracAtlasDataset(
        csv_path=str(split_csv),
        img_dir=str(IMAGE_DIR),
        transform=get_val_transforms(),
    )
    loader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    labels, probs = [], []
    for images, y in loader:
        images = images.to(device)
        logits = model(images).squeeze(1)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        probs.extend(prob.tolist())
        labels.extend(y.cpu().numpy().astype(int).tolist())

    df = pd.DataFrame(
        {
            "image_id": dataset.filenames,
            "image_path": [f"data/FracAtlas/images/{name}" for name in dataset.filenames],
            "true_label": labels,
            "predicted_probability": np.asarray(probs, dtype=float),
        }
    )
    return normalize_prediction_df(df, model_key, split)


def copy_existing_test_predictions(model_key: str) -> bool:
    cfg = MODEL_CONFIGS[model_key]
    source = ROOT / cfg["existing_test_predictions"]
    dest = normalized_prediction_path(model_key, "test")
    if not source.exists():
        return False
    df = pd.read_csv(source)
    normalize_prediction_df(df, model_key, "test").to_csv(dest, index=False)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(MODEL_CONFIGS), choices=list(MODEL_CONFIGS))
    parser.add_argument("--splits", nargs="*", default=["val", "test"], choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Regenerate normalized CSVs even if they already exist.")
    args = parser.parse_args()

    ensure_dirs()
    notes = ["# CNN Prediction Generation Notes", ""]
    generated = []

    for model_key in args.models:
        for split in args.splits:
            dest = normalized_prediction_path(model_key, split)
            if dest.exists() and not args.force:
                notes.append(f"- Existing normalized file kept: `{rel(dest)}`")
                continue
            try:
                if split == "test" and copy_existing_test_predictions(model_key):
                    generated.append(dest)
                    notes.append(f"- Normalized existing test predictions for `{model_key}`: `{rel(dest)}`")
                    continue
                df = infer_checkpoint(model_key, split, args.batch_size, args.num_workers)
                df.to_csv(dest, index=False)
                generated.append(dest)
                notes.append(f"- Generated `{split}` predictions for `{model_key}` from checkpoint: `{rel(dest)}`")
            except Exception as exc:
                notes.append(f"- Could not generate `{model_key}` `{split}` predictions: {exc}")

    notes.extend(
        [
            "",
            "To regenerate manually:",
            "",
            "```bash",
            "python experiments/journal_strengthening/generate_cnn_predictions.py --force",
            "```",
        ]
    )
    write_text(ROOT / "experiments" / "journal_strengthening" / "prediction_generation_notes.md", "\n".join(notes))
    print(f"Prediction normalization/generation complete. Files written under {rel(PRED_DIR)}")


if __name__ == "__main__":
    main()
