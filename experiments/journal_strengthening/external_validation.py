"""External validation scaffold for FracAtlas-trained ResNet50."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from _common import EXP_DIR, RESULTS_DIR, ROOT, TABLES_DIR, compute_metrics, ensure_dirs, save_latex_table, write_text, add_repo_to_path


EXTERNAL_DIRS = [ROOT / "data" / "MURA", ROOT / "data" / "GRAZPEDWRI-DX", ROOT / "data" / "external"]


def discover_external_csv(dataset_dir: Path) -> Path | None:
    for name in ["test.csv", "labels.csv", "metadata.csv", "dataset.csv"]:
        path = dataset_dir / name
        if path.exists():
            return path
    return None


def validate_label_mapping(df: pd.DataFrame) -> tuple[bool, str]:
    """Conservative mapping check; do not infer labels from vague columns."""
    columns = set(df.columns)
    if "fractured" in columns:
        vals = set(pd.to_numeric(df["fractured"], errors="coerce").dropna().astype(int).unique())
        if vals <= {0, 1}:
            return True, "`fractured` is binary with 0=no-fracture and 1=fracture."
    if "label" in columns:
        labels = set(df["label"].dropna().astype(str).str.lower().unique())
        allowed = {"fracture", "fractured", "positive", "no fracture", "non_fractured", "negative", "normal"}
        if labels and labels <= allowed:
            return True, "`label` contains recognizable fracture/no-fracture text labels."
    return False, (
        "No clean fracture/no-fracture mapping was found. Add a CSV with either a binary `fractured` column "
        "or a `label` column using explicit values such as fracture/no fracture."
    )


def mapped_labels(df: pd.DataFrame) -> pd.Series:
    if "fractured" in df.columns:
        return pd.to_numeric(df["fractured"], errors="coerce").astype(int)
    mapping = {
        "fracture": 1,
        "fractured": 1,
        "positive": 1,
        "no fracture": 0,
        "non_fractured": 0,
        "negative": 0,
        "normal": 0,
    }
    return df["label"].astype(str).str.lower().map(mapping).astype(int)


class ExternalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, dataset_dir: Path, labels: pd.Series, transform):
        self.df = df.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.labels = labels.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def resolve_path(self, row: pd.Series) -> Path:
        if "image_path" in row and pd.notna(row["image_path"]):
            p = Path(str(row["image_path"]))
            if not p.is_absolute():
                p = self.dataset_dir / p
            return p
        for col in ["image_id", "filename", "file", "path"]:
            if col in row and pd.notna(row[col]):
                name = str(row[col])
                direct = self.dataset_dir / name
                if direct.exists():
                    return direct
                matches = list(self.dataset_dir.rglob(Path(name).name))
                if matches:
                    return sorted(matches)[0]
        raise FileNotFoundError("No image path column could be resolved for external row.")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self.resolve_path(row)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(float(self.labels.iloc[idx])), str(path)


@torch.no_grad()
def evaluate_external(csv_path: Path, dataset_dir: Path) -> pd.DataFrame:
    add_repo_to_path()
    from src.dataset import get_val_transforms
    from src.models import get_model

    df = pd.read_csv(csv_path)
    labels = mapped_labels(df)
    dataset = ExternalDataset(df, dataset_dir, labels, get_val_transforms())
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(ROOT / "outputs" / "resnet50_5ep" / "best_model.pth", map_location=device, weights_only=False)
    model = get_model("resnet50", pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    y_true, y_prob, paths = [], [], []
    for images, labels_batch, batch_paths in loader:
        logits = model(images.to(device)).squeeze(1)
        y_prob.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        y_true.extend(labels_batch.numpy().astype(int).tolist())
        paths.extend(batch_paths)
    metrics = compute_metrics(y_true, y_prob, 0.5)
    return pd.DataFrame([{"dataset": dataset_dir.name, "csv": str(csv_path.relative_to(ROOT)), "threshold": 0.5, "n": len(y_true), **metrics}])


def main() -> None:
    ensure_dirs()
    mapping_lines = ["# External Validation Label Mapping", ""]
    present = [p for p in EXTERNAL_DIRS if p.exists()]
    if not present:
        mapping_lines.extend(
            [
                "No external dataset directory was found under:",
                "- `data/MURA/`",
                "- `data/GRAZPEDWRI-DX/`",
                "- `data/external/`",
                "",
                "Place external images and a labels CSV in one of these folders. The CSV should include `image_id` or `image_path` and a validated binary `fractured` label where 1=fracture and 0=no-fracture.",
            ]
        )
        write_text(EXP_DIR / "external_validation_label_mapping.md", "\n".join(mapping_lines))
        write_text(EXP_DIR / "external_validation_summary.md", "# External Validation Summary\n\nExternal validation was not run because no external dataset folder was present.")
        print("No external dataset found; wrote scaffold notes.")
        return

    for dataset_dir in present:
        csv_path = discover_external_csv(dataset_dir)
        if csv_path is None:
            mapping_lines.append(f"- `{dataset_dir.relative_to(ROOT)}` exists but no labels CSV was found.")
            continue
        df = pd.read_csv(csv_path)
        ok, msg = validate_label_mapping(df)
        mapping_lines.append(f"- `{csv_path.relative_to(ROOT)}`: {msg}")
        if not ok:
            write_text(EXP_DIR / "external_validation_label_mapping.md", "\n".join(mapping_lines))
            write_text(EXP_DIR / "external_validation_summary.md", "# External Validation Summary\n\nExternal validation was stopped because labels could not be mapped cleanly.")
            print("External data exists, but label mapping is not validated.")
            return
        results = evaluate_external(csv_path, dataset_dir)
        results.to_csv(RESULTS_DIR / "external_validation_results.csv", index=False)
        save_latex_table(results, TABLES_DIR / "external_validation_table.tex", "External validation of FracAtlas-trained ResNet50.", "tab:external_validation")
        write_text(EXP_DIR / "external_validation_label_mapping.md", "\n".join(mapping_lines))
        write_text(EXP_DIR / "external_validation_summary.md", "# External Validation Summary\n\nExternal validation was run for the first dataset with validated labels. Results are saved in `results/external_validation_results.csv`.")
        print("External validation complete.")
        return

    write_text(EXP_DIR / "external_validation_label_mapping.md", "\n".join(mapping_lines))
    write_text(
        EXP_DIR / "external_validation_summary.md",
        "# External Validation Summary\n\nExternal dataset directories were present, but no usable labels CSV was found.",
    )
    pd.DataFrame().to_csv(RESULTS_DIR / "external_validation_results.csv", index=False)
    write_text(TABLES_DIR / "external_validation_table.tex", "% External validation table not generated; validated external inference has not been run.")


if __name__ == "__main__":
    main()
