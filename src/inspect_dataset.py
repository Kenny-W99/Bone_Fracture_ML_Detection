"""
FracAtlas Dataset Inspector
============================

A diagnostic script that helps you understand the structure of your FracAtlas
CSV file BEFORE you start training. Run this first to identify:

  • Which column contains the image filenames  → set as FILENAME_COLUMN in dataset.py
  • Which column contains the binary labels    → set as LABEL_COLUMN in dataset.py
  • Whether image files actually exist on disk

Usage:
------
    python src/inspect_dataset.py --csv_path data/FracAtlas/dataset.csv --image_dir data/FracAtlas/images

    # If images are in subfolders or you want to search recursively:
    python src/inspect_dataset.py --csv_path data/FracAtlas/dataset.csv --image_dir data/FracAtlas/images --recursive
"""

import os
import argparse
from pathlib import Path

import pandas as pd


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(title: str) -> None:
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str) -> None:
    """Print a formatted sub-section header."""
    print()
    print(f"  --- {title} ---")


def guess_filename_columns(df: pd.DataFrame) -> list:
    """
    Heuristic: guess which columns likely contain image filenames.
    Looks for columns whose values contain file-extension-like patterns
    (e.g., .jpg, .png, .dcm) or common naming patterns (id, file, image, path).
    """
    candidates = []
    name_hints = ["file", "image", "img", "path", "name", "id", "filename"]

    for col in df.columns:
        col_lower = col.lower()
        # Check column name for hints
        if any(hint in col_lower for hint in name_hints):
            candidates.append(col)
            continue
        # Check first non-null value for file-extension patterns
        sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
        if isinstance(sample, str) and any(
            ext in sample.lower() for ext in [".jpg", ".jpeg", ".png", ".bmp", ".dcm", ".tif", ".tiff"]
        ):
            candidates.append(col)

    return candidates


def guess_label_columns(df: pd.DataFrame) -> list:
    """
    Heuristic: guess which columns likely contain classification labels.
    Looks for columns with a small number of unique values (2–10) or
    columns whose names suggest labels (label, class, fracture, target, etc.).
    """
    candidates = []
    name_hints = ["label", "class", "fracture", "fract", "target", "diagnosis", "category", "abnormal"]

    for col in df.columns:
        col_lower = col.lower()
        # Check column name for hints
        if any(hint in col_lower for hint in name_hints):
            candidates.append(col)
            continue
        # Check if column has 2–10 unique values (likely categorical)
        if df[col].dtype in ["int64", "float64", "object", "bool"]:
            n_unique = df[col].nunique()
            if 2 <= n_unique <= 10:
                candidates.append(col)

    # Deduplicate while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)
    return unique_candidates


# =============================================================================
# MAIN INSPECTION LOGIC
# =============================================================================

def inspect_dataset(csv_path: str, image_dir: str, recursive: bool = False) -> None:
    """
    Inspect a FracAtlas CSV file and corresponding image directory.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing image metadata and labels.
    image_dir : str
        Path to the directory containing image files.
    recursive : bool
        If True, search for images recursively in subdirectories.
    """

    # =========================================================================
    # 1. Load the CSV
    # =========================================================================
    print_header("1. LOADING CSV FILE")

    if not os.path.isfile(csv_path):
        print(f"  ❌ ERROR: CSV file not found at: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"  ✅ Loaded: {csv_path}")
    print(f"     Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    # =========================================================================
    # 2. Print all column names
    # =========================================================================
    print_header("2. COLUMN NAMES")
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        n_null = df[col].isnull().sum()
        print(f"  [{i:2d}] {col:<30s}  dtype={str(dtype):<10s}  unique={n_unique:<6d}  nulls={n_null}")

    # =========================================================================
    # 3. Print first 5 rows
    # =========================================================================
    print_header("3. FIRST 5 ROWS")
    # Use to_string for clean formatting without truncation
    print(df.head(5).to_string(index=True, max_colwidth=50))

    # =========================================================================
    # 4. Print DataFrame shape
    # =========================================================================
    print_header("4. DATAFRAME SHAPE")
    print(f"  Rows (images): {df.shape[0]:,}")
    print(f"  Columns:       {df.shape[1]}")

    # =========================================================================
    # 5. Print value counts for label-related columns
    # =========================================================================
    print_header("5. VALUE COUNTS FOR POTENTIAL LABEL COLUMNS")

    label_candidates = guess_label_columns(df)

    if not label_candidates:
        print("  ⚠️  No obvious label columns detected. Showing all columns with ≤10 unique values:")
        label_candidates = [col for col in df.columns if df[col].nunique() <= 10]

    if not label_candidates:
        print("  ❌ No columns with ≤10 unique values found. Your CSV may need preprocessing.")
    else:
        for col in label_candidates:
            print_subheader(f"Column: '{col}'")
            vc = df[col].value_counts(dropna=False)
            for value, count in vc.items():
                pct = 100.0 * count / len(df)
                bar = "█" * int(pct / 2)  # Simple visual bar
                print(f"    {str(value):>20s} : {count:>6,} ({pct:5.1f}%) {bar}")

    # =========================================================================
    # 6. Check whether image filenames exist in the image directory
    # =========================================================================
    print_header("6. IMAGE FILE EXISTENCE CHECK")

    if not os.path.isdir(image_dir):
        print(f"  ❌ ERROR: Image directory not found at: {image_dir}")
        print(f"     Please check the --image_dir argument.")
    else:
        # Build a set of all image files in the directory (for fast lookup)
        if recursive:
            print(f"  Scanning recursively: {image_dir}")
            all_files_on_disk = set()
            for root, dirs, files in os.walk(image_dir):
                for f in files:
                    # Store both the full relative path and just the filename
                    rel_path = os.path.relpath(os.path.join(root, f), image_dir)
                    all_files_on_disk.add(rel_path)
                    all_files_on_disk.add(f)  # Also store bare filename
        else:
            print(f"  Scanning (non-recursive): {image_dir}")
            all_files_on_disk = set(os.listdir(image_dir))

        print(f"  Found {len(all_files_on_disk):,} files on disk.")

        # Try each candidate filename column
        filename_candidates = guess_filename_columns(df)

        if not filename_candidates:
            # Fallback: try every string column
            filename_candidates = [col for col in df.columns if df[col].dtype == "object"]
            if filename_candidates:
                print(f"\n  ⚠️  No obvious filename column detected. Trying all string columns.")

        for col in filename_candidates:
            print_subheader(f"Checking column: '{col}'")
            filenames_in_csv = df[col].dropna().astype(str).tolist()
            n_total = len(filenames_in_csv)

            found = 0
            missing = []
            for fname in filenames_in_csv:
                # Check both the raw value and with common extensions
                if fname in all_files_on_disk:
                    found += 1
                else:
                    # Try appending common extensions
                    matched = False
                    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif"]:
                        if (fname + ext) in all_files_on_disk:
                            found += 1
                            matched = True
                            break
                    if not matched:
                        missing.append(fname)

            pct_found = 100.0 * found / n_total if n_total > 0 else 0
            print(f"    Total filenames in CSV : {n_total:,}")
            print(f"    Found on disk          : {found:,} ({pct_found:.1f}%)")
            print(f"    Missing                : {len(missing):,} ({100-pct_found:.1f}%)")

            # -----------------------------------------------------------------
            # 7. Print missing image examples
            # -----------------------------------------------------------------
            if missing:
                n_show = min(10, len(missing))
                print(f"\n    ⚠️  First {n_show} missing filenames:")
                for m in missing[:n_show]:
                    print(f"       • {m}")
                if len(missing) > n_show:
                    print(f"       ... and {len(missing) - n_show} more.")
            else:
                print(f"\n    ✅ ALL images found! This is likely your FILENAME_COLUMN.")

    # =========================================================================
    # 8. Recommendation: which columns to use in dataset.py
    # =========================================================================
    print_header("8. RECOMMENDATION FOR dataset.py")

    print("  Based on the analysis above, update these lines in src/dataset.py:\n")
    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │                                                                 │")

    # Filename recommendation
    if filename_candidates:
        best_filename = filename_candidates[0]
        print(f"  │  FILENAME_COLUMN = \"{best_filename}\"")
    else:
        print(f"  │  FILENAME_COLUMN = \"<CHECK COLUMN NAMES ABOVE>\"")

    # Label recommendation
    if label_candidates:
        # Prefer columns with exactly 2 unique values (binary)
        binary_cols = [c for c in label_candidates if df[c].nunique() == 2]
        best_label = binary_cols[0] if binary_cols else label_candidates[0]
        print(f"  │  LABEL_COLUMN    = \"{best_label}\"")
    else:
        print(f"  │  LABEL_COLUMN    = \"<CHECK VALUE COUNTS ABOVE>\"")

    print("  │                                                                 │")
    print("  └─────────────────────────────────────────────────────────────────┘")

    if label_candidates:
        binary_cols = [c for c in label_candidates if df[c].nunique() == 2]
        if binary_cols:
            print(f"\n  💡 Column '{binary_cols[0]}' has exactly 2 unique values — ideal for binary classification!")
        else:
            print(f"\n  ⚠️  No column has exactly 2 unique values. You may need to binarize a column.")
            print(f"      e.g., df['{label_candidates[0]}'] = (df['{label_candidates[0]}'] > 0).astype(int)")

    print()
    print("  Done! Use the recommendations above to configure src/dataset.py.")
    print()


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a FracAtlas CSV file to identify columns and verify image paths.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/inspect_dataset.py --csv_path data/FracAtlas/dataset.csv --image_dir data/FracAtlas/images
  python src/inspect_dataset.py --csv_path data/FracAtlas/classification.csv --image_dir data/FracAtlas/images --recursive
        """,
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the CSV file (e.g., data/FracAtlas/dataset.csv)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to the image directory (e.g., data/FracAtlas/images)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Search for images recursively in subdirectories (default: False)",
    )

    args = parser.parse_args()

    inspect_dataset(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        recursive=args.recursive,
    )
