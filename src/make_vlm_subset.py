"""
Create a small balanced subset from a FracAtlas split for VLM cost/debug testing.

Example:
    python src/make_vlm_subset.py \
        --input_csv data/FracAtlas/test.csv \
        --output_csv data/FracAtlas/test_vlm_debug_40.csv \
        --per_class 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--label_column", default="fractured")
    parser.add_argument("--per_class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.label_column not in df.columns:
        raise ValueError(f"Missing label column {args.label_column}. Available: {list(df.columns)}")

    parts = []
    for label in [0, 1]:
        subset = df[df[args.label_column] == label]
        n = min(args.per_class, len(subset))
        parts.append(subset.sample(n=n, random_state=args.seed))

    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Saved {len(out)} rows to {output_csv}")
    print(out[args.label_column].value_counts().sort_index())


if __name__ == "__main__":
    main()
