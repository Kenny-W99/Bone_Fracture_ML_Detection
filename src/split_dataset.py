import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def print_distribution(name, df, label_column):
    counts = df[label_column].value_counts().sort_index()
    total = len(df)
    print(f"\n{name}: {total} images")
    for label, count in counts.items():
        pct = count / total * 100
        print(f"  class {label}: {count} ({pct:.2f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label_column", default="fractured")
    parser.add_argument("--train_size", type=float, default=0.70)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if args.label_column not in df.columns:
        raise ValueError(f"Label column '{args.label_column}' not found. Columns: {list(df.columns)}")

    if abs(args.train_size + args.val_size + args.test_size - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    train_df, temp_df = train_test_split(
        df,
        train_size=args.train_size,
        stratify=df[args.label_column],
        random_state=args.seed,
    )

    relative_val_size = args.val_size / (args.val_size + args.test_size)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        stratify=temp_df[args.label_column],
        random_state=args.seed,
    )

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print_distribution("Train", train_df, args.label_column)
    print_distribution("Validation", val_df, args.label_column)
    print_distribution("Test", test_df, args.label_column)

    print("\nSaved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")


if __name__ == "__main__":
    main()
