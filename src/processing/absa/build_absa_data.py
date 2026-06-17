"""
Build core ABSA datasets from reviews_final_cleaned.csv using only
user-provided aspect ratings:

  - product_quality
  - seller_service
  - delivery_service

Output format matches src/training/absa/trainPhoBERT_absa.py:
  content, aspect, label

Label ids:
  0 = positive
  1 = negative
  2 = neutral
  3 = none
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_PATH = BASE_DIR / "data" / "raw_source" / "reviews_final_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "data" / "absa"

LABEL_MAP = {"positive": 0, "negative": 1, "neutral": 2, "none": 3}

ASPECT_COLUMNS = {
    "product_quality": ("product_quality", False),
    "service": ("seller_service", True),
    "delivery": ("delivery_service", True),
}


def map_star_to_label(star: int, *, is_none_if_zero: bool) -> str:
    if is_none_if_zero and star == 0:
        return "none"
    if star in (4, 5):
        return "positive"
    if star in (1, 2):
        return "negative"
    if star == 3:
        return "neutral"
    return "none"


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        content = str(row.get("comment", "")).strip()
        if len(content) < 10:
            continue

        for aspect, (column, is_none_if_zero) in ASPECT_COLUMNS.items():
            try:
                star = int(row.get(column, 0))
            except Exception:
                star = 0
            label = map_star_to_label(star, is_none_if_zero=is_none_if_zero)
            rows.append(
                {
                    "content": content,
                    "aspect": aspect,
                    "label": LABEL_MAP[label],
                }
            )

    return pd.DataFrame(rows)


def safe_stratify_series(df: pd.DataFrame) -> pd.Series | None:
    """Return a usable stratify target if the label distribution permits it."""
    if "rating_star" not in df.columns:
        return None

    target = df["rating_star"]
    if target.isna().any():
        return None

    counts = target.value_counts()
    if counts.empty or counts.min() < 2:
        return None

    return target


def split_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split at review level first to avoid leaking the same review across aspects."""
    stratify_all = safe_stratify_series(df)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=stratify_all,
    )

    stratify_temp = safe_stratify_series(temp_df)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=stratify_temp,
    )
    return train_df, val_df, test_df


def maybe_drop_none(df: pd.DataFrame, drop_none: bool) -> pd.DataFrame:
    if not drop_none:
        return df
    out = df[df["label"] != LABEL_MAP["none"]].copy()
    out["label"] = out["label"].map({0: 0, 1: 1, 2: 2})
    return out


def maybe_balance_train(df: pd.DataFrame, method: str, ratio: float) -> pd.DataFrame:
    if method == "none" or df.empty:
        return df

    if method == "aspect_label_undersample":
        parts = []
        for aspect, grp_aspect in df.groupby("aspect", sort=False):
            label_counts = grp_aspect["label"].value_counts()
            if label_counts.empty:
                parts.append(grp_aspect)
                continue
            min_count = int(label_counts.min())
            cap = max(min_count, int(min_count * ratio))
            for _, grp_label in grp_aspect.groupby("label", sort=False):
                if len(grp_label) > cap:
                    parts.append(grp_label.sample(cap, random_state=42))
                else:
                    parts.append(grp_label)
        return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

    label_counts = df["label"].value_counts()
    if method == "undersample":
        min_count = label_counts.min()
        cap = int(min_count * ratio)
        parts = []
        for _, grp in df.groupby("label"):
            parts.append(grp.sample(cap, random_state=42) if len(grp) > cap else grp)
        return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

    max_count = label_counts.max()
    parts = []
    for _, grp in df.groupby("label"):
        parts.append(grp.sample(max_count, replace=True, random_state=42) if len(grp) < max_count else grp)
    return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 3-aspect core ABSA dataset from reviews_final_cleaned.csv")
    parser.add_argument("--min-length", type=int, default=15)
    parser.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    parser.add_argument("--drop-none", action="store_true", help="Build 3-class dataset without label none")
    parser.add_argument(
        "--balance",
        type=str,
        default="none",
        choices=["none", "undersample", "oversample", "aspect_label_undersample"],
    )
    parser.add_argument("--balance-ratio", type=float, default=3.0)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building core ABSA dataset from reviews_final_cleaned.csv")
    print("=" * 60)
    print(f"Input   : {INPUT_PATH}")
    print(f"Aspects : {list(ASPECT_COLUMNS.keys())}")

    if not INPUT_PATH.exists():
        print(f"Missing input: {INPUT_PATH}", file=sys.stderr)
        raise SystemExit(1)

    df = pd.read_csv(INPUT_PATH)
    df["comment"] = df["comment"].astype(str).str.strip()
    df = df[df["comment"].str.len() >= args.min_length].copy()
    df = df.drop_duplicates(subset=["comment"])
    if args.max_rows > 0 and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=42)

    print(f"Rows after filtering: {len(df):,}")

    label_inv = {v: k for k, v in LABEL_MAP.items()}
    review_train, review_val, review_test = split_reviews(df)
    print(
        "Review split sizes: "
        f"train={len(review_train):,} val={len(review_val):,} test={len(review_test):,}"
    )

    train_df = build_pairs(review_train)
    val_df = build_pairs(review_val)
    test_df = build_pairs(review_test)

    print(
        "ABSA pair sizes before drop-none: "
        f"train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}"
    )

    suffix = "_3class" if args.drop_none else ""
    train_df = maybe_drop_none(train_df, args.drop_none)
    val_df = maybe_drop_none(val_df, args.drop_none)
    test_df = maybe_drop_none(test_df, args.drop_none)

    if args.drop_none:
        print(
            "ABSA pair sizes after drop-none: "
            f"train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}"
        )

    train_df = maybe_balance_train(train_df, args.balance, args.balance_ratio)
    if args.balance != "none":
        print(f"Train size after balance ({args.balance}): {len(train_df):,}")

    file_suffix = suffix
    if args.balance != "none":
        file_suffix = f"{file_suffix}_balanced"

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print("Final label distribution:")
    for lbl, cnt in full_df["label"].value_counts().sort_index().items():
        print(f"  {label_inv[lbl]:10s}: {cnt:>8,}")

    print("Final per-aspect label distribution:")
    aspect_dist = full_df.groupby(["aspect", "label"]).size().unstack(fill_value=0)
    print(aspect_dist.to_string())

    train_path = out_dir / f"train{file_suffix}.csv"
    val_path = out_dir / f"val{file_suffix}.csv"
    test_path = out_dir / f"test{file_suffix}.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved split sizes: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
    print("Saved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print("Train with:")
    print("  python src/training/absa/trainPhoBERT_absa.py")


if __name__ == "__main__":
    main()
