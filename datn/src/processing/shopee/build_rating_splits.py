import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _read_csv_with_fallbacks(path: Path) -> pd.DataFrame:
    encodings_to_try: Iterable[str | None] = ("utf-8", "utf-8-sig", "latin1", None)
    last_error: Exception | None = None

    for enc in encodings_to_try:
        try:
            if enc is None:
                return pd.read_csv(path, low_memory=False)
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError(f"Failed to read CSV: {path}") from last_error


def _ensure_col(df: pd.DataFrame, col: str, value) -> pd.DataFrame:
    if col in df.columns:
        return df
    out = df.copy()
    out[col] = value
    return out


def _stratified_split_indices(
    labels: pd.Series,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("Expected 0 < train_frac, 0 < val_frac, train_frac+val_frac < 1")

    rng = np.random.RandomState(seed)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    # Keep label dtype unchanged; only use it for grouping.
    label_values = labels.to_numpy()
    all_indices = np.arange(len(label_values))

    for label in pd.unique(labels):
        class_mask = label_values == label
        class_indices = all_indices[class_mask]
        rng.shuffle(class_indices)

        n_total = len(class_indices)
        n_train = int(round(train_frac * n_total))
        n_val = int(round(val_frac * n_total))

        # Ensure no overflow.
        if n_train + n_val > n_total:
            n_val = max(0, n_total - n_train)

        train_idx.extend(class_indices[:n_train].tolist())
        val_idx.extend(class_indices[n_train : n_train + n_val].tolist())
        test_idx.extend(class_indices[n_train + n_val :].tolist())

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def _distribution(df: pd.DataFrame, label_col: str) -> dict:
    vc = df[label_col].value_counts(dropna=False)
    # Convert numpy ints to plain ints for JSON.
    return {str(k): int(v) for k, v in vc.to_dict().items()}


def _build_key(df: pd.DataFrame, key_cols: list[str]) -> pd.Series:
    if not key_cols:
        raise ValueError("No key columns available")
    if len(key_cols) == 1:
        return df[key_cols[0]].astype(str)
    return df[key_cols].astype(str).agg("\u241f".join, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build stratified 80/10/10 splits from the original shopee_processed CSV and (optionally) "
            "append augmented rows to train only."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/source_pools/shopee_proccessed - Sao ch\u00e9p.csv",
        help="Path to original CSV.",
    )
    parser.add_argument(
        "--augment",
        type=str,
        default="data/augmented/shopee_augmented.csv",
        help="Path to augmentation CSV (train-only).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/rating_splits",
        help="Output directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--do-augment", action="store_true", help="Also write augmented train CSV")
    parser.add_argument(
        "--write-meta",
        action="store_true",
        help="Write a small JSON meta file with counts and label distributions.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_csv_with_fallbacks(input_path)
    if "rating_star" not in df.columns:
        raise ValueError("Expected column rating_star")

    df = df.copy()
    df["is_augmented"] = 0

    train_idx, val_idx, test_idx = _stratified_split_indices(
        df["rating_star"],
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Canonical filenames for the active rating branch.
    train_out = out_dir / "rating_train_base.csv"
    val_out = out_dir / "rating_val.csv"
    test_out = out_dir / "rating_test.csv"

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    meta = {
        "input": str(input_path),
        "seed": int(args.seed),
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
        "outputs": {
            "train": str(train_out),
            "val": str(val_out),
            "test": str(test_out),
        },
        "counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "label_distribution": {
            "train": _distribution(train_df, "rating_star"),
            "val": _distribution(val_df, "rating_star"),
            "test": _distribution(test_df, "rating_star"),
        },
    }

    if args.do_augment:
        augment_path = Path(args.augment)
        aug = _read_csv_with_fallbacks(augment_path)

        # Common column name mapping from augmentation sources.
        rename_map: dict[str, str] = {}
        if "rating_star" not in aug.columns and "rating" in aug.columns:
            rename_map["rating"] = "rating_star"
        if "comment" in df.columns and "comment" not in aug.columns and "content" in aug.columns:
            rename_map["content"] = "comment"
        if "item_id" in df.columns and "item_id" not in aug.columns and "review_id" in aug.columns:
            rename_map["review_id"] = "item_id"

        if rename_map:
            aug = aug.rename(columns=rename_map)

        if "rating_star" not in aug.columns:
            raise ValueError(
                "Augment CSV must contain a rating column (expected rating_star; also supports rating->rating_star)"
            )

        # Normalize rating_star to numeric when possible.
        try:
            aug["rating_star"] = pd.to_numeric(aug["rating_star"])
        except Exception:  # noqa: BLE001
            pass

        aug = _ensure_col(aug, "is_augmented", 1)

        # Align columns to the original schema (preserve original cols + is_augmented).
        base_cols = [c for c in df.columns]
        for c in base_cols:
            if c not in aug.columns:
                aug[c] = np.nan
        aug = aug[base_cols]

        # Remove overlaps with val/test to avoid leakage.
        key_candidates = [c for c in ["item_id", "comment"] if c in df.columns and c in aug.columns]
        if not key_candidates:
            # Fallback to any shared text column
            shared = [c for c in df.columns if c in aug.columns]
            if not shared:
                raise ValueError("No shared columns between base and augment to deduplicate")
            key_candidates = [shared[0]]

        holdout_keys = pd.concat(
            [
                _build_key(val_df, key_candidates),
                _build_key(test_df, key_candidates),
            ],
            ignore_index=True,
        )
        holdout_key_set = set(holdout_keys.tolist())

        aug_keys = _build_key(aug, key_candidates)
        aug = aug.loc[~aug_keys.isin(holdout_key_set)].reset_index(drop=True)

        augmented_train = pd.concat([train_df, aug], ignore_index=True)

        # Match the historical filename used in old logs.
        augmented_train_out = out_dir / "rating_train_augmented.csv"
        augmented_train.to_csv(augmented_train_out, index=False)

        meta["augment"] = {
            "path": str(augment_path),
            "output": str(augmented_train_out),
            "dedup_key_cols": key_candidates,
            "kept_aug_rows": int(len(aug)),
            "final_train_rows": int(len(augmented_train)),
        }

    if args.write_meta:
        meta_out = out_dir / "shopee_processed_split_meta.json"
        meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OK")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
