from __future__ import annotations

import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(BASE_DIR))

import src.training.shopee.trainPhoBERT as train_phobert


# Recovered legacy-style defaults from the strong `phobert_rating_star` run log.
train_phobert.MAX_LEN = 128
train_phobert.BATCH_SIZE = 24
train_phobert.EPOCHS = 12
train_phobert.GRAD_ACCUM_STEPS = 2
train_phobert.TRAIN_MICRO_BATCH_SIZE = 0
train_phobert.BERT_TRAINABLE_LAYERS = -1
train_phobert.DEFAULT_TRAIN_PATH = BASE_DIR / "data" / "rating_splits" / "rating_train_augmented.csv"
train_phobert.DEFAULT_VAL_PATH = BASE_DIR / "data" / "rating_splits" / "rating_val.csv"
train_phobert.DEFAULT_TEST_PATH = BASE_DIR / "data" / "tests" / "rating" / "rating_test.csv"


if __name__ == "__main__":
    train_phobert.main()
