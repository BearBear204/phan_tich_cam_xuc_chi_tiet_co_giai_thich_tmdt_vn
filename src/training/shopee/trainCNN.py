import argparse
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import BackupAndRestore, CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Input,
    SpatialDropout1D,
    concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import l2


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_PATH = BASE_DIR / "data" / "rating_splits" / "rating_train_augmented.csv"
DEFAULT_VAL_PATH = BASE_DIR / "data" / "rating_splits" / "rating_val.csv"
DEFAULT_TEST_PATH = BASE_DIR / "data" / "tests" / "rating" / "rating_test.csv"
MODEL_DIR = BASE_DIR / "models" / "cnn"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR = BASE_DIR / "results" / "training" / "cnn"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORDS = 30000
MAX_LEN = 150
EMBED_DIM = 128
BATCH = 64
EPOCHS = 20
LR = 2e-4
PATIENCE = 4


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "run"


def format_class_names(label_col: str, classes) -> list[str]:
    if label_col == "rating_star":
        return [f"{int(c)}⭐" for c in classes]
    return [str(c) for c in classes]


def classes_to_jsonable(classes) -> list:
    jsonable = []
    for item in classes:
        if isinstance(item, (np.integer, int)):
            jsonable.append(int(item))
        elif isinstance(item, (np.floating, float)):
            jsonable.append(float(item))
        else:
            jsonable.append(str(item))
    return jsonable


def build_cnn_model(vocab_size: int, num_classes: int) -> Model:
    inputs = Input(shape=(MAX_LEN,))

    embed = Embedding(
        input_dim=vocab_size,
        output_dim=EMBED_DIM,
    )(inputs)
    embed = SpatialDropout1D(0.3)(embed)

    conv3 = Conv1D(128, 3, activation="relu", kernel_regularizer=l2(0.001))(embed)
    pool3 = GlobalMaxPooling1D()(conv3)

    conv4 = Conv1D(128, 4, activation="relu", kernel_regularizer=l2(0.001))(embed)
    pool4 = GlobalMaxPooling1D()(conv4)

    conv5 = Conv1D(128, 5, activation="relu", kernel_regularizer=l2(0.001))(embed)
    pool5 = GlobalMaxPooling1D()(conv5)

    merged = concatenate([pool3, pool4, pool5])

    x = Dense(256, activation="relu", kernel_regularizer=l2(0.001))(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)

    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, out)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=LR),
        metrics=["accuracy"],
    )
    return model


def load_split_frame(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [text_col, label_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")

    df = df.dropna(subset=required).copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 5].copy()
    return df


def encode_texts(tokenizer: Tokenizer, texts: pd.Series) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(texts.astype(str).tolist())
    return pad_sequences(sequences, maxlen=MAX_LEN)


def train_and_evaluate(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    text_col: str,
    label_col: str,
    run_name: str,
    epochs: int,
    patience: int,
    resume: bool,
):
    print(f"\n{'='*60}")
    print(f"🔥 TRAINING: {run_name}")
    print(f"{'='*60}")
    print(f"📂 Train: {train_path}")
    print(f"📂 Val  : {val_path}")
    print(f"📂 Test : {test_path}")

    start_time = time.time()

    df_train = load_split_frame(train_path, text_col, label_col)
    df_val = load_split_frame(val_path, text_col, label_col)
    df_test = load_split_frame(test_path, text_col, label_col)

    print(f"\n📊 Rows: train={len(df_train):,} | val={len(df_val):,} | test={len(df_test):,}")
    print("📊 Train label distribution:")
    print(df_train[label_col].value_counts(dropna=False).sort_index().to_string())

    le = LabelEncoder()
    y_train = le.fit_transform(df_train[label_col].tolist())
    y_val = le.transform(df_val[label_col].tolist())
    y_test = le.transform(df_test[label_col].tolist())
    class_names = format_class_names(label_col, le.classes_)
    num_classes = len(le.classes_)

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df_train[text_col].tolist())

    x_train = encode_texts(tokenizer, df_train[text_col])
    x_val = encode_texts(tokenizer, df_val[text_col])
    x_test = encode_texts(tokenizer, df_test[text_col])

    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
    class_weights_arr = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights_arr)}

    print(f"\n🏗️  Model built")
    print(f"   ✓ text_col={text_col} | label_col={label_col}")
    print(f"   ✓ num_classes={num_classes}")
    print(f"   ✓ classes={class_names}")
    print(f"   ✓ vocab_size={vocab_size:,}")
    print(f"   ✓ class_weights={class_weight_dict}")

    model = build_cnn_model(vocab_size, num_classes)

    run_base = slugify(run_name)
    checkpoint_path = MODEL_DIR / f"best_cnn_{run_base}.keras"
    tokenizer_path = MODEL_DIR / f"tokenizer_{run_base}.pkl"
    label_meta_path = MODEL_DIR / f"label_map_{run_base}.json"
    backup_dir = MODEL_DIR / f"backup_cnn_{run_base}"
    csv_log_path = HISTORY_DIR / f"history_cnn_{run_base}.csv"

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        mode="min",
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        mode="min",
        verbose=1,
    )
    checkpoint = ModelCheckpoint(
        str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    )
    callbacks = [early_stop, reduce_lr, checkpoint, CSVLogger(str(csv_log_path), append=resume)]
    if resume:
        callbacks.append(
            BackupAndRestore(
                backup_dir=str(backup_dir),
                save_freq="epoch",
                delete_checkpoint=False,
            )
        )

    with open(tokenizer_path, "wb") as handle:
        pickle.dump(tokenizer, handle)
    label_meta_path.write_text(
        json.dumps(
            {
                "label_col": label_col,
                "classes": classes_to_jsonable(le.classes_),
                "class_names": class_names,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"💾 Saved tokenizer: {tokenizer_path.name}")
    print(f"💾 Saved label map: {label_meta_path.name}")
    if resume:
        print(f"💾 Resume backup dir: {backup_dir.name}")
    print(f"💾 History log: {csv_log_path.name}")

    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH,
        epochs=epochs,
        validation_data=(x_val, y_val),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    weighted_f1 = f1_score(y_test, y_pred_classes, average="weighted")
    macro_f1 = f1_score(y_test, y_pred_classes, average="macro")
    cm = confusion_matrix(y_test, y_pred_classes)
    train_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"✅ TEST ACCURACY: {acc * 100:.2f}%")
    print(f"📊 Weighted F1  : {weighted_f1:.4f}")
    print(f"📊 Macro F1     : {macro_f1:.4f}")
    print(f"⏱️  Training Time: {train_time/60:.2f} minutes")
    print(f"{'='*60}")
    print("\n📈 Classification Report:\n")
    print(
        classification_report(
            y_test,
            y_pred_classes,
            target_names=class_names,
            digits=4,
        )
    )

    return {
        "run_name": run_name,
        "accuracy": float(acc),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "loss": float(loss),
        "train_time": float(train_time),
        "history": history.history,
        "confusion_matrix": cm.tolist(),
        "checkpoint_path": str(checkpoint_path),
        "tokenizer_path": str(tokenizer_path),
        "label_meta_path": str(label_meta_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN for Shopee text classification")
    parser.add_argument("--train-path", type=str, default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--val-path", type=str, default=str(DEFAULT_VAL_PATH))
    parser.add_argument("--test-path", type=str, default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--text-col", type=str, default="comment")
    parser.add_argument("--label-col", type=str, default="rating_star")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--resume", action="store_true", help="Resume interrupted training from backup state for the same run-name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    test_path = Path(args.test_path)
    run_name = args.run_name or f"cnn_{args.label_col}_v2"

    print("\n" + "=" * 60)
    print("🚀 CNN TRAINING - Shopee Text Classification")
    print("=" * 60)

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise SystemExit(f"⚠️ File không tồn tại: {path}")

    result = train_and_evaluate(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        text_col=args.text_col,
        label_col=args.label_col,
        run_name=run_name,
        epochs=args.epochs,
        patience=args.patience,
        resume=args.resume,
    )

    print("\n" + "=" * 60)
    print("📊 TÓM TẮT KẾT QUẢ")
    print("=" * 60)
    print(f"Accuracy    : {result['accuracy']:.4f}")
    print(f"Weighted F1 : {result['weighted_f1']:.4f}")
    print(f"Macro F1    : {result['macro_f1']:.4f}")
    print(f"Model       : {result['checkpoint_path']}")
    print(f"Tokenizer   : {result['tokenizer_path']}")
    print(f"Label Map   : {result['label_meta_path']}")
