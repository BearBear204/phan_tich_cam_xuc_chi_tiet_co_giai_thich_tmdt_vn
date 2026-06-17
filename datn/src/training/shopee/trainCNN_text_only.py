"""
CNN Training - TEXT ONLY (no feature dependencies)
Models B and C refactored to work with just text input
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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
MODEL_DIR = BASE_DIR / "models" / "cnn"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORDS = 30000
MAX_LEN = 150
EMBED_DIM = 128
BATCH = 128       # Tuned
EPOCHS = 25       # Tuned
LR = 5e-5         # Reduced (was 2e-4)
PATIENCE = 5      # Tuned
MAX_CLASS_WEIGHT = 5.0
GRAD_CLIP_NORM = 1.0


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


def build_cnn_model_text_only(vocab_size: int, num_classes: int) -> Model:
    """CNN with TEXT ONLY - no feature dependencies"""

    text_input = Input(shape=(MAX_LEN,), name="text_input")

    embed = Embedding(
        input_dim=vocab_size,
        output_dim=EMBED_DIM,
    )(text_input)
    embed = SpatialDropout1D(0.3)(embed)

    # 3 parallel conv layers
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

    model = Model(inputs=[text_input], outputs=out)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=LR, clipnorm=GRAD_CLIP_NORM),
        metrics=["accuracy"],
        jit_compile=False,
    )
    return model


class TerminateOnNaNMetrics(Callback):
    """Stop training as soon as loss/val_loss becomes non-finite."""

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None and not np.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss detected at batch {batch}: {loss}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key in ("loss", "val_loss"):
            value = logs.get(key)
            if value is not None and not np.isfinite(value):
                raise RuntimeError(f"Non-finite {key} detected at epoch {epoch + 1}: {value}")


def assert_finite_model_weights(model: Model, context: str) -> None:
    bad = []
    for idx, weights in enumerate(model.get_weights()):
        if not np.isfinite(weights).all():
            bad.append(idx)
    if bad:
        raise RuntimeError(f"Non-finite weights detected {context}; bad arrays: {bad}")


def capped_class_weights(y_train: np.ndarray) -> dict[int, float]:
    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {
        int(cls): float(min(weight, MAX_CLASS_WEIGHT))
        for cls, weight in zip(classes, class_weights)
    }


def load_split_frame(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [text_col, label_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")

    df = df.dropna(subset=[text_col, label_col]).copy()
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
    text_col: str = "comment",
    label_col: str = "rating_star",
    run_name: str = "cnn_text_only",
):
    print(f"\n{'='*80}")
    print(f"CNN TEXT-ONLY (No Feature Dependencies)")
    print(f"{'='*80}")
    print(f"Run: {run_name}")
    print(f"Train: {train_path.name} ({len(pd.read_csv(train_path))} rows)")
    print(f"Val:   {val_path.name}")
    print(f"Test:  {test_path.name}")
    print(f"{'='*80}\n")

    # Load data
    df_train = load_split_frame(train_path, text_col, label_col)
    df_val = load_split_frame(val_path, text_col, label_col)
    df_test = load_split_frame(test_path, text_col, label_col)

    print(f"📊 Loaded - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # Tokenizer
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df_train[text_col].astype(str).tolist())

    X_train = encode_texts(tokenizer, df_train[text_col])
    X_val = encode_texts(tokenizer, df_val[text_col])
    X_test = encode_texts(tokenizer, df_test[text_col])

    y_train = df_train[label_col].values - 1  # Convert 1-5 → 0-4
    y_val = df_val[label_col].values - 1
    y_test = df_test[label_col].values - 1

    # Classes
    classes = np.unique(y_train)
    num_classes = len(classes)
    class_names = format_class_names(label_col, classes)

    print(f"📚 Classes: {class_names}")
    print(f"🔤 Vocab: {len(tokenizer.word_index)}, Max Len: {MAX_LEN}")

    # Class weights
    class_weight_dict = capped_class_weights(y_train)
    print(f"⚖️ Class weights: {class_weight_dict}")

    # Build model
    model = build_cnn_model_text_only(len(tokenizer.word_index) + 1, num_classes)

    model_file = MODEL_DIR / f"best_cnn_{slugify(run_name)}_text_only.keras"
    tok_file = MODEL_DIR / f"tokenizer_{slugify(run_name)}_text_only.pkl"

    # Callbacks
    callbacks = [
        TerminateOnNaNMetrics(),
        ModelCheckpoint(
            str(model_file),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            initial_value_threshold=1e9,
            verbose=0,
        ),
        EarlyStopping(monitor="val_loss", patience=PATIENCE, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=0),
    ]

    # Train
    print(f"\n🚀 Training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    if not model_file.exists():
        raise RuntimeError(
            f"No usable checkpoint was saved for {run_name}. Training likely diverged before a finite val_loss appeared."
        )

    best_model = tf.keras.models.load_model(str(model_file), safe_mode=False)
    assert_finite_model_weights(best_model, f"in checkpoint {model_file.name}")

    # Evaluate
    print(f"\n📈 Evaluating...")
    probs = best_model.predict(X_test, verbose=0)
    if not np.isfinite(probs).all():
        raise RuntimeError(f"Non-finite predictions detected for {run_name}")
    y_pred = np.argmax(probs, axis=1)
    y_pred = y_pred + 1  # Convert back 0-4 → 1-5
    y_test_orig = y_test + 1  # Also convert test labels back for comparison

    acc = (y_pred == y_test_orig).sum() / len(y_test_orig)
    f1_weighted = f1_score(y_test_orig, y_pred, average="weighted")
    f1_macro = f1_score(y_test_orig, y_pred, average="macro")

    print(f"\n{'='*80}")
    print(f"📊 Classification Report:\n")
    print(classification_report(y_test_orig, y_pred, target_names=class_names))
    print(f"{'='*80}\n")

    # Save tokenizer
    with open(tok_file, "wb") as f:
        pickle.dump(tokenizer, f)

    # Save label map
    label_map = {int(c): name for c, name in zip(classes, class_names)}
    label_map_file = MODEL_DIR / f"label_map_{slugify(run_name)}_text_only.json"
    with open(label_map_file, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"✅ Model saved: {model_file.name}")
    print(f"✅ Tokenizer saved: {tok_file.name}")
    print(f"✅ Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--val-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()

    train_and_evaluate(
        args.train_path,
        args.val_path,
        args.test_path,
        run_name=args.run_name,
    )
