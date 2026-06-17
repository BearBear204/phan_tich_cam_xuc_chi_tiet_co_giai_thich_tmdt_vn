"""
PhoBERT training for Shopee text classification using pre-split train/val/test CSVs.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(BASE_DIR))

# Helps reduce CUDA allocator fragmentation on shared GPUs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 96
BATCH_SIZE = 8
EPOCHS = 8
LR = 2e-5
DROPOUT = 0.3
LABEL_SMOOTHING = 0.1
WARMUP_RATIO = 0.1
GRAD_ACCUM_STEPS = 4
FREEZE_EPOCHS = 2
PATIENCE = 3
NUM_WORKERS = 0
TRAIN_MICRO_BATCH_SIZE = 2
BERT_TRAINABLE_LAYERS = 2

DEFAULT_TRAIN_PATH = BASE_DIR / "data" / "rating_splits" / "rating_train_augmented.csv"
DEFAULT_VAL_PATH = BASE_DIR / "data" / "rating_splits" / "rating_val.csv"
DEFAULT_TEST_PATH = BASE_DIR / "data" / "tests" / "rating" / "rating_test.csv"
MODEL_DIR = BASE_DIR / "models" / "phobert"
MODEL_DIR.mkdir(exist_ok=True)


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


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, features=None):
        self.texts = texts
        self.labels = labels
        self.features = features  # dict of {feature_name: array}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {
            "text": str(self.texts[idx]),
            "label": int(self.labels[idx]),
        }
        if self.features:
            for feat_name, feat_array in self.features.items():
                item[feat_name] = float(feat_array[idx])
        return item


class BatchTokenizer:
    """Tokenize per-batch to avoid per-sample tokenizer overhead and pad dynamically."""

    def __init__(self, tokenizer, max_len: int, feature_names=None, use_tensor_cores: bool = False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_to_multiple_of = 8 if use_tensor_cores else None
        self.feature_names = feature_names or []

    def __call__(self, batch):
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        encoded["label"] = labels

        # Add features if present
        for feat_name in self.feature_names:
            feat_values = torch.tensor([item[feat_name] for item in batch], dtype=torch.float32)
            encoded[feat_name] = feat_values

        return encoded


class PhoBERTClassifier(nn.Module):
    def __init__(self, num_classes, num_features=0, dropout=0.3):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.phobert.config.hidden_size

        # Concatenate text embedding + features
        feature_input_size = hidden_size + num_features

        self.classifier = nn.Sequential(
            nn.Linear(feature_input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, feature_tensor=None):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        pooled_output = self.dropout(pooled_output)

        # Concatenate with features if available
        if feature_tensor is not None:
            pooled_output = torch.cat([pooled_output, feature_tensor], dim=1)

        return self.classifier(pooled_output)


class SafeBatchNorm1d(nn.BatchNorm1d):
    """Allow training with micro-batch size 1 by falling back to running stats."""

    def forward(self, input):
        if self.training and input.dim() >= 2 and input.size(0) == 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )
        return super().forward(input)


def replace_batch_norm_with_safe(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm1d) and not isinstance(child, SafeBatchNorm1d):
            safe_bn = SafeBatchNorm1d(
                child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            )
            safe_bn.load_state_dict(child.state_dict())
            setattr(module, name, safe_bn)
        else:
            replace_batch_norm_with_safe(child)


def create_grad_scaler(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda" if use_amp else "cpu", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def get_bert_encoder_layers(model) -> list:
    encoder = getattr(model.phobert, "encoder", None)
    layers = getattr(encoder, "layer", None)
    return list(layers) if layers is not None else []


def set_gradient_checkpointing(model, enabled: bool):
    if enabled:
        if hasattr(model.phobert, "gradient_checkpointing_enable"):
            try:
                model.phobert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.phobert.gradient_checkpointing_enable()
        if hasattr(model.phobert, "enable_input_require_grads"):
            model.phobert.enable_input_require_grads()
        if hasattr(model.phobert.config, "use_cache"):
            model.phobert.config.use_cache = False
    else:
        if hasattr(model.phobert, "gradient_checkpointing_disable"):
            model.phobert.gradient_checkpointing_disable()
        if hasattr(model.phobert.config, "use_cache"):
            model.phobert.config.use_cache = True


def set_bert_trainability(model, freeze: bool, trainable_layers: int = -1) -> int:
    for param in model.phobert.parameters():
        param.requires_grad = False

    if freeze:
        return 0

    encoder_layers = get_bert_encoder_layers(model)
    total_layers = len(encoder_layers)

    if trainable_layers == 0:
        return 0

    if not encoder_layers or trainable_layers < 0 or trainable_layers >= total_layers:
        for param in model.phobert.parameters():
            param.requires_grad = True
        return total_layers

    for layer in encoder_layers[-trainable_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    return trainable_layers


def resolve_train_micro_batch_size(batch_size: int, requested_micro_batch_size: int, bert_is_trainable: bool):
    if not bert_is_trainable or requested_micro_batch_size <= 0:
        return None
    return min(batch_size, requested_micro_batch_size)


def load_checkpoint_cpu(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def move_optimizer_state_to_device(optimizer, device: torch.device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device, non_blocking=(device.type == "cuda"))


def load_split_frame(path: Path, text_col: str, label_col: str, feature_cols=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    feature_cols = feature_cols or []
    required = [text_col, label_col] + feature_cols
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")

    df = df.dropna(subset=required).copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 5].copy()
    return df


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch,
    class_weights=None,
    grad_accum_steps=1,
    scaler=None,
    use_amp=False,
    disable_progress: bool = False,
    log_every_steps: int = 2000,
    micro_batch_size=None,
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
        reduction="none",
    )
    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch+1} - Training",
        disable=disable_progress,
        mininterval=5.0,
    )
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        batch_size = labels.size(0)
        current_micro_batch_size = batch_size if micro_batch_size is None else min(batch_size, micro_batch_size)
        batch_loss_denominator = (
            class_weights.index_select(0, labels.to(class_weights.device, non_blocking=True)).sum().item()
            if class_weights is not None
            else float(batch_size)
        )
        batch_loss_denominator = max(batch_loss_denominator, 1e-12)
        batch_loss_sum = 0.0
        batch_correct = 0

        try:
            for start in range(0, batch_size, current_micro_batch_size):
                end = start + current_micro_batch_size
                input_ids_mb = input_ids[start:end].to(device, non_blocking=True)
                attention_mask_mb = attention_mask[start:end].to(device, non_blocking=True)
                labels_mb = labels[start:end].to(device, non_blocking=True)

                # Collect feature tensors if present
                features_mb = None
                feature_names = [k for k in batch.keys() if k not in ['input_ids', 'attention_mask', 'label', 'token_type_ids']]
                if feature_names:
                    feature_tensors = [batch[fname][start:end].to(device, non_blocking=True).reshape(-1, 1) for fname in feature_names]
                    features_mb = torch.cat(feature_tensors, dim=1) if feature_tensors else None

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(input_ids_mb, attention_mask_mb, feature_tensor=features_mb)
                    loss = loss_fn(logits, labels_mb).sum()
                    scaled_loss = loss / (batch_loss_denominator * grad_accum_steps)

                scaler.scale(scaled_loss).backward()
                batch_loss_sum += loss.item()
                preds = torch.argmax(logits, dim=1)
                batch_correct += (preds == labels_mb).sum().item()
                del input_ids_mb, attention_mask_mb, labels_mb, logits, loss, scaled_loss, preds, features_mb
        except torch.OutOfMemoryError as exc:
            if device.type == "cuda":
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM during training. Try lowering --batch-size, lowering --train-micro-batch-size, "
                "lowering --max-len, reducing --bert-trainable-layers, or freeing other GPU processes with nvidia-smi."
            ) from exc

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        batch_loss = batch_loss_sum / batch_loss_denominator
        total_loss += batch_loss
        correct += batch_correct
        total += batch_size

        if not disable_progress:
            progress_bar.set_postfix(
                loss=f"{batch_loss:.4f}",
                acc=f"{100 * correct / total:.2f}%",
            )
        elif log_every_steps and (
            step == 0 or (step + 1) % log_every_steps == 0 or (step + 1) == len(dataloader)
        ):
            pct = 100.0 * (step + 1) / max(1, len(dataloader))
            avg_loss = total_loss / max(1, (step + 1))
            acc = 100.0 * correct / max(1, total)
            try:
                lrs = scheduler.get_last_lr()
                lr_str = ",".join(f"{lr:.2e}" for lr in lrs)
            except Exception:
                lr_str = "?"
            print(
                f"Epoch {epoch+1} - Training: {step+1}/{len(dataloader)} ({pct:.1f}%) | "
                f"loss={avg_loss:.4f} | acc={acc:.2f}% | lr={lr_str}",
                flush=True,
            )

    return total_loss / len(dataloader), correct / total


def eval_model(model, dataloader, device, desc="Validation", use_amp=False, disable_progress: bool = False):
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=desc, disable=disable_progress, mininterval=5.0)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            # Collect feature tensors if present
            features = None
            feature_names = [k for k in batch.keys() if k not in ['input_ids', 'attention_mask', 'label', 'token_type_ids']]
            if feature_names:
                feature_tensors = [batch[fname].to(device, non_blocking=True).reshape(-1, 1) for fname in feature_names]
                features = torch.cat(feature_tensors, dim=1) if feature_tensors else None

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids, attention_mask, feature_tensor=features)
                loss = nn.CrossEntropyLoss()(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    weighted_f1 = f1_score(true_labels, predictions, average="weighted")
    macro_f1 = f1_score(true_labels, predictions, average="macro")
    return avg_loss, accuracy, weighted_f1, macro_f1, predictions, true_labels

def build_optimizer_scheduler(
    model,
    bert_frozen: bool,
    train_loader_len: int,
    grad_accum_steps: int,
    remaining_epochs: int,
    total_steps_override=None,
    warmup_steps_override=None,
):
    if bert_frozen:
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, eps=1e-8)
        total_steps = total_steps_override or max(1, (train_loader_len // grad_accum_steps) * remaining_epochs)
        warmup_steps = warmup_steps_override if warmup_steps_override is not None else int(total_steps * WARMUP_RATIO)
    else:
        bert_params = [param for param in model.phobert.parameters() if param.requires_grad]
        classifier_params = [param for param in model.classifier.parameters() if param.requires_grad]
        optimizer = AdamW(
            [
                {"params": bert_params, "lr": LR / 10},
                {"params": classifier_params, "lr": LR},
            ],
            eps=1e-8,
        )
        total_steps = total_steps_override or max(1, (train_loader_len // grad_accum_steps) * remaining_epochs)
        warmup_steps = warmup_steps_override if warmup_steps_override is not None else 0

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler, total_steps, warmup_steps


def parse_args():
    parser = argparse.ArgumentParser(description="Train PhoBERT for Shopee text classification")
    parser.add_argument("--train-path", type=str, default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--val-path", type=str, default=str(DEFAULT_VAL_PATH))
    parser.add_argument("--test-path", type=str, default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--text-col", type=str, default="comment")
    parser.add_argument("--label-col", type=str, default="rating_star")
    parser.add_argument("--feature-cols", type=str, default="", help="Comma-separated feature column names")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument(
        "--train-micro-batch-size",
        type=int,
        default=TRAIN_MICRO_BATCH_SIZE,
        help="Split each training batch into smaller GPU micro-batches to reduce activation memory (0 disables).",
    )
    parser.add_argument(
        "--bert-trainable-layers",
        type=int,
        default=BERT_TRAINABLE_LAYERS,
        help="How many final PhoBERT encoder layers to unfreeze after warmup (-1=all, 0=keep frozen).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing when PhoBERT is trainable to reduce VRAM usage.",
    )
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--resume", action="store_true", help="Resume interrupted training from the latest checkpoint for the same run-name")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (recommended when piping/tee-ing output to avoid pipe blocking)",
    )
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=2000,
        help="When progress is disabled, print training progress every N steps (0 disables).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    test_path = Path(args.test_path)
    run_name = args.run_name or f"phobert_{args.label_col}_v2"

    # Parse feature columns
    feature_cols = [f.strip() for f in args.feature_cols.split(",") if f.strip()]

    print("=" * 60)
    print("🚀 PhoBERT Training for Shopee Text Classification")
    print("=" * 60)
    print(f"📂 Train: {train_path}")
    print(f"📂 Val  : {val_path}")
    print(f"📂 Test : {test_path}")
    print(f"🧾 text_col={args.text_col} | label_col={args.label_col}")
    if feature_cols:
        print(f"🎯 feature_cols={feature_cols}")

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise SystemExit(f"⚠️ File không tồn tại: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # If stdout is not a TTY (e.g., piped through `tee`), tqdm's frequent carriage-return updates can
    # overwhelm the pipe/pty buffer and block the training process. Disable progress bars by default
    # in that scenario, or when explicitly requested.
    disable_progress = args.no_progress or (not sys.stdout.isatty())
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("⚡ Mixed precision: enabled")
        print("⚡ TF32: enabled")
        print(f"🧠 CUDA alloc conf: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}")

    df_train = load_split_frame(train_path, args.text_col, args.label_col, feature_cols)
    df_val = load_split_frame(val_path, args.text_col, args.label_col, feature_cols)
    df_test = load_split_frame(test_path, args.text_col, args.label_col, feature_cols)

    print(f"\n📊 Rows: train={len(df_train):,} | val={len(df_val):,} | test={len(df_test):,}")
    print("📊 Train label distribution:")
    print(df_train[args.label_col].value_counts(dropna=False).sort_index().to_string())

    le = LabelEncoder()
    train_labels = le.fit_transform(df_train[args.label_col].tolist())
    val_labels = le.transform(df_val[args.label_col].tolist())
    test_labels = le.transform(df_test[args.label_col].tolist())
    class_names = format_class_names(args.label_col, le.classes_)
    num_classes = len(le.classes_)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = BatchTokenizer(tokenizer, args.max_len, feature_names=feature_cols, use_tensor_cores=use_amp)

    # Prepare features dict
    train_features = {col: df_train[col].values for col in feature_cols} if feature_cols else None
    val_features = {col: df_val[col].values for col in feature_cols} if feature_cols else None
    test_features = {col: df_test[col].values for col in feature_cols} if feature_cols else None

    train_dataset = ReviewDataset(df_train[args.text_col].values, train_labels, features=train_features)
    val_dataset = ReviewDataset(df_val[args.text_col].values, val_labels, features=val_features)
    test_dataset = ReviewDataset(df_test[args.text_col].values, test_labels, features=test_features)

    num_workers = max(0, args.num_workers)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "pin_memory": use_amp,
        "collate_fn": collator,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    class_weights_array = compute_class_weight(
        "balanced",
        classes=np.arange(num_classes),
        y=train_labels,
    )
    class_weights = torch.tensor(class_weights_array, dtype=torch.float32).to(device)

    print(f"\n🏗️  Model built")
    print(f"   ✓ num_classes={num_classes}")
    print(f"   ✓ classes={class_names}")
    print(f"   ✓ batch_size={args.batch_size} | grad_accum_steps={args.grad_accum_steps}")
    print(f"   ✓ max_len={args.max_len} | num_workers={num_workers}")
    print(f"   ✓ train_micro_batch_size={args.train_micro_batch_size}")
    print(f"   ✓ bert_trainable_layers={args.bert_trainable_layers}")
    print(f"   ✓ gradient_checkpointing={args.gradient_checkpointing}")
    print(f"   ✓ num_features={len(feature_cols)}")
    print(f"   ✓ class_weights={[f'{w:.3f}' for w in class_weights_array]}")

    model = PhoBERTClassifier(num_classes, num_features=len(feature_cols), dropout=DROPOUT)
    replace_batch_norm_with_safe(model)
    model = model.to(device)
    trainable_bert_layers = set_bert_trainability(model, freeze=True)
    set_gradient_checkpointing(model, enabled=False)
    active_micro_batch_size = resolve_train_micro_batch_size(
        args.batch_size,
        args.train_micro_batch_size,
        bert_is_trainable=False,
    )
    scaler = create_grad_scaler(use_amp)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, eps=1e-8)
    total_steps = max(1, (len(train_loader) // args.grad_accum_steps) * args.epochs)
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    run_base = slugify(run_name)
    model_save_path = MODEL_DIR / f"best_phobert_{run_base}.pt"
    latest_save_path = MODEL_DIR / f"last_phobert_{run_base}.pt"
    label_meta_path = MODEL_DIR / f"label_map_{run_base}.json"
    label_meta_path.write_text(
        json.dumps(
            {
                "label_col": args.label_col,
                "classes": classes_to_jsonable(le.classes_),
                "class_names": class_names,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"💾 Saved label map: {label_meta_path.name}")
    print(f"💾 Latest checkpoint: {latest_save_path.name}")

    bert_is_frozen = True
    optimizer, scheduler, scheduler_total_steps, scheduler_warmup_steps = build_optimizer_scheduler(
        model,
        bert_frozen=bert_is_frozen,
        train_loader_len=len(train_loader),
        grad_accum_steps=args.grad_accum_steps,
        remaining_epochs=args.epochs,
    )

    best_val_macro_f1 = -1.0
    best_val_acc = 0.0
    no_improve_count = 0
    start_epoch = 0

    if args.resume and latest_save_path.exists():
        checkpoint = load_checkpoint_cpu(latest_save_path)
        bert_is_frozen = checkpoint.get("bert_is_frozen", True)
        saved_trainable_layers = checkpoint.get(
            "trainable_bert_layers",
            args.bert_trainable_layers if not bert_is_frozen else 0,
        )
        trainable_bert_layers = set_bert_trainability(
            model,
            freeze=bert_is_frozen,
            trainable_layers=saved_trainable_layers,
        )
        set_gradient_checkpointing(
            model,
            enabled=use_amp and args.gradient_checkpointing and trainable_bert_layers > 0,
        )
        active_micro_batch_size = resolve_train_micro_batch_size(
            args.batch_size,
            args.train_micro_batch_size,
            bert_is_trainable=trainable_bert_layers > 0,
        )
        optimizer, scheduler, scheduler_total_steps, scheduler_warmup_steps = build_optimizer_scheduler(
            model,
            bert_frozen=bert_is_frozen,
            train_loader_len=len(train_loader),
            grad_accum_steps=args.grad_accum_steps,
            remaining_epochs=max(1, checkpoint.get("remaining_epochs", args.epochs)),
            total_steps_override=checkpoint.get("scheduler_total_steps"),
            warmup_steps_override=checkpoint.get("scheduler_warmup_steps"),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        best_val_macro_f1 = checkpoint.get("best_val_macro_f1", -1.0)
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        no_improve_count = checkpoint.get("no_improve_count", 0)
        start_epoch = checkpoint.get("epoch", -1) + 1
        print(f"♻️  Resuming from epoch {start_epoch + 1}/{args.epochs}")
        print(f"   ✓ bert_frozen={bert_is_frozen}")
        print(f"   ✓ trainable_bert_layers={trainable_bert_layers}")
        print(f"   ✓ best_val_macro_f1={best_val_macro_f1:.4f}")
    else:
        optimizer, scheduler, scheduler_total_steps, scheduler_warmup_steps = build_optimizer_scheduler(
            model,
            bert_frozen=bert_is_frozen,
            train_loader_len=len(train_loader),
            grad_accum_steps=args.grad_accum_steps,
            remaining_epochs=args.epochs,
        )

    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'=' * 60}")
        print(f"📍 Epoch {epoch+1}/{args.epochs}")
        print(f"{'=' * 60}")

        if bert_is_frozen and epoch == FREEZE_EPOCHS:
            if args.bert_trainable_layers == 0:
                print("🧊 PhoBERT remains frozen (--bert-trainable-layers=0)")
            else:
                trainable_bert_layers = set_bert_trainability(
                    model,
                    freeze=False,
                    trainable_layers=args.bert_trainable_layers,
                )
                bert_is_frozen = False
                set_gradient_checkpointing(
                    model,
                    enabled=use_amp and args.gradient_checkpointing and trainable_bert_layers > 0,
                )
                active_micro_batch_size = resolve_train_micro_batch_size(
                    args.batch_size,
                    args.train_micro_batch_size,
                    bert_is_trainable=trainable_bert_layers > 0,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                optimizer, scheduler, scheduler_total_steps, scheduler_warmup_steps = build_optimizer_scheduler(
                    model,
                    bert_frozen=bert_is_frozen,
                    train_loader_len=len(train_loader),
                    grad_accum_steps=args.grad_accum_steps,
                    remaining_epochs=max(1, args.epochs - epoch),
                )
                # Reset scaler when changing optimizer to avoid gradient state mismatch
                scaler = create_grad_scaler(use_amp)
                total_bert_layers = len(get_bert_encoder_layers(model))
                layers_label = (
                    "all"
                    if total_bert_layers and trainable_bert_layers >= total_bert_layers
                    else f"last {trainable_bert_layers}/{total_bert_layers}"
                )
                print(
                    f"🔓 Unfreezing BERT ({layers_label} layers, bert_lr={LR/10:.0e}, "
                    f"head_lr={LR:.0e}, micro_batch={active_micro_batch_size or args.batch_size}, "
                    f"gradient_checkpointing={'on' if use_amp and args.gradient_checkpointing else 'off'})"
                )

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            class_weights=class_weights,
            grad_accum_steps=args.grad_accum_steps,
            scaler=scaler,
            use_amp=use_amp,
            disable_progress=disable_progress,
            log_every_steps=args.log_every_steps,
            micro_batch_size=active_micro_batch_size,
        )

        val_loss, val_acc, val_weighted_f1, val_macro_f1, _, _ = eval_model(
            model,
            val_loader,
            device,
            desc=f"Epoch {epoch+1} - Validation",
            use_amp=use_amp,
            disable_progress=disable_progress,
        )

        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Val Weighted F1: {val_weighted_f1:.4f} | Val Macro F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_weighted_f1": val_weighted_f1,
                    "val_macro_f1": val_macro_f1,
                    "config": {
                        "model_name": MODEL_NAME,
                        "max_len": args.max_len,
                        "num_classes": num_classes,
                        "dropout": DROPOUT,
                        "text_col": args.text_col,
                        "label_col": args.label_col,
                        "train_micro_batch_size": args.train_micro_batch_size,
                        "bert_trainable_layers": args.bert_trainable_layers,
                        "gradient_checkpointing": args.gradient_checkpointing,
                        "classes": classes_to_jsonable(le.classes_),
                        "class_names": class_names,
                    },
                },
                model_save_path,
            )
            print(f"  ✅ Saved best model to {model_save_path}")
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"  ⏳ No improvement ({no_improve_count}/{args.patience})")
            if no_improve_count >= args.patience:
                print(f"\n🛑 EarlyStopping after {args.patience} epochs without improvement.")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "best_val_acc": best_val_acc,
                        "best_val_macro_f1": best_val_macro_f1,
                        "no_improve_count": no_improve_count,
                        "bert_is_frozen": bert_is_frozen,
                        "trainable_bert_layers": trainable_bert_layers,
                        "remaining_epochs": max(1, args.epochs - epoch - 1),
                        "scheduler_total_steps": scheduler_total_steps,
                        "scheduler_warmup_steps": scheduler_warmup_steps,
                        "config": {
                            "model_name": MODEL_NAME,
                            "max_len": args.max_len,
                            "num_classes": num_classes,
                            "dropout": DROPOUT,
                            "text_col": args.text_col,
                            "label_col": args.label_col,
                            "train_micro_batch_size": args.train_micro_batch_size,
                            "bert_trainable_layers": args.bert_trainable_layers,
                            "gradient_checkpointing": args.gradient_checkpointing,
                            "classes": classes_to_jsonable(le.classes_),
                            "class_names": class_names,
                        },
                    },
                    latest_save_path,
                )
                break

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_acc": best_val_acc,
                "best_val_macro_f1": best_val_macro_f1,
                "no_improve_count": no_improve_count,
                "bert_is_frozen": bert_is_frozen,
                "trainable_bert_layers": trainable_bert_layers,
                "remaining_epochs": max(1, args.epochs - epoch - 1),
                "scheduler_total_steps": scheduler_total_steps,
                "scheduler_warmup_steps": scheduler_warmup_steps,
                "config": {
                    "model_name": MODEL_NAME,
                    "max_len": args.max_len,
                    "num_classes": num_classes,
                    "dropout": DROPOUT,
                    "text_col": args.text_col,
                    "label_col": args.label_col,
                    "train_micro_batch_size": args.train_micro_batch_size,
                    "bert_trainable_layers": args.bert_trainable_layers,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "classes": classes_to_jsonable(le.classes_),
                    "class_names": class_names,
                },
            },
            latest_save_path,
        )

    print(f"\n🏆 Best Val Accuracy : {best_val_acc:.4f}")
    print(f"🏆 Best Val Macro F1 : {best_val_macro_f1:.4f}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
    checkpoint = load_checkpoint_cpu(model_save_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_weighted_f1, test_macro_f1, test_preds, test_true = eval_model(
        model,
        test_loader,
        device,
        desc="Test",
        use_amp=use_amp,
        disable_progress=disable_progress,
    )

    print(f"\n{'=' * 60}")
    print("📊 Final Test Results")
    print(f"{'=' * 60}")
    print(f"Test Loss       : {test_loss:.4f}")
    print(f"Test Accuracy   : {test_acc:.4f}")
    print(f"Test Weighted F1: {test_weighted_f1:.4f}")
    print(f"Test Macro F1   : {test_macro_f1:.4f}")
    print(
        classification_report(
            test_true,
            test_preds,
            target_names=class_names,
            digits=4,
        )
        )
    print("🧩 Confusion Matrix:")
    print(confusion_matrix(test_true, test_preds))
    print(f"\n✅ Model saved to {model_save_path}")
    print(f"💾 Latest resume checkpoint: {latest_save_path}")


if __name__ == "__main__":
    main()
