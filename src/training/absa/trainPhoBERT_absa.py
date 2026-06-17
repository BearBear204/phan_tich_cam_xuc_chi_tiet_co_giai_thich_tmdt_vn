"""
PhoBERT Fine-tuning for ABSA (Aspect-Based Sentiment Analysis)

Input format: (review_text, aspect_name) → sentiment label
Sử dụng PhoBERT sentence-pair encoding:
  [CLS] review_content [SEP] aspect_name [SEP]

4 classes: positive(0) / negative(1) / neutral(2) / none(3)

Usage:
    python src/training/trainPhoBERT_absa.py
    python src/training/trainPhoBERT_absa.py --load-pretrained   # Dùng PhoBERT đã fine-tune sentiment
    python src/training/trainPhoBERT_absa.py --drop-none         # Train 3-class bỏ nhãn none
    python src/training/trainPhoBERT_absa.py --epochs 10 --lr 2e-5
    python src/training/trainPhoBERT_absa.py --drop-none --resume models/phobert/latest_phobert_absa_3class.pt
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(BASE_DIR))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoModel, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

# ==================================================
# CONFIG
# ==================================================
MODEL_NAME      = "vinai/phobert-base"
MAX_LEN         = 128
BATCH_SIZE      = 16
EPOCHS          = 8
LR              = 1.5e-5
DROPOUT         = 0.2
LABEL_SMOOTHING = 0.0
WARMUP_RATIO    = 0.06
GRAD_ACCUM      = 2
FREEZE_EPOCHS   = 0
PATIENCE        = 3
HEAD_LR_MULT    = 5.0

NUM_CLASSES_4   = 4       # positive / negative / neutral / none
NUM_CLASSES_3   = 3       # positive / negative / neutral

LABEL_NAMES_4   = ["positive", "negative", "neutral", "none"]
LABEL_NAMES_3   = ["positive", "negative", "neutral"]

ASPECT_NAMES = [
    "product_quality", "delivery", "service",
]
ASPECT_PROMPTS = {
    "product_quality": "chất lượng sản phẩm",
    "delivery": "giao hàng",
    "service": "dịch vụ người bán",
}

# Paths
DATA_DIR   = BASE_DIR / "data" / "absa"
TEST_DIR   = BASE_DIR / "data" / "tests" / "absa"
MODEL_DIR  = BASE_DIR / "models" / "phobert"
LOG_PATH   = BASE_DIR / "logs" / "phobert_absa_training.log"
REPORT_DIR = BASE_DIR / "results" / "training" / "absa"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# DATASET
# ==================================================
class ABSADataset(Dataset):
    """Dataset cho ABSA: encode cặp (content, aspect)"""

    def __init__(self, contents, aspects, labels, tokenizer, max_len):
        self.contents  = contents
        self.aspects   = aspects
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):
        aspect_text = ASPECT_PROMPTS.get(str(self.aspects[idx]), str(self.aspects[idx]))
        # Pair encoding: tokenizer tự thêm [CLS] text [SEP] aspect [SEP]
        encoding = self.tokenizer(
            str(self.contents[idx]),
            aspect_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids"      : encoding["input_ids"].flatten(),
            "attention_mask" : encoding["attention_mask"].flatten(),
            "label"          : torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ==================================================
# MODEL
# ==================================================
class PhoBERTABSA(nn.Module):
    """PhoBERT fine-tuned cho ABSA (sentence-pair classification)"""

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.phobert    = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout    = nn.Dropout(dropout)
        hidden          = self.phobert.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        token_emb    = out.last_hidden_state               # [B, seq, 768]
        mask_exp     = attention_mask.unsqueeze(-1).float()
        sum_emb      = (token_emb * mask_exp).sum(dim=1)
        sum_mask     = mask_exp.sum(dim=1).clamp(min=1e-9)
        pooled       = sum_emb / sum_mask                  # [B, 768]
        pooled       = self.dropout(pooled)
        return self.classifier(pooled)


# ==================================================
# TRAIN / EVAL
# ==================================================
def train_epoch(model, loader, optimizer, scheduler, device, epoch, loss_fn, *, scaler, use_amp: bool, grad_accum: int):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()

    bar = tqdm(loader, desc=f"Epoch {epoch+1} Train")
    for step, batch in enumerate(bar):
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(ids, mask)
            loss   = loss_fn(logits, labels) / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        preds      = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        bar.set_postfix(loss=f"{loss.item()*grad_accum:.4f}", acc=f"{correct/total*100:.2f}%")

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, device, desc="Val", *, use_amp: bool):
    model.eval()
    preds_all, labels_all = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(ids, mask)
                loss   = nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss.item()

            preds_all.extend(logits.argmax(dim=1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    acc  = accuracy_score(labels_all, preds_all)
    f1   = f1_score(labels_all, preds_all, average="macro")
    return total_loss / len(loader), acc, f1, preds_all, labels_all


def save_resume_checkpoint(
    path: Path,
    *,
    epoch: int,
    model,
    optimizer,
    scheduler,
    scaler,
    best_val_f1: float,
    patience_cnt: int,
    val_acc: float,
    val_f1: float,
    num_classes: int,
    label_names: list[str],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_f1": best_val_f1,
            "patience_cnt": patience_cnt,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "num_classes": num_classes,
            "label_names": label_names,
            "aspects": ASPECT_NAMES,
        },
        path,
    )


# ==================================================
# MAIN
# ==================================================
def main():
    parser = argparse.ArgumentParser(description="Train PhoBERT ABSA")
    parser.add_argument("--epochs",         type=int,   default=EPOCHS)
    parser.add_argument("--lr",             type=float, default=LR)
    parser.add_argument("--head-lr-mult",   type=float, default=HEAD_LR_MULT)
    parser.add_argument("--batch-size",     type=int,   default=BATCH_SIZE)
    parser.add_argument("--max-len",        type=int,   default=MAX_LEN)
    parser.add_argument("--grad-accum",     type=int,   default=GRAD_ACCUM)
    parser.add_argument("--dropout",        type=float, default=DROPOUT)
    parser.add_argument("--warmup-ratio",   type=float, default=WARMUP_RATIO)
    parser.add_argument("--freeze-epochs",  type=int,   default=FREEZE_EPOCHS)
    parser.add_argument("--patience",       type=int,   default=PATIENCE)
    parser.add_argument("--log-path",       type=str,   default=str(LOG_PATH))
    parser.add_argument("--train-path",     type=str,   default="")
    parser.add_argument("--val-path",       type=str,   default="")
    parser.add_argument("--test-path",      type=str,   default="")
    parser.add_argument("--artifact-suffix", type=str, default="",
                        help="Extra suffix for best/latest/history artifact names, e.g. balanced")
    parser.add_argument(
        "--class-weight-mode",
        type=str,
        default="none",
        choices=["none", "balanced"],
        help="Use class weights in CrossEntropyLoss. Default none for balanced train.",
    )
    parser.add_argument("--resume",         type=str,   default="",
                        help="Path to a resume checkpoint created by this trainer")
    parser.add_argument("--drop-none",      action="store_true",
                        help="Train 3-class (bỏ nhãn none)")
    parser.add_argument("--load-pretrained", action="store_true",
                        help="Load PhoBERT backbone từ best_phobert_phobert_rating_core.pt")
    args = parser.parse_args()

    suffix     = "_3class" if args.drop_none else ""
    num_classes = NUM_CLASSES_3 if args.drop_none else NUM_CLASSES_4
    label_names = LABEL_NAMES_3 if args.drop_none else LABEL_NAMES_4
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*60)
    print("🚀 PhoBERT ABSA Training")
    print("="*60)
    print(f"🖥️  Device     : {device}")
    print(f"🏷️  Classes    : {num_classes} ({', '.join(label_names)})")
    print(f"📦 Batch     : {args.batch_size} × {args.grad_accum} (eff. {args.batch_size*args.grad_accum})")
    print(f"📏 Max len   : {args.max_len}")
    print(f"📚 Epochs    : {args.epochs}")
    print(f"🎯 LR        : {args.lr}")
    print(f"🎯 Head LR×  : {args.head_lr_mult}")
    print(f"💧 Dropout   : {args.dropout}")
    print(f"🔥 Warmup    : {args.warmup_ratio}")
    print(f"🧊 Freeze ep : {args.freeze_epochs}")
    print(f"🛑 Patience  : {args.patience}")
    if torch.cuda.is_available():
        print(f"🎮 GPU        : {torch.cuda.get_device_name(0)}")

    # ── Load data ──────────────────────────────────────────
    if args.train_path:
        train_path = Path(args.train_path)
    elif args.drop_none:
        train_path = DATA_DIR / "train_3class_balanced.csv"
    else:
        train_path = DATA_DIR / f"train{suffix}.csv"

    val_path = Path(args.val_path) if args.val_path else DATA_DIR / f"val{suffix}.csv"
    test_path = Path(args.test_path) if args.test_path else TEST_DIR / f"test{suffix}.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            print(f"\n❌ Không tìm thấy {p}")
            print("   Chạy trước builder ABSA core hoặc truyền --train-path/--val-path/--test-path")
            sys.exit(1)

    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)
    df_test  = pd.read_csv(test_path)

    print(f"\n📂 Train : {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")
    print(f"📊 Train label dist: {df_train['label'].value_counts().sort_index().to_dict()}")

    # ── Tokenizer ───────────────────────────────────────────
    print(f"\n🔤 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Datasets / Loaders ─────────────────────────────────
    def make_loader(df, shuffle=False):
        ds = ABSADataset(
            df["content"].values,
            df["aspect"].values,
            df["label"].values,
            tokenizer,
            args.max_len,
        )
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = make_loader(df_train, shuffle=True)
    val_loader   = make_loader(df_val)
    test_loader  = make_loader(df_test)

    # ── Model ───────────────────────────────────────────────
    print(f"\n🧠 Initializing PhoBERT ABSA ({num_classes}-class)...")
    model = PhoBERTABSA(num_classes=num_classes, dropout=args.dropout)

    if args.load_pretrained:
        pretrained_path = MODEL_DIR / "best_phobert_phobert_rating_core.pt"
        if pretrained_path.exists():
            checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            state = checkpoint.get("model_state_dict", checkpoint)
            # Chỉ load PhoBERT backbone, bỏ classifier (khác num_classes)
            phobert_state = {k.replace("phobert.", ""): v
                             for k, v in state.items() if k.startswith("phobert.")}
            missing, unexpected = model.phobert.load_state_dict(phobert_state, strict=False)
            print(f"✅ Loaded PhoBERT backbone từ {pretrained_path}")
            print(f"   Missing: {len(missing)} | Unexpected: {len(unexpected)}")
        else:
            print(f"⚠️  Không tìm thấy {pretrained_path}, dùng pretrained từ HuggingFace")

    model = model.to(device)

    # ── Class weights ───────────────────────────────────────
    labels_arr = df_train["label"].values
    class_weights = None
    if args.class_weight_mode == "balanced":
        cw = compute_class_weight("balanced", classes=np.arange(num_classes), y=labels_arr)
        class_weights = torch.tensor(cw, dtype=torch.float).to(device)
        print(f"⚖️  Class weights: {[f'{w:.2f}' for w in cw]}")
    else:
        print("⚖️  Class weights: disabled")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    # ── Optimizer / Scheduler ───────────────────────────────
    optimizer = AdamW(
        [
            {"params": model.phobert.parameters(), "lr": args.lr, "weight_decay": 0.01},
            {"params": model.classifier.parameters(), "lr": args.lr * args.head_lr_mult, "weight_decay": 0.01},
        ]
    )
    total_steps  = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Freeze backbone đầu
    def set_backbone_grad(requires_grad: bool):
        for param in model.phobert.parameters():
            param.requires_grad = requires_grad

    # ── Training loop ───────────────────────────────────────
    best_val_f1   = 0.0
    patience_cnt  = 0
    start_epoch   = 0
    artifact_suffix = f"_{args.artifact_suffix.strip()}" if args.artifact_suffix.strip() else ""
    best_ckpt     = MODEL_DIR / f"best_phobert_absa{suffix}{artifact_suffix}.pt"
    latest_ckpt   = MODEL_DIR / f"latest_phobert_absa{suffix}{artifact_suffix}.pt"
    history_path  = REPORT_DIR / f"absa_training_history{suffix}{artifact_suffix}.csv"
    history       = []

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"\n❌ Resume checkpoint not found: {resume_path}")
            sys.exit(1)

        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        scaler_state = resume_ckpt.get("scaler_state_dict")
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        start_epoch = int(resume_ckpt.get("epoch", 0))
        best_val_f1 = float(resume_ckpt.get("best_val_f1", 0.0))
        patience_cnt = int(resume_ckpt.get("patience_cnt", 0))
        print(f"\n🔁 Resuming from {resume_path}")
        print(f"   start_epoch={start_epoch+1} best_val_f1={best_val_f1:.4f} patience={patience_cnt}")

    print(f"\n{'='*60}")
    print("📈 TRAINING")
    print(f"{'='*60}")

    for epoch in range(start_epoch, args.epochs):
        # Unfreeze sau FREEZE_EPOCHS
        if epoch == 0 and args.freeze_epochs > 0:
            set_backbone_grad(False)
            print(f"\n🔒 Epoch 1-{args.freeze_epochs}: Backbone frozen, chỉ train head")
        elif epoch == args.freeze_epochs and args.freeze_epochs > 0:
            set_backbone_grad(True)
            print(f"\n🔓 Epoch {epoch+1}+: Backbone unfrozen, full fine-tune")
        elif epoch == 0 and args.freeze_epochs == 0:
            set_backbone_grad(True)
            print("\n🔓 Full fine-tune from epoch 1")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, loss_fn,
            scaler=scaler, use_amp=use_amp, grad_accum=args.grad_accum
        )
        val_loss, val_acc, val_f1, _, _ = eval_epoch(
            model, val_loader, device, f"Epoch {epoch+1} Val", use_amp=use_amp
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss,     "val_acc": val_acc, "val_f1": val_f1,
        })

        print(f"\nEpoch {epoch+1}/{args.epochs} | "
              f"Train loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
              f"Val loss={val_loss:.4f} acc={val_acc*100:.2f}% f1={val_f1:.4f}")

        # Log to file
        with open(log_path, "a") as f:
            f.write(f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}\n")

        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_cnt = 0
            torch.save({
                "epoch"           : epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_acc"         : val_acc,
                "val_f1"          : val_f1,
                "best_val_f1"     : val_f1,
                "patience_cnt"    : patience_cnt,
                "num_classes"     : num_classes,
                "label_names"     : label_names,
                "aspects"         : ASPECT_NAMES,
            }, best_ckpt)
            print(f"  💾 Saved best → {best_ckpt.name} (F1={val_f1:.4f})")
        else:
            patience_cnt += 1
            print(f"  ⏳ No improvement ({patience_cnt}/{args.patience})")

        save_resume_checkpoint(
            latest_ckpt,
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_val_f1=best_val_f1,
            patience_cnt=patience_cnt,
            val_acc=val_acc,
            val_f1=val_f1,
            num_classes=num_classes,
            label_names=label_names,
        )

        pd.DataFrame(history).to_csv(history_path, index=False)

        if patience_cnt >= args.patience:
            print(f"\n⏹️  Early stopping tại epoch {epoch+1}")
            break

    # ── Test evaluation ────────────────────────────────────
    print(f"\n{'='*60}")
    print("🧪 TEST EVALUATION")
    print(f"{'='*60}")

    checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, test_acc, test_f1, test_preds, test_labels = eval_epoch(
        model, test_loader, device, "Test", use_amp=use_amp
    )

    print(f"\n📊 Test Accuracy : {test_acc*100:.2f}%")
    print(f"📊 Test Macro F1 : {test_f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=label_names))

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print("Confusion Matrix:")
    print(cm)

    # Per-aspect accuracy
    df_test["pred"] = test_preds[: len(df_test)]
    print(f"\n📊 Accuracy per aspect (test set):")
    for asp in ASPECT_NAMES:
        sub = df_test[df_test["aspect"] == asp]
        if len(sub) == 0:
            continue
        acc_asp = (sub["label"] == sub["pred"]).mean()
        print(f"  {asp:20s}: {acc_asp*100:.1f}% (n={len(sub)})")

    # Save history
    pd.DataFrame(history).to_csv(history_path, index=False)

    print(f"\n{'='*60}")
    print(f"✅ DONE")
    print(f"📊 Best Val F1  : {best_val_f1:.4f}")
    print(f"📊 Test Accuracy: {test_acc*100:.2f}%")
    print(f"📊 Test Macro F1: {test_f1:.4f}")
    print(f"💾 Best model   : {best_ckpt}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
