from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.training.shopee.trainPhoBERT import MODEL_NAME, PhoBERTClassifier
from src.training.absa.trainPhoBERT_absa import ASPECT_NAMES, ASPECT_PROMPTS


RATING_CKPT = (
    BASE_DIR
    / "models"
    / "best_phobert_phobert_rating_core"
    / "best_phobert_phobert_rating_core.pt"
)
ABSA_CKPT = (
    BASE_DIR
    / "models"
    / "best_phobert_absa_3class_balanced"
    / "best_phobert_absa_3class_balanced.pt"
)

RATING_LABELS = ["1_star", "2_star", "3_star", "4_star", "5_star"]
ABSA_LABELS = ["positive", "negative", "neutral"]
RATING_MAX_LEN = 96
ABSA_MAX_LEN = 128


class ExplainableRatingModel(PhoBERTClassifier):
    def forward_with_attention(self, input_ids, attention_mask):
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits, outputs.attentions

    def forward_embeds(self, inputs_embeds, attention_mask):
        outputs = self.phobert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


class ExplainableABSAModel(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        from transformers import AutoModel

        self.phobert = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.phobert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

    def forward_with_attention(self, input_ids, attention_mask):
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits, outputs.attentions

    def forward_embeds(self, inputs_embeds, attention_mask):
        outputs = self.phobert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


def normalize_aspect(aspect: str) -> str:
    return ASPECT_PROMPTS.get(aspect, aspect)


def get_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_rating_stack(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = ExplainableRatingModel(num_classes=5, num_features=0, dropout=0.3)
    checkpoint = torch.load(RATING_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, tokenizer


def load_absa_stack(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = ExplainableABSAModel(num_classes=3, dropout=0.3)
    checkpoint = torch.load(ABSA_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, tokenizer


def tokenize_rating(tokenizer, text: str, device: torch.device):
    return tokenizer(
        text,
        return_tensors="pt",
        max_length=RATING_MAX_LEN,
        truncation=True,
        padding="max_length",
    ).to(device)


def tokenize_absa(tokenizer, text: str, aspect: str, device: torch.device):
    return tokenizer(
        text,
        normalize_aspect(aspect),
        return_tensors="pt",
        max_length=ABSA_MAX_LEN,
        truncation=True,
        padding="max_length",
    ).to(device)


def softmax_result(logits: torch.Tensor, labels: list[str]) -> dict:
    probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return {
        "label": labels[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probs": {labels[i]: float(probs[i]) for i in range(len(labels))},
    }


def attention_explanation(model, tokenizer, encoded, labels):
    with torch.no_grad():
        logits, attentions = model.forward_with_attention(
            encoded["input_ids"], encoded["attention_mask"]
        )
        pred = softmax_result(logits, labels)

    last_attn = attentions[-1].mean(dim=1)
    cls_attn = last_attn[0, 0, :]
    seq_len = int(encoded["attention_mask"][0].sum().item())
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0][:seq_len].cpu().tolist())
    scores = cls_attn[:seq_len].detach().cpu().numpy()

    all_tokens = []
    for token, score in zip(tokens, scores):
        if token in ("<s>", "</s>", "<pad>"):
            continue
        all_tokens.append(
            {
                "token": token,
                "score": float(score),
            }
        )
    
    if all_tokens:
        max_score = max(abs(item["score"]) for item in all_tokens)
        if max_score > 0:
            for item in all_tokens:
                item["score"] = item["score"] / max_score

    top_tokens = sorted(all_tokens, key=lambda item: abs(item["score"]), reverse=True)

    return {
        "method": "attention",
        "prediction": pred,
        "top_tokens": top_tokens[:10],
        "all_tokens": all_tokens,
    }


def integrated_gradients_explanation(model, tokenizer, encoded, labels, n_steps: int = 24):
    with torch.no_grad():
        logits = model(encoded["input_ids"], encoded["attention_mask"])
        pred = softmax_result(logits, labels)
        pred_idx = int(np.argmax(list(pred["probs"].values())))

    embed_layer = model.phobert.embeddings.word_embeddings
    input_embeds = embed_layer(encoded["input_ids"])
    baseline = torch.zeros_like(input_embeds)
    integrated_grads = torch.zeros_like(input_embeds)

    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        interpolated = baseline + alpha * (input_embeds - baseline)
        interpolated = interpolated.detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits_interp = model.forward_embeds(interpolated, encoded["attention_mask"])
        score = logits_interp[0, pred_idx]
        score.backward()
        integrated_grads += interpolated.grad.detach()

    integrated_grads = integrated_grads / n_steps
    attributions = (input_embeds - baseline).detach() * integrated_grads
    attr_scores = attributions.sum(dim=-1).squeeze(0)

    seq_len = int(encoded["attention_mask"][0].sum().item())
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0][:seq_len].cpu().tolist())
    scores = attr_scores[:seq_len].detach().cpu().numpy()

    all_tokens = []
    for token, score in zip(tokens, scores):
        if token in ("<s>", "</s>", "<pad>"):
            continue
        all_tokens.append(
            {
                "token": token,
                "score": float(score),
            }
        )

    if all_tokens:
        max_score = max(abs(item["score"]) for item in all_tokens)
        if max_score > 0:
            for item in all_tokens:
                item["score"] = item["score"] / max_score

    top_tokens = sorted(all_tokens, key=lambda item: abs(item["score"]), reverse=True)

    return {
        "method": "integrated_gradients",
        "prediction": pred,
        "top_tokens": top_tokens[:10],
        "all_tokens": all_tokens,
    }


def explain(model, tokenizer, encoded, labels, method: str):
    if method == "none":
        with torch.no_grad():
            logits = model(encoded["input_ids"], encoded["attention_mask"])
        return {"method": "none", "prediction": softmax_result(logits, labels)}
    if method == "ig":
        return integrated_gradients_explanation(model, tokenizer, encoded, labels)
    return attention_explanation(model, tokenizer, encoded, labels)


def analyze_text(
    text: str,
    *,
    device: torch.device,
    xai_method: str = "attention",
    explain_absa: bool = False,
) -> dict:
    rating_model, rating_tok = load_rating_stack(device)
    absa_model, absa_tok = load_absa_stack(device)

    rating_encoded = tokenize_rating(rating_tok, text, device)
    rating_result = explain(rating_model, rating_tok, rating_encoded, RATING_LABELS, xai_method)

    aspects = []
    for aspect in ASPECT_NAMES:
        encoded = tokenize_absa(absa_tok, text, aspect, device)
        aspect_result = explain(
            absa_model,
            absa_tok,
            encoded,
            ABSA_LABELS,
            xai_method if explain_absa else "none",
        )
        aspects.append(
            {
                "aspect": aspect,
                "aspect_text": normalize_aspect(aspect),
                **aspect_result,
            }
        )

    return {
        "text": text,
        "device": str(device),
        "rating": rating_result,
        "absa": aspects,
    }


def main():
    parser = argparse.ArgumentParser(
        description="One-click review analysis: rating + ABSA + XAI"
    )
    parser.add_argument("--text", required=True, help="Review text")
    parser.add_argument(
        "--xai-method",
        default="attention",
        choices=["attention", "ig", "none"],
        help="Default attention for UI speed; use ig only when deeper attribution is needed.",
    )
    parser.add_argument(
        "--explain-absa",
        action="store_true",
        help="Also run XAI for each aspect prediction. Slower than explaining rating only.",
    )
    parser.add_argument("--device", default=None, help="Force device, e.g. cpu or cuda")
    args = parser.parse_args()

    device = get_device(args.device)
    result = analyze_text(
        args.text,
        device=device,
        xai_method=args.xai_method,
        explain_absa=args.explain_absa,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
