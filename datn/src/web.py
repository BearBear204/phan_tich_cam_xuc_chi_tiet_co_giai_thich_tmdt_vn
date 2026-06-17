from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any

from flask import Flask, Response, jsonify, request


DATN_DIR = Path(__file__).resolve().parent.parent
if str(DATN_DIR) not in sys.path:
    sys.path.insert(0, str(DATN_DIR))

LOGGER = logging.getLogger("datn.web")

try:
    from src.inference.analyze_review import (
        ABSA_LABELS,
        ASPECT_NAMES,
        RATING_LABELS,
        explain,
        get_device,
        load_absa_stack,
        load_rating_stack,
        normalize_aspect,
        tokenize_absa,
        tokenize_rating,
    )

    INFERENCE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - reported by /api/health.
    INFERENCE_IMPORT_ERROR = exc


VALID_XAI_METHODS = {"attention", "ig", "none"}
MAX_REVIEW_CHARS = 5000


class AnalyzerService:
    """Lazy, reusable wrapper around the DATN rating + ABSA inference stack."""

    def __init__(self, device_arg: str | None = None) -> None:
        self.device_arg = device_arg or os.environ.get("DATN_DEVICE")
        self.device = None
        self.rating_model = None
        self.rating_tokenizer = None
        self.absa_model = None
        self.absa_tokenizer = None
        self.loaded_at: float | None = None
        self._load_lock = Lock()

    @property
    def loaded(self) -> bool:
        return self.loaded_at is not None

    def set_device(self, device_arg: str | None) -> None:
        if self.loaded:
            raise RuntimeError("Model đã được nạp, không thể đổi device trong phiên này.")
        self.device_arg = device_arg

    def load(self) -> None:
        if INFERENCE_IMPORT_ERROR is not None:
            raise RuntimeError(f"Không import được pipeline DATN: {INFERENCE_IMPORT_ERROR}")

        if self.loaded:
            return

        with self._load_lock:
            if self.loaded:
                return

            start = time.perf_counter()
            self.device = get_device(self.device_arg)
            LOGGER.info("Loading DATN models on device=%s", self.device)
            self.rating_model, self.rating_tokenizer = load_rating_stack(self.device)
            self.absa_model, self.absa_tokenizer = load_absa_stack(self.device)
            self.loaded_at = time.time()
            LOGGER.info("DATN models loaded in %.2fs", time.perf_counter() - start)

    def analyze(
        self,
        text: str,
        *,
        xai_method: str = "attention",
        explain_absa: bool = False,
    ) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            raise ValueError("Bạn cần nhập nội dung review.")
        if len(text) > MAX_REVIEW_CHARS:
            raise ValueError(f"Review tối đa {MAX_REVIEW_CHARS} ký tự.")
        if xai_method not in VALID_XAI_METHODS:
            raise ValueError("xai_method không hợp lệ.")

        self.load()
        assert self.device is not None
        assert self.rating_model is not None
        assert self.rating_tokenizer is not None
        assert self.absa_model is not None
        assert self.absa_tokenizer is not None

        started = time.perf_counter()
        rating_encoded = tokenize_rating(self.rating_tokenizer, text, self.device)
        rating_result = explain(
            self.rating_model,
            self.rating_tokenizer,
            rating_encoded,
            RATING_LABELS,
            xai_method,
        )

        aspects: list[dict[str, Any]] = []
        aspect_xai = xai_method if explain_absa else "none"
        for aspect in ASPECT_NAMES:
            encoded = tokenize_absa(self.absa_tokenizer, text, aspect, self.device)
            aspect_result = explain(
                self.absa_model,
                self.absa_tokenizer,
                encoded,
                ABSA_LABELS,
                aspect_xai,
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
            "device": str(self.device),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "rating": rating_result,
            "absa": aspects,
        }


analyzer = AnalyzerService()
app = Flask(__name__)


PAGE_HTML = r"""<!doctype html>
<html lang="vi" class="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DATN Sentiment Engine</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #181b1f;
      --panel-2: #20242a;
      --line: #333941;
      --text: #eef2f5;
      --muted: #a8b0ba;
      --cyan: #54d6c4;
      --green: #8bd46e;
      --amber: #f5c451;
      --red: #ff7777;
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    body {
      min-height: 100vh;
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, textarea, select { font: inherit; }
    .text-xs { font-size: 12px; line-height: 1.35; }
    .text-lg { font-size: 18px; line-height: 1.35; }
    .text-2xl { font-size: 24px; line-height: 1.2; }
    .font-extrabold { font-weight: 800; }
    .font-bold { font-weight: 700; }
    .mb-2 { margin-bottom: 8px; }
    .mb-3 { margin-bottom: 12px; }
    .mt-1 { margin-top: 4px; }
    .text-right { text-align: right; }
    .overflow-x-auto { overflow-x: auto; }
    .muted { color: var(--muted); }
    .mono { font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace; }
    .material-symbols-outlined { font-size: 20px; line-height: 1; vertical-align: middle; }
    .app-shell { display: grid; grid-template-columns: 260px minmax(0, 1fr); min-height: 100vh; }
    .sidebar { border-right: 1px solid var(--line); background: #131619; padding: 28px 20px; }
    .brand { display: flex; align-items: center; gap: 12px; margin-bottom: 32px; }
    .brand-mark {
      width: 38px; height: 38px; display: grid; place-items: center;
      border-radius: 8px; background: var(--cyan); color: #07110f; font-weight: 800;
    }
    .nav-item {
      display: flex; align-items: center; gap: 12px; min-height: 44px;
      color: var(--muted); border-radius: 8px; padding: 0 12px; margin-bottom: 8px;
    }
    .nav-item.active { color: var(--text); background: #20262b; border: 1px solid var(--line); }
    .main { padding: 28px; }
    .topbar { display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; margin-bottom: 24px; }
    .status-pill {
      display: inline-flex; align-items: center; gap: 8px; min-height: 32px;
      border: 1px solid var(--line); border-radius: 999px; padding: 0 12px;
      color: var(--muted); background: #15181b;
    }
    .dot { width: 8px; height: 8px; border-radius: 99px; background: var(--amber); }
    .dot.ok { background: var(--green); }
    .workspace { display: grid; grid-template-columns: minmax(320px, 0.9fr) minmax(420px, 1.4fr); gap: 18px; align-items: start; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .panel-head { display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 18px 20px; border-bottom: 1px solid var(--line); }
    .panel-body { padding: 20px; }
    textarea {
      width: 100%; min-height: 292px; resize: vertical;
      color: var(--text); background: #111417; border: 1px solid var(--line);
      border-radius: 8px; padding: 14px 15px; line-height: 1.55; outline: none;
    }
    textarea:focus, select:focus { border-color: var(--cyan); box-shadow: 0 0 0 3px rgba(84, 214, 196, 0.16); }
    select {
      width: 100%; height: 42px; color: var(--text); background: #111417;
      border: 1px solid var(--line); border-radius: 8px; padding: 0 12px;
    }
    .control-grid { display: grid; grid-template-columns: 1fr 150px; gap: 12px; margin-top: 12px; }
    .check-row { display: flex; align-items: center; gap: 10px; min-height: 42px; color: var(--muted); }
    input[type="checkbox"] { accent-color: var(--cyan); width: 16px; height: 16px; }
    .primary-btn {
      width: 100%; min-height: 46px; margin-top: 16px;
      display: inline-flex; align-items: center; justify-content: center; gap: 10px;
      color: #07110f; background: var(--cyan); border: 0; border-radius: 8px;
      font-weight: 800; cursor: pointer;
    }
    .primary-btn:disabled { opacity: 0.62; cursor: progress; }
    .ghost-btn {
      min-height: 34px; border: 1px solid var(--line); background: #14171a;
      color: var(--muted); border-radius: 8px; padding: 0 10px; cursor: pointer;
    }
    .results-grid { display: grid; grid-template-columns: minmax(180px, 0.8fr) minmax(260px, 1.2fr); gap: 18px; }
    .score-value { font-size: clamp(42px, 7vw, 72px); font-weight: 800; line-height: 0.95; color: var(--text); }
    .score-caption { margin-top: 8px; color: var(--muted); }
    .prob-list { display: grid; gap: 10px; }
    .prob-row { display: grid; grid-template-columns: 54px 1fr 58px; align-items: center; gap: 10px; }
    .bar-track { height: 9px; background: #101316; border-radius: 99px; overflow: hidden; border: 1px solid #242a30; }
    .bar-fill { height: 100%; width: 0%; background: var(--cyan); border-radius: 99px; transition: width .28s ease; }
    .section-title { display: flex; align-items: center; gap: 8px; font-weight: 700; }
    .tokens { display: flex; flex-wrap: wrap; gap: 8px; }
    .token-chip {
      display: inline-flex; align-items: center; gap: 8px; min-height: 32px;
      padding: 0 10px; border-radius: 999px; border: 1px solid var(--line);
      background: #14181b; color: var(--text);
    }
    .token-score { color: var(--cyan); font-size: 12px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 13px 12px; border-bottom: 1px solid var(--line); text-align: left; }
    th { color: var(--muted); font-size: 12px; text-transform: uppercase; font-weight: 700; }
    tr:last-child td { border-bottom: 0; }
    .sentiment { font-weight: 700; }
    .sentiment.positive { color: var(--green); }
    .sentiment.negative { color: var(--red); }
    .sentiment.neutral { color: var(--amber); }
    .stack { display: grid; gap: 18px; }
    .json-box {
      max-height: 300px; overflow: auto; white-space: pre-wrap;
      background: #0e1114; border-top: 1px solid var(--line);
      padding: 14px; color: #c7d1da; font-size: 12px;
    }
    .alert {
      display: none; margin-top: 12px; color: #ffd6d6; background: rgba(255, 119, 119, 0.12);
      border: 1px solid rgba(255, 119, 119, 0.35); padding: 10px 12px; border-radius: 8px;
    }
    .alert.show { display: block; }
    .info {
      margin-top: 12px; color: #d6faf5; background: rgba(84, 214, 196, 0.10);
      border: 1px solid rgba(84, 214, 196, 0.30); padding: 10px 12px; border-radius: 8px;
    }
    .mobile-brand { display: none; }
    @media (max-width: 980px) {
      .app-shell { display: block; }
      .sidebar { display: none; }
      .mobile-brand { display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }
      .main { padding: 18px; }
      .topbar { display: block; }
      .topbar .status-pill { margin-top: 14px; }
      .workspace, .results-grid { grid-template-columns: 1fr; }
      textarea { min-height: 220px; }
    }
    @media (max-width: 560px) {
      .main { padding: 12px; }
      .panel-head, .panel-body { padding: 14px; }
      .control-grid { grid-template-columns: 1fr; }
      .prob-row { grid-template-columns: 48px 1fr 52px; }
      th, td { padding: 11px 8px; }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">AI</div>
        <div>
          <div class="text-lg font-extrabold">DATN Engine</div>
          <div class="mono text-xs muted">PhoBERT + ABSA</div>
        </div>
      </div>
      <nav>
        <div class="nav-item active"><span class="material-symbols-outlined">science</span><span>Phân tích</span></div>
        <div class="nav-item"><span class="material-symbols-outlined">table_chart</span><span>ABSA</span></div>
        <div class="nav-item"><span class="material-symbols-outlined">manage_search</span><span>XAI</span></div>
      </nav>
    </aside>

    <main class="main">
      <div class="mobile-brand">
        <div class="brand-mark">AI</div>
        <div>
          <div class="text-lg font-extrabold">DATN Engine</div>
          <div class="mono text-xs muted">PhoBERT + ABSA</div>
        </div>
      </div>

      <header class="topbar">
        <div>
          <h1 class="text-2xl md:text-3xl font-extrabold">Vietnamese Review Sentiment</h1>
          <p class="mt-1 muted">Rating prediction, aspect sentiment, and token attribution</p>
        </div>
        <div id="healthPill" class="status-pill mono text-xs"><span id="healthDot" class="dot"></span><span id="healthText">Đang kiểm tra</span></div>
      </header>

      <div class="workspace">
        <section class="panel">
          <div class="panel-head">
            <div class="section-title"><span class="material-symbols-outlined">edit_note</span><span>Input Workspace</span></div>
            <button id="sampleBtn" class="ghost-btn mono text-xs" type="button">Mẫu</button>
          </div>
          <div class="panel-body">
            <textarea id="reviewText" placeholder="Nhập review tiếng Việt cần phân tích...">Sản phẩm dùng tốt, giao hàng nhanh, shop tư vấn nhiệt tình</textarea>
            <div class="control-grid">
              <label>
                <div class="mono text-xs muted mb-2">XAI method</div>
                <select id="xaiMethod">
                  <option value="attention">Attention</option>
                  <option value="none">None</option>
                  <option value="ig">Integrated gradients</option>
                </select>
              </label>
              <label>
                <div class="mono text-xs muted mb-2">ABSA XAI</div>
                <div class="check-row"><input id="explainAbsa" type="checkbox"><span>Bật</span></div>
              </label>
            </div>
            <button id="analyzeBtn" class="primary-btn" type="button">
              <span class="material-symbols-outlined">auto_awesome</span>
              <span id="buttonText">Initialize Analysis</span>
            </button>
            <div id="infoBox" class="info mono text-xs">Sẵn sàng. Bấm Initialize Analysis để chạy model DATN.</div>
            <div id="errorBox" class="alert"></div>
          </div>
        </section>

        <div class="stack">
          <section class="panel">
            <div class="panel-head">
              <div class="section-title"><span class="material-symbols-outlined">monitoring</span><span>AI Insights</span></div>
              <div id="latencyText" class="mono text-xs muted">Latency: --</div>
            </div>
            <div class="panel-body">
              <div class="results-grid">
                <div>
                  <div class="mono text-xs muted mb-2">Overall Rating</div>
                  <div id="ratingValue" class="score-value">--</div>
                  <div id="ratingCaption" class="score-caption">Chưa có kết quả</div>
                </div>
                <div>
                  <div class="mono text-xs muted mb-3">Class probabilities</div>
                  <div id="probList" class="prob-list"></div>
                </div>
              </div>
            </div>
          </section>

          <section class="panel">
            <div class="panel-head">
              <div class="section-title"><span class="material-symbols-outlined">psychology</span><span>XAI Highlights</span></div>
              <div id="xaiLabel" class="mono text-xs muted">--</div>
            </div>
            <div class="panel-body">
              <div class="mono text-xs muted mb-2">Bản đồ nhiệt câu (Sentence Heatmap)</div>
              <div id="heatmapText" class="text-lg" style="line-height: 2.2; letter-spacing: 0.5px; background: #111417; border: 1px solid var(--line); border-radius: 8px; padding: 16px; margin-bottom: 20px; font-weight: 500;">
                <span class="muted">Chưa có dữ liệu phân tích</span>
              </div>
              <div class="mono text-xs muted mb-2">Top 10 Từ khóa ảnh hưởng nhất</div>
              <div id="tokenList" class="tokens"></div>
            </div>
          </section>

          <section class="panel">
            <div class="panel-head">
              <div class="section-title"><span class="material-symbols-outlined">splitscreen</span><span>Aspect Sentiment</span></div>
              <div id="deviceText" class="mono text-xs muted">Device: --</div>
            </div>
            <div class="overflow-x-auto">
              <table>
                <thead>
                  <tr>
                    <th>Aspect</th>
                    <th>Polarity</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody id="absaTable"></tbody>
              </table>
            </div>
          </section>

          <section class="panel">
            <div class="panel-head">
              <div class="section-title"><span class="material-symbols-outlined">data_object</span><span>Raw Output</span></div>
            </div>
            <pre id="rawJson" class="json-box">{}</pre>
          </section>
        </div>
      </div>
    </main>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);

    const samples = [
      "Sản phẩm dùng tốt, giao hàng nhanh, shop tư vấn nhiệt tình",
      "Kem dưỡng bị rít, mùi hơi nồng nhưng đóng gói chắc chắn",
      "Giao hàng chậm, hộp bị móp, chất lượng sản phẩm bình thường",
      "Shop phản hồi nhanh, sản phẩm đúng mô tả, sẽ mua lại"
    ];
    let sampleIndex = 0;

    function percent(value) {
      return `${Math.round((Number(value) || 0) * 100)}%`;
    }

    function cleanToken(token) {
      return String(token || "").replaceAll("@@", "").trim() || token;
    }

    function ratingLabel(label) {
      const star = String(label || "").replace("_star", "");
      return /^\d+$/.test(star) ? `${star} sao` : label;
    }

    function sentimentLabel(label) {
      return {
        positive: "Tích cực",
        negative: "Tiêu cực",
        neutral: "Trung tính"
      }[label] || label;
    }

    function showError(message) {
    $("errorBox").textContent = message || "";
    $("errorBox").classList.toggle("show", Boolean(message));
    if (message) {
      $("infoBox").hidden = true;
    }
    }

    function showInfo(message) {
      $("infoBox").textContent = message || "";
      $("infoBox").hidden = !message;
      if (message) {
        $("errorBox").classList.remove("show");
      }
    }

    function setBusy(isBusy) {
      $("analyzeBtn").disabled = isBusy;
      $("buttonText").textContent = isBusy ? "Đang phân tích..." : "Initialize Analysis";
    }

    function renderProbabilities(probs) {
      const list = $("probList");
      list.innerHTML = "";
      Object.entries(probs || {}).forEach(([label, value]) => {
        const row = document.createElement("div");
        row.className = "prob-row mono text-xs";

        const name = document.createElement("div");
        name.textContent = label.replace("_star", "★");

        const track = document.createElement("div");
        track.className = "bar-track";
        const fill = document.createElement("div");
        fill.className = "bar-fill";
        fill.style.width = percent(value);
        track.appendChild(fill);

        const score = document.createElement("div");
        score.className = "text-right muted";
        score.textContent = percent(value);

        row.append(name, track, score);
        list.appendChild(row);
      });
    }

    function renderTokens(tokens) {
      const list = $("tokenList");
      list.innerHTML = "";
      if (!tokens || tokens.length === 0) {
        const empty = document.createElement("div");
        empty.className = "muted";
        empty.textContent = "Không có token attribution cho chế độ hiện tại.";
        list.appendChild(empty);
        return;
      }

      tokens.forEach((item) => {
        const chip = document.createElement("span");
        chip.className = "token-chip mono";

        const token = document.createElement("span");
        token.textContent = cleanToken(item.token);

        const score = document.createElement("span");
        score.className = "token-score";
        score.textContent = Number(item.score || 0).toFixed(2);

        chip.append(token, score);
        list.appendChild(chip);
      });
    }

    function renderHeatmap(allTokens, method) {
      const container = $("heatmapText");
      container.innerHTML = "";
      if (!allTokens || allTokens.length === 0 || method === "none") {
        container.innerHTML = `<span class="muted">Không có bản đồ nhiệt ở chế độ này</span>`;
        return;
      }

      allTokens.forEach((item) => {
        const span = document.createElement("span");
        span.textContent = cleanToken(item.token) + " ";
        span.style.padding = "2px 5px";
        span.style.borderRadius = "4px";
        span.style.margin = "0 1px";
        span.style.display = "inline-block";
        span.style.transition = "background-color 0.2s ease";
        span.title = `Từ: "${cleanToken(item.token)}"\nĐiểm đóng góp: ${item.score.toFixed(4)}`;

        const score = Number(item.score || 0);
        if (method === "attention") {
          span.style.backgroundColor = `rgba(84, 214, 196, ${score * 0.45})`;
          if (score > 0.6) {
            span.style.fontWeight = "bold";
            span.style.border = "1px solid rgba(84, 214, 196, 0.4)";
          }
        } else if (method === "integrated_gradients" || method === "ig") {
          if (score > 0) {
            span.style.backgroundColor = `rgba(139, 212, 110, ${score * 0.45})`;
            if (score > 0.6) {
              span.style.fontWeight = "bold";
              span.style.border = "1px solid rgba(139, 212, 110, 0.4)";
            }
          } else if (score < 0) {
            span.style.backgroundColor = `rgba(255, 119, 119, ${Math.abs(score) * 0.45})`;
            if (score < -0.6) {
              span.style.fontWeight = "bold";
              span.style.border = "1px solid rgba(255, 119, 119, 0.4)";
            }
          }
        }
        container.appendChild(span);
      });
    }

    function renderAbsa(aspects) {
      const body = $("absaTable");
      body.innerHTML = "";
      (aspects || []).forEach((item) => {
        const row = document.createElement("tr");
        const prediction = item.prediction || {};
        const label = prediction.label || "";

        const aspectCell = document.createElement("td");
        aspectCell.textContent = item.aspect_text || item.aspect || "";

        const labelCell = document.createElement("td");
        const labelSpan = document.createElement("span");
        labelSpan.className = `sentiment ${label}`;
        labelSpan.textContent = sentimentLabel(label);
        labelCell.appendChild(labelSpan);

        const confidenceCell = document.createElement("td");
        confidenceCell.className = "mono";
        confidenceCell.textContent = percent(prediction.confidence);

        row.append(aspectCell, labelCell, confidenceCell);
        body.appendChild(row);
      });
    }

    function renderResult(data) {
      const prediction = data.rating?.prediction || {};
      $("ratingValue").textContent = ratingLabel(prediction.label || "--");
      $("ratingCaption").textContent = `Confidence ${percent(prediction.confidence)}`;
      $("latencyText").textContent = `Latency: ${data.elapsed_ms ?? "--"} ms`;
      $("deviceText").textContent = `Device: ${data.device || "--"}`;
      $("xaiLabel").textContent = data.rating?.method || "--";
      renderProbabilities(prediction.probs || {});
      renderTokens(data.rating?.top_tokens || []);
      renderHeatmap(data.rating?.all_tokens || [], data.rating?.method);
      renderAbsa(data.absa || []);
      $("rawJson").textContent = JSON.stringify(data, null, 2);
    }

    async function analyze() {
      const text = $("reviewText").value.trim();
      showError("");
      showInfo("Đang gửi review vào backend DATN. Lần đầu có thể mất vài giây để nạp model.");
      if (!text) {
        showError("Bạn cần nhập nội dung review.");
        return;
      }

      setBusy(true);
      try {
        const response = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text,
            xai_method: $("xaiMethod").value,
            explain_absa: $("explainAbsa").checked
          })
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || data.error || "Không phân tích được review.");
        }
        renderResult(data);
        showInfo(`Xong. Model chạy trên ${data.device || "unknown"}, latency ${data.elapsed_ms ?? "--"} ms.`);
      } catch (error) {
        showError(`${error.message || String(error)}. Nếu bạn đang mở file trực tiếp, hãy chạy: python datn/src/web.py rồi mở URL được in trong terminal.`);
      } finally {
        setBusy(false);
      }
    }

    async function loadHealth() {
      try {
        const response = await fetch("/api/health");
        const data = await response.json();
        $("healthDot").classList.toggle("ok", Boolean(data.ok));
        $("healthText").textContent = data.loaded ? "Model đã nạp" : (data.ok ? "Sẵn sàng" : "Lỗi cấu hình");
        if (!data.ok) {
          showError(data.error || "Backend chưa sẵn sàng.");
        }
      } catch {
        $("healthText").textContent = "Không kết nối";
        showError("Không kết nối được backend. Chạy: python datn/src/web.py rồi mở URL được in trong terminal.");
      }
    }

    $("analyzeBtn").addEventListener("click", analyze);
    $("sampleBtn").addEventListener("click", () => {
      sampleIndex = (sampleIndex + 1) % samples.length;
      $("reviewText").value = samples[sampleIndex];
    });
    $("reviewText").addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        analyze();
      }
    });

    loadHealth();
  </script>
</body>
</html>
"""


@app.get("/")
def index() -> Response:
    return Response(PAGE_HTML, mimetype="text/html; charset=utf-8")


@app.get("/api/health")
def health():
    ok = INFERENCE_IMPORT_ERROR is None
    return jsonify(
        {
            "ok": ok,
            "loaded": analyzer.loaded,
            "device": str(analyzer.device) if analyzer.device is not None else analyzer.device_arg,
            "datn_dir": str(DATN_DIR),
            "error": str(INFERENCE_IMPORT_ERROR) if INFERENCE_IMPORT_ERROR else None,
        }
    ), (200 if ok else 500)


@app.post("/api/analyze")
def analyze_api():
    payload = request.get_json(silent=True) or {}
    try:
        result = analyzer.analyze(
            str(payload.get("text", "")),
            xai_method=str(payload.get("xai_method", "attention")),
            explain_absa=bool(payload.get("explain_absa", False)),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        LOGGER.exception("Analysis failed")
        return jsonify({"error": "Không phân tích được review.", "detail": str(exc)}), 500
    return jsonify(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DATN sentiment web UI")
    parser.add_argument("--host", default=os.environ.get("DATN_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", default=int(os.environ.get("DATN_WEB_PORT", "5002")), type=int)
    parser.add_argument("--device", default=os.environ.get("DATN_DEVICE"))
    parser.add_argument("--preload", action="store_true", help="Load models before starting the server.")
    return parser.parse_args()


def find_available_port(host: str, preferred_port: int, attempts: int = 25) -> int:
    for port in range(preferred_port, preferred_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(
        f"Không tìm được port trống từ {preferred_port} đến {preferred_port + attempts - 1}."
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    analyzer.set_device(args.device)
    if args.preload:
        analyzer.load()
    port = find_available_port(args.host, args.port)
    if port != args.port:
        LOGGER.warning("Port %s đang bận, tự chuyển sang port %s.", args.port, port)
    print(f"\nDATN web UI: http://{args.host}:{port}\n", flush=True)
    app.run(host=args.host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
