#!/usr/bin/env python3
"""
Shopee Crawler using Playwright — tự động mở browser, lấy cookie, crawl reviews.
Không cần cookie thủ công!

Cách dùng:
  python shopee_crawler_playwright.py --links links_shopee.txt
  python shopee_crawler_playwright.py --links links_shopee.txt --deep
  python shopee_crawler_playwright.py --links links_shopee.txt --reset-progress
"""

import argparse
import csv
import glob
import json
import logging
import os
import re
import sys
import time
import random
from datetime import datetime
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Cần cài playwright: pip install playwright")
    print("Sau đó: python -m playwright install chromium")
    sys.exit(1)

import pandas as pd

OUTPUT_DIR = "output"
PROGRESS_FILE = "crawl_progress.json"
RATINGS_PER_PAGE = 50
MAX_RETRIES = 5
BACKOFF_BASE = 5
API_TIMEOUT_MS = 15000
LOG_DIR = "logs"
CHECKPOINT_FLUSH_PAGES = 3

# ─── Logging ───────────────────────────────────────────────────────────

def setup_logging():
    """Tạo logger ghi cả console lẫn file."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = f"crawl_pw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(LOG_DIR, log_filename)

    logger = logging.getLogger("shopee_pw")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger, log_path


log = logging.getLogger("shopee_pw")

EXPECTED_CSV_COLUMNS = [
    "product_name",
    "item_id",
    "rating_star",
    "comment",
    "product_quality",
    "seller_service",
    "delivery_service",
    "template_tags",
    "review_id",
]

LEGACY_SHIFTED_HEADER_MAP = {
    "product_namstar": "product_name",
    "comment": "item_id",
    "timee": "rating_star",
    "item_id": "comment",
}


def parse_shopee_url(url):
    url = url.strip()
    if not url:
        return None, None
    match = re.search(r'i\.(\d+)\.(\d+)', url)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r'/product/(\d+)/(\d+)', url)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def parse_rating(r, product_name, shop_id, item_id):
    """Parse 1 review — 13 trường cho phân tích ABSA."""
    comment = (r.get("comment") or "").strip()
    # Xóa ký tự Line Separator (U+2028) và Paragraph Separator (U+2029)
    # để tránh lỗi "Unusual Line Terminators" trong editor/CSV
    comment = comment.replace("\u2028", " ").replace("\u2029", " ")
    rating_star = r.get("rating_star", 0)
    product_items = r.get("product_items", [])

    # Lấy tên thật từ product_items hoặc original_item_info
    real_name = ""
    if product_items:
        real_name = product_items[0].get("name", "")
    if not real_name:
        real_name = (r.get("original_item_info") or {}).get("name", "")
    if real_name:
        product_name = real_name

    detailed_rating = r.get("detailed_rating", {}) or {}
    product_quality = detailed_rating.get("product_quality", 0)
    seller_service = detailed_rating.get("seller_service", 0)
    delivery_service = detailed_rating.get("delivery_service", 0)

    template_tags = r.get("template_tags", []) or []

    review_id = str(r.get("cmtid", "")).strip()

    return {
        "product_name": product_name,
        "item_id": item_id,
        "rating_star": rating_star,
        "comment": comment,
        "product_quality": product_quality,
        "seller_service": seller_service,
        "delivery_service": delivery_service,
        "template_tags": " | ".join(template_tags),
        "review_id": review_id,
    }


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def normalize_review_id(value):
    if pd.isna(value):
        return ""
    review_id = str(value).strip()
    if review_id.endswith(".0") and review_id[:-2].isdigit():
        review_id = review_id[:-2]
    return review_id


def normalize_loaded_row_schema(row_dict):
    if not isinstance(row_dict, dict):
        return row_dict

    if "product_name" in row_dict and "rating_star" in row_dict:
        return row_dict

    if all(key in row_dict for key in LEGACY_SHIFTED_HEADER_MAP):
        normalized = {}
        for key, value in row_dict.items():
            normalized[LEGACY_SHIFTED_HEADER_MAP.get(key, key)] = value
        return normalized

    return row_dict


def checkpoint_path_for(output_file):
    filename = Path(output_file).name
    stem = Path(filename).stem
    return os.path.join(OUTPUT_DIR, f"{stem}.checkpoint.jsonl")


def session_checkpoint_path_for(output_file, session_id):
    filename = Path(output_file).name
    stem = Path(filename).stem
    return os.path.join(OUTPUT_DIR, f"{stem}.checkpoint.{session_id}.jsonl")


def checkpoint_paths_for(output_file):
    filename = Path(output_file).name
    stem = Path(filename).stem
    pattern = os.path.join(OUTPUT_DIR, f"{stem}.checkpoint*.jsonl")
    paths = sorted(set(glob.glob(pattern)))
    legacy_path = checkpoint_path_for(output_file)
    if os.path.exists(legacy_path) and legacy_path not in paths:
        paths.insert(0, legacy_path)
    return paths


def clear_checkpoint_files_for(output_file):
    removed = 0
    for path in checkpoint_paths_for(output_file):
        try:
            os.remove(path)
            removed += 1
        except FileNotFoundError:
            continue
    return removed


def lock_path_for(output_file):
    filename = Path(output_file).name
    stem = Path(filename).stem
    return os.path.join(OUTPUT_DIR, f"{stem}.write.lock")


def pid_exists(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    except Exception:
        return True
    return True


class OutputWriteLock:
    def __init__(self, output_file, session_id):
        self.output_file = output_file
        self.session_id = session_id
        self.path = lock_path_for(output_file)
        self.acquired = False

    def acquire(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                payload = {
                    "pid": os.getpid(),
                    "session_id": self.session_id,
                    "created_at": datetime.now().isoformat(),
                    "output_file": self.output_file,
                }
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                self.acquired = True
                return
            except FileExistsError:
                try:
                    with open(self.path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                except Exception:
                    raise RuntimeError(
                        f"Phát hiện lock file {self.path}. "
                        "Hãy kiểm tra có phiên crawl khác đang chạy không."
                    )

                lock_pid = payload.get("pid")
                if not pid_exists(lock_pid):
                    try:
                        os.remove(self.path)
                        log.warning(
                            f"[!] Phát hiện lock cũ từ PID {lock_pid} — đã dọn để chạy lại an toàn"
                        )
                        continue
                    except Exception:
                        raise RuntimeError(
                            f"Lock file {self.path} có vẻ đã stale nhưng không xóa được."
                        )

                raise RuntimeError(
                    f"Đang có phiên khác giữ quyền ghi {self.output_file} "
                    f"(PID {lock_pid}, session {payload.get('session_id')}). "
                    "Dừng phiên kia trước khi chạy tiếp để tránh rollback dữ liệu."
                )

    def release(self):
        if not self.acquired:
            return
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        finally:
            self.acquired = False


def load_reviews_from_csv(filepath):
    if not os.path.exists(filepath):
        return []

    reviews = []
    df_old = pd.read_csv(filepath, encoding="utf-8-sig")
    for _, row in df_old.iterrows():
        row_dict = normalize_loaded_row_schema(row.to_dict())
        rid = normalize_review_id(row_dict.get("review_id", ""))
        if not rid:
            continue
        row_dict["review_id"] = rid
        reviews.append(row_dict)
    return reviews


def load_reviews_from_checkpoint(filepath):
    if not os.path.exists(filepath):
        return []

    reviews = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                log.warning(f"[!] Bo qua checkpoint dong {line_no}: {e}")
                continue
            rid = normalize_review_id(row.get("review_id", ""))
            if not rid:
                continue
            row["review_id"] = rid
            reviews.append(row)
    return reviews


def hydrate_reviews(rows, all_reviews, seen_ids):
    added = 0
    for row in rows:
        rid = normalize_review_id(row.get("review_id", ""))
        if not rid or rid in seen_ids:
            continue
        row["review_id"] = rid
        seen_ids.add(rid)
        all_reviews.append(row)
        added += 1
    return added


class CheckpointManager:
    def __init__(self, output_file, session_id):
        self.output_file = output_file
        self.session_id = session_id
        self.path = session_checkpoint_path_for(output_file, session_id)
        self.pending = []

    def add_review(self, review):
        self.pending.append(review)

    def flush(self, reason=""):
        if not self.pending:
            return 0

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            for review in self.pending:
                f.write(json.dumps(review, ensure_ascii=False))
                f.write("\n")

        flushed = len(self.pending)
        self.pending = []
        if reason:
            msg = f"[~] Checkpoint +{flushed} reviews ({reason})"
            if reason.startswith("periodic"):
                log.debug(msg)
            else:
                log.info(msg)
        return flushed

    def sync_progress(self, progress, progress_key, offset, done, count, reason=""):
        self.flush(reason=reason)
        progress[progress_key] = {"offset": offset, "done": done, "count": count}
        save_progress(progress)

    def clear_file(self):
        if os.path.exists(self.path):
            os.remove(self.path)
            log.info(f"[~] Da xoa checkpoint sidecar: {self.path}")


def is_page_closed(page):
    try:
        return page.is_closed()
    except Exception:
        return True


def is_page_closed_error(err):
    err_str = str(err).lower()
    return (
        "target page, context or browser has been closed" in err_str
        or "page has been closed" in err_str
    )


def is_navigation_or_context_error(err):
    err_str = str(err).lower()
    return (
        err_str == "timeout"
        or "execution context was destroyed" in err_str
        or "navigation" in err_str
        or "context" in err_str
    )


def refresh_session(page):
    """Navigate lại shopee.vn để refresh session/cookie khi bị lỗi."""
    if is_page_closed(page):
        log.warning("    [~] Refresh session skipped: page/context đã đóng")
        return False
    try:
        log.info("    [~] Refreshing session...")
        page.goto("https://shopee.vn/", wait_until="domcontentloaded", timeout=20000)
        time.sleep(3)
        log.info("    [~] Session refreshed OK")
        return True
    except Exception as e:
        log.warning(f"    [~] Refresh session failed: {e}")
        return False


def api_request(page, url, params=None):
    """Gọi API bằng fetch() trong browser context."""
    if is_page_closed(page):
        return {"error": "PAGE_CLOSED"}

    query = "&".join(f"{k}={v}" for k, v in (params or {}).items())
    full_url = f"{url}?{query}" if query else url

    try:
        page.set_default_timeout(30000)
        response = page.evaluate("""
            async ([url, timeoutMs]) => {
                const controller = new AbortController();
                const timer = setTimeout(() => controller.abort(), timeoutMs);
                try {
                    const resp = await fetch(url, {
                        credentials: 'include',
                        signal: controller.signal,
                        headers: {
                            'Accept': 'application/json',
                            'X-Shopee-Language': 'vi',
                            'X-Requested-With': 'XMLHttpRequest',
                        }
                    });
                    clearTimeout(timer);
                    const status = resp.status;
                    if (status === 403) {
                        return { error: 403, status: 403 };
                    }
                    const data = await resp.json();
                    return { data: data, status: status };
                } catch (e) {
                    clearTimeout(timer);
                    return { error: e.name === 'AbortError' ? 'TIMEOUT' : e.message };
                }
            }
        """, [full_url, API_TIMEOUT_MS])
        return response
    except Exception as e:
        err_str = str(e)
        if "timeout" in err_str.lower():
            return {"error": "TIMEOUT"}
        return {"error": err_str}


def get_product_info(page, shop_id, item_id):
    """Lấy thông tin sản phẩm."""
    for attempt in range(3):
        url = f"https://shopee.vn/api/v2/item/get?shopid={shop_id}&itemid={item_id}"
        result = api_request(page, url)

        if "error" in result:
            log.warning(f"    [!] Error getting info (lần {attempt+1}): {result['error']}")
            if result["error"] == "PAGE_CLOSED":
                return {}
            if attempt < 2:
                time.sleep(random.uniform(2, 5))
            continue

        data = result.get("data", {})
        if isinstance(data, dict):
            item = data.get("data") or data.get("item")
            if item:
                return {
                    "name": item.get("name", ""),
                    "rating_star": item.get("item_rating", {}).get("rating_star", 0),
                    "rating_count": item.get("item_rating", {}).get("rating_count", [0]*6),
                    "sold": item.get("historical_sold", 0) or item.get("sold", 0),
                }
            elif data.get("error"):
                log.warning(f"    [!] API error (lần {attempt+1}): {data.get('error')}")
        if attempt < 2:
            time.sleep(random.uniform(2, 5))
    return {}


def fetch_ratings_browser(page, shop_id, item_id, rating_type=0, offset=0,
                          limit=RATINGS_PER_PAGE, filter_val=0):
    """Gọi API lấy reviews với retry + backoff."""
    if is_page_closed(page):
        return {"error": "PAGE_CLOSED"}

    for attempt in range(MAX_RETRIES):
        url = (
            f"https://shopee.vn/api/v2/item/get_ratings"
            f"?itemid={item_id}&shopid={shop_id}"
            f"&type={rating_type}&offset={offset}&limit={limit}"
            f"&filter={filter_val}&flag=1"
        )
        result = api_request(page, url)

        if "error" in result:
            err = result["error"]
            if err == 403:
                return {"error": 403}
            if err == "PAGE_CLOSED" or is_page_closed_error(err):
                return {"error": "PAGE_CLOSED"}
            if err == "TIMEOUT" or is_navigation_or_context_error(err):
                log.warning(f"    [!] {err} — refresh session (lần {attempt+1}/{MAX_RETRIES})")
                refresh_session(page)
                time.sleep(random.uniform(2, 4))
                continue
            wait = BACKOFF_BASE * (attempt + 1) + random.uniform(1, 3)
            log.warning(f"    [!] Error: {err} — retry sau {wait:.0f}s (lần {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue

        data = result.get("data", {})
        status = result.get("status", 200)

        if status == 429:
            wait = BACKOFF_BASE * (2 ** attempt) + random.uniform(1, 3)
            log.warning(f"    [!] Rate-limited — chờ {wait:.0f}s (lần {attempt+1})")
            time.sleep(wait)
            continue

        if isinstance(data, dict):
            api_error = data.get("error")
            if api_error and api_error != 0:
                if api_error == 90309999:
                    return {"error": 90309999}
                wait = BACKOFF_BASE * (attempt + 1)
                log.warning(f"    [!] API error: {api_error} — retry sau {wait}s (lần {attempt+1})")
                time.sleep(wait)
                continue
            return data.get("data", {})

        return {"error": "unexpected_response"}
    log.warning(f"    [!] Hết {MAX_RETRIES} lần retry!")
    return {"error": "max_retries"}


# ─── Filter labels ─────────────────────────────────────────────────────
FILTER_LABELS = {0: "all", 1: "có text"}


def stale_limit_for_combo(rating_type, filter_val):
    return 3


def build_combo_specs(rating_types, deep):
    combos = []
    for rt in rating_types:
        combos.append((rt, 0, stale_limit_for_combo(rt, 0)))
    return combos


def _crawl_one_combo(page, shop_id, item_id, product_name, rt, filter_val,
                     progress, all_reviews, seen_ids, delay_range,
                     max_per_combo, include_no_text, checkpoint_manager,
                     stale_limit):
    """Crawl 1 tổ hợp (type, filter). Trả (ok, new_count)."""
    rt_label = f"{rt}★" if rt > 0 else "all"
    f_label = FILTER_LABELS.get(filter_val, f"f{filter_val}")
    combo_label = f"{rt_label}/f={f_label}"
    progress_key = f"{shop_id}_{item_id}_r{rt}_f{filter_val}"

    start_offset = progress.get(progress_key, {}).get("offset", 0)
    done = progress.get(progress_key, {}).get("done", False)

    if done:
        count_done = progress.get(progress_key, {}).get("count", "?")
        log.info(f"    [{combo_label}] Đã xong trước đó ({count_done} reviews) — bỏ qua")
        return True, 0

    log.info(f"    [{combo_label}] Bắt đầu từ offset={start_offset}")
    offset = start_offset
    page_count = 0
    combo_new = 0
    consecutive_empty = 0
    stale_pages = 0

    def sync_combo_progress(done, reason):
        checkpoint_manager.sync_progress(
            progress,
            progress_key,
            offset,
            done,
            combo_new,
            reason=reason,
        )

    while True:
        time.sleep(random.uniform(*delay_range))

        data = fetch_ratings_browser(page, shop_id, item_id, rating_type=rt,
                         offset=offset, filter_val=filter_val)

        if isinstance(data, dict) and "error" in data:
            err = data["error"]
            if err in (403, 90309999):
                log.error(f"    [{combo_label}] BLOCKED ({err}) — Dừng combo, chuyển SP tiếp.")
                sync_combo_progress(False, f"blocked:{combo_label}")
                return False, combo_new
            if err == "max_retries":
                consecutive_empty += 1
                if consecutive_empty >= 5:
                    log.warning(f"    [{combo_label}] Quá nhiều lỗi — dừng combo này")
                    sync_combo_progress(False, f"retry-stop:{combo_label}")
                    break
                offset += RATINGS_PER_PAGE
                continue
            log.warning(f"    [{combo_label}] Lỗi: {err}")
            consecutive_empty += 1
            if consecutive_empty >= 5:
                sync_combo_progress(False, f"error-stop:{combo_label}")
                break
            offset += RATINGS_PER_PAGE
            continue

        ratings_list = data.get("ratings", [])

        if not ratings_list:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                log.info(f"    [{combo_label}] Hết reviews (offset={offset}, mới={combo_new})")
                sync_combo_progress(True, f"empty:{combo_label}")
                break
            offset += RATINGS_PER_PAGE
            continue

        consecutive_empty = 0
        new_with_text = 0

        for r in ratings_list:
            cmtid = str(r.get("cmtid", "")).strip()
            if not cmtid:
                continue
            if cmtid in seen_ids:
                continue
            seen_ids.add(cmtid)

            try:
                parsed = parse_rating(r, product_name, shop_id, item_id)
            except Exception as e:
                log.warning(f"    [!] parse_rating lỗi (cmtid={cmtid}): {e} — bỏ qua")
                continue
            
            # Chỉ count nếu có text
            if parsed["comment"]:
                all_reviews.append(parsed)
                checkpoint_manager.add_review(parsed)
                combo_new += 1
                new_with_text += 1

        page_count += 1
        offset += len(ratings_list)

        # Nếu trang này có reviews nhưng tất cả NO TEXT → count stale
        if len(ratings_list) > 0 and new_with_text == 0:
            stale_pages += 1
            if stale_pages >= stale_limit:
                log.info(f"    [{combo_label}] {stale_limit} trang liên tiếp toàn NO TEXT → dừng")
                sync_combo_progress(True, f"no-text:{combo_label}")
                break
        else:
            stale_pages = 0

        if max_per_combo and combo_new >= max_per_combo:
            log.info(f"    [{combo_label}] Đạt giới hạn {max_per_combo}")
            sync_combo_progress(True, f"limit:{combo_label}")
            break

        if page_count % CHECKPOINT_FLUSH_PAGES == 0:
            sync_combo_progress(False, f"periodic:{combo_label}")
            log.info(f"    [{combo_label}] Page {page_count}, offset={offset}, mới={combo_new}")

        has_more = data.get("has_more", False)
        if not has_more:
            log.info(f"    [{combo_label}] ✓ Xong — {combo_new} reviews mới")
            sync_combo_progress(True, f"done:{combo_label}")
            break

    return True, combo_new


def crawl_product(page, shop_id, item_id, url, rating_types, progress, all_reviews,
                  delay_range, max_per_type, seen_ids=None, include_no_text=False,
                  deep=False, checkpoint_manager=None):
    """Crawl reviews cho 1 sản phẩm."""
    if seen_ids is None:
        seen_ids = set()

    log.info(f"\n{'='*60}")
    log.info(f"[*] Crawling: {url}")
    log.info(f"    shop_id={shop_id}, item_id={item_id}")

    info = get_product_info(page, shop_id, item_id)
    product_name = info.get("name", f"item_{item_id}")
    if info:
        log.info(f"    Tên: {product_name}")
        rc = info.get("rating_count", [])
        if rc and len(rc) > 5:
            total_reviews = rc[0]
            log.info(f"    Rating: {info.get('rating_star', '?')}★ "
                     f"(Tổng:{total_reviews} | "
                     f"5★:{rc[5]} 4★:{rc[4]} 3★:{rc[3]} 2★:{rc[2]} 1★:{rc[1]})")
    time.sleep(random.uniform(*delay_range))

    # Xây danh sách tổ hợp (type, filter) cần crawl
    combos = build_combo_specs(rating_types, deep)

    log.info(f"    Chiến lược: {len(combos)} tổ hợp (type, filter)")
    log.info(
        "    Chiến lược hiện tại — chỉ chạy 1★..5★/f=all, dừng sau 3 trang NO TEXT liên tiếp"
        if deep else
        "    Cơ bản — filter=0 only"
    )

    total_new = 0
    before_count = len(seen_ids)
    blocked_any = False

    for i, (rt, fv, stale_limit) in enumerate(combos, 1):
        log.info(f"  ── Combo {i}/{len(combos)} ──")
        ok, new_count = _crawl_one_combo(
            page, shop_id, item_id, product_name, rt, fv,
            progress, all_reviews, seen_ids, delay_range,
            max_per_type, include_no_text, checkpoint_manager,
            stale_limit,
        )
        total_new += new_count
        if not ok:
            blocked_any = True
            log.warning(f"    [!] Combo bị block — skip SP này, chuyển SP tiếp")
            break

    actual_unique = len(seen_ids) - before_count
    log.info(f"    → Tổng reviews mới cho SP này: {total_new} (unique IDs: {actual_unique})")
    return {
        "blocked": blocked_any,
        "new_count": actual_unique,
    }


def save_to_csv(reviews, output_file):
    if not reviews:
        log.warning("[!] Không có review nào để lưu")
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, output_file)

    df = pd.DataFrame(reviews)

    if "review_id" in df.columns:
        before = len(df)
        df.drop_duplicates(subset=["review_id"], keep="first", inplace=True)
        dupes = before - len(df)
        if dupes:
            log.info(f"[i] Loại {dupes} review trùng lặp")

    ordered_cols = [col for col in EXPECTED_CSV_COLUMNS if col in df.columns]
    extra_cols = [col for col in df.columns if col not in EXPECTED_CSV_COLUMNS]
    if ordered_cols:
        df = df[ordered_cols + extra_cols]

    if "review_id" in df.columns:
        df["_review_id_sort"] = pd.to_numeric(df["review_id"], errors="coerce")
        df.sort_values(by=["product_name", "rating_star", "_review_id_sort"],
                       ascending=[True, True, False], inplace=True)
        df.drop(columns=["_review_id_sort"], inplace=True)
    else:
        df.sort_values(by=["product_name", "rating_star"],
                       ascending=[True, True], inplace=True)
    tmp_path = f"{filepath}.tmp.{os.getpid()}"
    try:
        df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
        os.replace(tmp_path, filepath)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    log.info(f"[+] Đã lưu {len(df)} reviews → {filepath}")

    log.info("\n── Thống kê theo sản phẩm & số sao ──")
    summary = df.groupby(["product_name", "rating_star"]).size().reset_index(name="count")
    log.info(summary.to_string(index=False))

    log.info("\n── Tổng hợp ──")
    for name, group in df.groupby("product_name"):
        total = len(group)
        with_text = group["comment"].apply(lambda x: bool(x.strip()) if isinstance(x, str) else False).sum()
        log.info(f"  {name}: {total} reviews ({with_text} có text)")

    return filepath


def persist_reviews_snapshot(reviews, output_file, checkpoint_manager, reason=""):
    if checkpoint_manager is not None:
        try:
            checkpoint_manager.flush(reason=f"snapshot:{reason}")
        except Exception as e:
            log.error(f"[!] Flush checkpoint thất bại trước khi lưu CSV: {e}")

    filepath = save_to_csv(reviews, output_file)

    if filepath:
        try:
            removed = clear_checkpoint_files_for(output_file)
            if removed:
                log.info(f"[~] Đã dọn {removed} checkpoint sidecar sau khi merge vào CSV")
        except Exception as e:
            log.warning(f"[!] Không dọn được checkpoint sidecar: {e}")

    return filepath


def main():
    parser = argparse.ArgumentParser(description="Shopee Crawler with Playwright")
    parser.add_argument("--links", default="output/links_retry_uncrawled.txt")
    parser.add_argument("--output", default="")
    parser.add_argument("--delay", nargs=2, type=float, default=[10.0, 16.0])
    parser.add_argument("--max-per-type", type=int, default=0)
    parser.add_argument("--reset-progress", action="store_true")
    parser.add_argument("--headless", action="store_true",
                        help="Chạy ẩn browser (mặc định: hiện browser)")
    parser.add_argument("--deep", action="store_true",
                        help="Deep crawl: dùng combo all/text để vớt review text")
    parser.add_argument("--max-products", type=int, default=0,
                        help="Giới hạn số sản phẩm mỗi lần chạy (0 = không giới hạn)")
    parser.add_argument("--shuffle-products", action="store_true",
                        help="Random thứ tự sản phẩm trước khi crawl")
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="Seed random để tái lập thứ tự sản phẩm")
    parser.add_argument("--stop-on-blocked-streak", type=int, default=0,
                        help="Tự dừng nếu bị BLOCKED liên tiếp N sản phẩm (0 = không dừng)")
    parser.add_argument("--product-delay", nargs=2, type=float, default=[3.0, 6.0],
                        help="Delay giữa các sản phẩm (vd: --product-delay 8 15)")
    args = parser.parse_args()

    global log
    log, log_path = setup_logging()
    start_time = datetime.now()
    session_id = f"{start_time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    write_lock = None

    log.info("=" * 60)
    log.info("  SHOPEE CRAWLER — Playwright Edition")
    log.info(f"  Log: {log_path}")
    log.info(f"  Bắt đầu: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Session: {session_id}")
    log.info("=" * 60)

    try:
        links_file = Path(args.links)
        if not links_file.exists():
            log.error(f"[!] Không tìm thấy {args.links}")
            sys.exit(1)

        urls = [line.strip() for line in links_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")]
        products = []
        for url in urls:
            sid, iid = parse_shopee_url(url)
            if sid and iid:
                products.append((sid, iid, url))
            else:
                log.warning(f"    [!] URL không hợp lệ: {url}")

        if not products:
            log.error("[!] Không có URL hợp lệ!")
            sys.exit(1)

        if args.shuffle_products:
            if args.shuffle_seed is not None:
                random.Random(args.shuffle_seed).shuffle(products)
                log.info(f"[+] Random thứ tự sản phẩm với seed={args.shuffle_seed}")
            else:
                random.shuffle(products)
                log.info("[+] Random thứ tự sản phẩm")

        if args.max_products and args.max_products > 0:
            products = products[:args.max_products]
            log.info(f"[+] Chạy giới hạn {len(products)} sản phẩm")

        out_filename = args.output or "shopee_reviews_all.csv"
        checkpoint_manager = CheckpointManager(out_filename, session_id)
        write_lock = OutputWriteLock(out_filename, session_id)
        write_lock.acquire()

        log.info(f"[+] {len(products)} sản phẩm")
        log.info(f"[+] Delay: {args.delay[0]}-{args.delay[1]}s")
        log.info(f"[+] Product delay: {args.product_delay[0]}-{args.product_delay[1]}s")
        log.info(f"[+] Checkpoint sidecar: {checkpoint_manager.path}")
        log.info(f"[+] Write lock: {write_lock.path}")

        if args.reset_progress and os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
            log.info("[+] Đã xóa progress cũ")
        if args.reset_progress:
            existing_ckpts = checkpoint_paths_for(out_filename)
            if existing_ckpts:
                log.info(
                    f"[i] Giữ nguyên {len(existing_ckpts)} checkpoint sidecar cũ để tránh mất dữ liệu chưa merge"
                )
        progress = load_progress()

        log.info("\n[+] Đang mở browser...")
        with sync_playwright() as p:
            # Thử dùng Chrome thật (ít bị detect hơn Chromium của Playwright)
            try:
                browser = p.chromium.launch(
                    channel="chrome",
                    headless=args.headless,
                    args=["--disable-blink-features=AutomationControlled"]
                )
                log.info("[+] Dùng Chrome thật")
            except Exception:
                # Fallback: dùng Chromium bundled
                browser = p.chromium.launch(
                    headless=args.headless,
                    args=["--disable-blink-features=AutomationControlled"]
                )
                log.info("[+] Dùng Chromium (Chrome không tìm thấy)")
            context = browser.new_context(
                viewport={"width": 1366, "height": 768},
                locale="vi-VN",
            )
            page = context.new_page()

            log.info("[+] Đang vào shopee.vn...")
            try:
                page.goto("https://shopee.vn/", wait_until="networkidle", timeout=30000)
                time.sleep(3)
                log.info("[+] Đã vào Shopee thành công!")
            except Exception as e:
                log.warning(f"[!] Lỗi vào Shopee: {e}")
                log.warning("[!] Thử tiếp...")

            # Chờ người dùng đăng nhập
            print("\n" + "=" * 60)
            print("  Browser đã mở. Hãy ĐĂNG NHẬP Shopee nếu cần.")
            print("  Sau khi đăng nhập xong, quay lại đây nhấn ENTER.")
            print("=" * 60)
            input("  >> Nhấn ENTER để bắt đầu crawl... ")
            log.info("[+] Bắt đầu crawl!")
            time.sleep(2)

            all_reviews = []
            seen_ids = set()
            delay_range = tuple(args.delay)

            # Load data cũ nếu file đã tồn tại (resume session)
            existing_path = os.path.join(OUTPUT_DIR, out_filename)
            if os.path.exists(existing_path):
                try:
                    loaded_old = hydrate_reviews(
                        load_reviews_from_csv(existing_path),
                        all_reviews,
                        seen_ids,
                    )
                    log.info(f"[+] Load data cũ: {loaded_old} reviews từ {out_filename}")
                except Exception as e:
                    log.warning(f"[!] Không load được data cũ: {e}")

            checkpoint_paths = checkpoint_paths_for(out_filename)
            if checkpoint_paths:
                loaded_ckpt_total = 0
                for ckpt_path in checkpoint_paths:
                    try:
                        loaded_ckpt = hydrate_reviews(
                            load_reviews_from_checkpoint(ckpt_path),
                            all_reviews,
                            seen_ids,
                        )
                        loaded_ckpt_total += loaded_ckpt
                        log.info(f"[+] Resume từ checkpoint sidecar {Path(ckpt_path).name}: {loaded_ckpt} reviews")
                    except Exception as e:
                        log.warning(f"[!] Không load được checkpoint sidecar {ckpt_path}: {e}")
                log.info(f"[+] Tổng reviews nạp từ checkpoint: {loaded_ckpt_total}")

            def _emergency_save(reason=""):
                if all_reviews:
                    try:
                        persist_reviews_snapshot(all_reviews, out_filename, checkpoint_manager, reason)
                        log.info(f"[!] ĐÃ LƯU KHẨN CẤP {len(all_reviews)} reviews ({reason})")
                    except Exception as save_err:
                        log.error(f"[!] Lưu khẩn cấp thất bại: {save_err}")
                        try:
                            fallback = os.path.join(OUTPUT_DIR, f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                            os.makedirs(OUTPUT_DIR, exist_ok=True)
                            with open(fallback, "w", encoding="utf-8") as f:
                                json.dump(all_reviews, f, ensure_ascii=False, indent=2)
                            log.info(f"[!] Đã lưu fallback JSON → {fallback}")
                        except Exception:
                            pass

            try:
                blocked_streak = 0
                for i, (shop_id, item_id, url) in enumerate(products, 1):
                    log.info(f"\n[{i}/{len(products)}]")
                    result = crawl_product(
                        page, shop_id, item_id, url, [1, 2, 3, 4, 5],
                        progress, all_reviews, delay_range, args.max_per_type,
                        seen_ids=seen_ids,
                        deep=args.deep,
                        checkpoint_manager=checkpoint_manager,
                    )

                    if result.get("blocked") and result.get("new_count", 0) == 0:
                        blocked_streak += 1
                    else:
                        blocked_streak = 0

                    if args.stop_on_blocked_streak and blocked_streak >= args.stop_on_blocked_streak:
                        log.warning(f"[!] BLOCKED liên tiếp {blocked_streak} sản phẩm — dừng sớm để an toàn")
                        break

                    # Auto-save an toàn: chỉ flush checkpoint, không rewrite CSV giữa chừng
                    if all_reviews:
                        flushed = checkpoint_manager.flush(reason=f"end-product:{i}")
                        log.info(
                            f"    [✓] Checkpoint sau SP {i}: +{flushed} mới flush, tổng in-memory {len(all_reviews)} reviews"
                        )

                    if args.product_delay[1] > 0:
                        wait_between = random.uniform(args.product_delay[0], args.product_delay[1])
                        log.info(f"    [~] Nghỉ giữa sản phẩm: {wait_between:.1f}s")
                        time.sleep(wait_between)

            except KeyboardInterrupt:
                log.warning("\n[!] Ctrl+C — đang lưu dữ liệu...")
                _emergency_save("Ctrl+C")

            except Exception as e:
                log.exception(f"[!] Lỗi: {e}")
                _emergency_save(f"Exception: {e}")

            finally:
                try:
                    browser.close()
                except Exception:
                    pass

        try:
            if all_reviews:
                persist_reviews_snapshot(all_reviews, out_filename, checkpoint_manager, "final")
            else:
                log.warning("\n[!] Không crawl được review nào.")
        finally:
            if write_lock is not None:
                write_lock.release()

    except KeyboardInterrupt:
        log.warning("\n[!] Ctrl+C — thoát.")
    except Exception as e:
        log.exception(f"[!] LỖI: {e}")
    finally:
        if write_lock is not None:
            write_lock.release()

    elapsed = datetime.now() - start_time
    log.info(f"\n[i] Thời gian crawl: {elapsed}")
    log.info(f"{'='*60}")
    log.info(f"  HOÀN TẤT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
