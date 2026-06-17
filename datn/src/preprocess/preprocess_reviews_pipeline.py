#!/usr/bin/env python3
"""
Safe preprocessing pipeline for Shopee review CSV.

Pipeline order:
1. Light text normalization
2. Hard no-value / off-topic filtering
   2.1 Hard noise rules
   2.2 Contextual off-topic / rating-mismatch rules
3. Salvage explicit "nhan xu" comments by trimming boilerplate fragments
   3.1 Re-check contextual off-topic / rating-mismatch after cleanup
4. Optional exact-duplicate cap to reduce repetitive boilerplate

Notes:
- drops non-core metadata columns: time, like_count, is_repeated_purchase, model_name
- does not overwrite the source file
- writes an audit CSV for dropped rows
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "output" / "shopee_reviews_all.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "shopee_proccessed.csv"
DEFAULT_DROPPED_OUTPUT = PROJECT_ROOT / "output" / "shopee_proccessed_dropped.csv"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "output" / "shopee_proccessed_summary.json"
DEFAULT_EXCLUDED_ITEMS_CONFIG = PROJECT_ROOT / "configs" / "excluded_item_ids.json"
EXCLUDED_OUTPUT_COLUMNS = {
    "time",
    "like_count",
    "is_repeated_purchase",
    "model_name",
}

SPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
URL_RE = re.compile(r"(https?://|www\.|fb\.com|facebook|zalo|tiktok|instagram)", re.IGNORECASE)
HANGUL_RE = re.compile(r"[\uac00-\ud7af]{8,}")
ASCII_ESSAY_RE = re.compile(
    r"(the diagram illustrates|a neuron consists|transmits information|in conclusion|firstly[, ]|secondly[, ])",
    re.IGNORECASE,
)
SPLIT_RE = re.compile(r"[\r\n]+|[.!?;]+|,+")
TRAILING_DANGLING_PAREN_RE = re.compile(r"\(\s*[\wÀ-ỹ]{0,5}$")
NHAN_XU_RE = re.compile(r"\bnhan+\s*xu+\b")

VISUAL_HINTS = [
    "hinh anh",
    "video",
    "anh",
    "minh hoa",
    "minh hoạ",
    "mang tinh chat",
    "mang tc",
    "de nhan xu",
    "chi de nhan xu",
    "chi mang tinh chat",
]

EXPLICIT_SPAM_KEYWORDS = [
    "van mau",
    "nhan xu",
]

CLIPBOARD_KEYWORDS = [
    "gboard",
    "bang nho tam",
    "clipboard",
    "ghim doan do",
]

ESIM_KEYWORDS = [
    "esim",
    "nano sim",
    "sim ky thuat so",
    "doi esim",
]

OFFTOPIC_EVENT_KEYWORDS = [
    "workshop",
    "dia diem to chuc",
    "ngay to chuc",
    "villa coffee house",
]

POLICE_ALERT_KEYWORDS = [
    "co quan cong an",
    "cuc canh sat hinh su",
    "truc ban hinh su",
]

TELECOM_STRONG_KEYWORDS = [
    "[tb]",
    "[qc]",
    "quy khach",
    "thue bao",
    "my viettel",
    "tv360",
    "viettel",
    "mobifone",
    "vinaphone",
]

TELECOM_SOFT_KEYWORDS = [
    "nap tien",
    "khuyen mai",
    "goi cuoc",
    "tai khoan",
    "truy cap",
    "chi tiet goi",
]

HARD_REJECT_HINTS = [
    "viettel",
    "mobifone",
    "vinaphone",
    "my viettel",
    "gboard",
    "clipboard",
    "esim",
    "workshop",
    "villa coffee house",
    "co quan cong an",
    "cuc canh sat",
    "the diagram illustrates",
    "a neuron consists",
    "pubg",
]

BEAUTY_CONTEXT_HINTS = [
    "duong am",
    "cap am",
    "duong da",
    "da mun",
    "da dau",
    "da nhay cam",
    "tay trang",
    "sua rua mat",
    "toner",
    "serum",
    "chong nang",
    "mat na",
    "mask",
    "nang tone",
    "kich ung",
    "lo chan long",
    "dau goi",
    "duong toc",
    "makeup",
    "kcn",
    "lam sach",
    "tay te bao chet",
]

POSITIVE_RATING_HINTS = [
    "rat dang mua",
    "dang mua",
    "nen mua",
    "rat ok",
    "sieu thich",
    "qua ung",
    "hai long",
    "chan ai",
    "qua oke",
    "qua on",
    "rất đáng mua",
    "rất ok",
    "siêu thích",
    "hài lòng",
    "chân ái",
]

NEGATIVE_RATING_HINTS = [
    "khong ung",
    "that vong",
    "cau tha",
    "von cuc",
    "loang lo",
    "lua dao",
    "khon nan",
    "buc minh",
    "ghet",
    "khong co",
    "ko co",
    "giao sai",
    "thieu hang",
    "xuong tone",
    "kich ung",
    "noi mun",
    "rat da",
    "nang tone ao",
]

OFFTOPIC_GROUP_RULES = [
    {
        "reason": "cake_or_bakery_offtopic",
        "cues": [
            "banh sinh nhat",
            "bang gia",
            "size banh",
            "banh tron",
            "banh chu nhat",
            "duong kinh mat banh",
            "banh dot chay",
        ],
        "min_hits": 2,
        "max_beauty_hits": 1,
    },
    {
        "reason": "idol_or_fandom_news_offtopic",
        "cues": [
            "idol",
            "fan girl",
            "hot search",
            "kpop",
            "cbiz",
            "concert",
            "ticketbox",
            "trailer",
            "newjeans",
            "seventeen",
            "bts",
            "westlife",
            "taylor swift",
            "king & prince",
            "weibovietnam",
            "pann",
            "show dien",
            "line up",
            "fanpage",
        ],
        "min_hits": 4,
        "max_beauty_hits": 2,
    },
    {
        "reason": "movie_or_show_recap_offtopic",
        "cues": [
            "bo phim",
            "tap ",
            "nhan vat",
            "dien vien",
            "khan gia",
            "vai dien",
            "motip",
            "drama yeu duong",
            "dat dien",
            "xem phim",
        ],
        "min_hits": 3,
        "max_beauty_hits": 1,
    },
    {
        "reason": "restaurant_or_food_promo_offtopic",
        "cues": [
            "pho bo",
            "bat pho",
            "menu cac mon",
            "lau bo",
            "dong gia 30k",
            "freeship tu 2 suat",
            "quan an",
        ],
        "min_hits": 2,
        "max_beauty_hits": 1,
    },
    {
        "reason": "apparel_or_accessory_offtopic",
        "cues": [
            "chat vai",
            "vua van",
            "buoc len",
            "ao dep",
            "vay dep",
            "giay khi con uot",
            "fullbox",
            "authentic",
            "kinh local",
        ],
        "min_hits": 2,
        "max_beauty_hits": 1,
    },
    {
        "reason": "shoe_or_warranty_instruction_offtopic",
        "cues": [
            "khong giat tay",
            "khong ngam nuoc qua lau",
            "bao hanh va cskh",
            "doi tra san pham",
            "sai mau ma",
            "giay khi con uot",
            "ve sinh thuong xuyen",
        ],
        "min_hits": 2,
        "max_beauty_hits": 1,
    },
    {
        "reason": "horse_or_sports_text_offtopic",
        "cues": [
            "giu thang bang",
            "lam quen voi ngua",
            "huan luyen vien",
            "dieu khien bang tay chan",
            "bo mon nay",
        ],
        "min_hits": 2,
        "max_beauty_hits": 1,
    },
    {
        "reason": "admin_or_land_text_offtopic",
        "cues": [
            "quan ly hanh chinh",
            "dat dai",
            "huong dan cong tac",
        ],
        "min_hits": 2,
        "max_beauty_hits": 1,
    },
    {
        "reason": "nail_or_acetone_text_offtopic",
        "cues": [
            "mong gia",
            "acetone",
            "rach nen mong",
            "ton thuong be mat mong",
        ],
        "min_hits": 2,
        "max_beauty_hits": 1,
    },
]

LOW_VALUE_HINTS = [
    "chua dung",
    "chua biet",
    "danh gia sau",
    "se review sau",
    "se danh gia sau",
    "mua ho",
    "mua de nhan xu",
    "danh gia de nhan xu",
    "nhan xu hoi",
    "nhan xu thoi",
    "nhan xu thui",
    "chi de nhan xu",
    "de lay xu",
    "bo qua dum",
    "thong cam",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run safe preprocessing pipeline on Shopee reviews.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Final cleaned CSV path.")
    parser.add_argument("--dropped-output", default=str(DEFAULT_DROPPED_OUTPUT), help="Audit CSV of dropped rows.")
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_OUTPUT), help="Summary JSON path.")
    parser.add_argument(
        "--excluded-items-config",
        type=Path,
        default=DEFAULT_EXCLUDED_ITEMS_CONFIG,
        help="Optional JSON config listing item_id values to exclude from the dataset.",
    )
    parser.add_argument(
        "--duplicate-cap",
        type=int,
        default=5,
        help="Keep at most this many rows for each exact normalized comment after cleanup. Use 0 to disable.",
    )
    parser.add_argument("--salvage-min-words", type=int, default=6, help="Minimum word count after salvage cleanup.")
    parser.add_argument("--salvage-min-chars", type=int, default=20, help="Minimum char count after salvage cleanup.")
    return parser.parse_args()


def normalize_spaces(text: str) -> str:
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2028", " ").replace("\u2029", " ")
    text = CONTROL_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text.replace("\r", " ").replace("\n", " ").replace("\t", " "))
    return text.strip()


def fold_text(text: str) -> str:
    normalized = normalize_spaces(text).lower()
    decomposed = unicodedata.normalize("NFKD", normalized)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.replace("đ", "d")


def normalize_item_id(value: str) -> str:
    raw = normalize_spaces(value)
    if not raw:
        return ""
    try:
        return str(int(float(raw)))
    except ValueError:
        return raw


def load_excluded_item_ids(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        raw_items = payload.get("excluded_items", payload)
    else:
        raw_items = payload

    if isinstance(raw_items, dict):
        return {
            normalize_item_id(item_id): str(label).strip()
            for item_id, label in raw_items.items()
            if normalize_item_id(item_id)
        }

    if isinstance(raw_items, list):
        return {
            normalize_item_id(item_id): "excluded_item_id_config"
            for item_id in raw_items
            if normalize_item_id(item_id)
        }

    raise ValueError(f"Unsupported excluded item config format: {path}")


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def count_hits(text: str, phrases: list[str]) -> int:
    return sum(1 for phrase in phrases if phrase in text)


def classify_hard_drop(comment: str) -> str | None:
    normalized = normalize_spaces(comment)
    folded = fold_text(comment)

    if not normalized:
        return "empty_comment"

    strong_hits = count_hits(folded, TELECOM_STRONG_KEYWORDS)
    soft_hits = count_hits(folded, TELECOM_SOFT_KEYWORDS)

    if contains_any(folded, CLIPBOARD_KEYWORDS):
        return "clipboard_noise"
    if contains_any(folded, ESIM_KEYWORDS):
        return "esim_or_sim_ad"
    if contains_any(folded, POLICE_ALERT_KEYWORDS):
        return "police_or_alert_broadcast"
    if contains_any(folded, OFFTOPIC_EVENT_KEYWORDS):
        return "offtopic_event_or_invite"
    if strong_hits >= 2 or (strong_hits >= 1 and soft_hits >= 1):
        return "telecom_system_message"
    if URL_RE.search(normalized) and (strong_hits >= 1 or contains_any(folded, OFFTOPIC_EVENT_KEYWORDS)):
        return "url_with_system_or_event_text"
    if HANGUL_RE.search(normalized):
        return "foreign_offtopic_text"
    if ASCII_ESSAY_RE.search(folded):
        return "english_offtopic_text"

    return None


def classify_contextual_drop(product_name: str, comment: str, star: int) -> str | None:
    folded_comment = fold_text(comment)
    folded_product = fold_text(product_name)
    beauty_hits = count_hits(folded_comment, BEAUTY_CONTEXT_HINTS)
    product_beauty_hits = count_hits(folded_product, BEAUTY_CONTEXT_HINTS)

    if len(comment) >= 80 and (URL_RE.search(comment) or "#" in comment or "@" in comment) and beauty_hits == 0:
        return "social_or_news_post_offtopic"

    for rule in OFFTOPIC_GROUP_RULES:
        group_hits = count_hits(folded_comment, rule["cues"])
        if group_hits < rule["min_hits"]:
            continue

        max_beauty_hits = rule.get("max_beauty_hits", 0)
        long_comment_override = len(comment) >= 120 and group_hits >= rule["min_hits"] + 1
        if beauty_hits <= max_beauty_hits or long_comment_override:
            if product_beauty_hits >= 1 or rule["reason"] != "nail_or_acetone_text_offtopic":
                return rule["reason"]

    positive_hits = count_hits(folded_comment, POSITIVE_RATING_HINTS)
    negative_hits = count_hits(folded_comment, NEGATIVE_RATING_HINTS)

    if star in {1, 2} and positive_hits >= 3 and negative_hits <= 1:
        return "rating_comment_mismatch_too_positive_for_low_star"

    if star >= 4 and negative_hits >= 3 and positive_hits <= 1:
        return "rating_comment_mismatch_too_negative_for_high_star"

    return None


def is_explicit_spam_candidate(comment: str) -> bool:
    folded = fold_text(comment)
    return contains_any(folded, EXPLICIT_SPAM_KEYWORDS)


def should_drop_fragment(fragment_folded: str) -> bool:
    if not fragment_folded:
        return True

    alnum_count = sum(ch.isalnum() for ch in fragment_folded)
    word_count = len(fragment_folded.split())
    if alnum_count < 5:
        return True
    if word_count <= 1 and alnum_count < 8:
        return True

    if contains_any(fragment_folded, HARD_REJECT_HINTS) or URL_RE.search(fragment_folded):
        return True

    if NHAN_XU_RE.search(fragment_folded):
        return True

    if contains_any(fragment_folded, LOW_VALUE_HINTS):
        return True

    visual_hits = sum(1 for hint in VISUAL_HINTS if hint in fragment_folded)
    if visual_hits >= 2:
        return True

    return False


def cleanup_explicit_spam_comment(text: str) -> str:
    cleaned_source = normalize_spaces(text)
    pieces = [piece.strip(" -()[]{}\"'") for piece in SPLIT_RE.split(cleaned_source)]

    kept: list[str] = []
    seen_folded: set[str] = set()

    for piece in pieces:
        piece = normalize_spaces(piece)
        if not piece:
            continue

        folded = fold_text(piece)
        if should_drop_fragment(folded):
            continue

        if folded in seen_folded:
            continue

        seen_folded.add(folded)
        kept.append(piece)

    cleaned = normalize_spaces(". ".join(kept))
    cleaned = TRAILING_DANGLING_PAREN_RE.sub("", cleaned).rstrip(" ,;-")
    return normalize_spaces(cleaned)


def reject_after_salvage(cleaned_comment: str, min_words: int, min_chars: int) -> str | None:
    folded = fold_text(cleaned_comment)

    if not folded:
        return "empty_after_cleanup"

    if contains_any(folded, HARD_REJECT_HINTS) or URL_RE.search(folded):
        return "hard_offtopic_after_cleanup"

    if NHAN_XU_RE.search(folded) or contains_any(folded, LOW_VALUE_HINTS):
        return "still_contains_nhan_xu_or_low_value"

    if len(cleaned_comment) < min_chars or len(cleaned_comment.split()) < min_words:
        return "too_short_after_cleanup"

    return None


def format_int(value: int) -> str:
    return f"{value:,}"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    dropped_output_path = Path(args.dropped_output)
    summary_output_path = Path(args.summary_output)
    excluded_items_config_path = args.excluded_items_config
    excluded_item_ids = load_excluded_item_ids(excluded_items_config_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dropped_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    csv.field_size_limit(1024 * 1024 * 64)

    total_rows = 0
    kept_rows = 0
    dropped_rows = 0

    kept_by_star: Counter[int] = Counter()
    dropped_by_star: Counter[int] = Counter()

    stage_counts: Counter[str] = Counter()
    drop_reason_counts: Counter[str] = Counter()
    duplicate_seen: Counter[str] = Counter()

    with input_path.open("r", encoding="utf-8-sig", newline="") as src, output_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as dst, dropped_output_path.open("w", encoding="utf-8-sig", newline="") as dropped_dst:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise RuntimeError("Input CSV has no header.")

        output_fieldnames = [name for name in fieldnames if name not in EXCLUDED_OUTPUT_COLUMNS]

        writer = csv.DictWriter(dst, fieldnames=output_fieldnames, extrasaction="ignore")
        writer.writeheader()

        dropped_writer = csv.DictWriter(
            dropped_dst,
            fieldnames=output_fieldnames + ["drop_stage", "drop_reason"],
            extrasaction="ignore",
        )
        dropped_writer.writeheader()

        for row in reader:
            total_rows += 1
            raw_comment = row.get("comment", "")
            normalized_comment = normalize_spaces(raw_comment)
            row["comment"] = normalized_comment
            product_name = row.get("product_name", "")
            item_id = normalize_item_id(row.get("item_id", ""))

            try:
                star = int(float((row.get("rating_star") or "0").strip()))
            except Exception:
                star = 0

            if item_id and item_id in excluded_item_ids:
                dropped_rows += 1
                dropped_by_star[star] += 1
                stage_counts["excluded_product"] += 1
                drop_reason_counts["excluded_item_id_config"] += 1
                drop_row = dict(row)
                drop_row["drop_stage"] = "excluded_product"
                drop_row["drop_reason"] = excluded_item_ids[item_id] or "excluded_item_id_config"
                dropped_writer.writerow(drop_row)
                continue

            hard_reason = classify_hard_drop(normalized_comment)
            if hard_reason is not None:
                dropped_rows += 1
                dropped_by_star[star] += 1
                stage_counts["hard_drop"] += 1
                drop_reason_counts[hard_reason] += 1
                drop_row = dict(row)
                drop_row["drop_stage"] = "hard_drop"
                drop_row["drop_reason"] = hard_reason
                dropped_writer.writerow(drop_row)
                continue

            context_reason = classify_contextual_drop(product_name, normalized_comment, star)
            if context_reason is not None:
                dropped_rows += 1
                dropped_by_star[star] += 1
                stage_counts["context_drop"] += 1
                drop_reason_counts[context_reason] += 1
                drop_row = dict(row)
                drop_row["drop_stage"] = "context_drop"
                drop_row["drop_reason"] = context_reason
                dropped_writer.writerow(drop_row)
                continue

            if is_explicit_spam_candidate(normalized_comment):
                salvaged_comment = cleanup_explicit_spam_comment(normalized_comment)
                salvage_reject = reject_after_salvage(
                    salvaged_comment,
                    min_words=args.salvage_min_words,
                    min_chars=args.salvage_min_chars,
                )
                if salvage_reject is not None:
                    dropped_rows += 1
                    dropped_by_star[star] += 1
                    stage_counts["salvage_drop"] += 1
                    drop_reason_counts[salvage_reject] += 1
                    drop_row = dict(row)
                    drop_row["drop_stage"] = "salvage_drop"
                    drop_row["drop_reason"] = salvage_reject
                    dropped_writer.writerow(drop_row)
                    continue

                salvage_context_reason = classify_contextual_drop(product_name, salvaged_comment, star)
                if salvage_context_reason is not None:
                    dropped_rows += 1
                    dropped_by_star[star] += 1
                    stage_counts["context_drop_after_salvage"] += 1
                    drop_reason_counts[salvage_context_reason] += 1
                    drop_row = dict(row)
                    drop_row["drop_stage"] = "context_drop_after_salvage"
                    drop_row["drop_reason"] = salvage_context_reason
                    dropped_writer.writerow(drop_row)
                    continue

                row["comment"] = salvaged_comment
                stage_counts["salvaged_explicit_spam"] += 1
            else:
                stage_counts["kept_direct"] += 1

            if args.duplicate_cap > 0:
                comment_key = text_hash(fold_text(row.get("comment", "")))
                duplicate_seen[comment_key] += 1
                if duplicate_seen[comment_key] > args.duplicate_cap:
                    dropped_rows += 1
                    dropped_by_star[star] += 1
                    stage_counts["duplicate_cap_drop"] += 1
                    drop_reason_counts["duplicate_cap_exceeded"] += 1
                    drop_row = dict(row)
                    drop_row["drop_stage"] = "duplicate_cap"
                    drop_row["drop_reason"] = "duplicate_cap_exceeded"
                    dropped_writer.writerow(drop_row)
                    continue

            writer.writerow(row)
            kept_rows += 1
            kept_by_star[star] += 1

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "dropped_output": str(dropped_output_path),
        "excluded_items_config": str(excluded_items_config_path),
        "excluded_item_count": len(excluded_item_ids),
        "excluded_output_columns": sorted(EXCLUDED_OUTPUT_COLUMNS),
        "rows_read": total_rows,
        "rows_kept": kept_rows,
        "rows_dropped": dropped_rows,
        "duplicate_cap": args.duplicate_cap,
        "salvage_min_words": args.salvage_min_words,
        "salvage_min_chars": args.salvage_min_chars,
        "stage_counts": dict(stage_counts),
        "drop_reason_counts": dict(drop_reason_counts),
        "kept_by_star": {str(k): kept_by_star[k] for k in sorted(kept_by_star)},
        "dropped_by_star": {str(k): dropped_by_star[k] for k in sorted(dropped_by_star)},
    }

    with summary_output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Input:   {input_path}")
    print(f"Output:  {output_path}")
    print(f"Dropped: {dropped_output_path}")
    print(f"Summary: {summary_output_path}")
    print(f"Rows read:    {format_int(total_rows)}")
    print(f"Rows kept:    {format_int(kept_rows)}")
    print(f"Rows dropped: {format_int(dropped_rows)}")
    print("")
    print("Stage counts:")
    for key, value in stage_counts.most_common():
        print(f"  {key}: {format_int(value)}")
    print("")
    print("Drop reasons:")
    for key, value in drop_reason_counts.most_common():
        print(f"  {key}: {format_int(value)}")
    print("")
    print("Kept by star:")
    for star in range(1, 6):
        print(f"  {star} star: {format_int(kept_by_star[star])}")


if __name__ == "__main__":
    main()
