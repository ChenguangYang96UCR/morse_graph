#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import threading
import time
from pathlib import Path

import openreview
from tqdm import tqdm

DEFAULT_API_BASEURL = "https://api2.openreview.net"
THREAD_LOCAL = threading.local()
RATE_LIMIT_LOCK = threading.Lock()
LAST_CLIENT_REQUEST_TS = 0.0

TARGET_FIELDS = ("soundness", "presentation", "contribution")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch OpenReview review-dimension metadata for ICLR papers."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Input JSON file containing paper records.",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Output JSON file. Default: <input_stem>_with_review_dimensions.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N papers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_file if it already exists.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Base sleep in seconds before each paper request. Default: 0",
    )
    parser.add_argument(
        "--min_interval",
        type=float,
        default=0.0,
        help="Minimum seconds between note-read requests. Default: 0",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Number of attempts per paper before giving up. Default: 4",
    )
    parser.add_argument(
        "--user_name",
        default=None,
        help="OpenReview username. Defaults to OPENREVIEW_USERNAME if unset.",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="OpenReview password. Defaults to OPENREVIEW_PASSWORD if unset.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="OpenReview access token. Defaults to OPENREVIEW_TOKEN if unset.",
    )
    parser.add_argument(
        "--api_baseurl",
        default=DEFAULT_API_BASEURL,
        help=f"OpenReview API base URL. Default: {DEFAULT_API_BASEURL}",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_with_review_dimensions.json")


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def get_openreview_client(args: argparse.Namespace) -> openreview.api.OpenReviewClient:
    client = getattr(THREAD_LOCAL, "openreview_client", None)
    if client is not None:
        return client

    client = openreview.api.OpenReviewClient(
        baseurl=args.api_baseurl,
        username=args.user_name or os.environ.get("OPENREVIEW_USERNAME"),
        password=args.password or os.environ.get("OPENREVIEW_PASSWORD"),
        token=args.token or os.environ.get("OPENREVIEW_TOKEN"),
    )
    THREAD_LOCAL.openreview_client = client
    return client


def wait_for_client_rate_limit(min_interval: float) -> None:
    global LAST_CLIENT_REQUEST_TS
    if min_interval <= 0:
        return

    with RATE_LIMIT_LOCK:
        now = time.time()
        wait_seconds = LAST_CLIENT_REQUEST_TS + min_interval - now
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        LAST_CLIENT_REQUEST_TS = time.time()


def parse_openreview_exception_payload(exc: Exception) -> dict | None:
    if not exc.args:
        return None
    payload = exc.args[0]
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = ast.literal_eval(payload)
        except Exception:  # noqa: BLE001
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_retry_delay_seconds(exc: Exception) -> float | None:
    text = str(exc)
    match = re.search(r"try again in (\d+) seconds", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.0

    payload = parse_openreview_exception_payload(exc)
    if payload:
        message = str(payload.get("message", ""))
        match = re.search(r"try again in (\d+) seconds", message, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)) + 1.0
    return None


def unwrap_content_value(value):
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", key.lower())


def extract_numeric_score(value) -> float | None:
    value = unwrap_content_value(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match:
            return float(match.group(0))
    return None


def extract_dimension_scores(content: dict) -> dict[str, float]:
    normalized = {normalize_key(key): value for key, value in (content or {}).items()}
    scores = {}
    for field in TARGET_FIELDS:
        value = normalized.get(normalize_key(field))
        score = extract_numeric_score(value)
        if score is not None:
            scores[field] = score
    return scores


def note_looks_like_review(note) -> bool:
    invitation = getattr(note, "invitation", "") or ""
    invitation_lc = invitation.lower()
    if "review" in invitation_lc and "meta" not in invitation_lc:
        return True
    content = getattr(note, "content", {}) or {}
    scores = extract_dimension_scores(content)
    return any(field in scores for field in TARGET_FIELDS)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def build_dimension_payload(notes: list) -> dict:
    values = {field: [] for field in TARGET_FIELDS}
    matched_review_ids = []

    for note in notes:
        if not note_looks_like_review(note):
            continue
        scores = extract_dimension_scores(getattr(note, "content", {}) or {})
        if not scores:
            continue
        matched_review_ids.append(getattr(note, "id", None))
        for field in TARGET_FIELDS:
            if field in scores:
                values[field].append(scores[field])

    soundness_avg = mean_or_none(values["soundness"])
    presentation_avg = mean_or_none(values["presentation"])
    contribution_avg = mean_or_none(values["contribution"])

    coherence = None
    if soundness_avg is not None and presentation_avg is not None:
        coherence = (soundness_avg + presentation_avg) / 2.0

    return {
        "soundness": values["soundness"],
        "presentation": values["presentation"],
        "contribution": values["contribution"],
        "soundness_avg": soundness_avg,
        "presentation_avg": presentation_avg,
        "contribution_avg": contribution_avg,
        "coherence": coherence,
        "novelty": contribution_avg,
        "review_dimension_count": len(matched_review_ids),
        "review_dimension_note_ids": matched_review_ids,
    }


def fetch_forum_notes(forum_id: str, args: argparse.Namespace) -> list:
    wait_for_client_rate_limit(args.min_interval)
    client = get_openreview_client(args)
    return client.get_all_notes(forum=forum_id)


def process_record(record: dict, args: argparse.Namespace) -> dict:
    paper_id = str(record.get("id", "")).strip()
    if not paper_id:
        enriched = dict(record)
        enriched["review_dimensions_error"] = "missing_paper_id"
        return enriched

    if args.sleep > 0:
        time.sleep(args.sleep)

    last_error = "unknown error"
    for attempt in range(1, args.retries + 1):
        try:
            notes = fetch_forum_notes(paper_id, args)
            enriched = dict(record)
            enriched.update(build_dimension_payload(notes))
            return enriched
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc} (attempt {attempt}/{args.retries})"
            if attempt < args.retries:
                time.sleep(parse_retry_delay_seconds(exc) or min(2.0, 0.5 * attempt))

    enriched = dict(record)
    enriched.update(
        {
            "soundness": [],
            "presentation": [],
            "contribution": [],
            "soundness_avg": None,
            "presentation_avg": None,
            "contribution_avg": None,
            "coherence": None,
            "novelty": None,
            "review_dimension_count": 0,
            "review_dimension_note_ids": [],
            "review_dimensions_error": last_error,
        }
    )
    return enriched


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file) if args.output_file else default_output_path(input_path)

    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} already exists. Pass --overwrite to replace it.")

    records = load_records(input_path)
    if args.limit is not None:
        records = records[: args.limit]

    enriched_records = []
    for record in tqdm(records, desc="Fetching review dimensions"):
        enriched_records.append(process_record(record, args))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(enriched_records, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "input_file": str(input_path),
                "output_file": str(output_path),
                "processed": len(enriched_records),
                "api_baseurl": args.api_baseurl,
                "min_interval": args.min_interval,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
