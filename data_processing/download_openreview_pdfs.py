#!/usr/bin/env python3
import argparse
import ast
import concurrent.futures
import json
import os
import random
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download paper PDFs using the official OpenReview API client."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Input JSON file containing a list of paper records.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store downloaded PDFs. Default: <input_stem>_pdfs",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Base sleep in seconds before each task. Default: 0.2",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.4,
        help="Additional random sleep in seconds per task. Default: 0.4",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of attempts per paper before giving up. Default: 3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N papers, useful for testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent download workers. Default: 1",
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
    parser.add_argument(
        "--min_interval",
        type=float,
        default=21.0,
        help="Minimum seconds between authenticated PDF requests. Default: 21",
    )
    return parser.parse_args()


def default_output_dir(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_pdfs")


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def get_paper_id(record: dict) -> str:
    for key in ("paper_id", "id"):
        value = str(record.get(key, "")).strip()
        if value:
            return value
    return ""


def safe_filename(text: str, max_len: int = 160) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r'[<>:"/\\\\|?*]', "_", text)
    text = text.rstrip(". ")
    if not text:
        text = "untitled"
    if len(text) > max_len:
        text = text[:max_len].rstrip()
    return text


def build_output_path(output_dir: Path, record: dict) -> Path:
    paper_id = get_paper_id(record)
    title = str(record.get("title", "")).strip()
    stem = safe_filename(f"{paper_id} - {title}" if title else paper_id)
    return output_dir / f"{stem}.pdf"


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


def credentials_present(args: argparse.Namespace) -> bool:
    return bool(
        args.token
        or os.environ.get("OPENREVIEW_TOKEN")
        or args.user_name
        or os.environ.get("OPENREVIEW_USERNAME")
        or args.password
        or os.environ.get("OPENREVIEW_PASSWORD")
    )


def download_pdf_with_client(paper_id: str, destination: Path, args: argparse.Namespace) -> tuple[bool, str]:
    wait_for_client_rate_limit(args.min_interval)
    client = get_openreview_client(args)
    pdf_bytes = client.get_pdf(id=paper_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as f:
        f.write(pdf_bytes)
    return True, f"downloaded via openreview-py from {args.api_baseurl}/pdf?id={paper_id}"


def download_pdf(paper_id: str, destination: Path, args: argparse.Namespace) -> tuple[bool, str]:
    last_error = "unknown error"
    for attempt in range(1, args.retries + 1):
        try:
            return download_pdf_with_client(paper_id, destination, args)
        except Exception as exc:  # noqa: BLE001
            last_error = f"openreview-py -> {type(exc).__name__}: {exc} on attempt {attempt}/{args.retries}"
            if attempt < args.retries:
                time.sleep(parse_retry_delay_seconds(exc) or min(2.0, 0.5 * attempt))
    return False, last_error


def process_record(
    record: dict,
    output_dir: Path,
    overwrite: bool,
    sleep_seconds: float,
    jitter_seconds: float,
    args: argparse.Namespace,
) -> tuple[str, str, str]:
    paper_id = get_paper_id(record)
    title = str(record.get("title", "")).strip()

    if not paper_id:
        return "failed", "", "missing id, skipped"

    output_path = build_output_path(output_dir, record)
    if output_path.exists() and not overwrite:
        return "skipped", paper_id, output_path.name

    delay = max(0.0, sleep_seconds) + random.uniform(0.0, max(0.0, jitter_seconds))
    if delay > 0:
        time.sleep(delay)

    ok, detail = download_pdf(paper_id, output_path, args)
    if ok:
        return "downloaded", paper_id, title
    return "failed", paper_id, detail


def main() -> int:
    args = parse_args()
    if not credentials_present(args):
        raise SystemExit(
            "Missing OpenReview credentials. Set --user_name/--password, --token, "
            "or OPENREVIEW_USERNAME/OPENREVIEW_PASSWORD/OPENREVIEW_TOKEN."
        )

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(input_path)

    records = load_records(input_path)
    if args.limit is not None:
        records = records[: args.limit]

    downloaded = 0
    skipped = 0
    failed = 0
    start_time = time.time()
    workers = max(1, args.workers)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_record,
                record,
                output_dir,
                args.overwrite,
                args.sleep,
                args.jitter,
                args,
            )
            for record in records
        ]

        with tqdm(
            total=len(futures),
            desc="Downloading PDFs",
            unit="pdf",
            dynamic_ncols=True,
        ) as pbar:
            for future in concurrent.futures.as_completed(futures):
                status, paper_id, detail = future.result()
                if status == "downloaded":
                    downloaded += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    tqdm.write(f"failed: {paper_id or '<missing id>'} | {detail}")

                pbar.update(1)
                pbar.set_postfix(
                    downloaded=downloaded,
                    skipped=skipped,
                    failed=failed,
                    workers=workers,
                )

    elapsed = time.time() - start_time
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output_dir": str(output_dir),
                "total": len(records),
                "downloaded": downloaded,
                "skipped": skipped,
                "failed": failed,
                "elapsed_seconds": round(elapsed, 2),
                "retries": args.retries,
                "workers": workers,
                "sleep": args.sleep,
                "jitter": args.jitter,
                "api_baseurl": args.api_baseurl,
                "min_interval": args.min_interval,
                "used_credentials": True,
            },
            ensure_ascii=False,
        )
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
