#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trim all local PDF files in a directory to the first N pages. "
            "By default, this keeps the first 10 pages."
        )
    )
    parser.add_argument(
        "--pdf_dir",
        required=True,
        help="Directory containing source PDF files.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for trimmed PDFs. Default: <pdf_dir>_first10",
    )
    parser.add_argument(
        "--max_pages",
        type=int,
        default=10,
        help="Maximum number of pages to keep from each PDF. Default: 10",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output PDFs that already exist.",
    )
    return parser.parse_args()


def default_output_dir(pdf_dir: Path, max_pages: int) -> Path:
    return pdf_dir.with_name(f"{pdf_dir.name}_first{max_pages}")


def require_pypdf():
    try:
        from pypdf import PdfReader, PdfWriter
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: pypdf\n"
            "Install it with: pip install pypdf"
        ) from exc
    return PdfReader, PdfWriter


def trim_pdf(
    source_path: Path,
    output_path: Path,
    max_pages: int,
    pdf_reader_cls,
    pdf_writer_cls,
) -> tuple[int, int]:
    reader = pdf_reader_cls(str(source_path))
    original_pages = len(reader.pages)
    kept_pages = min(original_pages, max_pages)

    writer = pdf_writer_cls()
    for page_idx in range(kept_pages):
        writer.add_page(reader.pages[page_idx])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        writer.write(f)
    return original_pages, kept_pages


def main() -> int:
    args = parse_args()

    if args.max_pages <= 0:
        raise SystemExit("--max_pages must be a positive integer")

    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir(pdf_dir, args.max_pages).resolve()
    )

    if not pdf_dir.is_dir():
        raise SystemExit(f"PDF directory not found: {pdf_dir}")

    PdfReader, PdfWriter = require_pypdf()

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDF files found in: {pdf_dir}")

    processed = 0
    skipped = 0
    errors = 0

    progress = tqdm(pdf_paths, desc="trim", unit="pdf")
    for pdf_path in progress:
        progress.set_postfix_str(pdf_path.name)
        destination = output_dir / pdf_path.name
        if destination.exists() and not args.overwrite:
            progress.set_postfix_str(f"skip {pdf_path.name}")
            skipped += 1
            continue

        try:
            original_pages, kept_pages = trim_pdf(
                source_path=pdf_path,
                output_path=destination,
                max_pages=args.max_pages,
                pdf_reader_cls=PdfReader,
                pdf_writer_cls=PdfWriter,
            )
            progress.set_postfix_str(
                f"ok {pdf_path.name} {original_pages}->{kept_pages}"
            )
            processed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {pdf_path.name}: {exc}", file=sys.stderr)
            progress.set_postfix_str(f"error {pdf_path.name}")
            errors += 1

    print(
        f"Finished. processed={processed}, skipped={skipped}, errors={errors}, output_dir={output_dir}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
