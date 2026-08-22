#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from tqdm import tqdm


TARGET_STRINGS = (
    "Published as a conference paper at ICLR 2025",
    "Under review as a conference paper at ICLR",
    "Paper under double-blind review",
)

ABSTRACT_LABELS = (
    "ABSTRACT",
    "Abstract",
)

INTRODUCTION_LABELS = (
    "1 Introduction",
    "1  Introduction",
    "1 INTRODUCTION",
    "Introduction",
)

ACK_LABELS = (
    "Acknowledgement",
    "Acknowledgements",
    "Acknowledgment",
    "Acknowledgments",
)

REFERENCE_LABELS = (
    "References",
    "REFERENCES",
)

FIRST_PAGE_FOOTER_BAND_HEIGHT = 84


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove the standard ICLR review/publication banner from the first "
            "page of each PDF in a directory."
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
        help="Directory for cleaned PDFs. Default: <pdf_dir>_clean",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output PDFs that already exist.",
    )
    return parser.parse_args()


def default_output_dir(pdf_dir: Path) -> Path:
    return pdf_dir.with_name(f"{pdf_dir.name}_clean")


def require_pymupdf():
    try:
        import fitz  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: PyMuPDF\n"
            "Install it with: pip install pymupdf"
        ) from exc
    return fitz


def find_first_match_rect(page, needles: tuple[str, ...]):
    candidates = []
    for needle in needles:
        for rect in page.search_for(needle):
            candidates.append(rect)
    if not candidates:
        return None
    return min(candidates, key=lambda rect: (rect.y0, rect.x0))


def find_title_bottom(page) -> float | None:
    data = page.get_text("dict")
    candidates = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        lines = block.get("lines", [])
        text_parts = []
        max_size = 0.0
        for line in lines:
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    text_parts.append(text)
                    max_size = max(max_size, float(span.get("size", 0.0)))
        if not text_parts:
            continue
        x0, y0, x1, y1 = block["bbox"]
        if y0 > 240:
            continue
        text = " ".join(text_parts).strip()
        if text.isdigit() or text.replace(" ", "").isdigit():
            continue
        candidates.append((x0, y0, x1, y1, text, max_size))

    if not candidates:
        return None

    abstract_rect = find_first_match_rect(page, ABSTRACT_LABELS)
    abstract_y = abstract_rect.y0 if abstract_rect is not None else 10**9
    visible = []
    for block in candidates:
        _, y0, _, _, text, max_size = block
        lower = text.lower()
        if any(label.lower() in lower for label in TARGET_STRINGS):
            continue
        if "abstract" in lower:
            continue
        if y0 >= abstract_y:
            continue
        visible.append(block)

    if not visible:
        return None

    max_title_size = max(block[5] for block in visible)
    size_threshold = max_title_size * 0.88
    title_blocks = [block for block in visible if block[5] >= size_threshold]
    if not title_blocks:
        return None
    return max(block[3] for block in title_blocks)


def redact_text_blocks_between(page, top: float, bottom: float, fitz) -> bool:
    if bottom - top < 12:
        return False
    rect = fitz.Rect(0, top, page.rect.width, bottom)
    page.add_redact_annot(rect, fill=(1, 1, 1))
    return True


def redact_margin_bands(page, fitz) -> int:
    page_width = page.rect.width
    band_width = min(max(page_width * 0.15, 48), 96)
    left_rect = fitz.Rect(0, 0, band_width, page.rect.height)
    right_rect = fitz.Rect(page_width - band_width, 0, page_width, page.rect.height)
    page.add_redact_annot(left_rect, fill=(1, 1, 1))
    page.add_redact_annot(right_rect, fill=(1, 1, 1))
    return 2


def redact_bottom_page_numbers(page, fitz) -> int:
    removed = 0
    page_width = page.rect.width
    page_height = page.rect.height
    center_x = page_width / 2
    for block in page.get_text("blocks", sort=True):
        x0, y0, x1, y1, text = block[:5]
        text = str(text).strip()
        if not text:
            continue
        normalized = text.replace("\n", " ").replace(" ", "")
        if not normalized.isdigit():
            continue
        if y0 < page_height * 0.9:
            continue
        if abs(((x0 + x1) / 2) - center_x) > page_width * 0.12:
            continue
        if (x1 - x0) > 20:
            continue
        rect = fitz.Rect(x0 - 1, y0 - 1, x1 + 1, y1 + 1)
        page.add_redact_annot(rect, fill=(1, 1, 1))
        removed += 1
    return removed


def find_next_heading_rect(page, needles: tuple[str, ...], after_y: float):
    candidates = []
    for needle in needles:
        for rect in page.search_for(needle):
            if rect.y0 > after_y:
                candidates.append(rect)
    if not candidates:
        return None
    return min(candidates, key=lambda rect: (rect.y0, rect.x0))


def find_first_page_footer_separator_y(page) -> float | None:
    page_width = page.rect.width
    page_height = page.rect.height
    candidates = []
    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            if not item:
                continue
            op = item[0]
            if op != "l":
                continue
            p1, p2 = item[1], item[2]
            if abs(p1.y - p2.y) > 1.5:
                continue
            x_min, x_max = sorted((p1.x, p2.x))
            length = x_max - x_min
            y = (p1.y + p2.y) / 2
            if y < page_height * 0.65:
                continue
            if length < page_width * 0.28:
                continue
            candidates.append(y)
    if not candidates:
        return None
    return min(candidates)


def redact_first_page_footer_note_blocks(page, fitz) -> int:
    removed = 0
    page_height = page.rect.height
    footer_top = find_first_page_footer_separator_y(page)
    if footer_top is None:
        footer_top = page_height * 0.82

    keywords = (
        "equal contribution",
        "contributed equally",
        "corresponding author",
        "work done during",
        "during",
        "internship",
        "author",
    )

    for block in page.get_text("blocks", sort=True):
        x0, y0, x1, y1, text = block[:5]
        text = str(text).strip()
        if not text:
            continue
        if y0 < footer_top and y0 < page_height * 0.82:
            continue
        lower = " ".join(text.lower().split())
        has_contact_info = "@" in text or "http://" in lower or "https://" in lower or "www." in lower
        if not any(keyword in lower for keyword in keywords) and not has_contact_info:
            continue
        rect = fitz.Rect(x0 - 1, y0 - 1, x1 + 1, y1 + 1)
        page.add_redact_annot(rect, fill=(1, 1, 1))
        removed += 1
    return removed


def remove_banner_from_pdf(source_path: Path, output_path: Path, fitz) -> tuple[int, bool, int, int]:
    with fitz.open(source_path) as doc:
        if not doc:
            raise RuntimeError("empty pdf")

        total_matches = 0
        ack_changes = 0
        line_number_changes = 0
        footer_note_changes = 0
        for page in doc:
            page_matches = 0
            header_rect = fitz.Rect(0, 0, page.rect.width, min(42, page.rect.height))
            page.add_redact_annot(header_rect, fill=(1, 1, 1))
            page_matches += 1
            for target in TARGET_STRINGS:
                rects = page.search_for(target)
                page_matches += len(rects)
                for rect in rects:
                    page.add_redact_annot(rect, fill=(1, 1, 1))

            ack_rect = find_first_match_rect(page, ACK_LABELS)
            if ack_rect is not None:
                next_rect = find_next_heading_rect(page, REFERENCE_LABELS, ack_rect.y0)
                bottom = next_rect.y0 - 4 if next_rect is not None else page.rect.height
                redact_rect = fitz.Rect(0, max(0, ack_rect.y0 - 2), page.rect.width, bottom)
                page.add_redact_annot(redact_rect, fill=(1, 1, 1))
                ack_changes += 1

            line_number_changes += redact_margin_bands(page, fitz)
            line_number_changes += redact_bottom_page_numbers(page, fitz)

            if page_matches:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
                total_matches += page_matches
            elif ack_rect is not None or line_number_changes:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        author_band_removed = False
        first_page = doc[0]
        footer_note_changes = redact_first_page_footer_note_blocks(first_page, fitz)
        abstract_rect = find_first_match_rect(first_page, ABSTRACT_LABELS)
        if abstract_rect is None:
            abstract_rect = find_first_match_rect(first_page, INTRODUCTION_LABELS)

        title_bottom = find_title_bottom(first_page)
        if abstract_rect is not None and title_bottom is not None:
            top = title_bottom + 4
            bottom = abstract_rect.y0 - 4
            if bottom - top >= 12:
                author_band_removed = redact_text_blocks_between(
                    page=first_page,
                    top=top,
                    bottom=bottom,
                    fitz=fitz,
                )
                if author_band_removed:
                    first_page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
        if footer_note_changes and not author_band_removed:
            first_page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(
            output_path,
            garbage=3,
            deflate=True,
            clean=True,
        )
        return total_matches, author_band_removed, ack_changes, line_number_changes, footer_note_changes


def main() -> int:
    args = parse_args()
    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir(pdf_dir).resolve()
    )

    if not pdf_dir.is_dir():
        raise SystemExit(f"PDF directory not found: {pdf_dir}")

    fitz = require_pymupdf()

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDF files found in: {pdf_dir}")

    processed = 0
    skipped = 0
    errors = 0
    changed = 0
    unchanged = 0
    author_band_changes = 0
    ack_section_changes = 0
    line_number_total = 0
    footer_note_total = 0
    progress = tqdm(pdf_paths, desc="clean", unit="pdf")
    for pdf_path in progress:
        progress.set_postfix_str(pdf_path.name)
        destination = output_dir / pdf_path.name

        if destination.exists() and not args.overwrite:
            progress.set_postfix_str(f"skip {pdf_path.name}")
            skipped += 1
            continue

        try:
            matches, author_band_removed, ack_changes, line_number_changes, footer_note_changes = remove_banner_from_pdf(
                source_path=pdf_path,
                output_path=destination,
                fitz=fitz,
            )
            processed += 1
            if matches or author_band_removed or ack_changes or line_number_changes or footer_note_changes:
                changed += 1
                if author_band_removed:
                    author_band_changes += 1
                if ack_changes:
                    ack_section_changes += 1
                line_number_total += line_number_changes
                footer_note_total += footer_note_changes
                progress.set_postfix_str(
                    f"ok {pdf_path.name} matches={matches} author_band={int(author_band_removed)} ack={ack_changes} line_nums={line_number_changes} footer_notes={footer_note_changes}"
                )
            else:
                unchanged += 1
                progress.set_postfix_str(f"ok {pdf_path.name} no-match")
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {pdf_path.name}: {exc}", file=sys.stderr)
            progress.set_postfix_str(f"error {pdf_path.name}")
            errors += 1

    print(
        "Finished. "
        f"processed={processed}, skipped={skipped}, changed={changed}, "
        f"author_band_changes={author_band_changes}, ack_section_changes={ack_section_changes}, "
        f"line_number_changes={line_number_total}, footer_note_changes={footer_note_total}, "
        f"unchanged={unchanged}, errors={errors}, output_dir={output_dir}"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
