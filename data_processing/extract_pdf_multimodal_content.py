#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


FORMULA_SYMBOL_RE = re.compile(r"[=+\-*/^_<>≤≥≈≠∑∫√∞∂∆∇πσμλθαβγδωΩ{}[\]()]")
LATEX_TOKEN_RE = re.compile(r"\\[A-Za-z]+")
MULTISPACE_RE = re.compile(r"\s+")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract text, formula-like text, images, and formula-like images "
            "from the first N pages of every PDF in a directory."
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
        help="Output directory. Default: <pdf_dir>_content",
    )
    parser.add_argument(
        "--max_pages",
        type=int,
        default=10,
        help="Maximum number of pages to process per PDF. Default: 10",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser.parse_args()


def safe_name(text: str) -> str:
    cleaned = SAFE_NAME_RE.sub("_", text.strip())
    return cleaned.strip("._") or "untitled"


def default_output_dir(pdf_dir: Path) -> Path:
    return pdf_dir.with_name(f"{pdf_dir.name}_content")


def normalize_fragment(text: str) -> str:
    return MULTISPACE_RE.sub(" ", text.replace("\x00", " ")).strip()


def extract_text_lines(page) -> list[dict]:
    fragments: list[dict] = []

    def visitor(text, _cm, tm, _font_dict, font_size):
        normalized = normalize_fragment(text)
        if not normalized:
            return
        x = float(tm[4]) if len(tm) > 4 else 0.0
        y = float(tm[5]) if len(tm) > 5 else 0.0
        fragments.append(
            {
                "text": normalized,
                "x": x,
                "y": y,
                "font_size": float(font_size),
            }
        )

    page.extract_text(visitor_text=visitor)
    fragments.sort(key=lambda item: (-item["y"], item["x"]))

    lines: list[dict] = []
    y_tolerance = 2.5

    for fragment in fragments:
        target = None
        for line in lines:
            if abs(line["y"] - fragment["y"]) <= y_tolerance:
                target = line
                break
        if target is None:
            lines.append(
                {
                    "y": fragment["y"],
                    "parts": [fragment],
                }
            )
        else:
            target["parts"].append(fragment)

    merged_lines: list[dict] = []
    for line in lines:
        parts = sorted(line["parts"], key=lambda item: item["x"])
        text = normalize_fragment(" ".join(part["text"] for part in parts))
        if not text:
            continue
        merged_lines.append(
            {
                "text": text,
                "y": line["y"],
                "x_min": min(part["x"] for part in parts),
                "font_size": max(part["font_size"] for part in parts),
            }
        )
    return merged_lines


def is_line_number_only(text: str) -> bool:
    stripped = text.strip()
    return bool(re.fullmatch(r"\d{1,3}", stripped))


def detect_line_numbered_page(lines: list[dict]) -> bool:
    number_only = [line for line in lines if is_line_number_only(line["text"])]
    if len(number_only) < 5:
        return False

    ys = [line["y"] for line in number_only]
    y_span = max(ys) - min(ys) if ys else 0.0
    return y_span >= 100.0


def strip_line_number_prefix(text: str) -> str:
    stripped = re.sub(r"^\d{1,3}\s+", "", text).strip()
    return stripped


def clean_lines(lines: list[dict]) -> list[dict]:
    if not detect_line_numbered_page(lines):
        return lines

    cleaned: list[dict] = []
    for line in lines:
        text = line["text"]
        if is_line_number_only(text):
            continue
        updated = strip_line_number_prefix(text)
        if not updated:
            continue
        cleaned.append({**line, "text": updated})
    return cleaned


def is_formula_like_text(line: str) -> bool:
    text = line.strip()
    if not text:
        return False

    symbol_count = len(FORMULA_SYMBOL_RE.findall(text))
    latex_count = len(LATEX_TOKEN_RE.findall(text))
    digit_count = sum(char.isdigit() for char in text)
    alpha_count = sum(char.isalpha() for char in text)
    char_count = len(text)
    token_count = len(text.split())

    if latex_count > 0:
        return True
    if symbol_count >= 3 and symbol_count / max(char_count, 1) >= 0.08:
        return True
    if "=" in text and (digit_count > 0 or symbol_count >= 2):
        return True
    if token_count <= 3 and symbol_count >= 2:
        return True
    if alpha_count > 0 and symbol_count >= 2 and alpha_count / max(char_count, 1) < 0.65:
        return True
    return False


def split_page_text(lines: list[dict]) -> tuple[str, str]:
    body_lines: list[str] = []
    formula_lines: list[str] = []

    for line in lines:
        if is_formula_like_text(line["text"]):
            formula_lines.append(line["text"])
        else:
            body_lines.append(line["text"])

    return "\n".join(body_lines).strip(), "\n".join(formula_lines).strip()


def guess_extension(image: Image.Image) -> str:
    fmt = (image.format or "").lower()
    if fmt in {"jpeg", "jpg"}:
        return "jpg"
    if fmt in {"png", "tiff", "tif", "bmp", "gif", "jp2"}:
        return "png" if fmt == "jp2" else fmt.replace("tif", "tiff")
    return "png"


def is_formula_like_image(image: Image.Image) -> bool:
    width, height = image.size
    if width == 0 or height == 0:
        return False

    rgb = image.convert("RGB")
    colors = rgb.getcolors(maxcolors=4096)
    unique_colors = len(colors) if colors is not None else 4096

    grayscale_hits = 0
    sampled = 0
    step_x = max(1, width // 64)
    step_y = max(1, height // 64)
    for y in range(0, height, step_y):
        for x in range(0, width, step_x):
            r, g, b = rgb.getpixel((x, y))
            sampled += 1
            if abs(r - g) <= 8 and abs(g - b) <= 8:
                grayscale_hits += 1

    grayscale_ratio = grayscale_hits / max(sampled, 1)
    aspect_ratio = width / max(height, 1)
    small_area = width * height <= 900_000

    if grayscale_ratio >= 0.95 and unique_colors <= 64 and 1.5 <= aspect_ratio <= 12 and small_area:
        return True
    if grayscale_ratio >= 0.98 and unique_colors <= 16 and aspect_ratio >= 1.2:
        return True
    return False


def save_image_file(image_file, destination: Path) -> dict:
    image = image_file.image
    is_formula = is_formula_like_image(image)
    ext = guess_extension(image)
    output_path = destination.with_suffix(f".{ext}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {
        "path": str(output_path),
        "width": image.width,
        "height": image.height,
        "is_formula_like": is_formula,
    }


def write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def process_pdf(pdf_path: Path, output_dir: Path, max_pages: int, overwrite: bool) -> dict:
    pdf_name = safe_name(pdf_path.stem)
    paper_dir = output_dir / pdf_name
    manifest = {
        "pdf_name": pdf_path.name,
        "pdf_path": str(pdf_path),
        "pages": [],
    }

    reader = PdfReader(str(pdf_path))
    page_count = min(len(reader.pages), max_pages)

    for page_index in range(page_count):
        page_number = page_index + 1
        page = reader.pages[page_index]
        page_key = f"page_{page_number:03d}"

        text_output = paper_dir / "texts" / f"{page_key}.txt"
        formula_text_output = paper_dir / "formula_texts" / f"{page_key}.txt"
        images_dir = paper_dir / "images" / page_key
        formula_images_dir = paper_dir / "formula_images" / page_key

        if not overwrite and text_output.exists() and formula_text_output.exists():
            body_text = text_output.read_text(encoding="utf-8").strip()
            formula_text = formula_text_output.read_text(encoding="utf-8").strip()
        else:
            lines = clean_lines(extract_text_lines(page))
            body_text, formula_text = split_page_text(lines)
            write_text_file(text_output, body_text)
            write_text_file(formula_text_output, formula_text)

        page_record = {
            "page_number": page_number,
            "text_file": str(text_output),
            "formula_text_file": str(formula_text_output),
            "text_char_count": len(body_text),
            "formula_text_char_count": len(formula_text),
            "images": [],
            "formula_images": [],
        }

        image_index = 0
        for image_file in page.images:
            image_index += 1
            is_formula = is_formula_like_image(image_file.image)
            target_base = (
                formula_images_dir / f"image_{image_index:03d}"
                if is_formula
                else images_dir / f"image_{image_index:03d}"
            )
            try:
                image_record = save_image_file(image_file, target_base)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[warn] failed to save image from {pdf_path.name} page {page_number}: {exc}",
                    file=sys.stderr,
                )
                continue

            if image_record["is_formula_like"]:
                page_record["formula_images"].append(image_record)
            else:
                page_record["images"].append(image_record)

        manifest["pages"].append(page_record)

    manifest_path = paper_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    if args.max_pages <= 0:
        raise SystemExit("--max_pages must be a positive integer")

    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir(pdf_dir).resolve()
    )

    if not pdf_dir.is_dir():
        raise SystemExit(f"PDF directory not found: {pdf_dir}")

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDF files found in: {pdf_dir}")

    run_manifest = {
        "pdf_dir": str(pdf_dir),
        "output_dir": str(output_dir),
        "max_pages": args.max_pages,
        "pdfs": [],
    }

    processed = 0
    errors = 0

    for pdf_path in pdf_paths:
        try:
            pdf_manifest = process_pdf(
                pdf_path=pdf_path,
                output_dir=output_dir,
                max_pages=args.max_pages,
                overwrite=args.overwrite,
            )
            run_manifest["pdfs"].append(pdf_manifest)
            processed += 1
            print(f"[ok] {pdf_path.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {pdf_path.name}: {exc}", file=sys.stderr)
            errors += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Finished. processed={processed}, errors={errors}, output_dir={output_dir}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
