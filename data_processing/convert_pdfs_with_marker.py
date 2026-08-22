#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess PDF files with marker in two modes: "
            "`ocr` converts each PDF to markdown with marker; "
            "`image` only renders the first N pages to page images."
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
        help="Directory where outputs will be written. Default: <pdf_dir>_<mode>",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["ocr", "image"],
        help="Preprocessing mode.",
    )
    parser.add_argument(
        "--max_pages",
        type=int,
        default=10,
        help="Maximum number of pages to process per PDF. Default: 10",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Render DPI for PDF pages. Default: 200",
    )
    parser.add_argument(
        "--image_format",
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Rendered page image format. Default: png",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--start_ratio",
        type=float,
        default=0.0,
        help="Start ratio in [0, 1) over the sorted PDF list. Default: 0.0",
    )
    parser.add_argument(
        "--end_ratio",
        type=float,
        default=-1.0,
        help="End ratio in (0, 1]. Default: -1, which means process through the end.",
    )
    parser.add_argument(
        "--force_ocr",
        action="store_true",
        help="Force marker to OCR all text regions. Default: enabled",
    )
    parser.add_argument(
        "--no_force_ocr",
        dest="force_ocr",
        action="store_false",
        help="Disable forced OCR and let marker use embedded PDF text when available.",
    )
    parser.set_defaults(force_ocr=True)
    parser.add_argument(
        "--strip_existing_ocr",
        action="store_true",
        help="Keep digital text but strip out existing OCR text when marker parses PDFs.",
    )
    parser.add_argument(
        "--paginate_output",
        action="store_true",
        help="Insert page separators in marker markdown output.",
    )
    parser.add_argument(
        "--output_debug_json",
        action="store_true",
        help="Also write marker JSON output beside the markdown file.",
    )
    return parser.parse_args()


def require_pymupdf():
    try:
        import fitz  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: PyMuPDF\n"
            "Install with: pip install pymupdf"
        ) from exc
    return fitz


def require_marker():
    try:
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependencies for OCR mode.\n"
            "Install with: pip install marker-pdf"
        ) from exc
    return ConfigParser, PdfConverter, create_model_dict


def safe_stem(path: Path) -> str:
    return path.stem


def paper_id_from_path(path: Path) -> str:
    return path.name.split(" - ", 1)[0]


def default_output_dir(pdf_dir: Path, mode: str) -> Path:
    return pdf_dir.with_name(f"{pdf_dir.name}_{mode}")


def done_ids_path(output_dir: Path) -> Path:
    return output_dir / "done_ids.txt"


def load_done_ids(output_dir: Path) -> set[str]:
    path = done_ids_path(output_dir)
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def append_done_id(output_dir: Path, paper_id: str) -> None:
    path = done_ids_path(output_dir)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{paper_id}\n")


def select_pdf_slice(pdf_paths: list[Path], start_ratio: float, end_ratio: float) -> list[Path]:
    total = len(pdf_paths)
    if total == 0:
        return []

    if not (0.0 <= start_ratio < 1.0):
        raise SystemExit("--start_ratio must be in [0, 1)")
    if end_ratio != -1.0 and not (0.0 < end_ratio <= 1.0):
        raise SystemExit("--end_ratio must be -1 or in (0, 1]")

    # Use the same boundary mapping for both slice starts and slice ends so
    # adjacent ratio ranges (for example 0-0.1 and 0.1-0.2) partition the
    # sorted PDF list without overlap.
    start_index = math.floor(total * start_ratio)
    end_index = total if end_ratio == -1.0 else math.floor(total * end_ratio)

    start_index = max(0, min(start_index, total))
    end_index = max(0, min(end_index, total))

    if end_index < start_index:
        raise SystemExit("--end_ratio must be greater than or equal to --start_ratio")

    return pdf_paths[start_index:end_index]


def render_pdf_pages(
    pdf_path: Path,
    destination_dir: Path,
    max_pages: int,
    dpi: int,
    image_format: str,
) -> list[Path]:
    fitz = require_pymupdf()

    destination_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []

    with fitz.open(pdf_path) as doc:
        page_count = min(len(doc), max_pages)
        for page_index in range(page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            image_path = destination_dir / f"page {page_index + 1}.{image_format}"
            pix.save(str(image_path))
            image_paths.append(image_path)

    return image_paths


def build_marker_page_range(page_count: int, max_pages: int) -> str:
    last_page = min(page_count, max_pages) - 1
    if last_page < 0:
        raise RuntimeError("PDF has no pages")
    return f"0-{last_page}"


def run_marker_on_pdf(
    pdf_path: Path,
    markdown_path: Path,
    max_pages: int,
    force_ocr: bool,
    strip_existing_ocr: bool,
    paginate_output: bool,
    output_debug_json: bool,
) -> None:
    ConfigParser, PdfConverter, create_model_dict = require_marker()
    fitz = require_pymupdf()

    with fitz.open(pdf_path) as doc:
        page_count = len(doc)

    page_range = build_marker_page_range(page_count, max_pages)

    config = {
        "output_format": "markdown",
        "page_range": page_range,
        "force_ocr": force_ocr,
        "strip_existing_ocr": strip_existing_ocr,
        "paginate_output": paginate_output,
    }
    config_parser = ConfigParser(config)
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )

    rendered = converter(str(pdf_path))
    markdown = getattr(rendered, "markdown", None)
    if not isinstance(markdown, str):
        raise RuntimeError(f"marker did not return markdown for {pdf_path.name}")

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")

    if output_debug_json:
        json_renderer_config = ConfigParser(
            {
                "output_format": "json",
                "page_range": page_range,
                "force_ocr": force_ocr,
                "strip_existing_ocr": strip_existing_ocr,
            }
        )
        json_converter = PdfConverter(
            config=json_renderer_config.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=json_renderer_config.get_processors(),
            renderer=json_renderer_config.get_renderer(),
            llm_service=json_renderer_config.get_llm_service(),
        )
        json_rendered = json_converter(str(pdf_path))
        json_path = markdown_path.with_suffix(".json")
        if hasattr(json_rendered, "model_dump_json"):
            json_path.write_text(json_rendered.model_dump_json(indent=2), encoding="utf-8")
        elif hasattr(json_rendered, "model_dump"):
            import json

            json_path.write_text(
                json.dumps(json_rendered.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


def process_image_mode(
    pdf_paths: list[Path],
    output_dir: Path,
    max_pages: int,
    dpi: int,
    image_format: str,
    overwrite: bool,
) -> None:
    done_ids = load_done_ids(output_dir)
    progress = tqdm(pdf_paths, desc="image", unit="pdf")
    for pdf_path in progress:
        progress.set_postfix_str(pdf_path.name)
        paper_id = paper_id_from_path(pdf_path)
        pdf_output_dir = output_dir / safe_stem(pdf_path)
        existing_images = []
        if pdf_output_dir.exists():
            existing_images = sorted(
                [
                    path
                    for path in pdf_output_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
                ]
            )
        if not overwrite and (paper_id in done_ids or existing_images):
            progress.set_postfix_str(f"skip {pdf_path.name}")
            continue

        render_pdf_pages(
            pdf_path=pdf_path,
            destination_dir=pdf_output_dir,
            max_pages=max_pages,
            dpi=dpi,
            image_format=image_format,
        )
        if paper_id not in done_ids:
            append_done_id(output_dir, paper_id)
            done_ids.add(paper_id)
        progress.set_postfix_str(f"ok {pdf_path.name}")


def process_ocr_mode(
    pdf_paths: list[Path],
    output_dir: Path,
    max_pages: int,
    overwrite: bool,
    force_ocr: bool,
    strip_existing_ocr: bool,
    paginate_output: bool,
    output_debug_json: bool,
) -> None:
    done_ids = load_done_ids(output_dir)
    progress = tqdm(pdf_paths, desc="ocr", unit="pdf")
    for pdf_path in progress:
        progress.set_postfix_str(pdf_path.name)
        paper_id = paper_id_from_path(pdf_path)
        markdown_path = output_dir / f"{safe_stem(pdf_path)}.md"
        if not overwrite and (paper_id in done_ids or markdown_path.exists()):
            progress.set_postfix_str(f"skip {pdf_path.name}")
            continue

        run_marker_on_pdf(
            pdf_path=pdf_path,
            markdown_path=markdown_path,
            max_pages=max_pages,
            force_ocr=force_ocr,
            strip_existing_ocr=strip_existing_ocr,
            paginate_output=paginate_output,
            output_debug_json=output_debug_json,
        )
        if paper_id not in done_ids:
            append_done_id(output_dir, paper_id)
            done_ids.add(paper_id)
        progress.set_postfix_str(f"ok {pdf_path.name}")


def main() -> int:
    args = parse_args()
    if args.max_pages <= 0:
        raise SystemExit("--max_pages must be a positive integer")
    if args.dpi <= 0:
        raise SystemExit("--dpi must be a positive integer")

    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir(pdf_dir, args.mode).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_dir.is_dir():
        raise SystemExit(f"PDF directory not found: {pdf_dir}")

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDF files found in: {pdf_dir}")
    selected_pdf_paths = select_pdf_slice(
        pdf_paths=pdf_paths,
        start_ratio=args.start_ratio,
        end_ratio=args.end_ratio,
    )
    if not selected_pdf_paths:
        raise SystemExit(
            "No PDF files selected for the requested ratio range. "
            f"total_pdfs={len(pdf_paths)}, start_ratio={args.start_ratio}, end_ratio={args.end_ratio}"
        )

    if args.mode == "image":
        process_image_mode(
            pdf_paths=selected_pdf_paths,
            output_dir=output_dir,
            max_pages=args.max_pages,
            dpi=args.dpi,
            image_format=args.image_format,
            overwrite=args.overwrite,
        )
    else:
        process_ocr_mode(
            pdf_paths=selected_pdf_paths,
            output_dir=output_dir,
            max_pages=args.max_pages,
            overwrite=args.overwrite,
            force_ocr=args.force_ocr,
            strip_existing_ocr=args.strip_existing_ocr,
            paginate_output=args.paginate_output,
            output_debug_json=args.output_debug_json,
        )

    print(
        "Finished. "
        f"mode={args.mode}, total_pdfs={len(pdf_paths)}, selected_pdfs={len(selected_pdf_paths)}, "
        f"start_ratio={args.start_ratio}, end_ratio={args.end_ratio}, output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
