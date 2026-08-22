# Data Processing Utilities

This directory contains utilities for preparing paper-evaluation inputs.

- `download_openreview_pdfs.py`: download PDFs from OpenReview.
- `fetch_openreview_review_metadata.py`: fetch OpenReview review metadata.
- `redact_iclr_pdf_headers.py`: remove common ICLR PDF status banners and first-page metadata marks.
- `trim_pdfs_to_first_pages.py`: keep only the first `N` pages of each PDF.
- `convert_pdfs_with_marker.py`: run Marker OCR or render pages to images.
- `extract_pdf_multimodal_content.py`: extract text, formula-like text, embedded images, and formula-like images.
- `prepare_iclr_sample.py`: build sampled ICLR metadata from local parquet files.

All credentials must be provided at runtime through CLI flags or environment variables.
