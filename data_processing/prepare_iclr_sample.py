#!/usr/bin/env python3
import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PARQUETS = ["iclr24v2.parquet", "iclr25v1.parquet", "iclr26v1.parquet"]
DEFAULT_OUTPUT_DIR = "data/iclr2024_sample"
DEFAULT_SAMPLE_SIZE = 350
DEFAULT_SEED = 20240430
EXCLUDED_DECISION_PATTERNS = ("withdraw", "desk reject")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an ICLR 2024 sampled metadata directory from local parquet files."
    )
    parser.add_argument(
        "--parquets",
        nargs="+",
        default=DEFAULT_PARQUETS,
        help=f"Input parquet files. Default: {' '.join(DEFAULT_PARQUETS)}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output data directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of papers to sample. Default: {DEFAULT_SAMPLE_SIZE}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducible sampling. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--exemplar-source",
        default="data/exemplars/iclr2024_examples.json",
        help="Optional exemplar JSON to copy into the output directory.",
    )
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value


def normalize_record(row: pd.Series) -> dict[str, Any]:
    record = {
        key: to_jsonable(value)
        for key, value in row.to_dict().items()
        if not str(key).startswith("_")
    }
    record["year"] = int(record["year"])

    return record


def row_quality(row: pd.Series) -> tuple[int, int]:
    nonempty = 0
    for value in row.to_dict().values():
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        nonempty += 1
    has_author_ids = int(bool(str(row.get("author_ids", "") or "").strip()))
    return has_author_ids, nonempty


def load_candidates(parquet_paths: list[Path]) -> list[dict[str, Any]]:
    frames = []
    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path)
        df = df[df["year"].astype(int) == 2024].copy()
        df["_source_parquet"] = parquet_path.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    decision = combined["decision"].fillna("").astype(str).str.lower()
    excluded = pd.Series(False, index=combined.index)
    for pattern in EXCLUDED_DECISION_PATTERNS:
        excluded = excluded | decision.str.contains(pattern, regex=False)
    combined = combined[~excluded]

    records = []
    for _, group in combined.groupby("id", sort=False):
        best_idx = max(group.index, key=lambda idx: row_quality(group.loc[idx]))
        records.append(normalize_record(group.loc[best_idx].drop(labels=[])))
    return records


def main() -> int:
    args = parse_args()
    parquet_paths = [Path(path) for path in args.parquets]
    for path in parquet_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    candidates = load_candidates(parquet_paths)
    if len(candidates) < args.sample_size:
        raise ValueError(
            f"Only {len(candidates)} non-withdrawn ICLR 2024 papers available; "
            f"cannot sample {args.sample_size}."
        )

    rng = random.Random(args.seed)
    selected = rng.sample(candidates, args.sample_size)
    selected.sort(key=lambda record: str(record.get("id", "")))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in (
        "iclr2024_papers_pdfs",
        "iclr2024_papers_pdfs_image",
        "iclr2024_papers_pdfs_ocr",
    ):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "iclr2024_papers.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
        f.write("\n")

    exemplar_source = Path(args.exemplar_source)
    if exemplar_source.exists():
        shutil.copyfile(exemplar_source, output_dir / "iclr2024_examples.json")

    manifest = {
        "output_dir": str(output_dir),
        "metadata_file": str(metadata_path),
        "sample_size": len(selected),
        "seed": args.seed,
        "source_parquets": [str(path) for path in parquet_paths],
        "candidate_count_after_exclusions_unique": len(candidates),
        "excluded_decision_patterns": list(EXCLUDED_DECISION_PATTERNS),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
