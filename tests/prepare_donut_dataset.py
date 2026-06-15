"""
Prepare Donut metadata.jsonl from annotated prescription images.

This script intentionally refuses to create records from image-only datasets.
Every generated ground_truth must contain real doctor, patient, or medication
values from the paired annotation JSON.

Usage:
  python -m tests.prepare_donut_dataset
"""

from __future__ import annotations

import json
from pathlib import Path

from src.donut_dataset import build_metadata_from_annotations, iter_sample_summaries


BASE_DIR = Path("c:/Users/USER/Desktop/OJAAI")
EVAL_DIR = BASE_DIR / "data/evaluation"
ANNOTATIONS_DIR = EVAL_DIR / "annotations"
IMAGES_DIR = EVAL_DIR / "images"


def prepare_donut_dataset() -> None:
    print("Starting annotated Donut dataset preparation...")
    entries = build_metadata_from_annotations(
        annotations_dir=ANNOTATIONS_DIR,
        images_dir=IMAGES_DIR,
        output_dir=EVAL_DIR,
    )
    metadata_path = EVAL_DIR / "metadata.jsonl"
    print(f"Successfully wrote {len(entries)} validated metadata entries to {metadata_path}")
    print("")
    print("Sample training records:")
    for sample in iter_sample_summaries(entries, count=3):
        print(json.dumps(sample, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    prepare_donut_dataset()
