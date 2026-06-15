"""
Utilities for preparing and validating Donut prescription training data.

The source annotation files used by this project store one OCR-style string:
  <s_ocr> doctor_name: ... medications: - Drug 500 mg - After meals ... </s>

This module converts those strings into structured, non-empty metadata.jsonl
records for Donut fine-tuning. Empty labels are treated as hard failures.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


TRAINING_KEYS = (
    "doctor_name",
    "doctor_reg",
    "clinic_name",
    "patient_name",
    "patient_age",
    "patient_gender",
    "prescription_date",
    "diagnosis",
    "medications",
)

FIELD_NAMES = (
    "doctor_name",
    "clinic_name",
    "clinic_address",
    "patient_name",
    "patient_age",
    "patient_gender",
    "date",
    "diagnosis",
    "medications",
    "signature",
)

FIELD_BOUNDARY_RE = "|".join(f"{name}:" for name in FIELD_NAMES)

FREQUENCY_MAP = {
    "once daily": 1,
    "od": 1,
    "every 24 hours": 1,
    "every 12 hours": 2,
    "twice daily": 2,
    "bd": 2,
    "bid": 2,
    "three times daily": 3,
    "tds": 3,
    "tid": 3,
    "four times daily": 4,
    "qds": 4,
    "qid": 4,
    "as needed": 0,
    "sos": 0,
    "prn": 0,
}

NON_DRUG_INSTRUCTION_STARTS = {
    "after",
    "as",
    "at",
    "before",
    "every",
    "for",
    "if",
    "in",
    "on",
    "once",
    "take",
    "three",
    "twice",
    "with",
}

DOSAGE_RE = re.compile(
    r"(?P<drug>[A-Za-z][A-Za-z0-9 ./'-]*?)\s+"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>mg|mcg|ml|g|iu|units?|puffs?|drops?)\b",
    re.I,
)


def parse_header_field(gt_str: str, field_name: str) -> str:
    """Extract a named field from the source OCR-style ground_truth string."""
    pattern = rf"\b{re.escape(field_name)}:\s*(.*?)\s*(?={FIELD_BOUNDARY_RE}|</s>|$)"
    match = re.search(pattern, gt_str, re.I | re.S)
    return _clean_text(match.group(1)) if match else ""


def parse_gt_medications(gt_str: str) -> list[dict[str, Any]]:
    """Extract medication rows from the source OCR-style annotation string."""
    meds_match = re.search(
        rf"\bmedications:\s*(.*?)\s*(?=signature:|{FIELD_BOUNDARY_RE}|</s>|$)",
        gt_str,
        re.I | re.S,
    )
    if not meds_match:
        return []

    meds_block = _clean_text(meds_match.group(1))
    raw_items = [item.strip() for item in re.split(r"\s+-\s+", f" {meds_block}") if item.strip()]

    medications: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None

    for item in raw_items:
        item = _clean_text(item)
        if not item:
            continue

        dosage_match = DOSAGE_RE.search(item)
        if dosage_match:
            if pending:
                medications.append(pending)
            pending = {
                "drug_name": _clean_text(dosage_match.group("drug")),
                "dosage_value": float(dosage_match.group("value")),
                "dosage_unit": dosage_match.group("unit").lower(),
                "frequency": None,
                "freq_per_day": None,
                "duration_days": None,
                "route": "oral",
            }
            trailing = _clean_text(item[dosage_match.end():])
            if trailing:
                pending["frequency"] = trailing
                pending["freq_per_day"] = frequency_to_per_day(trailing)
            continue

        first_word = item.split()[0].lower() if item.split() else ""
        if pending and first_word in NON_DRUG_INSTRUCTION_STARTS:
            pending["frequency"] = item
            pending["freq_per_day"] = frequency_to_per_day(item)
            continue

        # Last-resort support for a drug item with no explicit dosage.
        if pending:
            medications.append(pending)
        pending = {
            "drug_name": item,
            "dosage_value": None,
            "dosage_unit": None,
            "frequency": None,
            "freq_per_day": None,
            "duration_days": None,
            "route": "oral",
        }

    if pending:
        medications.append(pending)

    return [med for med in medications if med.get("drug_name")]


def frequency_to_per_day(frequency: str | None) -> int | None:
    """Map common frequency phrases to numeric frequency where possible."""
    if not frequency:
        return None
    normalized = _clean_text(frequency).lower()
    for key, value in FREQUENCY_MAP.items():
        if key in normalized:
            return value
    return None


def parse_annotation_file(annotation_file: Path) -> dict[str, Any]:
    """Parse one source annotation JSON into the Donut structured target."""
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    gt_str = data.get("ground_truth", "")
    if not isinstance(gt_str, str) or not gt_str.strip():
        raise ValueError(f"{annotation_file.name}: missing source ground_truth")

    doctor_name = parse_header_field(gt_str, "doctor_name")
    doctor_reg = parse_doctor_reg(doctor_name)

    structured_gt = {
        "doctor_name": doctor_name,
        "doctor_reg": doctor_reg,
        "clinic_name": parse_header_field(gt_str, "clinic_name"),
        "patient_name": parse_header_field(gt_str, "patient_name"),
        "patient_age": parse_header_field(gt_str, "patient_age"),
        "patient_gender": parse_header_field(gt_str, "patient_gender"),
        "prescription_date": parse_header_field(gt_str, "date"),
        "diagnosis": parse_header_field(gt_str, "diagnosis"),
        "medications": parse_gt_medications(gt_str),
    }
    return compact_ground_truth(structured_gt)


def parse_doctor_reg(doctor_name: str) -> str | None:
    """Extract doctor registration if it appears inside the doctor name field."""
    reg_match = re.search(r"\b(?:mci|reg|nmc)[\s.:-]*([A-Z0-9/-]+)", doctor_name, re.I)
    return reg_match.group(1) if reg_match else None


def compact_ground_truth(value: Any) -> Any:
    """Remove empty strings, nulls, empty lists, and empty dicts recursively."""
    if isinstance(value, dict):
        compacted = {}
        for key, item in value.items():
            cleaned = compact_ground_truth(item)
            if cleaned not in (None, "", [], {}):
                compacted[key] = cleaned
        return compacted
    if isinstance(value, list):
        return [item for item in (compact_ground_truth(item) for item in value) if item not in (None, "", [], {})]
    if isinstance(value, str):
        return _clean_text(value)
    return value


def has_non_empty_labels(ground_truth: dict[str, Any]) -> bool:
    """Return True if a structured target contains trainable content."""
    if not isinstance(ground_truth, dict):
        return False
    if any(ground_truth.get(key) for key in ("doctor_name", "clinic_name", "patient_name", "patient_age", "prescription_date")):
        return True
    meds = ground_truth.get("medications")
    return isinstance(meds, list) and any(isinstance(med, dict) and med.get("drug_name") for med in meds)


def validate_ground_truth(ground_truth: dict[str, Any], source: str) -> None:
    """Raise immediately when a target is empty or unusable for training."""
    if not has_non_empty_labels(ground_truth):
        raise ValueError(f"{source}: empty Donut labels; refusing to train on null-only ground_truth")
    meds = ground_truth.get("medications", [])
    if "medications" in ground_truth and not isinstance(meds, list):
        raise ValueError(f"{source}: medications must be a list")
    for idx, med in enumerate(meds):
        if not isinstance(med, dict) or not med.get("drug_name"):
            raise ValueError(f"{source}: medication #{idx + 1} is missing drug_name")


def build_metadata_from_annotations(
    annotations_dir: Path,
    images_dir: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
) -> list[dict[str, str]]:
    """
    Build Donut metadata.jsonl from paired annotation/image directories.

    Images are copied into output_dir/images and metadata is written to
    output_dir/metadata.jsonl. Every emitted record is validated as non-empty.
    """
    annotations_dir = Path(annotations_dir)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    entries: list[dict[str, str]] = []
    annotation_files = sorted(annotations_dir.glob("*.json"))
    if limit is not None:
        annotation_files = annotation_files[:limit]

    for annotation_file in annotation_files:
        gt = parse_annotation_file(annotation_file)
        validate_ground_truth(gt, annotation_file.name)

        src_image = find_image_for_annotation(images_dir, annotation_file.stem)
        if src_image is None:
            raise FileNotFoundError(f"{annotation_file.name}: no matching image found in {images_dir}")

        dest_image = output_images_dir / src_image.name
        if src_image.resolve() != dest_image.resolve():
            shutil.copy2(src_image, dest_image)

        entries.append(
            {
                "file_name": f"images/{dest_image.name}",
                "ground_truth": json.dumps(gt, ensure_ascii=False, sort_keys=True),
            }
        )

    if not entries:
        raise ValueError(f"No valid Donut metadata entries generated from {annotations_dir}")

    metadata_file = output_dir / "metadata.jsonl"
    with metadata_file.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entries


def load_metadata(metadata_file: Path) -> list[dict[str, str]]:
    """Load and validate an existing metadata.jsonl file."""
    samples: list[dict[str, str]] = []
    with Path(metadata_file).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            gt = ground_truth_from_entry(entry, f"{metadata_file}:{line_no}")
            validate_ground_truth(gt, f"{metadata_file}:{line_no}")
            samples.append(entry)
    if not samples:
        raise ValueError(f"{metadata_file}: no valid training samples found")
    return samples


def ground_truth_from_entry(entry: dict[str, Any], source: str) -> dict[str, Any]:
    """Decode, compact, and validate a metadata entry's ground_truth object."""
    if "file_name" not in entry:
        raise ValueError(f"{source}: missing file_name")
    raw_gt = entry.get("ground_truth")
    if isinstance(raw_gt, str):
        gt = json.loads(raw_gt)
    elif isinstance(raw_gt, dict):
        gt = raw_gt
    else:
        raise ValueError(f"{source}: ground_truth must be a JSON string or object")
    if not isinstance(gt, dict):
        raise ValueError(f"{source}: decoded ground_truth must be an object")
    gt = compact_ground_truth(gt)
    validate_ground_truth(gt, source)
    return gt


def iter_sample_summaries(entries: Iterable[dict[str, str]], count: int = 3) -> list[dict[str, Any]]:
    """Return compact human-readable summaries for logging/debug printing."""
    summaries = []
    for idx, entry in enumerate(entries):
        if idx >= count:
            break
        gt = ground_truth_from_entry(entry, f"sample[{idx}]")
        summaries.append({"file_name": entry["file_name"], "ground_truth": gt})
    return summaries


def find_image_for_annotation(images_dir: Path, stem: str) -> Path | None:
    """Find the image file matching an annotation stem."""
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = Path(images_dir) / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _clean_text(value: str) -> str:
    """Normalize whitespace and strip source task tokens."""
    value = re.sub(r"</?s(?:_ocr)?>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\r\n:-")
