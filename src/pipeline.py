"""
pipeline.py — End-to-end orchestrator for OJAAI prescription processing.

Data flow:
  Image bytes
    → [Gemini Vision] (if GEMINI_API_KEY set) → structured JSON directly
    → [Fallback] preprocessor → Tesseract OCR → medical_ner (regex)
    → drug_normalizer (INDIA_BRAND_MAP + RxNorm)
    → ddi_checker (CRITICAL_DDI_DB + class rules + OpenFDA fallback)
    → clinical_checker (drug-lab safety cross-reference)
    → database (main tables if confidence >= threshold, else review_queue)
    → PrescriptionOutput (returned to API)

Safety rules:
  - Never log raw OCR text at INFO level.
  - Never store images as DB blobs — filepath only.
  - Low-confidence prescriptions go to review_queue only.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import pytesseract

from src import database as db_module
from src import ddi_checker, drug_normalizer, medical_ner, preprocessor
from src.database import (
    SessionLocal,
    get_active_medications,
    get_or_create_patient,
    save_prescription_to_db,
    save_to_review_queue,
)
from src.models import DrugInteraction, MedicationExtracted, NormalizedDrug, PrescriptionOutput

logger = logging.getLogger(__name__)

# Tesseract configuration per TRD Section 1
_TESS_CONFIG = "--oem 3 --psm 4 -l eng"

# Tesseract executable path for Windows
_TESS_WIN_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_TESS_WIN_PATH):
    pytesseract.pytesseract.tesseract_cmd = _TESS_WIN_PATH

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
IMAGE_STORAGE_PATH = Path(os.getenv("IMAGE_STORAGE_PATH", "./data/prescriptions"))
IMAGE_STORAGE_PATH.mkdir(parents=True, exist_ok=True)


def _save_image(image_bytes: bytes, filename: str) -> str:
    """
    Save uploaded image bytes to the local filesystem with a UUID filename.
    Returns the absolute path string.
    Never stores images as DB blobs.
    """
    suffix = Path(filename).suffix.lower() or ".jpg"
    unique_name = f"{uuid.uuid4()}{suffix}"
    dest = IMAGE_STORAGE_PATH / unique_name
    dest.write_bytes(image_bytes)
    logger.info("Image saved to %s", dest)
    return str(dest)


def _run_ocr(clean_image) -> str:
    """Run Tesseract OCR on a preprocessed numpy array. Returns raw text."""
    from PIL import Image
    pil_img = Image.fromarray(clean_image)
    raw_text = pytesseract.image_to_string(pil_img, config=_TESS_CONFIG)
    return raw_text


def _run_tesseract_ocr(image_bytes: bytes, filename: str) -> tuple[str, float]:
    """
    Run Tesseract preprocessing + OCR. Returns (raw_text, confidence).
    Uses the heavy binarisation pipeline (CLAHE + perspective correction).
    Confidence is derived from the NER pass on the text.
    """
    # Use the Tesseract-specific heavy pipeline (CLAHE, perspective, threshold)
    clean_image = preprocessor.preprocess_from_bytes(image_bytes, filename)
    raw_text = _run_ocr(clean_image)
    logger.debug("Tesseract OCR complete (text length=%d)", len(raw_text))

    # Quick NER pass just to get a confidence estimate
    extracted = medical_ner.extract(raw_text)
    return raw_text, extracted.confidence


def _extract_with_gemini_hybrid(
    image_bytes: bytes,
    filename: str,
    ocr_text: str,
    ocr_confidence: float,
) -> Optional[dict]:
    """
    Tier 2: Gemini hybrid extraction (text-only if confidence >= 0.7, else image+text).
    Uses the Gemini-specific light pipeline (EXIF rotation, sharpening, resize).
    Returns structured dict or None if Gemini is unavailable/fails.
    """
    try:
        from src.gemini_extractor import hybrid_extract_prescription, is_available
        if not is_available():
            return None
        # Pre-process for Gemini: EXIF correction, mild sharpening, resize
        gemini_bytes = preprocessor.preprocess_for_gemini(image_bytes, filename)
        return hybrid_extract_prescription(gemini_bytes, ocr_text, ocr_confidence)
    except Exception as e:
        logger.warning("Gemini hybrid extractor failed: %s", e)
        return None


def _extract_with_ner_fallback(image_bytes: bytes, filename: str, ocr_text: str) -> dict:
    """
    Tier 3: Local handwriting ML extraction fallback.
    First tries the local Donut ML model to parse the image.
    Falls back to rule-based NER on OCR text if ML model is unavailable or fails.
    """
    try:
        from src.ner_ml import extract_handwriting_ml
        ml_result = extract_handwriting_ml(image_bytes, filename)
        if ml_result:
            logger.info("Local handwriting ML extraction succeeded.")
            return ml_result
    except Exception as e:
        logger.debug("Local ML model unavailable or error: %s", e)

    logger.info("Using rule-based NER fallback on OCR text")
    extracted = medical_ner.extract(ocr_text)
    return {
        "medications":       [m.__dict__ for m in extracted.medications],
        "doctor_reg":        extracted.doctor_reg,
        "doctor_name":       None,
        "clinic_name":       None,
        "patient_name":      None,
        "patient_age":       extracted.patient_age,
        "patient_gender":    None,
        "diagnosis":         extracted.diagnosis,
        "prescription_date": extracted.prescription_date,
        "confidence":        extracted.confidence,
        "raw_text":          extracted.raw_text,
        "extractor_mode":    "tesseract_ner_fallback",
    }



def _dict_to_medication_extracted(med: dict) -> MedicationExtracted:
    """Convert a raw medication dict (from Gemini or Tesseract) to MedicationExtracted."""
    return MedicationExtracted(
        raw_drug_name=med.get("raw_drug_name") or med.get("drug_name", ""),
        dosage_value=med.get("dosage_value"),
        dosage_unit=med.get("dosage_unit"),
        frequency=med.get("frequency"),
        freq_per_day=med.get("freq_per_day"),
        duration_days=med.get("duration_days"),
        route=med.get("route", "oral"),
    )


def process_prescription(
    image_bytes: bytes,
    filename: str,
    phone: Optional[str] = None,
) -> PrescriptionOutput:
    """
    Full pipeline: image bytes → PrescriptionOutput.
    Writes to DB. Returns the output object regardless of confidence level.

    3-Tier Extraction Chain:
      Tier 1 — Tesseract OCR: Always runs first to produce raw_text + base confidence.
      Tier 2 — Gemini Hybrid: If API key is set, post-processes OCR text with LLM.
                               Uses text-only mode if confidence >= 0.7 (cheaper),
                               or image+text mode if confidence < 0.7 (more accurate).
      Tier 3 — Rule-Based NER Fallback: Used when Gemini is unavailable or fails.
    """
    prescription_id = str(uuid.uuid4())
    logger.info("Processing prescription_id=%s", prescription_id)

    # ── 1. Save image ──────────────────────────────────────────────────────
    image_path = _save_image(image_bytes, filename)

    # ── 2. Tier 1: Tesseract OCR (always runs — gives us raw text + confidence) ──
    ocr_text, tess_confidence = _run_tesseract_ocr(image_bytes, filename)
    logger.info(
        "Tesseract OCR: prescription_id=%s tess_confidence=%.4f text_len=%d",
        prescription_id, tess_confidence, len(ocr_text),
    )

    # ── 3. Tier 2: Gemini Hybrid post-processing ───────────────────────────
    extracted_dict = _extract_with_gemini_hybrid(image_bytes, filename, ocr_text, tess_confidence)
    extractor_used = extracted_dict.get("extractor_mode", "gemini_hybrid") if extracted_dict else None

    # ── 4. Tier 3: Rule-based NER fallback ────────────────────────────────
    if extracted_dict is None:
        extracted_dict = _extract_with_ner_fallback(image_bytes, filename, ocr_text)
        extractor_used = extracted_dict.get("extractor_mode", "tesseract_ner_fallback")


    # ── 5. Convert medication dicts to MedicationExtracted objects ─────────
    raw_meds = extracted_dict.get("medications", [])
    medication_objects: list[MedicationExtracted] = [
        _dict_to_medication_extracted(m) for m in raw_meds
        if (m.get("raw_drug_name") or m.get("drug_name", "")).strip()
    ]

    confidence = float(extracted_dict.get("confidence", 0.0))

    logger.info(
        "NER complete prescription_id=%s medications=%d confidence=%.4f extractor=%s",
        prescription_id, len(medication_objects), confidence, extractor_used,
    )

    # ── 6. Drug normalisation ──────────────────────────────────────────────
    normalized_drugs = drug_normalizer.normalize_all(medication_objects)

    # ── 7. DDI checking ────────────────────────────────────────────────────
    db = SessionLocal()
    try:
        patient = None
        existing_inn_list: list[str] = []

        if phone:
            patient = get_or_create_patient(db, phone)
            existing_meds = get_active_medications(db, patient.id)
            existing_inn_list = [m.inn for m in existing_meds if m.inn]

        new_inn_list = [m.inn for m in normalized_drugs if m.inn]
        all_active_inns = list(set(existing_inn_list + new_inn_list))
        interactions = ddi_checker.check_interactions(all_active_inns)

        # ── 7.5. Clinical safety check ─────────────────────────────────────
        safety_alerts = []
        if phone and patient:
            from src.clinical_checker import check_clinical_safety
            safety_alerts = check_clinical_safety(db, patient.id, normalized_drugs)

        # ── 8. Routing decision ────────────────────────────────────────────
        needs_human_review = confidence < CONFIDENCE_THRESHOLD
        has_major_interaction = any(i.severity == "major" for i in interactions)

        raw_text = extracted_dict.get("raw_text", "")

        if needs_human_review:
            reason = f"Confidence {confidence:.4f} < threshold {CONFIDENCE_THRESHOLD} (extractor={extractor_used})"
            save_to_review_queue(
                db=db,
                patient=patient,
                image_path=image_path,
                raw_ocr_text=raw_text,
                confidence=confidence,
                reason=reason,
            )
            logger.info(
                "prescription_id=%s routed to review_queue (confidence=%.4f)",
                prescription_id, confidence,
            )
        else:
            save_prescription_to_db(
                db=db,
                prescription_id=prescription_id,
                patient=patient,
                image_path=image_path,
                raw_ocr_text=raw_text,
                confidence=confidence,
                doctor_reg=extracted_dict.get("doctor_reg"),
                patient_age=extracted_dict.get("patient_age"),
                diagnosis=extracted_dict.get("diagnosis"),
                prescription_date=extracted_dict.get("prescription_date"),
                medications=normalized_drugs,
                interactions=interactions,
            )
            logger.info(
                "prescription_id=%s saved to main tables (confidence=%.4f)",
                prescription_id, confidence,
            )

        db.commit()

    except Exception as exc:
        db.rollback()
        logger.error("DB write failed for prescription_id=%s: %s", prescription_id, exc)
        raise
    finally:
        db.close()

    # ── 9. Build and return output ─────────────────────────────────────────
    return PrescriptionOutput(
        prescription_id=prescription_id,
        patient_phone=phone,
        medications=normalized_drugs,
        diagnosis=extracted_dict.get("diagnosis"),
        doctor_reg=extracted_dict.get("doctor_reg"),
        patient_age=extracted_dict.get("patient_age"),
        prescription_date=extracted_dict.get("prescription_date"),
        interactions=interactions,
        confidence=confidence,
        needs_human_review=needs_human_review,
        has_major_interaction=has_major_interaction,
        image_path=image_path,
        safety_alerts=safety_alerts,
    )

