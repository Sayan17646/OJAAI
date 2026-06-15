"""
gemini_extractor.py — Hybrid Tesseract + Gemini LLM extraction engine.

Architecture:
  Tesseract OCR runs first to produce raw text + confidence.
  Then this module post-processes with Gemini LLM:
    - If OCR confidence >= 0.7: sends TEXT-ONLY to Gemini (cheaper, faster)
    - If OCR confidence <  0.7: sends IMAGE + TEXT to Gemini (cross-references visual cues)

  Falls back gracefully if GEMINI_API_KEY is not set or the API errors out.

Models:
  - Default:  GEMINI_MODEL      env var → gemini-2.5-flash  (fast, cheap)
  - Upgrade:  GEMINI_PRO_MODEL  env var → gemini-2.5-pro    (complex reasoning)

Both prescription and lab report extraction are supported.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_PRO_MODEL = os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-pro")

# Fallback model chain: tried in order when the primary model's daily quota is exhausted.
# Only triggered on RESOURCE_EXHAUSTED (quota), not on rate-limit 429s (those are retried).
GEMINI_FALLBACK_MODELS = [
    GEMINI_MODEL,
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

# Confidence threshold for deciding text-only vs image+text mode
_HYBRID_IMAGE_THRESHOLD = 0.7


# ── Prompts ───────────────────────────────────────────────────────────────────

_RX_TEXT_ONLY_PROMPT = """You are a clinical data extraction assistant specializing in Indian prescriptions.

Below is raw OCR text extracted from a prescription image. The OCR may contain errors, misspellings, or formatting artifacts. Your job is to intelligently parse it and extract structured data.

OCR TEXT:
---
{ocr_text}
---

Return ONLY a valid JSON object with this exact structure (no markdown, no explanation):
{{
  "doctor_name": "<name or null>",
  "doctor_reg": "<registration/license number or null>",
  "clinic_name": "<clinic or hospital name or null>",
  "patient_name": "<full name or null>",
  "patient_age": "<age as string e.g. '45 yrs' or null>",
  "patient_gender": "<Male/Female/Other or null>",
  "prescription_date": "<date as DD/MM/YYYY or null>",
  "diagnosis": "<diagnosis/condition or null>",
  "medications": [
    {{
      "drug_name": "<exact drug name as written>",
      "dosage_value": <numeric dose value or null>,
      "dosage_unit": "<mg/ml/mcg/% etc or null>",
      "frequency": "<e.g. once daily, twice daily, 1-0-1 etc or null>",
      "freq_per_day": <integer times per day or null>,
      "duration_days": <number of days or null>,
      "route": "<oral/topical/ophthalmic/iv/im/inhaled/nasal/otic or 'oral'>",
      "instructions": "<any special instructions e.g. take with food, before bed>"
    }}
  ],
  "confidence": <your confidence in the extraction as float 0.0 to 1.0>
}}

Rules:
- Extract EVERY medication listed, even if partially legible.
- For freq_per_day: OD/once=1, BD/twice=2, TDS/TID/three times=3, QID=4, SOS/PRN/as needed=0.
- Indian frequency patterns: 1-0-1 means twice daily (freq_per_day=2), 1-1-1 means three times (3).
- Common Indian abbreviations: Tab=Tablet, Cap=Capsule, Inj=Injection, Syr=Syrup, ON=at night.
- If you cannot read something clearly, make your best guess and lower the confidence score.
- NEVER include markdown, code fences, or explanation. Return raw JSON only.
"""

_RX_IMAGE_TEXT_PROMPT = """You are a clinical data extraction assistant specializing in Indian prescriptions.

I am providing you with BOTH the original prescription image AND the raw OCR text extracted from it. The OCR text may be incomplete or inaccurate — use the image to verify and supplement the text.

OCR TEXT (may contain errors):
---
{ocr_text}
---

Cross-reference the image with the OCR text and extract ALL information accurately.

Return ONLY a valid JSON object with this exact structure (no markdown, no explanation):
{{
  "doctor_name": "<name or null>",
  "doctor_reg": "<registration/license number or null>",
  "clinic_name": "<clinic or hospital name or null>",
  "patient_name": "<full name or null>",
  "patient_age": "<age as string e.g. '45 yrs' or null>",
  "patient_gender": "<Male/Female/Other or null>",
  "prescription_date": "<date as DD/MM/YYYY or null>",
  "diagnosis": "<diagnosis/condition or null>",
  "medications": [
    {{
      "drug_name": "<exact drug name as written>",
      "dosage_value": <numeric dose value or null>,
      "dosage_unit": "<mg/ml/mcg/% etc or null>",
      "frequency": "<e.g. once daily, twice daily, 1-0-1 etc or null>",
      "freq_per_day": <integer times per day or null>,
      "duration_days": <number of days or null>,
      "route": "<oral/topical/ophthalmic/iv/im/inhaled/nasal/otic or 'oral'>",
      "instructions": "<any special instructions e.g. take with food, before bed>"
    }}
  ],
  "confidence": <your confidence in the extraction as float 0.0 to 1.0>
}}

Rules:
- Extract EVERY medication listed, even if partially legible.
- For freq_per_day: OD/once=1, BD/twice=2, TDS/TID/three times=3, QID=4, SOS/PRN/as needed=0.
- Indian frequency patterns: 1-0-1 means twice daily (freq_per_day=2), 1-1-1 means three times (3).
- Common Indian abbreviations: Tab=Tablet, Cap=Capsule, Inj=Injection, Syr=Syrup, ON=at night.
- Prefer what you SEE in the image over the OCR text if they conflict.
- NEVER include markdown, code fences, or explanation. Return raw JSON only.
"""

_LAB_TEXT_ONLY_PROMPT = """You are a clinical data extraction assistant specializing in Indian diagnostic lab reports.

Below is raw OCR text extracted from a lab report image. The OCR may contain errors. Parse it intelligently.

OCR TEXT:
---
{ocr_text}
---

Return ONLY a valid JSON object with this exact structure (no markdown, no explanation):
{{
  "lab_name": "<laboratory name or 'Unknown Diagnostics'>",
  "patient_name": "<patient name or null>",
  "patient_age": "<age string or null>",
  "patient_gender": "<Male/Female/Other or null>",
  "report_date": "<date as DD/MM/YYYY or null>",
  "referring_doctor": "<referring doctor name or null>",
  "results": [
    {{
      "test_name": "<exact test name as written>",
      "analyte_name": "<normalized: HEMOGLOBIN, TSH, HBA1C, CREATININE, LDL, HDL, TRIGLYCERIDES, FASTING_BLOOD_SUGAR, MCV, MCH, MCHC, RDW, PCV, TLC, PLATELET_COUNT, NEUTROPHILS, LYMPHOCYTES, or OTHER>",
      "value": <numeric value as float>,
      "unit": "<unit string>",
      "ref_range": "<reference range as written e.g. '13.0-17.0' or null>",
      "flag": "<HIGH/LOW/NORMAL based on reference range>"
    }}
  ],
  "clinical_notes": "<any advisor notes, conclusions, or observations>",
  "confidence": <your confidence in extraction as float 0.0 to 1.0>
}}

Rules:
- Extract EVERY test result present in the report.
- Determine flag (HIGH/LOW/NORMAL) by comparing value to reference range shown.
- For analyte_name use the standardized uppercase names listed above, or OTHER if not listed.
- NEVER include markdown, code fences, or explanation. Return raw JSON only.
"""

_LAB_IMAGE_TEXT_PROMPT = """You are a clinical data extraction assistant specializing in Indian diagnostic lab reports.

I am providing you with BOTH the original lab report image AND the raw OCR text. The OCR text may be incomplete — use the image to verify and supplement.

OCR TEXT (may contain errors):
---
{ocr_text}
---

Cross-reference the image with the OCR text and extract ALL information accurately.

Return ONLY a valid JSON object with this exact structure (no markdown, no explanation):
{{
  "lab_name": "<laboratory name or 'Unknown Diagnostics'>",
  "patient_name": "<patient name or null>",
  "patient_age": "<age string or null>",
  "patient_gender": "<Male/Female/Other or null>",
  "report_date": "<date as DD/MM/YYYY or null>",
  "referring_doctor": "<referring doctor name or null>",
  "results": [
    {{
      "test_name": "<exact test name as written>",
      "analyte_name": "<normalized: HEMOGLOBIN, TSH, HBA1C, CREATININE, LDL, HDL, TRIGLYCERIDES, FASTING_BLOOD_SUGAR, MCV, MCH, MCHC, RDW, PCV, TLC, PLATELET_COUNT, NEUTROPHILS, LYMPHOCYTES, or OTHER>",
      "value": <numeric value as float>,
      "unit": "<unit string>",
      "ref_range": "<reference range as written e.g. '13.0-17.0' or null>",
      "flag": "<HIGH/LOW/NORMAL based on reference range>"
    }}
  ],
  "clinical_notes": "<any advisor notes, conclusions, or observations>",
  "confidence": <your confidence in extraction as float 0.0 to 1.0>
}}

Rules:
- Extract EVERY test result present in the report.
- Determine flag (HIGH/LOW/NORMAL) by comparing value to reference range shown.
- Prefer what you SEE in the image over the OCR text if they conflict.
- NEVER include markdown, code fences, or explanation. Return raw JSON only.
"""


# ── Client & Helpers ──────────────────────────────────────────────────────────

def _get_client():
    """Create and return a google.genai.Client."""
    from google import genai
    return genai.Client(api_key=GEMINI_API_KEY)


def _parse_json_response(raw: str) -> Optional[dict]:
    """Clean up and parse a JSON response from Gemini."""
    if not raw:
        return None
    # Strip any accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Gemini returned non-JSON response: %s", e)
        logger.debug("Raw Gemini response (first 500 chars): %s", raw[:500])
        return None


def _call_gemini_text_only(prompt: str, model_name: str = None) -> Optional[dict]:
    """Send a text-only prompt to Gemini and return parsed JSON.
    
    Tries the primary model first, then falls back through GEMINI_FALLBACK_MODELS
    if the primary model's daily quota (RESOURCE_EXHAUSTED) is hit.
    """
    if not GEMINI_API_KEY:
        return None

    import time
    max_retries = 3

    # Build the model chain: explicit override first, then the configured fallbacks
    models_to_try = [model_name] if model_name else list(GEMINI_FALLBACK_MODELS)

    for model in models_to_try:
        delay = 2.0
        quota_exhausted = False
        for attempt in range(max_retries):
            try:
                client = _get_client()
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "temperature": 0.1,
                        "max_output_tokens": 4096,
                        "response_mime_type": "application/json",
                    },
                )
                logger.info("Gemini text-only succeeded with model=%s", model)
                return _parse_json_response(response.text)
            except Exception as e:
                err_msg = str(e)
                err_lower = err_msg.lower()
                # Daily quota exhausted — no point retrying same model
                is_quota_exhausted = (
                    "GenerateRequestsPerDayPerProjectPerModel" in err_msg
                    or ("resource_exhausted" in err_lower and "per_day" in err_lower.replace(" ", "_"))
                )
                is_transient = "429" in err_msg or "503" in err_lower or "unavailable" in err_lower
                if is_quota_exhausted:
                    logger.warning("Model %s daily quota exhausted — trying next fallback model.", model)
                    quota_exhausted = True
                    break
                elif is_transient and attempt < max_retries - 1:
                    logger.warning("Gemini text-only call failed (attempt %d/%d) with transient error, retrying in %.1fs: %s", attempt + 1, max_retries, delay, e)
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    logger.error("Gemini text-only call failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
                    return None
        if quota_exhausted:
            continue  # try next model in chain
    logger.error("All Gemini models exhausted quota or failed for text-only call.")
    return None


def _call_gemini_image_text(
    image_bytes: bytes,
    prompt: str,
    model_name: str = None,
) -> Optional[dict]:
    """Send image + text prompt to Gemini and return parsed JSON.
    
    Tries the primary model first, then falls back through GEMINI_FALLBACK_MODELS
    if the primary model's daily quota (RESOURCE_EXHAUSTED) is hit.
    """
    if not GEMINI_API_KEY:
        return None

    import time
    max_retries = 3

    models_to_try = [model_name] if model_name else list(GEMINI_FALLBACK_MODELS)

    for model in models_to_try:
        delay = 2.0
        quota_exhausted = False
        for attempt in range(max_retries):
            try:
                from google.genai import types
                client = _get_client()

                # Determine mime type from file header
                mime_type = "image/jpeg"  # default (our preprocessor always outputs JPEG)
                if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
                    mime_type = "image/png"
                elif image_bytes.startswith(b'%PDF'):
                    mime_type = "application/pdf"

                image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                text_part = types.Part.from_text(text=prompt)

                response = client.models.generate_content(
                    model=model,
                    contents=[image_part, text_part],
                    config={
                        "temperature": 0.1,
                        "max_output_tokens": 4096,
                        "response_mime_type": "application/json",
                    },
                )
                logger.info("Gemini image+text succeeded with model=%s", model)
                return _parse_json_response(response.text)
            except Exception as e:
                err_msg = str(e)
                err_lower = err_msg.lower()
                is_quota_exhausted = (
                    "GenerateRequestsPerDayPerProjectPerModel" in err_msg
                    or ("resource_exhausted" in err_lower and "per_day" in err_lower.replace(" ", "_"))
                )
                is_transient = "429" in err_msg or "503" in err_lower or "unavailable" in err_lower
                if is_quota_exhausted:
                    logger.warning("Model %s daily quota exhausted — trying next fallback model.", model)
                    quota_exhausted = True
                    break
                elif is_transient and attempt < max_retries - 1:
                    logger.warning("Gemini image+text call failed (attempt %d/%d) with transient error, retrying in %.1fs: %s", attempt + 1, max_retries, delay, e)
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    logger.error("Gemini image+text call failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
                    return None
        if quota_exhausted:
            continue
    logger.error("All Gemini models exhausted quota or failed for image+text call.")
    return None


# ── Hybrid Extraction: Prescriptions ──────────────────────────────────────────

def hybrid_extract_prescription(
    image_bytes: bytes,
    ocr_text: str,
    ocr_confidence: float,
) -> Optional[dict]:
    """
    Hybrid extraction: uses Tesseract OCR output + Gemini LLM post-processing.
    Always uses image + text mode for maximum accuracy.
    Falls back through GEMINI_FALLBACK_MODELS automatically on quota exhaustion.
    """
    if not GEMINI_API_KEY:
        logger.info("GEMINI_API_KEY not set — hybrid extraction unavailable.")
        return None

    mode = "image_and_text"
    logger.info(
        "Hybrid Rx extraction: mode=%s (ocr_confidence=%.2f)",
        mode, ocr_confidence,
    )
    prompt = _RX_IMAGE_TEXT_PROMPT.format(ocr_text=ocr_text)

    # image_bytes has already been pre-processed by pipeline.py via preprocess_for_gemini()
    result = _call_gemini_image_text(image_bytes, prompt)

    if not result:
        logger.warning("Hybrid Rx extraction returned no result (mode=%s)", mode)
        return None

    return _normalize_rx_result(result, mode)


def _normalize_rx_result(result: dict, mode: str) -> dict:
    """Convert Gemini's JSON response into a pipeline-compatible dict."""
    medications = []
    for med in result.get("medications", []):
        medications.append({
            "raw_drug_name": med.get("drug_name", ""),
            "dosage_value": med.get("dosage_value"),
            "dosage_unit": med.get("dosage_unit"),
            "frequency": med.get("frequency"),
            "freq_per_day": med.get("freq_per_day"),
            "duration_days": med.get("duration_days"),
            "route": med.get("route", "oral"),
            "instructions": med.get("instructions"),
        })

    return {
        "medications":       medications,
        "doctor_reg":        result.get("doctor_reg"),
        "doctor_name":       result.get("doctor_name"),
        "clinic_name":       result.get("clinic_name"),
        "patient_name":      result.get("patient_name"),
        "patient_age":       result.get("patient_age"),
        "patient_gender":    result.get("patient_gender"),
        "diagnosis":         result.get("diagnosis"),
        "prescription_date": result.get("prescription_date"),
        "confidence":        float(result.get("confidence", 0.8)),
        "raw_text":          json.dumps(result),
        "extractor_mode":    f"gemini_hybrid_{mode}",
    }


# ── Hybrid Extraction: Lab Reports ───────────────────────────────────────────

def hybrid_extract_lab_report(
    image_bytes: bytes,
    ocr_text: str,
    ocr_confidence: float,
) -> Optional[dict]:
    """
    Hybrid extraction for lab reports.
    Always uses image + text mode for maximum accuracy.
    """
    if not GEMINI_API_KEY:
        logger.info("GEMINI_API_KEY not set — hybrid lab extraction unavailable.")
        return None

    mode = "image_and_text"
    logger.info(
        "Hybrid Lab extraction: mode=%s (ocr_confidence=%.2f)",
        mode, ocr_confidence,
    )
    prompt = _LAB_IMAGE_TEXT_PROMPT.format(ocr_text=ocr_text)

    from src.preprocessor import preprocess_for_gemini
    try:
        light_image_bytes = preprocess_for_gemini(image_bytes, "lab_report.png")
    except Exception as e:
        logger.warning("Light preprocessing for Gemini failed, using raw bytes: %s", e)
        light_image_bytes = image_bytes

    result = _call_gemini_image_text(light_image_bytes, prompt)

    if not result:
        logger.warning("Hybrid Lab extraction returned no result (mode=%s)", mode)
        return None

    return _normalize_lab_result(result, mode)


def _normalize_lab_result(result: dict, mode: str) -> dict:
    """Convert Gemini's JSON response into a pipeline-compatible dict."""
    results = []
    for r in result.get("results", []):
        try:
            val = float(r.get("value", 0))
        except (TypeError, ValueError):
            continue

        flag = str(r.get("flag", "NORMAL")).lower()
        if flag not in ("high", "low", "normal"):
            flag = "normal"

        results.append({
            "raw_name":     r.get("test_name", ""),
            "analyte_name": r.get("analyte_name", "OTHER"),
            "value":        val,
            "unit":         r.get("unit"),
            "ref_range":    r.get("ref_range"),
            "flag":         flag,
        })

    return {
        "lab_name":        result.get("lab_name", "Unknown Diagnostics"),
        "patient_name":    result.get("patient_name"),
        "patient_age":     result.get("patient_age"),
        "report_date":     result.get("report_date"),
        "results":         results,
        "clinical_notes":  result.get("clinical_notes"),
        "confidence":      float(result.get("confidence", 0.8)),
        "raw_text":        json.dumps(result),
        "extractor_mode":  f"gemini_hybrid_{mode}",
    }


# ── Legacy Pure-Vision Extraction (backward compat) ──────────────────────────

def extract_prescription(image_bytes: bytes) -> Optional[dict]:
    """
    Pure vision extraction: sends image directly to Gemini (no OCR text).
    Kept for backward compatibility. Prefer hybrid_extract_prescription().
    """
    if not GEMINI_API_KEY:
        return None

    prompt = _RX_IMAGE_TEXT_PROMPT.format(ocr_text="[No OCR text available]")
    result = _call_gemini_image_text(image_bytes, prompt)
    if not result:
        return None
    return _normalize_rx_result(result, "pure_vision")


def extract_lab_report(image_bytes: bytes) -> Optional[dict]:
    """
    Pure vision extraction for lab reports.
    Kept for backward compatibility. Prefer hybrid_extract_lab_report().
    """
    if not GEMINI_API_KEY:
        return None

    prompt = _LAB_IMAGE_TEXT_PROMPT.format(ocr_text="[No OCR text available]")
    result = _call_gemini_image_text(image_bytes, prompt)
    if not result:
        return None
    return _normalize_lab_result(result, "pure_vision")


# ── Availability Check ────────────────────────────────────────────────────────

def is_available() -> bool:
    """Return True if Gemini is configured and the google-genai package is installed."""
    if not GEMINI_API_KEY:
        return False
    try:
        from google import genai  # noqa: F401
        return True
    except ImportError:
        return False
