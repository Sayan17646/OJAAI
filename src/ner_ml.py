"""
ner_ml.py — Local ML inference engine for handwriting prescription parsing.

Uses a fine-tuned Donut (VisionEncoderDecoderModel) model to extract structured
clinical fields directly from a prescription image without intermediate OCR.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from PIL import Image
import io


logger = logging.getLogger(__name__)

# Model loading config
_MODEL_DIR = os.getenv("DONUT_MODEL_PATH", "./models/donut-rx")
_PROCESSOR = None
_MODEL = None
TASK_START_TOKEN = "<s_rx>"
EOS_TOKEN = "</s>"


def _load_model_and_processor():
    """Lazily load the Donut model and processor into memory."""
    global _PROCESSOR, _MODEL
    if _PROCESSOR is not None and _MODEL is not None:
        return _PROCESSOR, _MODEL

    model_path = Path(_MODEL_DIR)
    if not model_path.exists():
        logger.warning(
            f"Local ML model directory not found at {model_path.absolute()}. "
            "Handwriting ML extraction will be unavailable."
        )
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    try:
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        import torch
        
        logger.info(f"Loading local ML handwriting model from {model_path}...")
        _PROCESSOR = DonutProcessor.from_pretrained(model_path)
        _MODEL = VisionEncoderDecoderModel.from_pretrained(model_path)
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL.to(device)
        _MODEL.eval()
        logger.info(f"ML handwriting model successfully loaded on device: {device}")
        
    except Exception as e:
        logger.error(f"Failed to load local ML handwriting model: {e}")
        _PROCESSOR = None
        _MODEL = None
        raise

    return _PROCESSOR, _MODEL


def _clean_donut_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardize the raw dict output from Donut to match our pipeline requirements.
    Extracts nested medication lists and handles default values.
    """
    normalized = {
        "doctor_name":       data.get("doctor_name"),
        "doctor_reg":        data.get("doctor_reg"),
        "clinic_name":       data.get("clinic_name"),
        "patient_name":      data.get("patient_name"),
        "patient_age":       data.get("patient_age"),
        "patient_gender":    data.get("patient_gender"),
        "prescription_date": data.get("prescription_date"),
        "diagnosis":         data.get("diagnosis"),
        "medications":       [],
        "confidence":        float(data.get("confidence", 0.75)),
        "raw_text":          json.dumps(data),
        "extractor_mode":    "local_donut_ml",
    }

    # Extract medications list
    meds = data.get("medications")
    if meds:
        # Handle cases where medications is a single dict instead of list
        if isinstance(meds, dict):
            meds = [meds]
        
        # In Donut, a list of dicts is often nested under elements or "el" list wrappers
        if isinstance(meds, list):
            for m in meds:
                if not isinstance(m, dict):
                    continue
                # Donut list elements are often wrapped in {"el": {...}} or similar
                if "el" in m and isinstance(m["el"], dict):
                    m = m["el"]
                
                normalized["medications"].append({
                    "raw_drug_name": m.get("drug_name") or m.get("raw_drug_name") or "",
                    "dosage_value":  m.get("dosage_value"),
                    "dosage_unit":   m.get("dosage_unit"),
                    "frequency":     m.get("frequency"),
                    "freq_per_day":  m.get("freq_per_day"),
                    "duration_days": m.get("duration_days"),
                    "route":         m.get("route", "oral"),
                    "instructions":  m.get("instructions"),
                })
    return normalized


def extract_handwriting_ml(image_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
    """
    Run local Donut handwriting extraction on raw image bytes.
    Returns a normalized dictionary compatible with the pipeline, or None if failed.
    """
    try:
        processor, model = _load_model_and_processor()
    except Exception:
        # Graceful fallback: warning logged inside loader
        return None

    try:
        import torch
        # Load PIL image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Prepare inputs
        device = next(model.parameters()).device
        pixel_values = processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        # Generate output tokens
        logger.info("Running local ML generation on image...")
        
        # Start generation with the prescription task token used during training.
        # Without this, Donut can fall back to its base <s_iitcdip> task token.
        task_prompt = TASK_START_TOKEN
        decoder_input_ids = processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)

        eos_token_id = processor.tokenizer.convert_tokens_to_ids(EOS_TOKEN)
        if eos_token_id == processor.tokenizer.unk_token_id:
            eos_token_id = processor.tokenizer.eos_token_id

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=512,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                use_cache=True,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        # Decode output string
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.pad_token or "", "").strip()
        if sequence.startswith(TASK_START_TOKEN):
            sequence = sequence[len(TASK_START_TOKEN):]
        sequence = sequence.replace(processor.tokenizer.eos_token or "", "")
        sequence = sequence.replace(EOS_TOKEN, "").strip()

        if not sequence or sequence == "<s_iitcdip>":
            logger.warning("Local Donut model returned base task token or empty sequence: %r", sequence)
            return None
        
        # Convert generated sequence back to JSON
        try:
            raw_dict = processor.token2json(sequence)
        except Exception as json_err:
            logger.error(f"Failed to parse generated Donut sequence to JSON. Sequence: {sequence!r}, Error: {json_err}")
            return None

        if not raw_dict:
            logger.warning("Local Donut model returned empty dictionary.")
            return None

        # Clean and map to pipeline schema
        result = _clean_donut_json(raw_dict)
        
        # Verify that the parsed result contains at least one non-empty structured field
        # to prevent returning an empty dict with a hardcoded high confidence when Donut fails.
        clinical_keys = [
            "doctor_name",
            "doctor_reg",
            "clinic_name",
            "patient_name",
            "patient_age",
            "patient_gender",
            "prescription_date",
            "diagnosis",
        ]
        has_clinical_field = any(result.get(key) is not None for key in clinical_keys)
        has_medications = bool(result.get("medications"))

        if not (has_clinical_field or has_medications):
            logger.warning("Local Donut model did not extract any structured clinical fields (fallback to rule-based NER).")
            return None

        return result

    except Exception as e:
        logger.error(f"Error during Donut handwriting extraction: {e}")
        return None
