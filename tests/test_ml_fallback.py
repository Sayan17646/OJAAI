"""
test_ml_fallback.py — Test suite for the local ML handwriting fallback routing and ner_ml.py.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
import io

# Setup mock database first to avoid import-time database errors
db_mock = MagicMock()
sys.modules['src.database'] = db_mock

# Mock deep learning libraries so tests can run without them installed
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()

import src.database as db
db.SessionLocal = MagicMock()
db.get_or_create_patient = MagicMock()
db.get_active_medications = MagicMock(return_value=[])
db.save_prescription_to_db = MagicMock()
db.save_to_review_queue = MagicMock()

import torch  # import mocked torch for testing context

from src import ner_ml
from src.pipeline import _extract_with_ner_fallback, process_prescription


def test_clean_donut_json():
    """Verify that _clean_donut_json cleans and normalizes raw Donut outputs."""
    raw_donut_output = {
        "doctor_reg": "MCI-1234",
        "diagnosis": "Hypertension",
        "medications": [
            {
                "el": {
                    "drug_name": "Amlodipine",
                    "dosage_value": 5,
                    "dosage_unit": "mg",
                    "frequency": "once daily"
                }
            }
        ]
    }
    
    cleaned = ner_ml._clean_donut_json(raw_donut_output)
    
    assert cleaned["doctor_reg"] == "MCI-1234"
    assert cleaned["diagnosis"] == "Hypertension"
    assert len(cleaned["medications"]) == 1
    assert cleaned["medications"][0]["raw_drug_name"] == "Amlodipine"
    assert cleaned["medications"][0]["dosage_value"] == 5
    assert cleaned["medications"][0]["dosage_unit"] == "mg"
    assert cleaned["extractor_mode"] == "local_donut_ml"


@patch("src.ner_ml._load_model_and_processor")
def test_extract_handwriting_ml_success(mock_load):
    """Test successful local ML extraction with mocked model/processor."""
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_load.return_value = (mock_processor, mock_model)
    
    # Mock pre-processing inputs
    mock_processor.return_value.pixel_values = MagicMock()
    mock_processor.tokenizer.return_value.input_ids = MagicMock()
    
    # Mock model generation output
    mock_model.parameters.return_value = iter([MagicMock(device="cpu")])
    mock_outputs = MagicMock()
    mock_outputs.sequences = [[1, 2, 3]]
    mock_model.generate.return_value = mock_outputs
    
    # Mock sequence decoding
    mock_processor.batch_decode.return_value = ["<s><s_doctor_reg>MCI-555</s_doctor_reg></s>"]
    mock_processor.tokenizer.eos_token = "</s>"
    mock_processor.tokenizer.pad_token = "<pad>"
    
    # Mock token2json conversion
    mock_processor.token2json.return_value = {
        "doctor_reg": "MCI-555",
        "medications": [{"drug_name": "Metformin", "dosage_value": 500, "dosage_unit": "mg"}]
    }
    
    dummy_image = Image.new("RGB", (100, 100))
    img_byte_arr = io.BytesIO()
    dummy_image.save(img_byte_arr, format="PNG")
    image_bytes = img_byte_arr.getvalue()
    
    result = ner_ml.extract_handwriting_ml(image_bytes, "test.png")
    
    assert result is not None
    assert result["doctor_reg"] == "MCI-555"
    assert len(result["medications"]) == 1
    assert result["medications"][0]["raw_drug_name"] == "Metformin"
    assert result["extractor_mode"] == "local_donut_ml"


@patch("src.ner_ml.extract_handwriting_ml")
def test_extract_with_ner_fallback_routing(mock_ml_extract):
    """Verify routing logic: tries ML handwriting model first, falls back to Tesseract + Regex NER if ML fails."""
    # 1. Simulate ML Success
    mock_ml_extract.return_value = {
        "doctor_reg": "MCI-8888",
        "medications": [{"drug_name": "Amlodipine", "dosage_value": 5, "dosage_unit": "mg"}],
        "extractor_mode": "local_donut_ml"
    }
    
    res = _extract_with_ner_fallback(b"fake_bytes", "test.png", "raw text")
    assert res["extractor_mode"] == "local_donut_ml"
    assert res["doctor_reg"] == "MCI-8888"
    
    # 2. Simulate ML failure (returns None) -> should fall back to Tesseract + regex NER
    mock_ml_extract.return_value = None
    
    # Simple regex NER input that contains a doctor registration tag and a dosage line
    mock_ocr_text = "Dr. Regist: MCI 12345\nMetformin 500mg OD\n"
    
    res = _extract_with_ner_fallback(b"fake_bytes", "test.png", mock_ocr_text)
    assert res["extractor_mode"] == "tesseract_ner_fallback"
    assert res["doctor_reg"] == "12345"
    assert len(res["medications"]) == 1
    assert res["medications"][0]["raw_drug_name"] == "Metformin"
