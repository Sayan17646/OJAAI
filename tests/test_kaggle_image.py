"""
test_kaggle_image.py — Test script to process a real handwritten prescription from the Kaggle dataset.

Usage:
  python -m tests.test_kaggle_image
"""

import os
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock

from unittest.mock import patch

# Now import pipeline
from src.pipeline import process_prescription

@patch("src.pipeline._try_get_existing_prescription")
@patch("src.pipeline.save_prescription_to_db")
@patch("src.pipeline.get_or_create_patient")
@patch("src.pipeline.get_active_medications")
@patch("src.pipeline.save_to_review_queue")
def test_single_kaggle_prescription(mock_save_queue, mock_get_meds, mock_get_patient, mock_save_db, mock_get_existing):
    mock_get_meds.return_value = []
    mock_get_existing.return_value = None
    # Path to dataset image
    dataset_base = Path(os.path.expanduser("~/.cache/kagglehub/datasets/mehaksingal/illegible-medical-prescription-images-dataset/versions/1/data"))
    img_path = dataset_base / "10.jpg"
    
    if not img_path.exists():
        print(f"Image not found at {img_path}")
        return
        
    print(f"Loading Kaggle prescription image: {img_path.name}")
    image_bytes = img_path.read_bytes()
    
    print("Running prescription through processing pipeline...")
    try:
        output = process_prescription(image_bytes, img_path.name)
    except Exception as e:
        print(f"Failed to process prescription: {e}")
        return
        
    print("\n================== PIPELINE OUTPUT ==================")
    print(f"Prescription ID:      {output.prescription_id}")
    print(f"Doctor Registration:  {output.doctor_reg}")
    print(f"Patient Age:          {output.patient_age}")
    print(f"Diagnosis:            {output.diagnosis}")
    print(f"Prescription Date:    {output.prescription_date}")
    print(f"Confidence Score:     {output.confidence:.4f}")
    print(f"Needs Human Review:   {output.needs_human_review}")
    
    print("\nExtracted Medications:")
    if not output.medications:
        print("  None")
    for i, med in enumerate(output.medications, 1):
        print(f"  {i}. {med.raw_drug_name}")
        print(f"     INN Normalisation:  {med.inn}")
        print(f"     Dosage:            {med.dosage_value} {med.dosage_unit}")
        print(f"     Frequency:         {med.frequency} (per day: {med.freq_per_day})")
        print(f"     Duration:          {med.duration_days} days")
        print(f"     Route:             {med.route}")
        
    print("\nDrug-Drug Interactions (DDI):")
    if not output.interactions:
        print("  None")
    for i, ddi in enumerate(output.interactions, 1):
        print(f"  {i}. Drugs: {ddi.drug1} + {ddi.drug2} ({ddi.severity})")
        print(f"     Description: {ddi.description}")
        
    print("=====================================================")

if __name__ == "__main__":
    test_single_kaggle_prescription()
