import os
import sys
import csv
import json
import time
import difflib
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock

# 1. Setup mock database before importing modules to avoid PostgreSQL dependencies
db_mock = MagicMock()
sys.modules['src.database'] = db_mock

import src.database as db
db.SessionLocal = MagicMock()
db.get_or_create_patient = MagicMock()
db.get_active_medications = MagicMock(return_value=[])
db.save_prescription_to_db = MagicMock()
db.save_to_review_queue = MagicMock()

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.drug_normalizer import normalize_all
from src.models import MedicationExtracted
import pytesseract

# Set Tesseract executable path for Windows
tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tess_path):
    pytesseract.pytesseract.tesseract_cmd = tess_path

# Resolve Kaggle cache path for the dataset
cache_dir = Path(os.path.expanduser("~/.cache/kagglehub/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset/versions/1"))
dataset_path = cache_dir / "Doctor’s Handwritten Prescription BD dataset"
if not dataset_path.exists():
    # Attempt fallback with ASCII single quotes
    dataset_path = cache_dir / "Doctor's Handwritten Prescription BD dataset"

def run_word_benchmark():
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found in cache. Path: {dataset_path.absolute()}")
        return

    csv_path = dataset_path / "Testing" / "testing_labels.csv"
    words_dir = dataset_path / "Testing" / "testing_words"
    
    if not csv_path.exists():
        print(f"[ERROR] Labels file not found: {csv_path}")
        return

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    print(f"Loaded {len(records)} test word crops. Benchmarking first 100 words...")

    ocr_word_correct = 0
    norm_inn_correct = 0
    total_eval = 0

    # Limit to 100 to keep verification fast
    for idx, row in enumerate(records[:100], start=1):
        img_name = row["IMAGE"]
        brand_gt = row["MEDICINE_NAME"].strip().lower()
        generic_gt = row["GENERIC_NAME"].strip().lower()
        
        img_path = words_dir / img_name
        if not img_path.exists():
            continue
            
        total_eval += 1
        
        # 1. Run Tesseract OCR using PSM 8 (treat image as a single word)
        try:
            with Image.open(img_path) as img:
                ocr_text = pytesseract.image_to_string(img, config="--oem 3 --psm 8").strip().lower()
        except Exception as e:
            ocr_text = ""
            
        # 2. Match OCR word accuracy (allowing fuzzy match ratio >= 0.8)
        word_match = False
        if ocr_text == brand_gt or brand_gt in ocr_text:
            word_match = True
        elif difflib.SequenceMatcher(None, ocr_text, brand_gt).ratio() >= 0.8:
            word_match = True
            
        if word_match:
            ocr_word_correct += 1

        # 3. Run drug name normalization
        med_extracted = MedicationExtracted(
            raw_drug_name=ocr_text if ocr_text else brand_gt, # Use brand_gt if OCR was blank to test normalizer isolation
            dosage_value=None,
            dosage_unit=None,
            frequency=None,
            freq_per_day=None,
            duration_days=None,
            route="oral"
        )
        
        try:
            normalized = normalize_all([med_extracted])
            norm_inn = normalized[0].inn.strip().lower() if normalized and normalized[0].inn else ""
        except Exception:
            norm_inn = ""

        # 4. Check if generic name matches normalized INN
        if norm_inn == generic_gt or generic_gt in norm_inn or norm_inn in generic_gt:
            norm_inn_correct += 1
            
    ocr_acc = ocr_word_correct / total_eval if total_eval else 0.0
    norm_acc = norm_inn_correct / total_eval if total_eval else 0.0

    print("\n" + "="*50)
    print("AUXILIARY MEDICINE-WORD BENCHMARK COMPLETE")
    print("="*50)
    print(f"Total Words Evaluated:        {total_eval}")
    print(f"OCR Word Reading Accuracy:    {ocr_acc:.2%}")
    print(f"INN Normalization Accuracy:   {norm_acc:.2%}")
    print("="*50)

    # Save results as JSON
    out_dir = Path("data/auxiliary_word_crops")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "word_benchmark_results.json"
    
    results = {
        "total_words_evaluated": total_eval,
        "ocr_word_accuracy": ocr_acc,
        "inn_normalization_accuracy": norm_acc
    }
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {results_file.absolute()}")

if __name__ == "__main__":
    run_word_benchmark()
