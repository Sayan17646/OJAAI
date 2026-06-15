import os
import re
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Load API key from env/dotenv
from dotenv import load_dotenv
load_dotenv()

# Setup mock database before importing pipeline
db_mock = MagicMock()
sys.modules['src.database'] = db_mock
import src.database as db
db.SessionLocal = MagicMock()
db.get_or_create_patient = MagicMock()
db.get_active_medications = MagicMock(return_value=[])
db.save_prescription_to_db = MagicMock()
db.save_to_review_queue = MagicMock()

from src.pipeline import process_prescription
from src.drug_normalizer import _clean_name, INDIA_BRAND_MAP
from tests.evaluate_pipeline import parse_gt_medications

EVAL_DIR = Path("c:/Users/USER/Desktop/OJAAI/data/evaluation")
IMAGES_DIR = EVAL_DIR / "images"
ANNOTATIONS_DIR = EVAL_DIR / "annotations"

def run_audit():
    print("Running audit of strict mismatches...")
    annotations = sorted(list(ANNOTATIONS_DIR.glob("*.json")))[:100]
    
    audit_records = []
    
    for ann_path in annotations:
        base_name = ann_path.stem
        img_path = IMAGES_DIR / f"{base_name}.png"
        
        if not img_path.exists():
            continue
            
        with open(ann_path, "r", encoding="utf-8") as f:
            ann_data = json.load(f)
            
        gt_str = ann_data.get("ground_truth", "")
        gt_meds = parse_gt_medications(gt_str)
        
        if not gt_meds:
            continue
            
        img_bytes = img_path.read_bytes()
        try:
            output = process_prescription(img_bytes, img_path.name)
        except Exception as e:
            continue
            
        extracted_meds = output.medications
        ext_meds_cleaned = [_clean_name(m.raw_drug_name) for m in extracted_meds]
        
        for gt in gt_meds:
            gt_name = gt["raw_drug_name"]
            gt_cleaned = _clean_name(gt_name)
            
            # Check strict match
            strict_match = False
            matched_extraction = None
            
            for idx, ext_clean in enumerate(ext_meds_cleaned):
                if gt_cleaned in ext_clean or ext_clean in gt_cleaned:
                    strict_match = True
                    matched_extraction = extracted_meds[idx].raw_drug_name
                    break
            
            if not strict_match:
                # Find the closest extracted drug (if any) to categorize
                closest_ext = None
                is_substring = False
                in_brand_map = False
                
                # Check if there is any extraction at all
                if extracted_meds:
                    # Look for any substring relation or closest ratio
                    for ext_med in extracted_meds:
                        ext_name = ext_med.raw_drug_name
                        ext_clean = _clean_name(ext_name)
                        
                        # Substring check
                        if gt_cleaned in ext_clean or ext_clean in gt_cleaned:
                            is_substring = True
                            closest_ext = ext_name
                            break
                        
                        # Brand map check
                        # If the ground truth name is mapped to the extracted name in our INDIA_BRAND_MAP
                        # e.g., if one is brand and other is INN
                        gt_inn = INDIA_BRAND_MAP.get(gt_cleaned)
                        ext_inn = INDIA_BRAND_MAP.get(ext_clean)
                        if (gt_inn and gt_inn == ext_clean) or (ext_inn and ext_inn == gt_cleaned) or (gt_inn and ext_inn and gt_inn == ext_inn):
                            in_brand_map = True
                            closest_ext = ext_name
                            break
                    
                    if not closest_ext:
                        # Default to the first extracted drug just to display what was found
                        closest_ext = extracted_meds[0].raw_drug_name
                        
                audit_records.append({
                    "filename": ann_path.name,
                    "gt_name": gt_name,
                    "extracted_name": closest_ext or "[No Medications Extracted]",
                    "is_substring": is_substring,
                    "in_brand_map": in_brand_map
                })
                
    print(f"\n--- STRICT MISSES AUDIT (Total: {len(audit_records)}) ---")
    # Display the first 20 records for analysis
    for idx, rec in enumerate(audit_records[:30]):
        sub_str = "YES" if rec["is_substring"] else "NO"
        brand_str = "YES" if rec["in_brand_map"] else "NO"
        print(f"{idx+1:02d}. File: {rec['filename']} | GT: {rec['gt_name']:<15} | Extracted: {rec['extracted_name']:<25} | Substring?: {sub_str:<3} | In Brand Map?: {brand_str}")

if __name__ == "__main__":
    run_audit()
