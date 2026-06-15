import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

def validate_entry(entry: Dict[str, Any], dataset_dir: Path, line_no: int) -> bool:
    file_name = entry.get("file_name")
    gt_str = entry.get("ground_truth")

    if not file_name:
        print(f"[ERROR] Line {line_no}: Missing 'file_name' key.")
        return False
        
    if not gt_str:
        print(f"[ERROR] Line {line_no} ({file_name}): Missing 'ground_truth' key.")
        return False

    # Check if image file exists
    image_path = dataset_dir / file_name
    if not image_path.exists():
        print(f"[ERROR] Line {line_no}: Image path does not exist: {image_path.absolute()}")
        return False

    # Parse ground truth JSON
    try:
        gt = json.loads(gt_str)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Line {line_no} ({file_name}): Invalid JSON string in ground_truth: {e}")
        return False

    # Validate ground truth fields
    clinical_keys = [
        "doctor_name", "doctor_reg", "clinic_name", 
        "patient_name", "patient_age", "patient_gender", 
        "prescription_date", "diagnosis"
    ]
    has_clinical_field = any(gt.get(key) is not None and str(gt.get(key)).strip() != "" for key in clinical_keys)
    
    meds = gt.get("medications", [])
    if not isinstance(meds, list):
        print(f"[ERROR] Line {line_no} ({file_name}): 'medications' field must be a list.")
        return False
        
    has_medications = len(meds) > 0

    if not (has_clinical_field or has_medications):
        print(f"[ERROR] Line {line_no} ({file_name}): Rejecting entry because all clinical fields and medications list are empty/null.")
        return False

    for idx, med in enumerate(meds, start=1):
        if not isinstance(med, dict):
            print(f"[ERROR] Line {line_no} ({file_name}): Medication entry #{idx} is not a dictionary.")
            return False
        
        drug_name = med.get("drug_name")
        if not drug_name or str(drug_name).strip() == "":
            print(f"[ERROR] Line {line_no} ({file_name}): Medication entry #{idx} is missing required 'drug_name'.")
            return False

    return True

def main():
    parser = argparse.ArgumentParser(description="Validate a dataset's metadata.jsonl and image directories.")
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        required=True,
        help="Path to the dataset directory (containing metadata.jsonl and images/)"
    )
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    metadata_file = dataset_dir / "metadata.jsonl"
    
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory does not exist: {dataset_dir.absolute()}")
        sys.exit(1)
        
    if not metadata_file.exists():
        print(f"[ERROR] metadata.jsonl not found in {dataset_dir.absolute()}")
        sys.exit(1)

    print(f"Validating dataset at: {dataset_dir.absolute()}")
    
    valid_count = 0
    errors = 0
    
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_no}: Invalid JSON: {e}")
                errors += 1
                continue
                
            if validate_entry(entry, dataset_dir, line_no):
                valid_count += 1
            else:
                errors += 1

    print("\n" + "="*40)
    if errors > 0:
        print(f"[FAIL] Validation FAILED! Found {errors} error(s). Valid records: {valid_count}.")
        sys.exit(1)
    else:
        print(f"[SUCCESS] Validation SUCCEEDED! All {valid_count} records are valid.")
        sys.exit(0)

if __name__ == "__main__":
    main()
