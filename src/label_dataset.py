import os
import json
import shutil
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

BASE_DIR = Path("data/real_prescriptions")
UNLABELED_DIR = BASE_DIR / "unlabeled/images"
EVAL_DIR = BASE_DIR / "evaluation"
TRAIN_DIR = BASE_DIR / "training"

def get_input(prompt: str, default: str = "") -> str:
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default

def open_image(path: Path):
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", str(path)], check=True)
        else:
            subprocess.run(["xdg-open", str(path)], check=True)
    except Exception as e:
        print(f"Warning: Could not open image automatically: {e}")

def validate_clinical_data(data: Dict[str, Any]) -> bool:
    # Check if at least one clinical detail is populated
    clinical_keys = [
        "doctor_name", "doctor_reg", "clinic_name", 
        "patient_name", "patient_age", "patient_gender", 
        "prescription_date", "diagnosis"
    ]
    has_clinical = any(data.get(key) for key in clinical_keys)
    has_meds = len(data.get("medications", [])) > 0
    return has_clinical or has_meds

def label_prescriptions():
    if not UNLABELED_DIR.exists():
        print(f"Unlabeled directory not found: {UNLABELED_DIR}")
        return

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    unlabeled_files = sorted([
        f for f in UNLABELED_DIR.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not unlabeled_files:
        print("No unlabeled prescriptions found in data/real_prescriptions/unlabeled/images/")
        return

    print(f"Found {len(unlabeled_files)} unlabeled prescriptions.")
    print("Starting human-in-the-loop labeling flow...")

    for idx, img_path in enumerate(unlabeled_files, start=1):
        print("\n" + "="*60)
        print(f"Prescription {idx}/{len(unlabeled_files)}: {img_path.name}")
        print("="*60)
        
        open_image(img_path)
        
        # Collect clinical metadata
        doc_name = get_input("Doctor Name").strip() or None
        doc_reg = get_input("Doctor Reg/License No").strip() or None
        clinic_name = get_input("Clinic Name").strip() or None
        pat_name = get_input("Patient Name").strip() or None
        pat_age = get_input("Patient Age (e.g. 45 or 45 yrs)").strip() or None
        pat_gender = get_input("Patient Gender (M/F/Other)").strip() or None
        rx_date = get_input("Prescription Date (YYYY-MM-DD)").strip() or None
        diagnosis = get_input("Diagnosis").strip() or None
        
        # Collect medications
        medications: List[Dict[str, Any]] = []
        print("\n--- Medications List ---")
        while True:
            add_med = input("Add a medication? (y/n) [y]: ").strip().lower()
            if add_med == 'n':
                break
            
            drug_name = input("  Drug Name (Required): ").strip()
            if not drug_name:
                print("  Error: Drug name is required to add a medication.")
                continue
                
            dosage_val_str = input("  Dosage Value (e.g. 500, empty for None): ").strip()
            dosage_val = None
            if dosage_val_str:
                try:
                    dosage_val = float(dosage_val_str)
                except ValueError:
                    print(f"  Warning: Invalid dosage value '{dosage_val_str}', setting to None.")
            
            dosage_unit = input("  Dosage Unit (e.g. mg, ml, empty for None): ").strip() or None
            frequency = input("  Frequency (e.g. twice daily, OD, BD, empty for None): ").strip() or None
            route = input("  Route [oral]: ").strip() or "oral"
            instructions = input("  Instructions (empty for None): ").strip() or None
            
            medications.append({
                "drug_name": drug_name,
                "dosage_value": dosage_val,
                "dosage_unit": dosage_unit,
                "frequency": frequency,
                "route": route,
                "instructions": instructions
            })
            print(f"  Added: {drug_name} {dosage_val or ''}{dosage_unit or ''} {frequency or ''}")
            
        # Compile structure
        structured_gt = {
            "doctor_name": doc_name,
            "doctor_reg": doc_reg,
            "clinic_name": clinic_name,
            "patient_name": pat_name,
            "patient_age": pat_age,
            "patient_gender": pat_gender,
            "prescription_date": rx_date,
            "diagnosis": diagnosis,
            "medications": medications
        }
        
        # Validate entry
        if not validate_clinical_data(structured_gt):
            print("\n[ERROR] The ground truth is completely empty. You must enter at least one clinical detail or medication.")
            choice = input("Would you like to (r)etry labeling this image, or (s)kip it for now? [r]: ").strip().lower()
            if choice == 's':
                continue
            os.system('cls' if os.name == 'nt' else 'clear')
            unlabeled_files.insert(0, img_path) # Insert back to retry
            continue

        # Partition routing
        while True:
            dest = input("\nSave to (e)valuation set, (t)raining set, or (s)kip for now? [e/t/s]: ").strip().lower()
            if dest in ('e', 'evaluation'):
                target_dir = EVAL_DIR
                break
            elif dest in ('t', 'training'):
                target_dir = TRAIN_DIR
                break
            elif dest in ('s', 'skip'):
                target_dir = None
                break
            else:
                print("Invalid choice. Enter 'e', 't', or 's'.")
                
        if target_dir is None:
            print("Prescription skipped.")
            continue
            
        # Target preparation
        target_img_dir = target_dir / "images"
        target_img_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image to target directory
        dest_img_path = target_img_dir / img_path.name
        shutil.copy2(img_path, dest_img_path)
        
        # Save annotation
        metadata_file = target_dir / "metadata.jsonl"
        entry = {
            "file_name": f"images/{img_path.name}",
            "ground_truth": json.dumps(structured_gt, ensure_ascii=False)
        }
        
        with open(metadata_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
        # Delete original file from unlabeled folder
        img_path.unlink()
        print(f"[SUCCESS] Successfully saved {img_path.name} to {target_dir.name} dataset!")

if __name__ == "__main__":
    label_prescriptions()
