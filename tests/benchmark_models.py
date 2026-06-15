import os
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import time
import difflib
from typing import Dict, Any, List, Set
from unittest.mock import MagicMock

# 1. Setup mock database before importing pipeline to avoid database connections in benchmark script
db_mock = MagicMock()
sys.modules['src.database'] = db_mock

import src.database as db
db.SessionLocal = MagicMock()
db.get_or_create_patient = MagicMock()
db.get_active_medications = MagicMock(return_value=[])
db.save_prescription_to_db = MagicMock()
db.save_to_review_queue = MagicMock()

# Now import the pipeline and ML extraction functions
from src.pipeline import process_prescription
from src.ner_ml import extract_handwriting_ml

BASE_DIR = Path("c:/Users/USER/Desktop/OJAAI")
EVAL_DIR = BASE_DIR / "data/real_prescriptions/evaluation"
METADATA_FILE = EVAL_DIR / "metadata.jsonl"
OUTPUT_REPORT = EVAL_DIR / "benchmark_report.md"
OUTPUT_JSON = EVAL_DIR / "benchmark_results.json"
ARTIFACTS_DIR = Path("C:/Users/USER/.gemini/antigravity/brain/3373bcad-6c06-46da-b8f8-ff3a1f57bba4")

def clean_str(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()

def is_fuzzy_match(s1: str, s2: str, threshold: float = 0.8) -> bool:
    c1 = clean_str(s1)
    c2 = clean_str(s2)
    if not c1 or not c2:
        return False
    if c1 == c2 or c1 in c2 or c2 in c1:
        return True
    return difflib.SequenceMatcher(None, c1, c2).ratio() >= threshold

def calculate_medication_metrics(gt_meds: List[Dict[str, Any]], pred_meds: List[str]) -> Dict[str, float]:
    gt_names = [clean_str(m.get("drug_name")) for m in gt_meds if m.get("drug_name")]
    pred_names = [clean_str(n) for n in pred_meds if n]
    
    if not gt_names and not pred_names:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gt_names or not pred_names:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched_gt = set()
    matched_pred = set()
    
    for gt_idx, gt in enumerate(gt_names):
        for pred_idx, pred in enumerate(pred_names):
            if is_fuzzy_match(gt, pred, threshold=0.8):
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)

    precision = len(matched_pred) / len(pred_names)
    recall = len(matched_gt) / len(gt_names)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}

def evaluate_pipeline():
    if not METADATA_FILE.exists():
        print(f"[WARNING] No evaluation metadata found at {METADATA_FILE.absolute()}")
        print("Please run 'python src/label_dataset.py' to add labeled prescriptions first.")
        return

    records = []
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print("[WARNING] Evaluation metadata file is empty.")
        return

    print(f"Loaded {len(records)} evaluation records. Running comparative benchmark...")

    ocr_times = []
    donut_times = []
    
    fields_to_eval = ["doctor_name", "doctor_reg", "patient_name", "patient_age", "prescription_date"]
    
    # Structure to store accuracy metrics
    # Pipeline -> Field -> 'strict_correct' / 'fuzzy_correct' / 'total'
    scores = {
        "ocr": {f: {"strict": 0, "fuzzy": 0, "total": 0} for f in fields_to_eval},
        "donut": {f: {"strict": 0, "fuzzy": 0, "total": 0} for f in fields_to_eval}
    }
    
    # Medication metrics totals
    ocr_med_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    donut_med_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    med_total_count = 0

    for idx, record in enumerate(records, start=1):
        img_name = record["file_name"]
        gt = json.loads(record["ground_truth"])
        img_path = EVAL_DIR / img_name
        
        if not img_path.exists():
            print(f"[WARNING] Image not found: {img_path.absolute()}")
            continue
            
        print(f"[{idx}/{len(records)}] Processing {img_name}...")
        img_bytes = img_path.read_bytes()
        
        # --- 1. Run Tesseract OCR + NER pipeline ---
        t0 = time.time()
        try:
            ocr_res = process_prescription(img_bytes, img_path.name)
            ocr_latency = time.time() - t0
            ocr_times.append(ocr_latency)
            
            # Extract names of drugs
            ocr_drugs = [m.raw_drug_name for m in ocr_res.medications if m.raw_drug_name]
            
            # Map fields for evaluation
            ocr_fields = {
                "doctor_name": None,  # Tesseract OCR pipeline doesn't extract doctor name in Tier 3
                "doctor_reg": ocr_res.doctor_reg,
                "patient_name": None, # Tesseract OCR pipeline doesn't extract patient name in Tier 3
                "patient_age": ocr_res.patient_age,
                "prescription_date": ocr_res.prescription_date
            }
        except Exception as e:
            print(f"  [ERROR] OCR pipeline execution failed: {e}")
            ocr_fields = {f: None for f in fields_to_eval}
            ocr_drugs = []
            ocr_latency = 0.0

        # --- 2. Run Local Donut ML Pipeline ---
        t0 = time.time()
        try:
            donut_res = extract_handwriting_ml(img_bytes, img_path.name)
            donut_latency = time.time() - t0
            donut_times.append(donut_latency)
            
            if donut_res:
                donut_drugs = [m.get("raw_drug_name") for m in donut_res.get("medications", []) if m.get("raw_drug_name")]
                donut_fields = {
                    "doctor_name": donut_res.get("doctor_name"),
                    "doctor_reg": donut_res.get("doctor_reg"),
                    "patient_name": donut_res.get("patient_name"),
                    "patient_age": donut_res.get("patient_age"),
                    "prescription_date": donut_res.get("prescription_date")
                }
            else:
                donut_drugs = []
                donut_fields = {f: None for f in fields_to_eval}
        except Exception as e:
            print(f"  [ERROR] Donut pipeline execution failed: {e}")
            donut_fields = {f: None for f in fields_to_eval}
            donut_drugs = []
            donut_latency = 0.0

        # --- 3. Evaluate clinical fields ---
        for field in fields_to_eval:
            gt_val = clean_str(gt.get(field))
            if gt_val:
                # OCR
                ocr_val = clean_str(ocr_fields.get(field))
                scores["ocr"][field]["total"] += 1
                if ocr_val == gt_val:
                    scores["ocr"][field]["strict"] += 1
                if is_fuzzy_match(gt_val, ocr_val, threshold=0.8):
                    scores["ocr"][field]["fuzzy"] += 1
                    
                # Donut
                donut_val = clean_str(donut_fields.get(field))
                scores["donut"][field]["total"] += 1
                if donut_val == gt_val:
                    scores["donut"][field]["strict"] += 1
                if is_fuzzy_match(gt_val, donut_val, threshold=0.8):
                    scores["donut"][field]["fuzzy"] += 1

        # --- 4. Evaluate medications ---
        gt_meds = gt.get("medications", [])
        if gt_meds:
            med_total_count += 1
            
            ocr_m = calculate_medication_metrics(gt_meds, ocr_drugs)
            ocr_med_metrics["precision"] += ocr_m["precision"]
            ocr_med_metrics["recall"] += ocr_m["recall"]
            ocr_med_metrics["f1"] += ocr_m["f1"]
            
            donut_m = calculate_medication_metrics(gt_meds, donut_drugs)
            donut_med_metrics["precision"] += donut_m["precision"]
            donut_med_metrics["recall"] += donut_m["recall"]
            donut_med_metrics["f1"] += donut_m["f1"]

    # --- 5. Compile aggregate metrics ---
    ocr_avg_time = sum(ocr_times) / len(ocr_times) if ocr_times else 0.0
    donut_avg_time = sum(donut_times) / len(donut_times) if donut_times else 0.0
    
    ocr_final_med = {k: v / med_total_count if med_total_count else 0.0 for k, v in ocr_med_metrics.items()}
    donut_final_med = {k: v / med_total_count if med_total_count else 0.0 for k, v in donut_med_metrics.items()}

    # Print results to stdout
    print("\n" + "="*50)
    print("COMPARATIVE BENCHMARK RUN COMPLETE")
    print("="*50)
    print(f"Total Prescriptions Evaluated: {len(records)}")
    print(f"Average Latency (seconds per image):")
    print(f"  - Tesseract OCR Pipeline: {ocr_avg_time:.2f}s")
    print(f"  - Donut ML Model:         {donut_avg_time:.2f}s")
    print("\n[MEDICATION ENTITY ACCURACY]")
    print(f"  Tesseract OCR: Precision={ocr_final_med['precision']:.2%}, Recall={ocr_final_med['recall']:.2%}, F1={ocr_final_med['f1']:.2%}")
    print(f"  Donut ML:      Precision={donut_final_med['precision']:.2%}, Recall={donut_final_med['recall']:.2%}, F1={donut_final_med['f1']:.2%}")

    # Build Markdown report content
    report_md = f"""# OJAAI Pipeline Benchmark Evaluation Report

This report presents a comparative evaluation of the **Tesseract OCR + Regex NER Pipeline** against the **Donut ML Model** on the real-world evaluation dataset.

## Summary Metrics

| Metric | Tesseract OCR Pipeline | Donut ML Model |
| :--- | :---: | :---: |
| **Total Prescriptions Evaluated** | {len(records)} | {len(records)} |
| **Average Latency per Image** | {ocr_avg_time:.2f}s | {donut_avg_time:.2f}s |
| **Medication Precision** | {ocr_final_med['precision']:.2%} | {donut_final_med['precision']:.2%} |
| **Medication Recall** | {ocr_final_med['recall']:.2%} | {donut_final_med['recall']:.2%} |
| **Medication F1-Score** | {ocr_final_med['f1']:.2%} | {donut_final_med['f1']:.2%} |

---

## Field-by-Field Clinical Extraction Accuracy

### Tesseract OCR Pipeline
| Clinical Field | Total Present | Strict Accuracy | Fuzzy Accuracy (Levenshtein >= 80%) |
| :--- | :---: | :---: | :---: |
"""
    for f in fields_to_eval:
        tot = scores["ocr"][f]["total"]
        strict_acc = scores["ocr"][f]["strict"] / tot if tot else 0.0
        fuzzy_acc = scores["ocr"][f]["fuzzy"] / tot if tot else 0.0
        report_md += f"| **{f}** | {tot} | {strict_acc:.2%} | {fuzzy_acc:.2%} |\n"

    report_md += """
### Donut ML Model
| Clinical Field | Total Present | Strict Accuracy | Fuzzy Accuracy (Levenshtein >= 80%) |
| :--- | :---: | :---: | :---: |
"""
    for f in fields_to_eval:
        tot = scores["donut"][f]["total"]
        strict_acc = scores["donut"][f]["strict"] / tot if tot else 0.0
        fuzzy_acc = scores["donut"][f]["fuzzy"] / tot if tot else 0.0
        report_md += f"| **{f}** | {tot} | {strict_acc:.2%} | {fuzzy_acc:.2%} |\n"

    # Conclude recommendation based on results
    rec_donut_primary = donut_final_med['f1'] > ocr_final_med['f1'] + 0.20
    recommendation_text = ""
    if rec_donut_primary:
        recommendation_text = "Recommendation: Donut ML should be transitioned to the Primary Extractor for handwriting as it has met the performance criteria (exceeding Tesseract OCR pipeline F1 score by more than 20% on the evaluation set)."
    else:
        recommendation_text = "Recommendation: Donut ML must remain as Tier 3 Fallback. The Tesseract OCR + Regex pipeline maintains higher reliability/F1-score, or Donut's improvement margin does not yet justify the latency cost."

    report_md += f"""
---

## Deployment Recommendation

{recommendation_text}

*Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # Write report and JSON output
    OUTPUT_REPORT.write_text(report_md, encoding="utf-8")
    print(f"Report saved to: {OUTPUT_REPORT.absolute()}")
    
    if ARTIFACTS_DIR.exists():
        artifact_report = ARTIFACTS_DIR / "performance_evaluation_report.md"
        artifact_report.write_text(report_md, encoding="utf-8")
        print(f"Artifact report saved to: {artifact_report.absolute()}")

    # Output JSON metrics
    json_metrics = {
        "dataset_size": len(records),
        "ocr": {
            "avg_latency": ocr_avg_time,
            "medication": ocr_final_med,
            "clinical_fields": {f: {"strict": scores["ocr"][f]["strict"] / scores["ocr"][f]["total"] if scores["ocr"][f]["total"] else 0.0, "fuzzy": scores["ocr"][f]["fuzzy"] / scores["ocr"][f]["total"] if scores["ocr"][f]["total"] else 0.0} for f in fields_to_eval}
        },
        "donut": {
            "avg_latency": donut_avg_time,
            "medication": donut_final_med,
            "clinical_fields": {f: {"strict": scores["donut"][f]["strict"] / scores["donut"][f]["total"] if scores["donut"][f]["total"] else 0.0, "fuzzy": scores["donut"][f]["fuzzy"] / scores["donut"][f]["total"] if scores["donut"][f]["total"] else 0.0} for f in fields_to_eval}
        }
    }
    
    OUTPUT_JSON.write_text(json.dumps(json_metrics, indent=4), encoding="utf-8")
    print(f"JSON metrics saved to: {OUTPUT_JSON.absolute()}")

if __name__ == "__main__":
    evaluate_pipeline()
