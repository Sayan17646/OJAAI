"""
run_use_case_resolution.py — Simulates the doctor submitting the resolved audit form from the dashboard.
"""
import os
import sys
import requests

# Make sure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def simulate_clinical_resolution():
    print("=========================================================================")
    print("SIMULATING CLINICIAN DASHBOARD RESOLUTION FOR AMIT PATEL")
    print("=========================================================================")
    
    # 1. Fetch the queue from local API (localhost debug bypass active)
    print("\nStep 1: Fetching unresolved queue items from dashboard API...")
    try:
        r_queue = requests.get("http://localhost:8000/api/review/queue")
        if r_queue.status_code != 200:
            print(f"[ERROR] Failed to fetch queue: Status {r_queue.status_code}")
            return
        
        queue = r_queue.json()
        print(f"[OK] Retrieved {len(queue)} pending item(s) from the audit queue.")
        
        # Find Amit Patel's item (phone: 9876501234)
        target_item = None
        for item in queue:
            if item.get("patient_phone") == "9876501234":
                target_item = item
                break
                
        if not target_item:
            print("[ERROR] Could not find pending clinical case for Amit Patel in the queue.")
            return
            
        item_id = target_item["id"]
        print(f"[OK] Found Amit Patel's queue item. ID: {item_id}")
        
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return

    # 2. Fetch the detailed item showing on-the-fly suggestions
    print("\nStep 2: Retrieving item details & non-mutating suggestions...")
    try:
        r_detail = requests.get(f"http://localhost:8000/api/review/{item_id}")
        if r_detail.status_code != 200:
            print(f"[ERROR] Failed to fetch detail: Status {r_detail.status_code}")
            return
            
        detail = r_detail.json()
        print("[OK] Detail loaded.")
        print(f"     Reason: {detail['reason']}")
        print(f"     Raw OCR Text: \n\"\"\"\n{detail['raw_ocr_text']}\n\"\"\"")
        
        draft = detail["draft_suggestion"]
        print(f"     Isolated draft suggestion detected:")
        print(f"       - Doctor Reg No: {draft.get('doctor_reg')}")
        print(f"       - Patient Age: {draft.get('patient_age')}")
        print(f"       - Diagnosis: {draft.get('diagnosis')}")
        print(f"       - Medications ({len(draft['medications'])} items):")
        for m in draft["medications"]:
            print(f"         * {m['raw_drug_name']} (Dose: {m['dosage_value']}{m['dosage_unit']}, Freq: {m['frequency']})")
            
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return

    # 3. Simulate doctor correcting and submitting the resolved prescription
    # Note: Ecosprin -> aspirin, Brufen -> ibuprofen (triggers major interaction warning)
    print("\nStep 3: Simulating doctor corrections and submitting resolved form...")
    corrected_payload = {
        "patient_phone": "9876501234",
        "doctor_reg": "MCI/98765",
        "patient_age": "62 yrs",
        "diagnosis": "Coronary Artery Disease & Severe Joint Osteoarthritis",
        "prescription_date": "23/05/2026",
        "medications": [
            {
                "raw_drug_name": "Ecosprin",
                "dosage_value": 75.0,
                "dosage_unit": "mg",
                "frequency": "once daily",
                "freq_per_day": 1,
                "duration_days": None,  # chronic
                "route": "oral"
            },
            {
                "raw_drug_name": "Brufen",
                "dosage_value": 400.0,
                "dosage_unit": "mg",
                "frequency": "twice daily",
                "freq_per_day": 2,
                "duration_days": 7,     # 7 days course
                "route": "oral"
            }
        ]
    }

    try:
        # Submit resolve request programmatically (simulating Approve & Merge)
        r_resolve = requests.post(
            f"http://localhost:8000/api/review/{item_id}/resolve",
            json=corrected_payload
        )
        
        if r_resolve.status_code != 200:
            print(f"[ERROR] Resolution failed: Status {r_resolve.status_code}")
            print(r_resolve.text)
            return
            
        result = r_resolve.json()
        print("[SUCCESS] Prescription audited and merged atomically inside a single transaction!")
        
        print("\n=========================================================================")
        print("DATABASE MERGE VERIFICATION RESULTS")
        print("=========================================================================")
        print(f"Prescription UUID:    {result['prescription_id']}")
        print(f"Confidence Level:     {result['confidence']} (Clinician Approved)")
        print(f"Doctor Reg:           {result['doctor_reg']}")
        print(f"Diagnosis:            {result['diagnosis']}")
        print("Medications Extracted & RxNorm Normalised:")
        for med in result["medications"]:
            print(f"  - {med['raw_drug_name']} -> INN: {med['inn']} (RxCUI: {med['rxcui']})")
            print(f"    Dosage: {med['dosage_value']}{med['dosage_unit']} | Route: {med['route']} | Active: {med['is_active']}")
            
        print(f"\nDrug Interactions Detected Across History: {len(result['interactions'])} alert(s)")
        for ddi in result["interactions"]:
            print(f"  [! ALERT !] {ddi['drug_1']} + {ddi['drug_2']} ({ddi['severity'].upper()})")
            print(f"    Source: {ddi['source']} | Description: {ddi['description']}")
        print("=========================================================================")
        
        # 4. Verify that the queue is now empty for Amit Patel
        r_queue_check = requests.get("http://localhost:8000/api/review/queue")
        queue_check = r_queue_check.json()
        resolved_ok = not any(x["id"] == item_id for x in queue_check)
        if resolved_ok:
            print("\n[OK] Queue Verification: Checked queue list, item successfully cleared!")
            print("=========================================================================\n")
        else:
            print("\n[WARNING] Item is still showing in the unresolved queue.")
            
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return

if __name__ == "__main__":
    simulate_clinical_resolution()
