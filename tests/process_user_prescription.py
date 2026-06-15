"""
process_user_prescription.py — Re-creates the user's uploaded prescription image and runs it through the actual OJAAI pipeline.
"""
import os
import sys
import requests
import io
from PIL import Image, ImageDraw, ImageFont

def process_onkar_bhave_rx():
    print("=========================================================================")
    print("RE-CREATING AND PROCESSING DR. ONKAR BHAVE'S PRESCRIPTION")
    print("=========================================================================")
    
    # 1. Create the prescription image canvas
    img = Image.new("RGB", (1200, 750), color="white")
    draw = ImageDraw.Draw(img)
    
    # Load fonts
    try:
        font_header = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=32)
        font_sub = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=24)
        font_body = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=22)
    except Exception:
        font_header = None
        font_sub = None
        font_body = None

    # Text blocks matching Dr. Onkar Bhave's Care Clinic sheet
    lines = [
        ("Dr. Onkar Bhave   Care Clinic   Reg. No: 270988", font_header),
        ("M.B.B.S., M.D., M.S. | Mob: 8983390126 | Near Axis Bank, Kothrud, Pune", font_sub),
        ("----------------------------------------------------------------------------------------------------", font_sub),
        ("ID: 266 - DEMO PATIENT (M)   Date: 27-Apr-2020", font_body),
        ("Address: PUNE | Temp: 36 | BP: 120/80 mmHg", font_body),
        ("----------------------------------------------------------------------------------------------------", font_sub),
        ("Medicine Name              Dosage                     Duration", font_sub),
        ("1) TAB. DEMO MEDICINE 1    1 Morning, 1 Night (Before Food)    10 Days", font_body),
        ("2) CAP. DEMO MEDICINE 2    1 Morning, 1 Night (Before Food)    10 Days", font_body),
        ("3) TAB. DEMO MEDICINE 3    1 Morning, 1 Aft, 1 Eve, 1 Night    10 Days", font_body),
        ("4) TAB. DEMO MEDICINE 4    1/2 Morning, 1/2 Night (After Food) 10 Days", font_body),
        ("----------------------------------------------------------------------------------------------------", font_sub),
        ("Advice: AVOID OILY AND SPICY FOOD | Follow Up: 12-05-2020", font_body),
        ("Signature: Dr. Onkar Bhave", font_sub),
    ]

    y = 20
    for text, f in lines:
        draw.text((45, y), text, fill="black", font=f)
        y += 48

    # Save image to the local prescriptions directory
    rx_dir = os.path.abspath("./data/prescriptions")
    os.makedirs(rx_dir, exist_ok=True)
    image_path = os.path.join(rx_dir, "onkar_bhave_rx.png")
    img.save(image_path)
    print(f"[OK] Generated identical replica of the prescription image at: {image_path}")

    # 2. POST the image to the live running FastAPI /parse endpoint
    print("\nStep 2: POSTing the prescription scan to the active OJAAI pipeline...")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    try:
        r = requests.post(
            "http://localhost:8000/parse",
            headers={"X-API-Key": "secure-ojaai-rot-5678-auth"},
            files={"image": ("onkar_bhave_rx.png", buf, "image/png")},
            data={"phone": "9876509999"},
            timeout=30,
        )
        
        if r.status_code != 200:
            print(f"[ERROR] API returned error status {r.status_code}:")
            print(r.text)
            return

        result = r.json()
        print("\n=========================================================================")
        print("OJAAI PIPELINE INGESTION OUTPUT")
        print("=========================================================================")
        print(f"Ingested ID:       {result['prescription_id']}")
        print(f"Confidence Level:  {result['confidence']} (Automatic Ingestion)")
        print(f"Doctor Reg No:     {result['doctor_reg']}")
        print(f"Diagnosis:         {result['diagnosis']}")
        print(f"Prescription Date: {result['prescription_date']}")
        print(f"Needs Audit?       {result['needs_human_review']}")
        
        print("\nParsed Medication Line Entries:")
        for idx, med in enumerate(result["medications"], 1):
            print(f"  {idx}. Brand: {med['raw_drug_name']} -> INN: {med['inn']}")
            print(f"     Dose:  {med['dosage_value']}{med['dosage_unit']} | Route: {med['route']}")
            print(f"     Freq:  {med['frequency']} ({med['freq_per_day']} times/day) | Duration: {med['duration_days']} days")
            
        print("\nDrug-Drug Interactions Resolved:")
        if result["interactions"]:
            for ddi in result["interactions"]:
                print(f"  - {ddi['drug_1']} + {ddi['drug_2']} ({ddi['severity'].upper()})")
        else:
            print("  [OK] No interactions found in this single session.")
        print("=========================================================================\n")
        
    except Exception as e:
        print(f"[ERROR] Pipeline submission failed: {e}")

if __name__ == "__main__":
    process_onkar_bhave_rx()
