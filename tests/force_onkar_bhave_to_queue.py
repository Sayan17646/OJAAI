"""
force_onkar_bhave_to_queue.py — Manually forces the Dr. Onkar Bhave prescription into the review queue for live browser auditing.
"""
import os
import sys

# Make sure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database import SessionLocal, get_or_create_patient, save_to_review_queue

def force_use_case():
    print("Forcing Dr. Onkar Bhave's Care Clinic prescription into the review queue...")
    
    rx_image = os.path.abspath("./data/prescriptions/onkar_bhave_rx.png")
    if not os.path.exists(rx_image):
        print("[ERROR] replicas image onkar_bhave_rx.png not found. Please run setup first.")
        return

    db = SessionLocal()
    try:
        # Patient Phone from the prescription: 8983390126
        phone = "8983390126"
        patient = get_or_create_patient(db, phone)
        print(f"[OK] Patient mapped/created (Phone: {phone})")

        raw_ocr_text = (
            "Dr. Onkar Bhave   Care Clinic   Reg. No: 270988\n"
            "M.B.B.S., M.D., M.S. | Mob: 8983390126 | Near Axis Bank, Kothrud, Pune\n"
            "----------------------------------------------------------------------\n"
            "ID: 266 - DEMO PATIENT (M)   Date: 27-Apr-2020, 04:37 PM\n"
            "Address: PUNE | Temp: 36 | BP: 120/80 mmHg\n"
            "----------------------------------------------------------------------\n"
            "Medicine Name              Dosage                     Duration\n"
            "1) TAB. DEMO MEDICINE 1    1 Morning, 1 Night (Before Food)    10 Days\n"
            "2) CAP. DEMO MEDICINE 2    1 Morning, 1 Night (Before Food)    10 Days\n"
            "3) TAB. DEMO MEDICINE 3    1 Morning, 1 Aft, 1 Eve, 1 Night    10 Days\n"
            "4) TAB. DEMO MEDICINE 4    1/2 Morning, 1/2 Night (After Food) 10 Days\n"
            "----------------------------------------------------------------------\n"
            "Advice: AVOID OILY AND SPICY FOOD | Follow Up: 12-05-2020\n"
            "Signature: Dr. Onkar Bhave"
        )

        item = save_to_review_queue(
            db=db,
            patient=patient,
            image_path=rx_image,
            raw_ocr_text=raw_ocr_text,
            confidence=0.35,  # Force under 0.5 to trigger review
            reason="OCR spelling artifacts ('Moming') and custom Brand Names ('DEMO MEDICINE')"
        )
        db.commit()
        print(f"[SUCCESS] Dr. Onkar Bhave's prescription saved to review queue with ID: {item.id}")
        print("\n=========================================================================")
        print("DR. ONKAR BHAVE'S CASE LOADED LIVE!")
        print("=========================================================================")
        print("1. Go to your dashboard tab in your browser: http://localhost:8000/dashboard")
        print("2. Refresh the browser page.")
        print("3. You will see a new card in the sidebar showing '35% Conf' (Dr. Onkar Bhave).")
        print("4. Click on it. The Care Clinic prescription scan will render beautifully on the left,")
        print("   and you can audit the parsed Demo Medicines and metadata live on your screen!")
        print("=========================================================================\n")
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to save to review queue: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    force_use_case()
