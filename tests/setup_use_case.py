"""
setup_use_case.py — Sets up a real-world clinical use case in the review queue database.
"""
import os
import sys
import shutil

# Make sure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database import SessionLocal, get_or_create_patient, save_to_review_queue

def setup_clinical_case():
    print("Setting up a clinical use case in the database...")
    
    # 1. Ensure the prescriptions directory exists and has a test scan image
    rx_dir = os.path.abspath("./data/prescriptions")
    os.makedirs(rx_dir, exist_ok=True)
    
    src_image = os.path.abspath("./smoke_test_rx.png")
    dest_image = os.path.join(rx_dir, "clinical_use_case_rx.png")
    
    if os.path.exists(src_image):
        shutil.copy(src_image, dest_image)
        print(f"[OK] Copied test scan image to {dest_image}")
    else:
        # Create a tiny dummy file if the base smoke image is not found
        with open(dest_image, "wb") as f:
            f.write(b"dummy image data")
        print(f"[OK] Created placeholder image at {dest_image}")

    db = SessionLocal()
    try:
        # 2. Get or create a patient
        phone = "9876501234"
        patient = get_or_create_patient(db, phone)
        print(f"[OK] Bound/Created patient with Phone: {phone}")

        # 3. Create a low-confidence review queue item representing a real-world messy prescription
        # This prescription has a critical drug-drug interaction (Ecosprin / Aspirin + Brufen / Ibuprofen)
        raw_ocr_text = (
            "Dr. S. K. Sharma   MCI/98765\n"
            "Pat Name: Amit Patel  Age: 62 yrs  Date: 23/05/2026\n"
            "Diagnosis: Coronary Artery Disease & Severe Joint Pain\n"
            "--------------------------------------------------\n"
            "1. Tab. Ecosprin 75mg OD\n"
            "2. Tab. Brufen 400mg BD\n"
            "--------------------------------------------------\n"
            "### Messy doctor scribble sequence & water stain artifact @#$% !!!\n"
            "No active signature match detected by machine NER."
        )

        item = save_to_review_queue(
            db=db,
            patient=patient,
            image_path=dest_image,
            raw_ocr_text=raw_ocr_text,
            confidence=0.28,  # Low confidence (< 0.5 threshold) triggers human audit
            reason="Low confidence OCR text & missing doctor signature match (Confidence: 28%)"
        )
        db.commit()
        print(f"[OK] Saved clinical review item ID: {item.id}")
        print("\n=========================================================================")
        print("CLINICAL USE CASE READY FOR AUDITING!")
        print("=========================================================================")
        print("Here is what you can do to test this actual use case:")
        print(f"1. Open your browser and go to: http://localhost:8000/dashboard")
        print("2. You will see a new pending card in the left sidebar marked '28% Conf'.")
        print("3. Click on the card. The workspace will load:")
        print("   - The prescription scan will render on the left panel (interactive scale/rotate).")
        print("   - Parsed details (Dr. S. K. Sharma, patient age, diagnosis) will load in the form.")
        print("   - In the medications table, you will see two raw brand entries:")
        print("     - Ecosprin (Aspirin)")
        print("     - Brufen (Ibuprofen)")
        print("4. Notice the real-time 'Drug Interaction Preview' at the bottom!")
        print("   - It will display a high-visibility MAJOR alert: Ecosprin + Brufen.")
        print("   - This alerts the clinician of the severe gastrointestinal bleeding risk.")
        print("5. Make any edits you like, then click 'Approve & Merge'.")
        print("6. The item will resolve atomically in the DB, merge to main tables, and vanish from the queue!")
        print("=========================================================================\n")
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to setup clinical use case: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    setup_clinical_case()
