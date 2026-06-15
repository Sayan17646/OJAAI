from src.database import SessionLocal, Prescription, Medication
db = SessionLocal()
try:
    rx = db.query(Prescription).order_by(Prescription.created_at.desc()).first()
    if rx:
        print("Last Prescription:")
        print(f"ID: {rx.id}")
        print(f"Confidence: {rx.confidence}")
        print(f"Doctor Reg: {rx.doctor_reg}")
        print(f"Diagnosis: {rx.diagnosis}")
        print(f"OCR text: {repr(rx.raw_ocr_text)}")
        print("Medications:")
        for m in rx.medications:
            print(f"  raw={m.raw_drug_name} inn={m.inn} standard={m.standard_name} dose={m.dosage_value}{m.dosage_unit} freq={m.frequency}")
    else:
        print("No prescriptions found in DB.")
except Exception as e:
    print("Error:", e)
finally:
    db.close()
