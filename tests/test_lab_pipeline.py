"""
test_lab_pipeline.py — Integration and unit tests for OJAAI Option C: Lab Ingestion & Clinical CDS.
"""
from __future__ import annotations

import pytest
from src.lab_ner import extract_lab_report
from src.clinical_checker import check_clinical_safety
from src.models import NormalizedDrug, LabResultExtracted
from src.database import SessionLocal, get_or_create_patient, create_all_tables

# Ensure all new lab tables are registered in the test DB
create_all_tables()

@pytest.fixture(autouse=True)
def clear_dynamic_rules():
    """Clear DB rules so fallback hardcoded rules are tested."""
    db = SessionLocal()
    try:
        from src.database import ClinicalSafetyRule
        db.query(ClinicalSafetyRule).delete()
        db.commit()
        yield
    finally:
        db.close()


# ===========================================================================
# 1. UNIT TESTS: Lab NER Extractor
# ===========================================================================

def test_lab_name_and_date_extraction():
    raw_text = (
        "APOLLO DIAGNOSTICS   Kothrud Centre\n"
        "Date of Registration: 24/05/2026   Report Date: 25-May-2026\n"
        "Patient: Amit Patel   Age: 62 Y / Male\n"
        "Serum Creatinine   1.8   mg/dL   0.6 - 1.2 mg/dL\n"
    )
    extracted = extract_lab_report(raw_text)
    assert extracted.lab_name == "Apollo Diagnostics"
    assert extracted.report_date in ("24/05/2026", "25-May-2026")
    assert len(extracted.results) == 1
    
    creat = extracted.results[0]
    assert creat.analyte_name == "CREATININE"
    assert creat.value == 1.8
    assert creat.unit == "mg/dL"
    assert creat.flag == "high"


def test_multiple_biomarkers_extraction():
    raw_text = (
        "SRL DIAGNOSTICS\n"
        "Hemoglobin (Hb)    8.5   g/dL   12.0 - 16.0 g/dL\n"
        "HbA1c              7.2   %      < 5.7 %\n"
        "TSH (Thyroid)      0.15  uIU/mL  0.4 - 4.5 uIU/mL\n"
    )
    extracted = extract_lab_report(raw_text)
    assert extracted.lab_name == "Srl Diagnostics"
    assert len(extracted.results) == 3
    
    hb = next(r for r in extracted.results if r.analyte_name == "HEMOGLOBIN")
    assert hb.value == 8.5
    assert hb.flag == "low"
    
    hba1c = next(r for r in extracted.results if r.analyte_name == "HBA1C")
    assert hba1c.value == 7.2
    assert hba1c.flag == "high"
    
    tsh = next(r for r in extracted.results if r.analyte_name == "TSH")
    assert tsh.value == 0.15
    assert tsh.flag == "low"


# ===========================================================================
# 2. UNIT TESTS: Clinical Decision Support Safety Rules
# ===========================================================================

def test_metformin_elevated_creatinine_contraindication():
    # Setup mock lab results
    db = SessionLocal()
    try:
        # Create a mock patient and save high creatinine lab report
        patient = get_or_create_patient(db, "9999911111")
        patient_id = patient.id
        
        # Save a lab result showing creatinine = 1.9 mg/dL (Female limit: 1.4)
        from src.database import save_lab_report_to_db
        res_list = [
            LabResultExtracted(
                raw_name="Serum Creatinine 1.9 mg/dL",
                analyte_name="CREATININE",
                value=1.9,
                unit="mg/dL",
                ref_range="0.5 - 1.1 mg/dL",
                flag="high"
            )
        ]
        save_lab_report_to_db(db, patient, "./test_lab.png", "Apollo Diagnostics", "24/05/2026", res_list)
        db.commit()
        
        # Active drugs in current prescription: Metformin
        meds = [
            NormalizedDrug(
                raw_drug_name="Glycomet 500",
                inn="metformin",
                standard_name="metformin",
                dosage_value=500.0,
                dosage_unit="mg",
                frequency="twice daily",
                freq_per_day=2,
                duration_days=30,
                route="oral",
                is_active=True
            )
        ]
        
        # Run safety checker
        alerts = check_clinical_safety(db, patient_id, meds)
        
        assert len(alerts) >= 1
        alert = next(a for a in alerts if a.analyte_name == "CREATININE")
        assert alert.severity == "critical"
        assert "contraindicated" in alert.description.lower()
        assert alert.drug_name == "Metformin"
        
    finally:
        # Clean up database
        db.rollback()
        from src.database import (
            LabResult, LabReport, Patient, ReviewQueue,
            PhgEvent, PatientCondition, MedicationDosageHistory, MedicationEpisode, Doctor
        )
        db.query(PhgEvent).delete()
        db.query(PatientCondition).delete()
        db.query(MedicationDosageHistory).delete()
        db.query(MedicationEpisode).delete()
        db.query(Doctor).delete()
        db.query(LabResult).delete()
        db.query(LabReport).delete()
        db.query(ReviewQueue).delete()
        db.query(Patient).filter(Patient.phone.in_(["9999911111", "9999922222", "9876599999"])).delete()
        db.commit()
        db.close()


def test_levothyroxine_out_of_range_tsh_alerts():
    db = SessionLocal()
    try:
        patient = get_or_create_patient(db, "9999922222")
        patient_id = patient.id
        
        # Scenario 1: High TSH = 8.5 uIU/mL (underdosing warning)
        from src.database import save_lab_report_to_db
        res_list_high = [
            LabResultExtracted(
                raw_name="TSH 8.5 uIU/mL",
                analyte_name="TSH",
                value=8.5,
                unit="uIU/mL",
                ref_range="0.4 - 4.5 uIU/mL",
                flag="high"
            )
        ]
        save_lab_report_to_db(db, patient, "./test_lab.png", "SRL Diagnostics", "24/05/2026", res_list_high)
        db.commit()
        
        meds = [
            NormalizedDrug(
                raw_drug_name="Thyronorm 50",
                inn="levothyroxine",
                standard_name="levothyroxine",
                dosage_value=50.0,
                dosage_unit="mcg",
                frequency="once daily",
                freq_per_day=1,
                route="oral",
                is_active=True
            )
        ]
        
        alerts_high = check_clinical_safety(db, patient_id, meds)
        assert len(alerts_high) == 1
        assert alerts_high[0].severity == "warning"
        assert "under-dosing" in alerts_high[0].description.lower()
        
    finally:
        db.rollback()
        from src.database import (
            LabResult, LabReport, Patient, ReviewQueue,
            PhgEvent, PatientCondition, MedicationDosageHistory, MedicationEpisode, Doctor
        )
        db.query(PhgEvent).delete()
        db.query(PatientCondition).delete()
        db.query(MedicationDosageHistory).delete()
        db.query(MedicationEpisode).delete()
        db.query(Doctor).delete()
        db.query(LabResult).delete()
        db.query(LabReport).delete()
        db.query(ReviewQueue).delete()
        db.query(Patient).filter(Patient.phone.in_(["9999911111", "9999922222", "9876599999"])).delete()
        db.commit()
        db.close()


# ===========================================================================
# 3. INTEGRATION TESTS: FastAPI REST Endpoint & Bypass Checks
# ===========================================================================

class TestLabApiEndpoints:
    
    @pytest.fixture(autouse=True)
    def setup_client(self):
        from fastapi.testclient import TestClient
        from src.api import app
        self.client = TestClient(app)

    def test_parse_lab_report_endpoint(self, monkeypatch):
        """Upload a lab report image programmatically and check returned JSON."""
        monkeypatch.setenv("DEBUG_LOCAL_DASHBOARD", "true")
        
        # Create a synthetic lab report text image
        import io
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.new("RGB", (1200, 300), color="white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=32)
        except Exception:
            font = None
            
        draw.text((20, 20), "APOLLO DIAGNOSTICS", fill="black", font=font)
        draw.text((20, 80), "Patient: Rohan Sen   Date: 12-May-2026", fill="black", font=font)
        draw.text((20, 140), "Serum Creatinine  1.7 mg/dL  0.6 - 1.2 mg/dL", fill="black", font=font)
        
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        
        # POST to local lab parser endpoint
        resp = self.client.post(
            "/api/lab/parse",
            headers={"X-API-Key": "secure-ojaai-rot-5678-auth"},
            files={"image": ("lab.png", buf, "image/png")},
            data={"phone": "9876599999"}
        )
        
        assert resp.status_code == 200
        result = resp.json()
        assert result["lab_name"].upper() == "APOLLO DIAGNOSTICS"
        assert result["patient_phone"] == "9876599999"
        assert len(result["results"]) == 1
        assert result["results"][0]["analyte_name"] == "CREATININE"
        assert result["results"][0]["value"] == 1.7
        assert result["results"][0]["flag"] == "high"
        
        # Clean up database writes
        from src.database import (
            SessionLocal, LabResult, LabReport, Patient, ReviewQueue,
            PhgEvent, PatientCondition, MedicationDosageHistory, MedicationEpisode, Doctor
        )
        db = SessionLocal()
        try:
            db.query(PhgEvent).delete()
            db.query(PatientCondition).delete()
            db.query(MedicationDosageHistory).delete()
            db.query(MedicationEpisode).delete()
            db.query(Doctor).delete()
            db.query(LabResult).delete()
            db.query(LabReport).delete()
            db.query(ReviewQueue).delete()
            db.query(Patient).filter(Patient.phone.in_(["9999911111", "9999922222", "9876599999"])).delete()
            db.commit()
        finally:
            db.close()

    def test_low_confidence_lab_report_routing(self, monkeypatch):
        """Upload a completely empty or low confidence lab report image and verify routing to ReviewQueue."""
        monkeypatch.setenv("DEBUG_LOCAL_DASHBOARD", "true")
        
        # Create a blank white image (which will have 0 OCR text / 0 confidence)
        import io
        from PIL import Image
        
        img = Image.new("RGB", (300, 100), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        
        # POST to local lab parser endpoint
        resp = self.client.post(
            "/api/lab/parse",
            headers={"X-API-Key": "secure-ojaai-rot-5678-auth"},
            files={"image": ("blank_lab.png", buf, "image/png")},
            data={"phone": "9876599999"}
        )
        
        assert resp.status_code == 200
        result = resp.json()
        assert result["needs_human_review"] is True
        
        # Verify it went to the ReviewQueue
        from src.database import SessionLocal, ReviewQueue
        db = SessionLocal()
        try:
            queue_item = db.query(ReviewQueue).filter(ReviewQueue.id == result["lab_report_id"]).first()
            assert queue_item is not None
            assert queue_item.item_type == "lab"
            assert queue_item.resolved is False
        finally:
            db.close()

    def test_resolve_lab_review_queue_item(self, monkeypatch):
        """Manually resolve a low confidence lab review queue item and verify writes to main tables."""
        monkeypatch.setenv("DEBUG_LOCAL_DASHBOARD", "true")
        
        from src.database import SessionLocal, save_to_review_queue, get_or_create_patient, LabReport, LabResult, ReviewQueue, Patient
        db = SessionLocal()
        try:
            patient = get_or_create_patient(db, "9876599999")
            queue_item = save_to_review_queue(
                db=db,
                patient=patient,
                image_path="./data/lab_reports/fake_lab.png",
                raw_ocr_text="BLANK OCR TEXT",
                confidence=0.1,
                reason="No analytes detected",
                item_type="lab"
            )
            db.commit()
            queue_id = str(queue_item.id)
            
            # Now trigger manual resolution
            payload = {
                "patient_phone": "9876599999",
                "lab_name": "Apollo Diagnostics",
                "report_date": "24/05/2026",
                "results": [
                    {
                        "raw_name": "Serum Creatinine 1.8",
                        "analyte_name": "CREATININE",
                        "value": 1.8,
                        "unit": "mg/dL",
                        "ref_range": "0.6 - 1.2 mg/dL",
                        "flag": "high"
                    }
                ]
            }
            
            resp = self.client.post(
                f"/api/review/lab/{queue_id}/resolve",
                headers={"X-API-Key": "secure-ojaai-rot-5678-auth"},
                json=payload
            )
            
            assert resp.status_code == 200
            result = resp.json()
            assert result["needs_human_review"] is False
            assert result["confidence"] == 1.0
            
            # Verify the review queue item is marked resolved
            db.refresh(queue_item)
            assert queue_item.resolved is True
            
            # Verify lab report is committed to main tables
            lab_report = db.query(LabReport).filter(LabReport.id == result["lab_report_id"]).first()
            assert lab_report is not None
            assert lab_report.lab_name == "Apollo Diagnostics"
            assert len(lab_report.results) == 1
            assert lab_report.results[0].analyte_name == "CREATININE"
            assert lab_report.results[0].value == 1.8
            assert lab_report.results[0].flag == "high"
            
        finally:
            db.rollback()
            from src.database import (
                PhgEvent, PatientCondition, MedicationDosageHistory, MedicationEpisode, Doctor
            )
            db.query(PhgEvent).delete()
            db.query(PatientCondition).delete()
            db.query(MedicationDosageHistory).delete()
            db.query(MedicationEpisode).delete()
            db.query(Doctor).delete()
            db.query(LabResult).delete()
            db.query(LabReport).delete()
            db.query(ReviewQueue).delete()
            db.query(Patient).filter(Patient.phone == "9876599999").delete()
            db.commit()
            db.close()
