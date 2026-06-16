"""
test_clinical_checker.py — Unit tests for the 11 Clinical Decision Support (CDS) rules.
"""
from __future__ import annotations

import pytest
from src.clinical_checker import check_clinical_safety
from src.models import NormalizedDrug, LabResultExtracted
from src.database import SessionLocal, get_or_create_patient, create_all_tables, save_lab_report_to_db

# Initialize database schema if not present
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


def setup_mock_patient_with_labs(phone: str, results: list[LabResultExtracted]) -> object:
    """Helper to create a test patient and seed their latest lab report."""
    db = SessionLocal()
    try:
        patient = get_or_create_patient(db, phone)
        patient_id = patient.id
        save_lab_report_to_db(
            db=db,
            patient=patient,
            image_path="./test_lab_report.png",
            lab_name="Pune Diagnostic Center",
            report_date="24/05/2026",
            results=results
        )
        db.commit()
        return patient_id
    finally:
        db.close()

def test_rule_1_metformin_creatinine_critical():
    """Rule 1: Metformin + Creatinine > 1.5 mg/dL (Critical contraindication)"""
    patient_id = setup_mock_patient_with_labs(
        phone="9876000001",
        results=[
            LabResultExtracted(
                raw_name="Serum Creatinine",
                analyte_name="CREATININE",
                value=1.9,
                unit="mg/dL",
                ref_range="0.6 - 1.2 mg/dL",
                flag="high"
            )
        ]
    )
    
    db = SessionLocal()
    try:
        meds = [
            NormalizedDrug(
                raw_drug_name="Glycomet 500mg",
                inn="metformin",
                standard_name="metformin",
                is_active=True
            )
        ]
        alerts = check_clinical_safety(db, patient_id, meds)
        assert len(alerts) == 1
        assert alerts[0].analyte_name == "CREATININE"
        assert alerts[0].severity == "critical"
        assert "contraindicated" in alerts[0].description
    finally:
        db.close()

def test_rule_2_levothyroxine_tsh_elevated():
    """Rule 2: Levothyroxine + TSH > 4.5 (Underdosing check)"""
    patient_id = setup_mock_patient_with_labs(
        phone="9876000002",
        results=[
            LabResultExtracted(
                raw_name="Thyroid Stimulating Hormone",
                analyte_name="TSH",
                value=7.2,
                unit="uIU/mL",
                ref_range="0.4 - 4.5 uIU/mL",
                flag="high"
            )
        ]
    )
    
    db = SessionLocal()
    try:
        meds = [
            NormalizedDrug(
                raw_drug_name="Thyronorm 100mcg",
                inn="levothyroxine",
                standard_name="levothyroxine",
                is_active=True
            )
        ]
        alerts = check_clinical_safety(db, patient_id, meds)
        assert len(alerts) == 1
        assert alerts[0].analyte_name == "TSH"
        assert alerts[0].severity == "warning"
        assert "under-dosing" in alerts[0].description
    finally:
        db.close()

def test_rule_3_aspirin_hemoglobin_suppressed():
    """Rule 3: Aspirin + Hemoglobin < 10.0 (Severe anemia bleeding check)"""
    patient_id = setup_mock_patient_with_labs(
        phone="9876000003",
        results=[
            LabResultExtracted(
                raw_name="Hemoglobin",
                analyte_name="HEMOGLOBIN",
                value=8.5,
                unit="g/dL",
                ref_range="12.0 - 16.0 g/dL",
                flag="low"
            )
        ]
    )
    
    db = SessionLocal()
    try:
        meds = [
            NormalizedDrug(
                raw_drug_name="Ecosprin 75mg",
                inn="aspirin",
                standard_name="aspirin",
                is_active=True
            )
        ]
        alerts = check_clinical_safety(db, patient_id, meds)
        # We might have both Rule 3 (Hemoglobin < 10) and Rule 10 (Aspirin/Antiplatelets + mild/mod Hb check)
        # Let's verify at least Rule 3 is found and flagged as warning
        assert len(alerts) >= 1
        rule_3_alert = next((a for a in alerts if a.severity == "warning"), None)
        assert rule_3_alert is not None
        assert rule_3_alert.analyte_name == "HEMOGLOBIN"
        assert "Severe anemia" in rule_3_alert.description
    finally:
        db.close()

def test_rule_4_statins_ldl_elevated():
    """Rule 4: Statins + LDL > 100 (Efficacy check)"""
    patient_id = setup_mock_patient_with_labs(
        phone="9876000004",
        results=[
            LabResultExtracted(
                raw_name="LDL Cholesterol",
                analyte_name="LDL",
                value=145.0,
                unit="mg/dL",
                ref_range="< 100 mg/dL",
                flag="high"
            )
        ]
    )
    
    db = SessionLocal()
    try:
        meds = [
            NormalizedDrug(
                raw_drug_name="Atorva 20mg",
                inn="atorvastatin",
                standard_name="atorvastatin",
                is_active=True
            )
        ]
        alerts = check_clinical_safety(db, patient_id, meds)
        assert len(alerts) == 1
        assert alerts[0].analyte_name == "LDL"
        assert alerts[0].severity == "info"
        assert "efficacy" in alerts[0].description
    finally:
        db.close()

def test_rule_5_sulfonylureas_insulin_hypo():
    """Rule 5: Insulin or Sulfonylureas + FBS < 70 (Critical hypoglycemia)"""
    patient_id = setup_mock_patient_with_labs(
        phone="9876000005",
        results=[
            LabResultExtracted(
                raw_name="Fasting Blood Sugar",
                analyte_name="FASTING_BLOOD_SUGAR",
                value=58.0,
                unit="mg/dL",
                ref_range="70 - 100 mg/dL",
                flag="low"
            )
        ]
    )
    
    db = SessionLocal()
    try:
        meds = [
            NormalizedDrug(
                raw_drug_name="Glimepiride 2mg",
                inn="glimepiride",
                standard_name="glimepiride",
                is_active=True
            )
        ]
        alerts = check_clinical_safety(db, patient_id, meds)
        assert len(alerts) == 1
        assert alerts[0].analyte_name == "FASTING_BLOOD_SUGAR"
        assert alerts[0].severity == "critical"
        assert "hypoglycemia" in alerts[0].description
    finally:
        db.close()

def test_rule_6_antidiabetic_hba1c_elevated():
    """Rule 6: Antidiabetic Medications + HbA1c > 8.0 (Sub-optimal glycemic control)"""
    patient_id = setup_mock_patient_with_labs(
        phone="9876000006",
        results=[
            LabResultExtracted(
                raw_name="HbA1c",
                analyte_name="HBA1C",
                value=8.9,
                unit="%",
                ref_range="< 5.7 %",
                flag="high"
            )
        ]
    )
    
    db = SessionLocal()
    try:
        meds = [
            NormalizedDrug(
                raw_drug_name="Glycomet 500mg",
                inn="metformin",
                standard_name="metformin",
                is_active=True
            )
        ]
        alerts = check_clinical_safety(db, patient_id, meds)
        assert len(alerts) == 1
        assert alerts[0].analyte_name == "HBA1C"
        assert alerts[0].severity == "warning"
        assert "Poor glycemic control" in alerts[0].description
    finally:
        db.close()

def test_rule_7_antiplatelets_platelets_suppressed():
    """Rule 7: Antiplatelets + Platelet Count < 100k (Thrombocytopenia check)"""
    patient_id = setup_mock_patient_with_labs(
        phone="9876000007",
        results=[
            LabResultExtracted(
                raw_name="Platelet Count",
                analyte_name="PLATELET_COUNT",
                value=85.0,
                unit="x10^3/uL",
                ref_range="150 - 450 x10^3/uL",
                flag="low"
            )
        ]
    )
    
    db = SessionLocal()
    try:
        meds = [
            NormalizedDrug(
                raw_drug_name="Clopivas 75mg",
                inn="clopidogrel",
                standard_name="clopidogrel",
                is_active=True
            )
        ]
        alerts = check_clinical_safety(db, patient_id, meds)
        assert len(alerts) == 1
        assert alerts[0].analyte_name == "PLATELET_COUNT"
        assert alerts[0].severity == "critical"
        assert "thrombocytopenia" in alerts[0].description
    finally:
        db.close()

def test_rule_8_ace_arbs_creatinine_elevated():
    """Rule 8: ACE Inhibitor / ARB + Creatinine > 1.4 (AKI AKI/hyperkalemia risk)"""
    patient_id = setup_mock_patient_with_labs(
        phone="9876000008",
        results=[
            LabResultExtracted(
                raw_name="Serum Creatinine",
                analyte_name="CREATININE",
                value=1.65,
                unit="mg/dL",
                ref_range="0.6 - 1.2 mg/dL",
                flag="high"
            )
        ]
    )
    
    db = SessionLocal()
    try:
        meds = [
            NormalizedDrug(
                raw_drug_name="Telma 40mg",
                inn="telmisartan",
                standard_name="telmisartan",
                is_active=True
            )
        ]
        alerts = check_clinical_safety(db, patient_id, meds)
        assert len(alerts) == 1
        assert alerts[0].analyte_name == "CREATININE"
        assert alerts[0].severity == "warning"
        assert "RAS blocker" in alerts[0].description
    finally:
        db.close()

def test_rule_9_statins_triglycerides_elevated():
    """Rule 9: Statins + Triglycerides > 200 mg/dL"""
    patient_id = setup_mock_patient_with_labs(
        phone="9876000009",
        results=[
            LabResultExtracted(
                raw_name="Triglycerides",
                analyte_name="TRIGLYCERIDES",
                value=240.0,
                unit="mg/dL",
                ref_range="< 150 mg/dL",
                flag="high"
            )
        ]
    )
    
    db = SessionLocal()
    try:
        meds = [
            NormalizedDrug(
                raw_drug_name="Rozavel 10mg",
                inn="rosuvastatin",
                standard_name="rosuvastatin",
                is_active=True
            )
        ]
        alerts = check_clinical_safety(db, patient_id, meds)
        assert len(alerts) == 1
        tg_alert = next((a for a in alerts if a.analyte_name == "TRIGLYCERIDES"), None)
        assert tg_alert is not None
        assert tg_alert.severity == "info"
        assert "hypertriglyceridemia" in tg_alert.description
    finally:
        db.close()

def test_rule_10_antiplatelets_mild_anemia():
    """Rule 10: Antiplatelets + Mild Anemia (Hemoglobin 10.0-11.5)"""
    patient_id = setup_mock_patient_with_labs(
        phone="9876000010",
        results=[
            LabResultExtracted(
                raw_name="Hemoglobin",
                analyte_name="HEMOGLOBIN",
                value=10.8,
                unit="g/dL",
                ref_range="12.0 - 16.0 g/dL",
                flag="low"
            )
        ]
    )
    
    db = SessionLocal()
    try:
        meds = [
            NormalizedDrug(
                raw_drug_name="Ecosprin 75mg",
                inn="aspirin",
                standard_name="aspirin",
                is_active=True
            )
        ]
        alerts = check_clinical_safety(db, patient_id, meds)
        assert len(alerts) == 1
        assert alerts[0].analyte_name == "HEMOGLOBIN"
        assert alerts[0].severity == "info"
        assert "Mild anemia" in alerts[0].description
    finally:
        db.close()

def test_rule_11_myelosuppression_tlc_neut():
    """Rule 11: Immunosuppressants + TLC < 4.0 or Neutrophils < 40%"""
    # Test case A: Low TLC
    patient_id_tlc = setup_mock_patient_with_labs(
        phone="9876000011",
        results=[
            LabResultExtracted(
                raw_name="Total Leucocyte Count",
                analyte_name="TLC",
                value=3.2,
                unit="10^3/uL",
                ref_range="4.0 - 11.0 10^3/uL",
                flag="low"
            )
        ]
    )
    
    db = SessionLocal()
    try:
        meds = [
            NormalizedDrug(
                raw_drug_name="Methotrexate 7.5mg",
                inn="methotrexate",
                standard_name="methotrexate",
                is_active=True
            )
        ]
        alerts = check_clinical_safety(db, patient_id_tlc, meds)
        assert len(alerts) == 1
        assert alerts[0].analyte_name == "TLC"
        assert alerts[0].severity == "critical"
        assert "Leucopenia" in alerts[0].description
    finally:
        db.close()
        
    # Test case B: Suppressed Neutrophils < 40%
    patient_id_neut = setup_mock_patient_with_labs(
        phone="9876000012",
        results=[
            LabResultExtracted(
                raw_name="Polymorphs/Neutrophils",
                analyte_name="NEUTROPHILS",
                value=35.0,
                unit="%",
                ref_range="40.0 - 70.0 %",
                flag="low"
            )
        ]
    )
    
    db = SessionLocal()
    try:
        meds = [
            NormalizedDrug(
                raw_drug_name="Azathioprine 50mg",
                inn="azathioprine",
                standard_name="azathioprine",
                is_active=True
            )
        ]
        alerts = check_clinical_safety(db, patient_id_neut, meds)
        assert len(alerts) == 1
        assert alerts[0].analyte_name == "NEUTROPHILS"
        assert alerts[0].severity == "warning"
        assert "Relative neutropenia" in alerts[0].description
    finally:
        db.close()
