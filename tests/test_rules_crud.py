"""
tests/test_rules_crud.py — Security, RBAC, CRUD, and dynamic CDS evaluator integration tests.
"""
from __future__ import annotations

from datetime import date
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from src.api import app
from src.database import (
    SessionLocal, Clinician, Facility, AuditLog, Patient,
    LabReport, LabResult, ClinicalSafetyRule, ReviewQueue, create_all_tables, hash_password,
    Doctor, MedicationEpisode, MedicationDosageHistory, PatientCondition, PhgEvent
)
from src.models import LabResultExtracted, NormalizedDrug
from src.clinical_checker import check_clinical_safety

create_all_tables()

@pytest.fixture
def db_session():
    """Database session fixture."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture(autouse=True)
def disable_local_bypass(monkeypatch):
    """Enforce CSRF and RBAC during testing by turning off the local bypass flag."""
    monkeypatch.setenv("DEBUG_LOCAL_DASHBOARD", "false")

@pytest.fixture
def setup_rules_data(db_session: Session):
    """Set up baseline data: facility, admin clinician, auditor clinician, and a test patient."""
    db_session.query(ClinicalSafetyRule).delete()
    db_session.query(AuditLog).delete()
    db_session.query(ReviewQueue).delete()
    db_session.query(PhgEvent).delete()
    db_session.query(PatientCondition).delete()
    db_session.query(MedicationDosageHistory).delete()
    db_session.query(MedicationEpisode).delete()
    db_session.query(Doctor).delete()
    db_session.query(LabResult).delete()
    db_session.query(LabReport).delete()
    db_session.query(Patient).delete()
    db_session.query(Clinician).delete()
    db_session.query(Facility).delete()
    db_session.commit()


    # 1. Facility
    fac = Facility(name="Test Rule Clinic", code="RULE_CLINIC", address="Pune")
    db_session.add(fac)
    db_session.flush()

    # 2. Clinicians
    c_admin = Clinician(
        email="admin_rules@ojaai.com",
        hashed_password=hash_password("admin123"),
        name="Dr. Rules Admin",
        role="admin",
        scopes="both"
    )
    c_admin.facilities.append(fac)

    c_auditor = Clinician(
        email="auditor_rules@ojaai.com",
        hashed_password=hash_password("password123"),
        name="Dr. Rules Auditor",
        role="auditor",
        scopes="both"
    )
    c_auditor.facilities.append(fac)

    db_session.add_all([c_admin, c_auditor])
    db_session.flush()

    # 3. Patient
    patient = Patient(phone="9876543299", facility_id=fac.id)
    db_session.add(patient)
    db_session.commit()

    return {
        "fac_id": str(fac.id),
        "c_admin": c_admin,
        "c_auditor": c_auditor,
        "patient": patient,
    }


def test_rules_endpoints_rbac_and_csrf(db_session, setup_rules_data):
    """Verify standard auditors are blocked (403), admins are allowed, and CSRF is enforced on mutations."""
    client = TestClient(app)

    # 1. Access by standard auditor -> GET should return 403 Forbidden
    client.post("/api/auth/login", json={"email": "auditor_rules@ojaai.com", "password": "password123"})
    resp_get_auditor = client.get("/api/admin/rules")
    assert resp_get_auditor.status_code == 403
    assert "privileges required" in resp_get_auditor.json()["detail"].lower()

    # 2. Access by Admin -> GET should return 200 OK
    client.post("/api/auth/login", json={"email": "admin_rules@ojaai.com", "password": "admin123"})
    csrf_val = client.cookies.get("csrf_token")
    resp_get_admin = client.get("/api/admin/rules")
    assert resp_get_admin.status_code == 200
    assert len(resp_get_admin.json()) == 0

    # 3. Attempt POST rule creation WITHOUT CSRF -> Should return 403 Forbidden
    payload = {
        "rule_name": "Test Rule Creatinine",
        "drug_inn": "metformin",
        "analyte_name": "CREATININE",
        "operator": ">",
        "threshold_value": 1.5,
        "description_template": "Elevated creatinine detected: {value} > {threshold}",
        "severity": "critical",
        "is_enabled": True
    }
    resp_post_no_csrf = client.post("/api/admin/rules", json=payload)
    assert resp_post_no_csrf.status_code == 403
    assert "csrf token validation failed" in resp_post_no_csrf.json()["detail"].lower()

    # 4. Attempt POST rule WITH CSRF -> Should succeed
    resp_post_csrf = client.post(
        "/api/admin/rules",
        json=payload,
        headers={"X-CSRF-Token": csrf_val}
    )
    assert resp_post_csrf.status_code == 200
    data = resp_post_csrf.json()
    assert data["status"] == "success"
    assert "rule_id" in data

    # Verify audit log recorded "create_rule"
    db_session.expire_all()
    audit = db_session.query(AuditLog).filter(AuditLog.action == "create_rule").first()
    assert audit is not None
    assert "Test Rule Creatinine" in audit.override_reason


def test_rules_crud_lifecycle(db_session, setup_rules_data):
    """Test entire Rule CRUD lifecycle: create, retrieve, update, toggle enabled status, and delete."""
    client = TestClient(app)

    # Login as Admin
    client.post("/api/auth/login", json={"email": "admin_rules@ojaai.com", "password": "admin123"})
    csrf_val = client.cookies.get("csrf_token")

    # 1. Create Rule
    payload = {
        "rule_name": "TSH Dose Check",
        "drug_inn": "levothyroxine",
        "analyte_name": "TSH",
        "operator": ">",
        "threshold_value": 4.5,
        "description_template": "High TSH: {value} uIU/mL > 4.5",
        "severity": "warning",
        "is_enabled": True
    }
    r_create = client.post("/api/admin/rules", json=payload, headers={"X-CSRF-Token": csrf_val})
    assert r_create.status_code == 200
    rule_id = r_create.json()["rule_id"]

    # 2. Retrieve Rules list
    r_get = client.get("/api/admin/rules")
    assert r_get.status_code == 200
    rules = r_get.json()
    assert len(rules) == 1
    assert rules[0]["rule_name"] == "TSH Dose Check"
    assert rules[0]["is_enabled"] is True
    assert rules[0]["version"] == 1
    assert rules[0]["is_deleted"] is False

    # 3. Update Rule (Toggle Active Status to False)
    payload["is_enabled"] = False
    r_update = client.put(f"/api/admin/rules/{rule_id}", json=payload, headers={"X-CSRF-Token": csrf_val})
    assert r_update.status_code == 200
    assert r_update.json()["status"] == "success"

    # Verify update logged in audit_logs
    db_session.expire_all()
    audit_update = db_session.query(AuditLog).filter(AuditLog.action == "update_rule").first()
    assert audit_update is not None
    assert "v2" in audit_update.override_reason

    # Retrieve rules list again and confirm toggle and version increment
    rules_toggled = client.get("/api/admin/rules").json()
    assert rules_toggled[0]["is_enabled"] is False
    assert rules_toggled[0]["version"] == 2

    # 4. Delete Rule (Soft Delete)
    r_delete = client.delete(f"/api/admin/rules/{rule_id}", headers={"X-CSRF-Token": csrf_val})
    assert r_delete.status_code == 200
    assert r_delete.json()["status"] == "success"

    # Verify delete logged in audit_logs as soft-delete with version
    db_session.expire_all()
    audit_delete = db_session.query(AuditLog).filter(AuditLog.action == "delete_rule").first()
    assert audit_delete is not None
    assert "Soft-deleted" in audit_delete.override_reason
    assert "v2" in audit_delete.override_reason

    # Retrieve rules list and verify it is empty (since soft-deleted rules are excluded)
    rules_empty = client.get("/api/admin/rules").json()
    assert len(rules_empty) == 0

    # Query database directly to confirm the rule record still exists but is soft-deleted
    deleted_rule_in_db = db_session.query(ClinicalSafetyRule).filter(ClinicalSafetyRule.id == rule_id).first()
    assert deleted_rule_in_db is not None
    assert deleted_rule_in_db.is_deleted is True
    assert deleted_rule_in_db.version == 2


def test_dynamic_rules_evaluator(db_session, setup_rules_data):
    """Verify dynamic safety checker rules evaluate correctly on lab results and fallback to hardcoded rules when empty."""
    # Step 1: Baseline fallback check (rules table empty)
    # Metformin + Creatinine = 1.9 (critical warning alert should trigger from baseline fallback rule 1)
    patient = setup_rules_data["patient"]
    patient_id = patient.id

    # Seed lab report
    lab_report = LabReport(
        patient_id=patient_id,
        image_path="./data/lab_reports/test_lab.png",
        lab_name="Test Rule Clinic",
        report_date=date(2026, 5, 28)
    )
    db_session.add(lab_report)
    db_session.flush()

    res_creat = LabResult(
        lab_report_id=lab_report.id,
        raw_name="Serum Creatinine",
        analyte_name="CREATININE",
        value=1.9,
        unit="mg/dL",
        ref_range="0.6 - 1.2 mg/dL",
        flag="high"
    )
    db_session.add(res_creat)
    db_session.commit()

    # Call clinical safety checks
    meds = [NormalizedDrug(raw_drug_name="Glycomet 500mg", inn="metformin", standard_name="metformin", is_active=True)]
    
    # Empty clinical safety rules table -> should fallback to baseline hardcoded rule 1
    alerts_fallback = check_clinical_safety(db_session, patient_id, meds)
    assert len(alerts_fallback) == 1
    assert alerts_fallback[0].analyte_name == "CREATININE"
    assert alerts_fallback[0].severity == "critical"
    assert "lactic acidosis" in alerts_fallback[0].description.lower()

    # Step 2: Seed custom dynamic rule and check it overrides fallback
    # Let's seed a custom, less severe Creatinine check (severity: info)
    custom_rule = ClinicalSafetyRule(
        rule_name="Custom Creatinine Info Alert",
        drug_inn="metformin",
        analyte_name="CREATININE",
        operator=">",
        threshold_value=1.0,
        description_template="Custom creatinine limit exceeded: {value} {unit} > {threshold} {unit}",
        severity="info",
        is_enabled=True,
        management_plan="Custom monitoring instruction."
    )
    db_session.add(custom_rule)
    db_session.commit()

    # Call clinical safety checks again -> rules table is populated, should run custom rules only and skip fallbacks
    alerts_dynamic = check_clinical_safety(db_session, patient_id, meds)
    assert len(alerts_dynamic) == 1
    assert alerts_dynamic[0].analyte_name == "CREATININE"
    assert alerts_dynamic[0].severity == "info"
    assert "custom creatinine limit exceeded" in alerts_dynamic[0].description.lower()
    assert alerts_dynamic[0].management == "Custom monitoring instruction."
