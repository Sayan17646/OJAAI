"""
tests/test_security_rbac.py — Security, RBAC, Scoping, and Safety Override acceptance tests.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from src.api import app, sign_session, SECRET_KEY
from src.database import (
    SessionLocal, Clinician, Facility, ReviewQueue, AuditLog, Patient,
    Medication, DrugInteractionRecord, Prescription, LabResult, LabReport,
    save_to_review_queue, hash_password,
    Doctor, MedicationEpisode, MedicationDosageHistory, PatientCondition, PhgEvent
)
from src.models import (
    ClinicianLoginInput, ResolvePrescriptionInput, ResolveMedicationInput,
    ResolveLabInput, ResolveLabResultInput
)

@pytest.fixture
def db_session():
    """Database session fixture."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture
def setup_rbac_data(db_session: Session):
    """Set up regional facilities, clinicians with specific scopes/roles, and patient files."""
    # Clean up any existing data first to prevent unique constraint failures
    db_session.query(AuditLog).delete()
    db_session.query(ReviewQueue).delete()
    db_session.query(PhgEvent).delete()
    db_session.query(PatientCondition).delete()
    db_session.query(MedicationDosageHistory).delete()
    db_session.query(MedicationEpisode).delete()
    db_session.query(Doctor).delete()
    db_session.query(Medication).delete()
    db_session.query(DrugInteractionRecord).delete()
    db_session.query(Prescription).delete()
    db_session.query(LabResult).delete()
    db_session.query(LabReport).delete()
    db_session.query(Patient).delete()
    db_session.query(Clinician).delete()
    db_session.query(Facility).delete()
    db_session.commit()


    # 1. Create Facilities
    fac_pune = Facility(name="Pune Diagnostics", code="PUNE_DIAGNOSTICS", address="Pune")
    fac_kothrud = Facility(name="Kothrud Pharmacy", code="KOTHRUD_PHARM", address="Kothrud")
    db_session.add_all([fac_pune, fac_kothrud])
    db_session.flush()

    # 2. Create Clinicians
    # Clinician 1: Auditor at Pune, scoped to Rx only
    c_pune_rx = Clinician(
        email="pune_rx@ojaai.com",
        hashed_password=hash_password("password123"),
        name="Dr. Pune Rx",
        role="auditor",
        scopes="rx"
    )
    c_pune_rx.facilities.append(fac_pune)

    # Clinician 2: Auditor at Pune, scoped to Lab only
    c_pune_lab = Clinician(
        email="pune_lab@ojaai.com",
        hashed_password=hash_password("password123"),
        name="Dr. Pune Lab",
        role="auditor",
        scopes="lab"
    )
    c_pune_lab.facilities.append(fac_pune)

    # Clinician 3: Auditor at Kothrud, scoped to both Rx & Lab
    c_kothrud_both = Clinician(
        email="kothrud_both@ojaai.com",
        hashed_password=hash_password("password123"),
        name="Dr. Kothrud Both",
        role="auditor",
        scopes="both"
    )
    c_kothrud_both.facilities.append(fac_kothrud)

    # Clinician 4: Admin at both clinics
    c_admin = Clinician(
        email="admin_all@ojaai.com",
        hashed_password=hash_password("admin123"),
        name="Super Admin",
        role="admin",
        scopes="both"
    )
    c_admin.facilities.extend([fac_pune, fac_kothrud])

    db_session.add_all([c_pune_rx, c_pune_lab, c_kothrud_both, c_admin])
    db_session.flush()

    # 3. Create patients
    patient = Patient(phone="9876543210", facility_id=fac_pune.id)
    db_session.add(patient)
    db_session.flush()

    # 4. Create review queue items
    # Item 1: Rx at Pune
    q_pune_rx = ReviewQueue(
        patient_id=patient.id,
        facility_id=fac_pune.id,
        image_path="./data/prescriptions/rx1.png",
        raw_ocr_text="Dr. Sharma. 1. Tab. Warfarin 5mg OD\n2. Tab. Ibuprofen 400mg BD",
        confidence=0.35,
        reason="Low confidence Rx",
        item_type="prescription",
        resolved=False
    )
    # Item 2: Lab at Pune
    q_pune_lab = ReviewQueue(
        patient_id=patient.id,
        facility_id=fac_pune.id,
        image_path="./data/lab_reports/lab1.png",
        raw_ocr_text="CREATININE: 1.8 mg/dL\nHEMOGLOBIN: 9.2 g/dL",
        confidence=0.4,
        reason="Low confidence Lab",
        item_type="lab",
        resolved=False
    )
    # Item 3: Rx at Kothrud
    q_kothrud_rx = ReviewQueue(
        patient_id=patient.id,
        facility_id=fac_kothrud.id,
        image_path="./data/prescriptions/rx2.png",
        raw_ocr_text="Tab. Glycomet 500mg BD",
        confidence=0.45,
        reason="Low confidence Rx at Kothrud",
        item_type="prescription",
        resolved=False
    )

    db_session.add_all([q_pune_rx, q_pune_lab, q_kothrud_rx])
    db_session.commit()

    return {
        "fac_pune_id": str(fac_pune.id),
        "fac_kothrud_id": str(fac_kothrud.id),
        "c_pune_rx": c_pune_rx,
        "c_pune_lab": c_pune_lab,
        "c_kothrud_both": c_kothrud_both,
        "c_admin": c_admin,
        "q_pune_rx_id": str(q_pune_rx.id),
        "q_pune_lab_id": str(q_pune_lab.id),
        "q_kothrud_rx_id": str(q_kothrud_rx.id),
        "patient": patient,
    }


class TestSecurityRBAC:
    """Security and RBAC Integration Test Cases."""

    @pytest.fixture(autouse=True)
    def disable_local_bypass(self, monkeypatch):
        monkeypatch.setenv("DEBUG_LOCAL_DASHBOARD", "false")

    def test_login_success(self, db_session, setup_rbac_data):
        """Verify successful clinician login sets HttpOnly session and CSRF cookies."""
        client = TestClient(app)
        login_payload = {
            "email": "pune_rx@ojaai.com",
            "password": "password123"
        }
        resp = client.post("/api/auth/login", json=login_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["clinician"]["email"] == "pune_rx@ojaai.com"
        assert len(data["clinician"]["facilities"]) == 1
        
        # Verify cookies
        assert "session_id" in client.cookies
        assert "csrf_token" in client.cookies
        
        # Logout
        resp_logout = client.post("/api/auth/logout")
        assert resp_logout.status_code == 200
        assert "session_id" not in client.cookies
        assert "csrf_token" not in client.cookies

    def test_login_failure(self, db_session, setup_rbac_data):
        """Verify invalid credentials return 401."""
        client = TestClient(app)
        login_payload = {
            "email": "pune_rx@ojaai.com",
            "password": "wrongpassword"
        }
        resp = client.post("/api/auth/login", json=login_payload)
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid email or password."

    def test_double_submit_csrf_enforcement(self, db_session, setup_rbac_data):
        """Verify mutative POST requests reject calls missing proper CSRF headers."""
        client = TestClient(app)
        # Login to establish session
        client.post("/api/auth/login", json={"email": "pune_rx@ojaai.com", "password": "password123"})
        
        # Attempt resolution without X-CSRF-Token header
        item_id = setup_rbac_data["q_pune_rx_id"]
        resolve_payload = {
            "patient_phone": "9876543210",
            "medications": [
                {
                    "raw_drug_name": "Glycomet",
                    "dosage_value": 500.0,
                    "dosage_unit": "mg",
                    "frequency": "twice daily",
                    "freq_per_day": 2,
                    "duration_days": 30,
                    "route": "oral"
                }
            ]
        }
        resp = client.post(f"/api/review/{item_id}/resolve", json=resolve_payload)
        assert resp.status_code == 403
        assert "CSRF token validation failed" in resp.json()["detail"]

        # Attempt resolution WITH correct X-CSRF-Token header
        csrf_val = client.cookies.get("csrf_token")
        resp_csrf = client.post(
            f"/api/review/{item_id}/resolve",
            json=resolve_payload,
            headers={"X-CSRF-Token": csrf_val}
        )
        assert resp_csrf.status_code == 200

    def test_facility_scope_isolation(self, db_session, setup_rbac_data):
        """Verify a clinician cannot query or resolve items belonging to another facility."""
        client = TestClient(app)
        # Login as Pune Rx clinician
        client.post("/api/auth/login", json={"email": "pune_rx@ojaai.com", "password": "password123"})
        csrf_val = client.cookies.get("csrf_token")
        
        # 1. Verify Pune queue only returns Pune items, not Kothrud items
        resp_queue = client.get(
            "/api/review/queue",
            headers={"X-Active-Facility": setup_rbac_data["fac_pune_id"]}
        )
        assert resp_queue.status_code == 200
        items = resp_queue.json()
        assert len(items) > 0
        assert all(setup_rbac_data["fac_kothrud_id"] not in str(x.get("facility_id", "")) for x in items)

        # 2. Try to set active facility to Kothrud (unassigned facility) -> should return 403
        resp_forbidden_fac = client.get(
            "/api/review/queue",
            headers={"X-Active-Facility": setup_rbac_data["fac_kothrud_id"]}
        )
        assert resp_forbidden_fac.status_code == 403

        # 3. Try to query Kothrud detail directly -> should return 403
        kothrud_item_id = setup_rbac_data["q_kothrud_rx_id"]
        resp_detail = client.get(f"/api/review/{kothrud_item_id}")
        assert resp_detail.status_code == 403

    def test_document_scope_restrictions(self, db_session, setup_rbac_data):
        """Verify rx-only clinician cannot retrieve/resolve lab documents, and vice versa."""
        client = TestClient(app)
        
        # Case A: Pune RX clinician (scopes = rx)
        client.post("/api/auth/login", json={"email": "pune_rx@ojaai.com", "password": "password123"})
        csrf_val = client.cookies.get("csrf_token")
        
        # Querying queue should return ONLY prescriptions, not labs
        resp_q = client.get("/api/review/queue", headers={"X-Active-Facility": setup_rbac_data["fac_pune_id"]})
        assert resp_q.status_code == 200
        assert all(x["item_type"] == "prescription" for x in resp_q.json())
        
        # Direct query to lab detail should return 403
        resp_det = client.get(f"/api/review/{setup_rbac_data['q_pune_lab_id']}")
        assert resp_det.status_code == 403

        # Direct resolve of lab should return 403
        lab_payload = {
            "patient_phone": "9876543210",
            "results": []
        }
        resp_res = client.post(
            f"/api/review/lab/{setup_rbac_data['q_pune_lab_id']}/resolve",
            json=lab_payload,
            headers={"X-CSRF-Token": csrf_val}
        )
        assert resp_res.status_code == 403

    def test_supervised_safety_overrides_escalation(self, db_session, setup_rbac_data):
        """Verify resolving a high-severity alert without override_reason fails, and with reason, escalates to Admin oversight."""
        # Clean up database queues/audit logs to have a clean state for this test
        db_session.query(AuditLog).delete()
        db_session.query(ReviewQueue).delete()
        db_session.commit()
        
        # Create a fresh Pune Rx item containing Warfarin + Ibuprofen (DDI)
        fac_id = setup_rbac_data["fac_pune_id"]
        item = save_to_review_queue(
            db=db_session,
            patient=setup_rbac_data["patient"],
            image_path="./data/prescriptions/ddi_rx.png",
            raw_ocr_text="Dr. Sharma. 1. Tab. Warfarin 5mg OD\n2. Tab. Ibuprofen 400mg BD",
            confidence=0.35,
            reason="Low confidence Rx with DDI",
            item_type="prescription",
            facility_id=db_session.query(Facility).filter(Facility.code == "PUNE_DIAGNOSTICS").first().id
        )
        db_session.commit()
        item_id = str(item.id)

        client = TestClient(app)
        # Login as Clinician with both scopes
        client.post("/api/auth/login", json={"email": "kothrud_both@ojaai.com", "password": "password123"})
        csrf_val = client.cookies.get("csrf_token")
        
        # Set active facility to Kothrud (wait, kothrud_both is unassigned to Pune! We must seed it to Pune or login as admin_all!)
        # Let's login as admin_all@ojaai.com who is assigned to both Pune and Kothrud!
        client.post("/api/auth/login", json={"email": "admin_all@ojaai.com", "password": "admin123"})
        csrf_val = client.cookies.get("csrf_token")

        resolve_payload = {
            "patient_phone": "9876543210",
            "medications": [
                {
                    "raw_drug_name": "Warfarin",
                    "dosage_value": 5.0,
                    "dosage_unit": "mg",
                    "frequency": "once daily",
                    "freq_per_day": 1,
                    "duration_days": 10,
                    "route": "oral"
                },
                {
                    "raw_drug_name": "Ibuprofen",
                    "dosage_value": 400.0,
                    "dosage_unit": "mg",
                    "frequency": "twice daily",
                    "freq_per_day": 2,
                    "duration_days": 10,
                    "route": "oral"
                }
            ],
            "override_reason": ""  # Missing/empty override reason!
        }

        # 1. Expect 400 Bad Request because of missing justification
        resp_fail = client.post(
            f"/api/review/{item_id}/resolve",
            json=resolve_payload,
            headers={"X-CSRF-Token": csrf_val, "X-Active-Facility": fac_id}
        )
        assert resp_fail.status_code == 400
        assert "override reason is required" in resp_fail.json()["detail"].lower()

        # 2. Add override justification and submit -> should succeed
        resolve_payload["override_reason"] = "Patient is closely monitored in inpatient setting with frequent INR monitoring."
        resp_success = client.post(
            f"/api/review/{item_id}/resolve",
            json=resolve_payload,
            headers={"X-CSRF-Token": csrf_val, "X-Active-Facility": fac_id}
        )
        assert resp_success.status_code == 200
        
        # 3. Verify safety override logged correctly and flagged for admin oversight
        db_session.expire_all()
        audit_entry = db_session.query(AuditLog).filter(AuditLog.item_id == item.id).first()
        assert audit_entry is not None
        assert audit_entry.action == "override_safety"
        assert audit_entry.override_reason == "Patient is closely monitored in inpatient setting with frequent INR monitoring."
        assert audit_entry.needs_admin_oversight is True

    def test_admin_oversight_endpoints(self, db_session, setup_rbac_data):
        """Verify Admin Oversight panel endpoints: role validation, CSRF, and approvals."""
        client = TestClient(app)
        
        # 1. Access by standard auditor (Scopes = rx) -> Should return 403 Forbidden
        client.post("/api/auth/login", json={"email": "pune_rx@ojaai.com", "password": "password123"})
        
        resp_get_auditor = client.get("/api/admin/audit-logs")
        assert resp_get_auditor.status_code == 403
        assert "privileges required" in resp_get_auditor.json()["detail"].lower()
        
        # 2. Access by Admin -> Should return 200 OK
        client.post("/api/auth/login", json={"email": "admin_all@ojaai.com", "password": "admin123"})
        csrf_val = client.cookies.get("csrf_token")
        
        # Ingest a safety override to audit first
        # We need an AuditLog entry with needs_admin_oversight = True
        audit_entry = AuditLog(
            clinician_id=setup_rbac_data["c_pune_rx"].id,
            facility_id=db_session.query(Facility).filter(Facility.code == "PUNE_DIAGNOSTICS").first().id,
            item_id=setup_rbac_data["q_pune_rx_id"],
            action="override_safety",
            override_reason="Test override reason.",
            needs_admin_oversight=True
        )
        db_session.add(audit_entry)
        db_session.commit()
        log_id = str(audit_entry.id)
        
        # Get active audit logs
        resp_get_admin = client.get("/api/admin/audit-logs")
        assert resp_get_admin.status_code == 200
        logs = resp_get_admin.json()
        assert len(logs) > 0
        assert any(x["id"] == log_id for x in logs)
        
        # Try to approve audit log WITHOUT CSRF -> Should return 403 Forbidden
        resp_approve_no_csrf = client.post(f"/api/admin/audit-logs/{log_id}/approve")
        assert resp_approve_no_csrf.status_code == 403
        assert "csrf token validation failed" in resp_approve_no_csrf.json()["detail"].lower()
        
        # Approve audit log WITH CSRF -> Should return 200 OK
        resp_approve = client.post(
            f"/api/admin/audit-logs/{log_id}/approve",
            headers={"X-CSRF-Token": csrf_val}
        )
        assert resp_approve.status_code == 200
        assert resp_approve.json()["status"] == "success"
        
        # Verify database shows needs_admin_oversight = False
        db_session.expire_all()
        updated_log = db_session.query(AuditLog).filter(AuditLog.id == audit_entry.id).first()
        assert updated_log.needs_admin_oversight is False

    def test_auth_me_endpoint(self, db_session, setup_rbac_data):
        """Verify GET /api/auth/me returns details for logged in clinician, and 401 if unauthenticated."""
        client = TestClient(app)
        
        # 1. Unauthenticated -> Should return 401
        resp_unauth = client.get("/api/auth/me")
        assert resp_unauth.status_code == 401
        
        # 2. Authenticated -> Should return clinician details
        client.post("/api/auth/login", json={"email": "pune_rx@ojaai.com", "password": "password123"})
        resp_auth = client.get("/api/auth/me")
        assert resp_auth.status_code == 200
        data = resp_auth.json()
        assert data["status"] == "success"
        assert data["clinician"]["email"] == "pune_rx@ojaai.com"
        assert data["clinician"]["role"] == "auditor"


