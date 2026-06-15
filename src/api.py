"""
api.py — FastAPI application for OJAAI.

Endpoints:
  POST /parse              → upload prescription image → PrescriptionOutput
  GET  /patient/{phone}/history      → last 50 prescriptions for a patient
  GET  /patient/{phone}/interactions → all active DDIs for a patient

Security: hardcoded API key via X-API-Key header (Phase 1 internal testing only).
Error handling: global exception handler — never exposes stack traces.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, Request, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

import hmac
import hashlib
import base64
import json
import uuid as uuid_lib
from datetime import datetime, timedelta

from src import database as db_module
from src import ddi_checker
from src import medical_ner
from src.database import (
    DrugInteractionRecord,
    Medication,
    Patient,
    Prescription,
    ReviewQueue,
    SessionLocal,
    create_all_tables,
    get_db,
    get_active_medications,
    get_or_create_patient,
    get_unresolved_review_items,
    get_review_item_by_id,
    resolve_review_item,
)
from src.models import (
    DrugInteraction,
    ErrorResponse,
    NormalizedDrug,
    PatientHistory,
    PatientInteractions,
    PrescriptionOutput,
    ReviewQueueItemResponse,
    ReviewQueueDetailResponse,
    ResolvePrescriptionInput,
    LabReportOutput,
    ResolveLabInput,
    ClinicianLoginInput,
    ClinicalSafetyRuleSchema,
    CreateClinicalSafetyRuleInput,
)
from src.pipeline import process_prescription

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("API_KEY", "secure-ojaai-rot-5678-auth")

app = FastAPI(
    title="OJAAI Prescription Intelligence API",
    description=(
        "Parses Indian prescription images and returns structured drug data "
        "with drug-drug interaction alerts. Phase 1 MVP."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def on_startup() -> None:
    """Create DB tables on server start if they don't exist."""
    create_all_tables()
    logger.info("OJAAI API started. DB tables verified.")


# ---------------------------------------------------------------------------
# Security dependency
# ---------------------------------------------------------------------------

def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    """Reject requests without the correct API key."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Catch-all: log the error, return safe JSON. Never expose stack trace."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. Please contact support.", "detail": None},
    )


# ---------------------------------------------------------------------------
# POST /parse
# ---------------------------------------------------------------------------

@app.post(
    "/parse",
    response_model=PrescriptionOutput,
    summary="Parse a prescription image",
    description=(
        "Upload a prescription image (JPEG/PNG/PDF). "
        "Optionally provide patient phone for drug history cross-check. "
        "Returns structured drug data and interaction alerts."
    ),
)
async def parse_prescription(
    image: UploadFile = File(..., description="Prescription image (JPEG/PNG/PDF, max 10MB)"),
    phone: Optional[str] = Form(None, description="Patient mobile number (10-digit)"),
    _: None = Depends(verify_api_key),
) -> PrescriptionOutput:
    # Validate file size (10MB limit)
    contents = await image.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Maximum file size is 10MB.")

    # Validate format
    filename = image.filename or "upload.jpg"
    suffix = filename.lower().split(".")[-1] if "." in filename else ""
    if suffix not in ("jpg", "jpeg", "png", "pdf"):
        raise HTTPException(
            status_code=422,
            detail="Unsupported file type. Accepted: JPEG, PNG, PDF.",
        )

    try:
        result = process_prescription(
            image_bytes=contents,
            filename=filename,
            phone=phone,
        )
    except ValueError as exc:
        # Corrupt/unreadable image
        raise HTTPException(status_code=422, detail=f"Could not read image: {exc}")

    return result


# ---------------------------------------------------------------------------
# GET /patient/{phone}/history
# ---------------------------------------------------------------------------

@app.get(
    "/patient/{phone}/history",
    response_model=PatientHistory,
    summary="Get prescription history for a patient",
    description="Returns the 50 most recent prescriptions for the given phone number.",
)
def get_patient_history(
    phone: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> PatientHistory:
    patient = db.query(Patient).filter(Patient.phone == phone).first()
    if not patient:
        return PatientHistory(phone=phone, total=0, prescriptions=[])

    prescriptions_orm = (
        db.query(Prescription)
        .filter(Prescription.patient_id == patient.id)
        .order_by(Prescription.created_at.desc())
        .limit(50)
        .all()
    )

    output_list: List[PrescriptionOutput] = []
    for rx in prescriptions_orm:
        meds = [
            NormalizedDrug(
                raw_drug_name=m.raw_drug_name,
                inn=m.inn or m.raw_drug_name,
                rxcui=m.rxcui,
                standard_name=m.standard_name or m.raw_drug_name,
                dosage_value=m.dosage_value,
                dosage_unit=m.dosage_unit,
                frequency=m.frequency,
                freq_per_day=m.freq_per_day,
                duration_days=m.duration_days,
                route=m.route or "oral",
                is_active=m.is_active,
            )
            for m in rx.medications
        ]
        interactions = [
            DrugInteraction(
                drug_1=i.drug_1,
                drug_2=i.drug_2,
                severity=i.severity,
                description=i.description or "",
                management=i.management,
                source=i.source or "CRITICAL_DDI_DB",
            )
            for i in rx.interactions
        ]
        has_major = any(i.severity == "major" for i in interactions)

        output_list.append(
            PrescriptionOutput(
                prescription_id=str(rx.id),
                patient_phone=phone,
                medications=meds,
                diagnosis=rx.diagnosis,
                doctor_reg=rx.doctor_reg,
                patient_age=rx.patient_age,
                prescription_date=str(rx.prescription_date) if rx.prescription_date else None,
                interactions=interactions,
                confidence=rx.confidence,
                needs_human_review=rx.confidence < float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),
                has_major_interaction=has_major,
                image_path=rx.image_path,
            )
        )

    return PatientHistory(phone=phone, total=len(output_list), prescriptions=output_list)


# ---------------------------------------------------------------------------
# GET /patient/{phone}/interactions
# ---------------------------------------------------------------------------

@app.get(
    "/patient/{phone}/interactions",
    response_model=PatientInteractions,
    summary="Get active drug interactions for a patient",
    description=(
        "Returns all drug-drug interactions across the patient's active medication list "
        "(prescribed within last 90 days or chronic). Sorted: major first."
    ),
)
def get_patient_interactions(
    phone: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> PatientInteractions:
    patient = db.query(Patient).filter(Patient.phone == phone).first()
    if not patient:
        return PatientInteractions(
            phone=phone,
            active_drugs=[],
            interactions=[],
            has_major_interaction=False,
        )

    active_meds = get_active_medications(db, patient.id)
    active_inns = list({m.inn for m in active_meds if m.inn})

    interactions = ddi_checker.check_interactions(active_inns)
    has_major = any(i.severity == "major" for i in interactions)

    return PatientInteractions(
        phone=phone,
        active_drugs=active_inns,
        interactions=interactions,
        has_major_interaction=has_major,
    )


# ---------------------------------------------------------------------------
# Health check (no auth required)
# ---------------------------------------------------------------------------

@app.get("/health", include_in_schema=False)
def health() -> dict:
    return {"status": "ok", "service": "OJAAI", "version": "0.1.0"}


# ---------------------------------------------------------------------------
# Session & CSRF Helper Functions
# ---------------------------------------------------------------------------

SECRET_KEY = os.getenv("SECRET_KEY", "ojaai-super-secret-key-12345-secured-clinical")

def sign_session(payload: dict) -> str:
    """Sign a session payload using HMAC-SHA256."""
    serialized = json.dumps(payload).encode('utf-8')
    b64_payload = base64.urlsafe_b64encode(serialized).decode('utf-8')
    sig = hmac.new(SECRET_KEY.encode('utf-8'), b64_payload.encode('utf-8'), hashlib.sha256).digest()
    b64_sig = base64.urlsafe_b64encode(sig).decode('utf-8')
    return f"{b64_payload}.{b64_sig}"


def verify_session(signed_token: str) -> Optional[dict]:
    """Verify a signed session token and return the payload."""
    if not signed_token or "." not in signed_token:
        return None
    try:
        b64_payload, b64_sig = signed_token.split(".", 1)
        sig = hmac.new(SECRET_KEY.encode('utf-8'), b64_payload.encode('utf-8'), hashlib.sha256).digest()
        expected_sig = base64.urlsafe_b64encode(sig).decode('utf-8')
        if not hmac.compare_digest(b64_sig, expected_sig):
            return None
        serialized = base64.urlsafe_b64decode(b64_payload.encode('utf-8'))
        return json.loads(serialized)
    except Exception:
        return None


def get_current_clinician(request: Request) -> dict:
    """Validate HttpOnly session cookie and return currently authenticated clinician."""
    signed_token = request.cookies.get("session_id")
    if not signed_token:
        # Fallback to Authorization Header for testing
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            signed_token = auth_header.split(" ", 1)[1]
            
    if not signed_token:
        raise HTTPException(status_code=401, detail="Authentication session expired or missing.")
        
    payload = verify_session(signed_token)
    if not payload:
        raise HTTPException(status_code=401, detail="Authentication session invalid.")
        
    exp_str = payload.get("exp")
    if not exp_str or datetime.fromisoformat(exp_str) <= datetime.utcnow():
        raise HTTPException(status_code=401, detail="Authentication session expired.")
        
    return payload


def verify_csrf_token(request: Request) -> None:
    """Validate double-submit CSRF cookie against request header for mutations."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
        
    is_debug = os.getenv("DEBUG_LOCAL_DASHBOARD", "false").lower() == "true"
    client_host = request.client.host if request.client else ""
    is_local = client_host in ("127.0.0.1", "localhost", "::1", "testclient")
    x_api_key = request.headers.get("X-API-Key")
    
    if (is_debug and is_local) or (x_api_key == API_KEY):
        return  # Bypass for local debug or API key
        
    csrf_cookie = request.cookies.get("csrf_token")
    csrf_header = request.headers.get("X-CSRF-Token")
    if not csrf_cookie or csrf_cookie != csrf_header:
        raise HTTPException(status_code=403, detail="CSRF token validation failed.")


def get_dashboard_clinician(request: Request, db: Session = Depends(get_db)) -> Optional[db_module.Clinician]:
    """
    Look up the clinician making the request, if authenticated via session cookie.
    If no session exists but local debug mode/API key is bypassed, returns None.
    """
    session_id = request.cookies.get("session_id")
    if session_id:
        payload = verify_session(session_id)
        if payload:
            exp_str = payload.get("exp")
            if exp_str and datetime.fromisoformat(exp_str) > datetime.utcnow():
                clinician_id = payload.get("clinician_id")
                if clinician_id:
                    import uuid
                    return db.query(db_module.Clinician).filter(db_module.Clinician.id == uuid.UUID(clinician_id)).first()
    
    is_debug = os.getenv("DEBUG_LOCAL_DASHBOARD", "false").lower() == "true"
    client_host = request.client.host if request.client else ""
    is_local = client_host in ("127.0.0.1", "localhost", "::1", "testclient")
    x_api_key = request.headers.get("X-API-Key")
    
    if (is_debug and is_local) or (x_api_key == API_KEY):
        return None  # Bypass, no scoped clinician
        
    raise HTTPException(status_code=401, detail="Unauthorized dashboard access.")


def get_active_facility_id(request: Request, clinician: Optional[db_module.Clinician]) -> Optional[str]:
    """Get the active facility ID from request headers or query parameters."""
    if not clinician:
        return None
        
    facility_id = request.headers.get("X-Active-Facility") or request.query_params.get("facility_id")
    if not facility_id:
        if clinician.facilities:
            return str(clinician.facilities[0].id)
        raise HTTPException(status_code=400, detail="No active facility selected, and clinician has no facilities assigned.")
        
    allowed_ids = {str(f.id) for f in clinician.facilities}
    if facility_id not in allowed_ids:
        raise HTTPException(status_code=403, detail="Access denied to the specified facility.")
        
    return facility_id



# ---------------------------------------------------------------------------
# Dashboard Security Dependency
# ---------------------------------------------------------------------------

def verify_dashboard_access(request: Request, x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> None:
    """
    Allow access to dashboard APIs if:
    1. DEBUG_LOCAL_DASHBOARD is 'true' in env AND request comes from localhost/127.0.0.1.
    2. A valid X-API-Key header is provided.
    3. OR, a valid HttpOnly session cookie is present!
    """
    is_debug = os.getenv("DEBUG_LOCAL_DASHBOARD", "false").lower() == "true"
    client_host = request.client.host if request.client else ""
    is_local = client_host in ("127.0.0.1", "localhost", "::1", "testclient")
    
    if is_debug and is_local:
        return
        
    if x_api_key == API_KEY:
        return
        
    # Check cookie session
    session_id = request.cookies.get("session_id")
    if session_id:
        payload = verify_session(session_id)
        if payload:
            exp_str = payload.get("exp")
            if exp_str and datetime.fromisoformat(exp_str) > datetime.utcnow():
                return
            
    raise HTTPException(
        status_code=401,
        detail="Unauthorized. Dashboard debug mode is inactive or request is not authenticated."
    )


# ---------------------------------------------------------------------------
# Dashboard REST APIs
# ---------------------------------------------------------------------------

@app.get(
    "/api/review/queue",
    response_model=List[ReviewQueueItemResponse],
    summary="Get all unresolved review queue items",
)
def get_review_queue(
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_dashboard_access),
) -> List[ReviewQueueItemResponse]:
    # Enforce facility scoping and scopes
    clinician = get_dashboard_clinician(request, db)
    facility_id = get_active_facility_id(request, clinician)
    
    query = db.query(ReviewQueue).filter(ReviewQueue.resolved == False)
    
    if facility_id:
        import uuid
        query = query.filter(ReviewQueue.facility_id == uuid.UUID(facility_id))
        
    if clinician:
        if clinician.scopes == "rx":
            query = query.filter(ReviewQueue.item_type == "prescription")
        elif clinician.scopes == "lab":
            query = query.filter(ReviewQueue.item_type == "lab")
            
    items = query.order_by(ReviewQueue.created_at.desc()).all()
    
    res = []
    for item in items:
        phone = item.patient.phone if item.patient else None
        res.append(
            ReviewQueueItemResponse(
                id=str(item.id),
                patient_id=str(item.patient_id) if item.patient_id else None,
                image_path=item.image_path,
                raw_ocr_text=item.raw_ocr_text or "",
                confidence=item.confidence,
                reason=item.reason or "",
                resolved=item.resolved,
                created_at=item.created_at.isoformat(),
                patient_phone=phone,
                item_type=getattr(item, "item_type", "prescription"),
            )
        )
    return res


@app.get(
    "/api/review/{id}",
    response_model=ReviewQueueDetailResponse,
    summary="Get full details for a review queue item including isolated dynamic suggestions",
)
def get_review_detail(
    id: str,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_dashboard_access),
) -> ReviewQueueDetailResponse:
    clinician = get_dashboard_clinician(request, db)
    
    item = get_review_item_by_id(db, id)
    if not item:
        raise HTTPException(status_code=404, detail="Review item not found.")
        
    # Enforce facility scoping and scopes
    if clinician:
        allowed_facility_ids = {f.id for f in clinician.facilities}
        if item.facility_id and item.facility_id not in allowed_facility_ids:
            raise HTTPException(status_code=403, detail="Access denied to this review item's facility.")
            
        if clinician.scopes == "rx" and item.item_type != "prescription":
            raise HTTPException(status_code=403, detail="Access denied. Document scope mismatch.")
        if clinician.scopes == "lab" and item.item_type != "lab":
            raise HTTPException(status_code=403, detail="Access denied. Document scope mismatch.")
            
    raw_text = item.raw_ocr_text or ""
    item_type = getattr(item, "item_type", "prescription")
    phone = item.patient.phone if item.patient else None
    
    draft_suggestion = None
    draft_lab_suggestion = None
    
    # Try parsing raw_text as JSON first (if populated by Gemini Vision)
    parsed_json = None
    if raw_text.strip().startswith("{") and raw_text.strip().endswith("}"):
        try:
            parsed_json = json.loads(raw_text)
        except Exception:
            pass

    if item_type == "lab":
        if parsed_json and ("results" in parsed_json or "results_list" in parsed_json):
            from src.models import LabResultExtracted, LabReportExtracted
            results = []
            for r in parsed_json.get("results", []):
                results.append(
                    LabResultExtracted(
                        raw_name=r.get("raw_name") or r.get("test_name") or "",
                        analyte_name=r.get("analyte_name") or "OTHER",
                        value=float(r.get("value", 0.0)),
                        unit=r.get("unit"),
                        ref_range=r.get("ref_range"),
                        flag=r.get("flag", "normal")
                    )
                )
            draft_lab_suggestion = LabReportExtracted(
                lab_name=parsed_json.get("lab_name"),
                report_date=parsed_json.get("report_date"),
                results=results,
                confidence=float(parsed_json.get("confidence", 0.8)),
                raw_text=raw_text
            )
        else:
            from src.lab_ner import extract_lab_report
            draft_lab_suggestion = extract_lab_report(raw_text)
    else:
        if parsed_json and "medications" in parsed_json:
            from src.models import MedicationExtracted, ExtractedPrescription
            meds = []
            for m in parsed_json.get("medications", []):
                meds.append(
                    MedicationExtracted(
                        raw_drug_name=m.get("raw_drug_name") or m.get("drug_name") or "",
                        dosage_value=m.get("dosage_value"),
                        dosage_unit=m.get("dosage_unit"),
                        frequency=m.get("frequency"),
                        freq_per_day=m.get("freq_per_day"),
                        duration_days=m.get("duration_days"),
                        route=m.get("route") or "oral"
                    )
                )
            draft_suggestion = ExtractedPrescription(
                medications=meds,
                diagnosis=parsed_json.get("diagnosis"),
                doctor_reg=parsed_json.get("doctor_reg"),
                patient_age=parsed_json.get("patient_age"),
                prescription_date=parsed_json.get("prescription_date"),
                confidence=float(parsed_json.get("confidence", 0.8)),
                raw_text=raw_text
            )
        else:
            # Dyn extraction isolated to prevent side-effects/mutations
            draft_suggestion = medical_ner.extract(raw_text)
    
    return ReviewQueueDetailResponse(
        id=str(item.id),
        image_path=item.image_path,
        raw_ocr_text=raw_text,
        confidence=item.confidence,
        reason=item.reason or "",
        created_at=item.created_at.isoformat(),
        patient_phone=phone,
        item_type=item_type,
        draft_suggestion=draft_suggestion,
        draft_lab_suggestion=draft_lab_suggestion,
    )



@app.post(
    "/api/review/{id}/resolve",
    response_model=PrescriptionOutput,
    summary="Resolve a review queue item with clinician corrected data",
)
def resolve_review_queue_item(
    id: str,
    payload: ResolvePrescriptionInput,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_dashboard_access),
    _csrf: None = Depends(verify_csrf_token),
) -> PrescriptionOutput:
    clinician = get_dashboard_clinician(request, db)
    
    item = get_review_item_by_id(db, id)
    if not item:
        raise HTTPException(status_code=404, detail="Review item not found.")
        
    # Enforce facility scoping and scopes
    if clinician:
        allowed_facility_ids = {f.id for f in clinician.facilities}
        if item.facility_id and item.facility_id not in allowed_facility_ids:
            raise HTTPException(status_code=403, detail="Access denied to this review item's facility.")
            
        if clinician.scopes == "lab":
            raise HTTPException(status_code=403, detail="Access denied. Document scope mismatch.")
            
    try:
        # Get or create patient if phone provided
        patient = None
        existing_inn_list: List[str] = []
        if payload.patient_phone:
            patient = get_or_create_patient(db, payload.patient_phone)
            existing_meds = get_active_medications(db, patient.id)
            existing_inn_list = [m.inn for m in existing_meds if m.inn]
            
        # Normalise the corrected drug inputs
        from src.models import MedicationExtracted
        from src.drug_normalizer import normalize_all
        import uuid as uuid_lib
        
        extracted_meds = [
            MedicationExtracted(
                raw_drug_name=m.raw_drug_name,
                dosage_value=m.dosage_value,
                dosage_unit=m.dosage_unit,
                frequency=m.frequency,
                freq_per_day=m.freq_per_day,
                duration_days=m.duration_days,
                route=m.route,
            )
            for m in payload.medications
        ]
        
        normalized_drugs = normalize_all(extracted_meds)
        
        # Trigger DDI check
        new_inn_list = [m.inn for m in normalized_drugs if m.inn]
        all_active_inns = list(set(existing_inn_list + new_inn_list))
        interactions = ddi_checker.check_interactions(all_active_inns)
        
        # Generate new prescription id
        prescription_id = str(uuid_lib.uuid4())
        
        # Convert date string if present
        date_str = payload.prescription_date or ""
        
        # Save prescription to main tables
        from src.database import save_prescription_to_db
        rx = save_prescription_to_db(
            db=db,
            prescription_id=prescription_id,
            patient=patient,
            image_path=item.image_path,
            raw_ocr_text=item.raw_ocr_text or "",
            confidence=1.0,  # clinical resolution always has 1.0 confidence
            doctor_reg=payload.doctor_reg,
            patient_age=payload.patient_age,
            diagnosis=payload.diagnosis,
            prescription_date=date_str if date_str else None,
            medications=normalized_drugs,
            interactions=interactions,
            facility_id=item.facility_id,
        )
        
        # Check clinical safety alerts (drug-laboratory warnings)
        safety_alerts = []
        if patient:
            from src.clinical_checker import check_clinical_safety
            safety_alerts = check_clinical_safety(db, patient.id, normalized_drugs)

        has_major = any(i.severity == "major" for i in interactions)
        has_critical_safety = any(s.severity == "critical" for s in safety_alerts)
        
        # Enforce supervised overrides if high-priority alerts triggered
        if has_major or has_critical_safety:
            if not payload.override_reason or not payload.override_reason.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Safety override reason is required for major drug-drug interactions or critical lab conflicts."
                )
                
            # Log audited safety override
            audit_clinician_id = clinician.id if clinician else db.query(db_module.Clinician).first().id
            audit_facility_id = item.facility_id if item.facility_id else (clinician.facilities[0].id if clinician and clinician.facilities else db.query(db_module.Facility).first().id)
            
            audit = db_module.AuditLog(
                clinician_id=audit_clinician_id,
                facility_id=audit_facility_id,
                item_id=item.id,
                action="override_safety",
                override_reason=payload.override_reason,
                needs_admin_oversight=True
            )
            db.add(audit)
        else:
            # Standard resolution log
            if clinician:
                audit_facility_id = item.facility_id if item.facility_id else (clinician.facilities[0].id if clinician.facilities else db.query(db_module.Facility).first().id)
                audit = db_module.AuditLog(
                    clinician_id=clinician.id,
                    facility_id=audit_facility_id,
                    item_id=item.id,
                    action="resolve_rx",
                    override_reason=None,
                    needs_admin_oversight=False
                )
                db.add(audit)

        # Mark review item resolved and bind to patient if provided
        item.resolved = True
        if patient:
            item.patient_id = patient.id
        
        # Commit single atomic transaction
        db.commit()
        
        return PrescriptionOutput(
            prescription_id=prescription_id,
            patient_phone=payload.patient_phone,
            medications=normalized_drugs,
            diagnosis=payload.diagnosis,
            doctor_reg=payload.doctor_reg,
            patient_age=payload.patient_age,
            prescription_date=payload.prescription_date,
            interactions=interactions,
            confidence=1.0,
            needs_human_review=False,
            has_major_interaction=has_major,
            image_path=item.image_path,
            safety_alerts=safety_alerts,
        )
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        logger.error("Failed to resolve review queue item id=%s: %s", id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database transaction rolled back due to error: {exc}")



@app.post(
    "/api/lab/parse",
    response_model=LabReportOutput,
    summary="Parse a diagnostic lab report image",
    description=(
        "Upload a lab report image (JPEG/PNG/PDF). "
        "Optionally provide patient phone to register results in their health history. "
        "Extracts biomarkers and reference ranges."
    ),
)
async def parse_lab_report(
    image: UploadFile = File(..., description="Lab report image (JPEG/PNG/PDF, max 10MB)"),
    phone: Optional[str] = Form(None, description="Patient mobile number (10-digit)"),
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
) -> LabReportOutput:
    from pathlib import Path
    
    # Validate file size (10MB limit)
    contents = await image.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Maximum file size is 10MB.")

    # Validate format
    filename = image.filename or "upload.jpg"
    suffix = filename.lower().split(".")[-1] if "." in filename else ""
    if suffix not in ("jpg", "jpeg", "png", "pdf"):
        raise HTTPException(
            status_code=422,
            detail="Unsupported file type. Accepted: JPEG, PNG, PDF.",
        )

    # Save image to filesystem
    from src.pipeline import _save_image, _run_ocr
    from src import preprocessor
    from src.lab_ner import extract_lab_report
    from src.database import save_lab_report_to_db, get_or_create_patient
    import uuid as uuid_lib

    # 1. Save image
    suffix = Path(filename).suffix.lower() or ".jpg"
    unique_name = f"{uuid_lib.uuid4()}{suffix}"
    dest_dir = Path("./data/lab_reports")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / unique_name
    dest.write_bytes(contents)
    image_path = str(dest)

    # 2. Extraction: Gemini Vision -> Fallback Preprocess & OCR & regex NER
    from src.gemini_extractor import is_available as is_gemini_available
    
    extracted = None
    raw_text = ""
    
    if is_gemini_available():
        try:
            from src.gemini_extractor import extract_lab_report as gemini_extract_lab
            logger.info("Using Gemini Vision extractor for lab report")
            gemini_res = gemini_extract_lab(contents)
            if gemini_res:
                from src.models import LabResultExtracted, LabReportExtracted
                results_list = [
                    LabResultExtracted(
                        raw_name=r.get("raw_name") or r.get("test_name", ""),
                        analyte_name=r.get("analyte_name", "OTHER"),
                        value=r.get("value", 0.0),
                        unit=r.get("unit"),
                        ref_range=r.get("ref_range"),
                        flag=r.get("flag", "normal").lower(),
                    )
                    for r in gemini_res.get("results", [])
                ]
                extracted = LabReportExtracted(
                    lab_name=gemini_res.get("lab_name"),
                    report_date=gemini_res.get("report_date"),
                    results=results_list,
                    confidence=float(gemini_res.get("confidence", 0.7)),
                    raw_text=gemini_res.get("raw_text", ""),
                )
        except Exception as e:
            logger.warning("Gemini lab extractor failed, falling back to Tesseract: %s", e)
            
    if not extracted:
        # Fallback Preprocess & OCR & regex NER
        try:
            clean_image = preprocessor.preprocess_from_bytes(contents, filename)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not preprocess image: {exc}")

        raw_text = _run_ocr(clean_image)
        extracted = extract_lab_report(raw_text)

    needs_human_review = extracted.confidence < 0.5

    # 4. Save to DB
    try:
        patient = None
        if phone:
            patient = get_or_create_patient(db, phone)
            
        if needs_human_review:
            from src.database import save_to_review_queue
            reason = "Lab report confidence below threshold."
            if not extracted.report_date:
                reason = "Missing report date."
            elif len(extracted.results) == 0:
                reason = "No analytes detected."
                
            queue_item = save_to_review_queue(
                db=db,
                patient=patient,
                image_path=image_path,
                raw_ocr_text=raw_text,
                confidence=extracted.confidence,
                reason=reason,
                item_type="lab"
            )
            db.commit()
            lab_report_id = str(queue_item.id)
        else:
            report_orm = save_lab_report_to_db(
                db=db,
                patient=patient,
                image_path=image_path,
                lab_name=extracted.lab_name,
                report_date=extracted.report_date,
                results=extracted.results,
            )
            db.commit()
            lab_report_id = str(report_orm.id)
            
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database write failed: {exc}")

    return LabReportOutput(
        lab_report_id=lab_report_id,
        patient_phone=phone,
        lab_name=extracted.lab_name,
        report_date=extracted.report_date,
        results=extracted.results,
        confidence=extracted.confidence,
        needs_human_review=needs_human_review,
        image_path=image_path,
    )


@app.post(
    "/api/review/lab/{id}/resolve",
    response_model=LabReportOutput,
    summary="Resolve a lab review queue item with clinician corrected data",
)
def resolve_lab_review_queue_item(
    id: str,
    payload: ResolveLabInput,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_dashboard_access),
    _csrf: None = Depends(verify_csrf_token),
) -> LabReportOutput:
    clinician = get_dashboard_clinician(request, db)
    
    item = get_review_item_by_id(db, id)
    if not item:
        raise HTTPException(status_code=404, detail="Review item not found.")
        
    # Enforce facility scoping and scopes
    if clinician:
        allowed_facility_ids = {f.id for f in clinician.facilities}
        if item.facility_id and item.facility_id not in allowed_facility_ids:
            raise HTTPException(status_code=403, detail="Access denied to this review item's facility.")
            
        if clinician.scopes == "rx":
            raise HTTPException(status_code=403, detail="Access denied. Document scope mismatch.")
            
    try:
        # Get or create patient if phone provided
        patient = None
        if payload.patient_phone:
            patient = get_or_create_patient(db, payload.patient_phone)
            
        # Parse the results
        from src.models import LabResultExtracted
        results_extracted = [
            LabResultExtracted(
                raw_name=r.raw_name,
                analyte_name=r.analyte_name,
                value=r.value,
                unit=r.unit,
                ref_range=r.ref_range,
                flag=r.flag,
            )
            for r in payload.results
        ]
        
        # Save lab report to main tables
        from src.database import save_lab_report_to_db
        report_orm = save_lab_report_to_db(
            db=db,
            patient=patient,
            image_path=item.image_path,
            lab_name=payload.lab_name,
            report_date=payload.report_date,
            results=results_extracted,
            facility_id=item.facility_id,
        )
        
        # Flush to DB so the latest results can be queried by check_clinical_safety
        db.flush()
        
        # Check clinical safety alerts (drug-laboratory warnings)
        safety_alerts = []
        if patient:
            active_meds = get_active_medications(db, patient.id)
            pydantic_meds = [
                NormalizedDrug(
                    raw_drug_name=m.raw_drug_name,
                    inn=m.inn or m.raw_drug_name,
                    rxcui=m.rxcui,
                    standard_name=m.standard_name or m.raw_drug_name,
                    dosage_value=m.dosage_value,
                    dosage_unit=m.dosage_unit,
                    frequency=m.frequency,
                    freq_per_day=m.freq_per_day,
                    duration_days=m.duration_days,
                    route=m.route or "oral",
                    is_active=m.is_active
                )
                for m in active_meds
            ]
            from src.clinical_checker import check_clinical_safety
            safety_alerts = check_clinical_safety(db, patient.id, pydantic_meds)

        has_critical_safety = any(s.severity == "critical" for s in safety_alerts)
        
        # Enforce supervised overrides if high-priority alerts triggered
        if has_critical_safety:
            if not payload.override_reason or not payload.override_reason.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Safety override reason is required for major drug-drug interactions or critical lab conflicts."
                )
                
            # Log audited safety override
            audit_clinician_id = clinician.id if clinician else db.query(db_module.Clinician).first().id
            audit_facility_id = item.facility_id if item.facility_id else (clinician.facilities[0].id if clinician and clinician.facilities else db.query(db_module.Facility).first().id)
            
            audit = db_module.AuditLog(
                clinician_id=audit_clinician_id,
                facility_id=audit_facility_id,
                item_id=item.id,
                action="override_safety",
                override_reason=payload.override_reason,
                needs_admin_oversight=True
            )
            db.add(audit)
        else:
            # Standard resolution log
            if clinician:
                audit_facility_id = item.facility_id if item.facility_id else (clinician.facilities[0].id if clinician.facilities else db.query(db_module.Facility).first().id)
                audit = db_module.AuditLog(
                    clinician_id=clinician.id,
                    facility_id=audit_facility_id,
                    item_id=item.id,
                    action="resolve_lab",
                    override_reason=None,
                    needs_admin_oversight=False
                )
                db.add(audit)

        # Mark review item resolved and bind to patient if provided
        item.resolved = True
        if patient:
            item.patient_id = patient.id
            
        # Commit single atomic transaction
        db.commit()
        
        return LabReportOutput(
            lab_report_id=str(report_orm.id),
            patient_phone=payload.patient_phone,
            lab_name=payload.lab_name,
            report_date=payload.report_date,
            results=results_extracted,
            confidence=1.0,  # human resolved is 1.0 confidence
            needs_human_review=False,
            image_path=item.image_path,
        )
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        logger.error("Failed to resolve lab review queue item id=%s: %s", id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Database transaction rolled back due to error: {exc}")



@app.post(
    "/api/review/upload",
    response_model=ReviewQueueItemResponse,
    summary="Directly upload any image from the dashboard for instant parsing and dynamic auditing",
)
async def upload_for_dashboard_audit(
    request: Request,
    image: UploadFile = File(..., description="Prescription or lab report image"),
    item_type: str = Form(..., description="prescription or lab"),
    phone: Optional[str] = Form(None, description="Patient mobile number (10-digit)"),
    db: Session = Depends(get_db),
    _: None = Depends(verify_dashboard_access),
    _csrf: None = Depends(verify_csrf_token),
) -> ReviewQueueItemResponse:
    clinician = get_dashboard_clinician(request, db)
    facility_id = get_active_facility_id(request, clinician)
    
    # Enforce scopes
    if clinician:
        if clinician.scopes == "rx" and item_type != "prescription":
            raise HTTPException(status_code=403, detail="Access denied. Document scope mismatch.")
        if clinician.scopes == "lab" and item_type != "lab":
            raise HTTPException(status_code=403, detail="Access denied. Document scope mismatch.")
            
    from pathlib import Path
    import uuid as uuid_lib
    from src.database import save_to_review_queue, get_or_create_patient

    # Validate file size (10MB limit)
    contents = await image.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Maximum file size is 10MB.")

    # Validate format
    filename = image.filename or "upload.jpg"
    suffix = filename.lower().split(".")[-1] if "." in filename else ""
    if suffix not in ("jpg", "jpeg", "png", "pdf"):
        raise HTTPException(
            status_code=422,
            detail="Unsupported file type. Accepted: JPEG, PNG, PDF.",
        )

    # Save image to the correct directory based on type
    unique_name = f"{uuid_lib.uuid4()}.{suffix}"
    
    if item_type == "lab":
        dest_dir = Path("./data/lab_reports")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / unique_name
        dest.write_bytes(contents)
        image_path = str(dest)
        
        # Parse it — 3-tier hybrid chain
        from src.preprocessor import preprocess_from_bytes
        from src.pipeline import _run_ocr
        from src.lab_ner import extract_lab_report as regex_extract_lab
        from src.gemini_extractor import is_available as is_gemini_available, hybrid_extract_lab_report
        from src import medical_ner

        extracted = None
        raw_text = ""

        # Tier 1: Tesseract OCR (always run first)
        try:
            clean_image = preprocess_from_bytes(contents, filename)
            raw_text = _run_ocr(clean_image)
            tess_extracted = medical_ner.extract(raw_text)
            tess_confidence = tess_extracted.confidence
        except Exception as exc:
            raw_text = f"[OCR Failure] {exc}"
            tess_confidence = 0.0

        # Tier 2: Gemini hybrid post-processing
        if is_gemini_available():
            try:
                gemini_res = hybrid_extract_lab_report(contents, raw_text, tess_confidence)
                if gemini_res:
                    from src.models import LabResultExtracted, LabReportExtracted
                    results_list = [
                        LabResultExtracted(
                            raw_name=r.get("raw_name") or r.get("test_name", ""),
                            analyte_name=r.get("analyte_name", "OTHER"),
                            value=r.get("value", 0.0),
                            unit=r.get("unit"),
                            ref_range=r.get("ref_range"),
                            flag=r.get("flag", "normal").lower(),
                        )
                        for r in gemini_res.get("results", [])
                    ]
                    extracted = LabReportExtracted(
                        lab_name=gemini_res.get("lab_name"),
                        report_date=gemini_res.get("report_date"),
                        results=results_list,
                        confidence=float(gemini_res.get("confidence", 0.7)),
                        raw_text=gemini_res.get("raw_text", ""),
                    )
            except Exception as e:
                logger.warning("Gemini hybrid lab extractor failed: %s", e)

        # Tier 3: Rule-based NER fallback
        if not extracted:
            extracted = regex_extract_lab(raw_text)
        else:
            raw_text = extracted.raw_text

        confidence = extracted.confidence
        # Force low confidence to trigger review queue
        confidence = min(0.45, confidence)
        reason = "Direct upload for clinical audit."
        
    else:
        # Prescription
        dest_dir = Path("./data/prescriptions")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / unique_name
        dest.write_bytes(contents)
        image_path = str(dest)
        
        # Parse it — 3-tier hybrid chain
        from src.preprocessor import preprocess_from_bytes
        from src.pipeline import _run_ocr
        from src.medical_ner import extract as regex_extract_rx
        from src.gemini_extractor import is_available as is_gemini_available, hybrid_extract_prescription

        extracted_dict = None
        raw_text = ""

        # Tier 1: Tesseract OCR (always run first)
        try:
            clean_image = preprocess_from_bytes(contents, filename)
            raw_text = _run_ocr(clean_image)
            tess_extracted = regex_extract_rx(raw_text)
            tess_confidence = tess_extracted.confidence
        except Exception as exc:
            raw_text = f"[OCR Failure] {exc}"
            tess_confidence = 0.0

        # Tier 2: Gemini hybrid post-processing
        if is_gemini_available():
            try:
                extracted_dict = hybrid_extract_prescription(contents, raw_text, tess_confidence)
            except Exception as e:
                logger.warning("Gemini hybrid Rx extractor failed: %s", e)

        # Tier 3: Rule-based NER fallback
        if not extracted_dict:
            extracted_dict = {
                "confidence": tess_confidence,
                "raw_text": raw_text,
            }
            
        confidence = float(extracted_dict.get("confidence", 0.45))
        confidence = min(0.45, confidence)
        raw_text = extracted_dict.get("raw_text", "")
        reason = "Direct upload for clinical audit."

    patient = None
    if phone:
        patient = get_or_create_patient(db, phone)

    import uuid
    db_facility_id = uuid.UUID(facility_id) if facility_id else None
    
    queue_item = save_to_review_queue(
        db=db,
        patient=patient,
        image_path=image_path,
        raw_ocr_text=raw_text,
        confidence=confidence,
        reason=reason,
        item_type=item_type,
        facility_id=db_facility_id,
    )
    db.commit()

    return ReviewQueueItemResponse(
        id=str(queue_item.id),
        patient_id=str(queue_item.patient_id) if queue_item.patient_id else None,
        image_path=queue_item.image_path,
        raw_ocr_text=queue_item.raw_ocr_text or "",
        confidence=queue_item.confidence,
        reason=queue_item.reason or "",
        resolved=queue_item.resolved,
        created_at=queue_item.created_at.isoformat(),
        patient_phone=phone,
        item_type=item_type,
    )


# ---------------------------------------------------------------------------
# Authentication REST APIs
# ---------------------------------------------------------------------------

@app.post("/api/auth/login")
def login(
    payload: ClinicianLoginInput,
    response: Response,
    db: Session = Depends(get_db)
):
    """Authenticate clinician and set signed cookies."""
    clinician = db_module.get_clinician_by_email(db, payload.email)
    if not clinician or not db_module.verify_password(payload.password, clinician.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
        
    # Generate session payload
    payload_data = {
        "clinician_id": str(clinician.id),
        "email": clinician.email,
        "name": clinician.name,
        "role": clinician.role,
        "scopes": clinician.scopes,
        "exp": (datetime.utcnow() + timedelta(days=1)).isoformat()
    }
    
    # Sign session
    signed_token = sign_session(payload_data)
    
    # Generate CSRF token
    csrf_token = uuid_lib.uuid4().hex
    
    # Set signed HttpOnly session cookie
    response.set_cookie(
        key="session_id",
        value=signed_token,
        httponly=True,
        samesite="lax",
        path="/"
    )
    
    # Set JS-readable CSRF cookie
    response.set_cookie(
        key="csrf_token",
        value=csrf_token,
        httponly=False,
        samesite="lax",
        path="/"
    )
    
    return {
        "status": "success",
        "clinician": {
            "id": str(clinician.id),
            "email": clinician.email,
            "name": clinician.name,
            "role": clinician.role,
            "scopes": clinician.scopes,
            "facilities": [
                {"id": str(f.id), "name": f.name, "code": f.code} for f in clinician.facilities
            ]
        }
    }


@app.get("/api/auth/me")
def get_auth_me(
    request: Request,
    db: Session = Depends(get_db)
):
    """Get the currently logged in clinician's details from session."""
    clinician = get_dashboard_clinician(request, db)
    if not clinician:
        raise HTTPException(status_code=401, detail="Not authenticated.")
        
    return {
        "status": "success",
        "clinician": {
            "id": str(clinician.id),
            "email": clinician.email,
            "name": clinician.name,
            "role": clinician.role,
            "scopes": clinician.scopes,
            "facilities": [
                {"id": str(f.id), "name": f.name, "code": f.code} for f in clinician.facilities
            ]
        }
    }


@app.post("/api/auth/logout")
def logout(response: Response):
    """Log out clinician by clearing cookies."""
    response.delete_cookie(key="session_id", path="/")
    response.delete_cookie(key="csrf_token", path="/")
    return {"status": "success", "message": "Logged out successfully."}


# ---------------------------------------------------------------------------
# Admin Oversight REST APIs
# ---------------------------------------------------------------------------

def verify_admin_role(request: Request, db: Session = Depends(get_db)) -> Optional[db_module.Clinician]:
    """Ensure clinician is authenticated and holds the admin role."""
    clinician = get_dashboard_clinician(request, db)
    if clinician is None:
        return None  # Bypassed local debug or API key
    if clinician.role != "admin":
        raise HTTPException(status_code=403, detail="Administrative privileges required.")
    return clinician


@app.get(
    "/api/admin/audit-logs",
    summary="Get all safety override audit logs requiring admin review",
)
def get_admin_audit_logs(
    request: Request,
    db: Session = Depends(get_db),
    admin: Optional[db_module.Clinician] = Depends(verify_admin_role),
) -> list:
    """Query pending safety override audit logs flagged for admin oversight."""
    logs = (
        db.query(db_module.AuditLog)
        .filter(db_module.AuditLog.needs_admin_oversight == True)
        .order_by(db_module.AuditLog.created_at.desc())
        .all()
    )
    
    result = []
    for log in logs:
        clinician = db.query(db_module.Clinician).filter(db_module.Clinician.id == log.clinician_id).first()
        facility = db.query(db_module.Facility).filter(db_module.Facility.id == log.facility_id).first()
        
        result.append({
            "id": str(log.id),
            "clinician_name": clinician.name if clinician else "Unknown",
            "clinician_email": clinician.email if clinician else "Unknown",
            "facility_name": facility.name if facility else "Unknown",
            "action": log.action,
            "item_id": str(log.item_id) if log.item_id else None,
            "override_reason": log.override_reason,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        })
        
    return result


@app.post(
    "/api/admin/audit-logs/{id}/approve",
    summary="Approve and clear a safety override audit log",
)
def approve_admin_audit_log(
    id: str,
    request: Request,
    db: Session = Depends(get_db),
    admin: Optional[db_module.Clinician] = Depends(verify_admin_role),
    _csrf: None = Depends(verify_csrf_token),
) -> dict:
    """Mark a clinical safety override audited, cleared, and approved."""
    import uuid
    try:
        log_id = uuid.UUID(id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid audit log ID format.")
        
    log = db.query(db_module.AuditLog).filter(db_module.AuditLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Audit log entry not found.")
        
    log.needs_admin_oversight = False
    db.commit()
    
    return {"status": "success", "message": "Audit log approved and cleared."}


@app.get(
    "/api/admin/rules",
    response_model=List[ClinicalSafetyRuleSchema],
    summary="Get all clinical safety rules",
)
def get_admin_rules(
    request: Request,
    db: Session = Depends(get_db),
    admin: Optional[db_module.Clinician] = Depends(verify_admin_role),
) -> List[ClinicalSafetyRuleSchema]:
    rules = db.query(db_module.ClinicalSafetyRule).filter(
        db_module.ClinicalSafetyRule.is_deleted == False
    ).order_by(db_module.ClinicalSafetyRule.created_at.desc()).all()
    result = []
    for r in rules:
        result.append(
            ClinicalSafetyRuleSchema(
                id=str(r.id),
                rule_name=r.rule_name,
                drug_inn=r.drug_inn,
                analyte_name=r.analyte_name,
                operator=r.operator,
                threshold_value=r.threshold_value,
                threshold_value_max=r.threshold_value_max,
                flag_match=r.flag_match,
                gender_specific=r.gender_specific or "both",
                severity=r.severity or "warning",
                description_template=r.description_template,
                management_plan=r.management_plan,
                is_enabled=r.is_enabled,
                is_deleted=r.is_deleted,
                version=r.version or 1,
                created_at=r.created_at.isoformat() if r.created_at else datetime.now().isoformat(),
                updated_at=r.updated_at.isoformat() if r.updated_at else datetime.now().isoformat(),
            )
        )
    return result


@app.post(
    "/api/admin/rules",
    summary="Create a new clinical safety rule",
)
def create_admin_rule(
    payload: CreateClinicalSafetyRuleInput,
    request: Request,
    db: Session = Depends(get_db),
    admin: Optional[db_module.Clinician] = Depends(verify_admin_role),
    _csrf: None = Depends(verify_csrf_token),
) -> dict:
    """Create a new clinical safety rule and log the action."""
    # Check duplicate rule name
    existing = db.query(db_module.ClinicalSafetyRule).filter(db_module.ClinicalSafetyRule.rule_name == payload.rule_name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Rule name already exists.")

    rule = db_module.ClinicalSafetyRule(
        rule_name=payload.rule_name,
        drug_inn=payload.drug_inn.lower(),
        analyte_name=payload.analyte_name.upper(),
        operator=payload.operator,
        threshold_value=payload.threshold_value,
        threshold_value_max=payload.threshold_value_max,
        flag_match=payload.flag_match,
        gender_specific=payload.gender_specific,
        severity=payload.severity,
        description_template=payload.description_template,
        management_plan=payload.management_plan,
        is_enabled=payload.is_enabled,
    )
    db.add(rule)
    db.flush()

    # Log to audit_logs with structural snapshot and version
    clinician_id = admin.id if admin else db.query(db_module.Clinician).first().id
    facility_id = admin.facilities[0].id if admin and admin.facilities else db.query(db_module.Facility).first().id
    
    op_desc = f" Op: {payload.operator} Val: {payload.threshold_value}" if payload.operator else f" Flag: {payload.flag_match}"
    snapshot = f"Drug: {payload.drug_inn}, Analyte: {payload.analyte_name},{op_desc}, Severity: {payload.severity}, Enabled: {payload.is_enabled}"
    
    audit = db_module.AuditLog(
        clinician_id=clinician_id,
        facility_id=facility_id,
        item_id=rule.id,
        action="create_rule",
        override_reason=f"Created clinical safety rule v1: {payload.rule_name} ({snapshot})",
        needs_admin_oversight=False
    )
    db.add(audit)
    db.commit()

    return {"status": "success", "message": "Rule created successfully.", "rule_id": str(rule.id)}


@app.put(
    "/api/admin/rules/{id}",
    summary="Update an existing clinical safety rule",
)
def update_admin_rule(
    id: str,
    payload: CreateClinicalSafetyRuleInput,
    request: Request,
    db: Session = Depends(get_db),
    admin: Optional[db_module.Clinician] = Depends(verify_admin_role),
    _csrf: None = Depends(verify_csrf_token),
) -> dict:
    """Update clinical safety rule configuration and log audit."""
    import uuid
    try:
        rule_id = uuid.UUID(id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid rule ID format.")

    rule = db.query(db_module.ClinicalSafetyRule).filter(
        db_module.ClinicalSafetyRule.id == rule_id,
        db_module.ClinicalSafetyRule.is_deleted == False
    ).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Clinical safety rule not found.")

    # Check duplicate rule name if changed
    if rule.rule_name != payload.rule_name:
        existing = db.query(db_module.ClinicalSafetyRule).filter(db_module.ClinicalSafetyRule.rule_name == payload.rule_name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Rule name already exists.")

    rule.rule_name = payload.rule_name
    rule.drug_inn = payload.drug_inn.lower()
    rule.analyte_name = payload.analyte_name.upper()
    rule.operator = payload.operator
    rule.threshold_value = payload.threshold_value
    rule.threshold_value_max = payload.threshold_value_max
    rule.flag_match = payload.flag_match
    rule.gender_specific = payload.gender_specific
    rule.severity = payload.severity
    rule.description_template = payload.description_template
    rule.management_plan = payload.management_plan
    rule.is_enabled = payload.is_enabled
    
    # Increment version
    rule.version = (rule.version or 1) + 1

    # Log to audit_logs with structural snapshot and version
    clinician_id = admin.id if admin else db.query(db_module.Clinician).first().id
    facility_id = admin.facilities[0].id if admin and admin.facilities else db.query(db_module.Facility).first().id
    
    op_desc = f" Op: {payload.operator} Val: {payload.threshold_value}" if payload.operator else f" Flag: {payload.flag_match}"
    snapshot = f"Drug: {payload.drug_inn}, Analyte: {payload.analyte_name},{op_desc}, Severity: {payload.severity}, Enabled: {payload.is_enabled}"
    
    audit = db_module.AuditLog(
        clinician_id=clinician_id,
        facility_id=facility_id,
        item_id=rule.id,
        action="update_rule",
        override_reason=f"Updated clinical safety rule v{rule.version}: {payload.rule_name} ({snapshot})",
        needs_admin_oversight=False
    )
    db.add(audit)
    db.commit()

    return {"status": "success", "message": "Rule updated successfully."}


@app.delete(
    "/api/admin/rules/{id}",
    summary="Delete a clinical safety rule",
)
def delete_admin_rule(
    id: str,
    request: Request,
    db: Session = Depends(get_db),
    admin: Optional[db_module.Clinician] = Depends(verify_admin_role),
    _csrf: None = Depends(verify_csrf_token),
) -> dict:
    """Delete a custom clinical safety rule from the system and log audit."""
    import uuid
    try:
        rule_id = uuid.UUID(id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid rule ID format.")

    rule = db.query(db_module.ClinicalSafetyRule).filter(
        db_module.ClinicalSafetyRule.id == rule_id,
        db_module.ClinicalSafetyRule.is_deleted == False
    ).first()
    if not rule:
        raise HTTPException(status_code=404, detail="Clinical safety rule not found.")

    # Log to audit_logs before deleting
    clinician_id = admin.id if admin else db.query(db_module.Clinician).first().id
    facility_id = admin.facilities[0].id if admin and admin.facilities else db.query(db_module.Facility).first().id
    audit = db_module.AuditLog(
        clinician_id=clinician_id,
        facility_id=facility_id,
        item_id=rule.id,
        action="delete_rule",
        override_reason=f"Soft-deleted clinical safety rule v{rule.version or 1}: {rule.rule_name}",
        needs_admin_oversight=False
    )
    db.add(audit)
    rule.is_deleted = True
    db.commit()

    return {"status": "success", "message": "Rule deleted successfully."}


# ---------------------------------------------------------------------------
# Dashboard Static Mounts
# ---------------------------------------------------------------------------


# Static data folder
data_dir = os.path.abspath("./data")
os.makedirs(data_dir, exist_ok=True)
app.mount("/data", StaticFiles(directory=data_dir), name="data")

# Static prescriptions folder
prescriptions_dir = os.path.abspath("./data/prescriptions")
if os.path.exists(prescriptions_dir):
    app.mount("/images", StaticFiles(directory=prescriptions_dir), name="images")

# Static assets dashboard folder
os.makedirs("./src/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="./src/static"), name="static")

@app.get("/dashboard", include_in_schema=False)
def serve_dashboard() -> FileResponse:
    """Serve the clinical audit dashboard SPA."""
    dashboard_path = os.path.abspath("./src/static/index.html")
    if not os.path.exists(dashboard_path):
        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write("<h1>OJAAI Clinical Audit Dashboard</h1>")
    return FileResponse(dashboard_path)

