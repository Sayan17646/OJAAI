"""
database.py — SQLAlchemy ORM models and session management for OJAAI.

Tables (per TRD architecture):
  patients, prescriptions, medications, drug_interactions, review_queue

Rules:
  - Never store image blobs — filepath only.
  - Low-confidence records go to review_queue, NOT main tables.
  - raw_ocr_text stored for audit but never logged at INFO level.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from typing import List, Optional

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, ForeignKey,
    Integer, String, Text, Table, create_engine, func, text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ojaai:password@localhost:5432/ojaai_dev",
)
MAX_ACTIVE_MED_DAYS = int(os.getenv("MAX_ACTIVE_MED_DAYS", "90"))

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


# Many-to-many join table for Clinicians and Facilities
clinician_facilities = Table(
    "clinician_facilities",
    Base.metadata,
    Column("clinician_id", PG_UUID(as_uuid=True), ForeignKey("clinicians.id", ondelete="CASCADE"), primary_key=True),
    Column("facility_id", PG_UUID(as_uuid=True), ForeignKey("facilities.id", ondelete="CASCADE"), primary_key=True)
)


class Facility(Base):
    __tablename__ = "facilities"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    name = Column(String(100), nullable=False)
    code = Column(String(20), unique=True, nullable=False)
    address = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    clinicians = relationship("Clinician", secondary=clinician_facilities, back_populates="facilities")


class Clinician(Base):
    __tablename__ = "clinicians"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    email = Column(String(150), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    name = Column(String(100), nullable=False)
    role = Column(String(20), default="viewer") # viewer | auditor | admin
    scopes = Column(String(50), default="both") # rx | lab | both
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    facilities = relationship("Facility", secondary=clinician_facilities, back_populates="clinicians")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    clinician_id = Column(PG_UUID(as_uuid=True), ForeignKey("clinicians.id"), nullable=False)
    facility_id = Column(PG_UUID(as_uuid=True), ForeignKey("facilities.id"), nullable=False)
    item_id = Column(PG_UUID(as_uuid=True), nullable=True)
    action = Column(String(50), nullable=False) # "resolve_rx" | "resolve_lab" | "override_safety"
    override_reason = Column(Text, nullable=True)
    needs_admin_oversight = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class Patient(Base):
    __tablename__ = "patients"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    phone = Column(String(15), unique=True, nullable=False, index=True)
    facility_id = Column(PG_UUID(as_uuid=True), ForeignKey("facilities.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    prescriptions = relationship("Prescription", back_populates="patient")
    interactions = relationship("DrugInteractionRecord", back_populates="patient")
    review_items = relationship("ReviewQueue", back_populates="patient")
    lab_reports = relationship("LabReport", back_populates="patient")


class Prescription(Base):
    __tablename__ = "prescriptions"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(PG_UUID(as_uuid=True), ForeignKey("patients.id"), nullable=True)
    facility_id = Column(PG_UUID(as_uuid=True), ForeignKey("facilities.id"), nullable=True)
    image_path = Column(Text, nullable=False)
    raw_ocr_text = Column(Text, nullable=True)        # audit only — never log at INFO
    confidence = Column(Float, nullable=False)
    doctor_reg = Column(String(50), nullable=True)
    patient_age = Column(String(20), nullable=True)
    diagnosis = Column(Text, nullable=True)
    prescription_date = Column(Date, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("Patient", back_populates="prescriptions")
    medications = relationship("Medication", back_populates="prescription")
    interactions = relationship("DrugInteractionRecord", back_populates="prescription")


class Medication(Base):
    __tablename__ = "medications"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    prescription_id = Column(PG_UUID(as_uuid=True), ForeignKey("prescriptions.id"), nullable=False)
    raw_drug_name = Column(Text, nullable=False)
    inn = Column(Text, nullable=True)
    rxcui = Column(String(20), nullable=True)
    standard_name = Column(Text, nullable=True)
    dosage_value = Column(Float, nullable=True)
    dosage_unit = Column(String(20), nullable=True)
    frequency = Column(Text, nullable=True)
    freq_per_day = Column(Integer, nullable=True)
    duration_days = Column(Integer, nullable=True)
    route = Column(String(30), default="oral")
    is_active = Column(Boolean, default=True)

    prescription = relationship("Prescription", back_populates="medications")


class DrugInteractionRecord(Base):
    __tablename__ = "drug_interactions"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    prescription_id = Column(PG_UUID(as_uuid=True), ForeignKey("prescriptions.id"), nullable=True)
    patient_id = Column(PG_UUID(as_uuid=True), ForeignKey("patients.id"), nullable=True)
    drug_1 = Column(Text, nullable=False)
    drug_2 = Column(Text, nullable=False)
    severity = Column(String(20), nullable=False)
    description = Column(Text, nullable=True)
    management = Column(Text, nullable=True)
    source = Column(String(50), nullable=True)
    detected_at = Column(DateTime(timezone=True), server_default=func.now())

    prescription = relationship("Prescription", back_populates="interactions")
    patient = relationship("Patient", back_populates="interactions")


class ReviewQueue(Base):
    __tablename__ = "review_queue"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(PG_UUID(as_uuid=True), ForeignKey("patients.id"), nullable=True)
    facility_id = Column(PG_UUID(as_uuid=True), ForeignKey("facilities.id"), nullable=True)
    image_path = Column(Text, nullable=False)
    raw_ocr_text = Column(Text, nullable=True)
    confidence = Column(Float, nullable=False)
    reason = Column(Text, nullable=True)
    resolved = Column(Boolean, default=False)
    item_type = Column(String(20), default="prescription", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("Patient", back_populates="review_items")


class LabReport(Base):
    __tablename__ = "lab_reports"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(PG_UUID(as_uuid=True), ForeignKey("patients.id"), nullable=True)
    facility_id = Column(PG_UUID(as_uuid=True), ForeignKey("facilities.id"), nullable=True)
    image_path = Column(Text, nullable=False)
    lab_name = Column(String(100), nullable=True)
    report_date = Column(Date, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("Patient", back_populates="lab_reports")
    results = relationship("LabResult", back_populates="lab_report", cascade="all, delete-orphan")


class LabResult(Base):
    __tablename__ = "lab_results"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    lab_report_id = Column(PG_UUID(as_uuid=True), ForeignKey("lab_reports.id"), nullable=False)
    raw_name = Column(Text, nullable=False)
    analyte_name = Column(String(50), nullable=False, index=True) # Normalized (e.g. HBA1C, CREATININE, HEMOGLOBIN, TSH, LDL)
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=True)
    ref_range = Column(String(50), nullable=True)
    flag = Column(String(20), default="normal") # normal | high | low

    lab_report = relationship("LabReport", back_populates="results")


class ClinicalSafetyRule(Base):
    __tablename__ = "clinical_safety_rules"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    rule_name = Column(String(150), unique=True, nullable=False)
    drug_inn = Column(String(100), nullable=False, index=True)
    analyte_name = Column(String(50), nullable=False, index=True)
    
    operator = Column(String(20), nullable=True) # ">", "<", ">=", "<=", "=", "between"
    threshold_value = Column(Float, nullable=True)
    threshold_value_max = Column(Float, nullable=True) # for "between"
    flag_match = Column(String(10), nullable=True) # "high", "low", "normal"
    
    gender_specific = Column(String(10), default="both") # "male", "female", "both"
    severity = Column(String(20), default="warning") # "critical", "warning", "info"
    
    description_template = Column(Text, nullable=False)
    management_plan = Column(Text, nullable=True)
    is_enabled = Column(Boolean, default=True)
    is_deleted = Column(Boolean, default=False, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------

# DB helpers
# ---------------------------------------------------------------------------

def create_all_tables() -> None:
    """Create all tables if they do not exist. Called on server startup."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified/created.")
    
    # Check/add item_type column for review_queue table (robust schema update)
    with engine.begin() as conn:
        # Check/add is_deleted and version columns for clinical_safety_rules table (robust schema update)
        try:
            conn.execute(text("ALTER TABLE clinical_safety_rules ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN DEFAULT FALSE NOT NULL"))
            logger.info("Successfully checked/added is_deleted column to clinical_safety_rules.")
        except Exception as e:
            logger.warning("Could not add is_deleted to clinical_safety_rules: %s. Trying generic check.", e)
            try:
                conn.execute(text("ALTER TABLE clinical_safety_rules ADD COLUMN is_deleted BOOLEAN DEFAULT FALSE NOT NULL"))
                logger.info("Successfully added is_deleted column to clinical_safety_rules via fallback.")
            except Exception as e2:
                logger.debug("Column is_deleted probably already exists in clinical_safety_rules: %s", e2)

        try:
            conn.execute(text("ALTER TABLE clinical_safety_rules ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1 NOT NULL"))
            logger.info("Successfully checked/added version column to clinical_safety_rules.")
        except Exception as e:
            logger.warning("Could not add version to clinical_safety_rules: %s. Trying generic check.", e)
            try:
                conn.execute(text("ALTER TABLE clinical_safety_rules ADD COLUMN version INTEGER DEFAULT 1 NOT NULL"))
                logger.info("Successfully added version column to clinical_safety_rules via fallback.")
            except Exception as e2:
                logger.debug("Column version probably already exists in clinical_safety_rules: %s", e2)

        try:
            conn.execute(text("ALTER TABLE review_queue ADD COLUMN IF NOT EXISTS item_type VARCHAR(20) DEFAULT 'prescription' NOT NULL"))
            logger.info("Successfully checked/added item_type column to review_queue.")
        except Exception as e:
            logger.warning("Could not run migration using ALTER TABLE IF NOT EXISTS: %s. Trying generic check.", e)
            try:
                conn.execute(text("ALTER TABLE review_queue ADD COLUMN item_type VARCHAR(20) DEFAULT 'prescription' NOT NULL"))
                logger.info("Successfully added item_type column to review_queue via fallback.")
            except Exception as e2:
                logger.debug("Column item_type probably already exists in review_queue: %s", e2)
        
        # Check/add facility_id column for patients, prescriptions, lab_reports, and review_queue
        for table_name in ("patients", "prescriptions", "lab_reports", "review_queue"):
            try:
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS facility_id UUID REFERENCES facilities(id) ON DELETE SET NULL"))
                logger.info("Successfully checked/added facility_id column to %s.", table_name)
            except Exception as e:
                logger.warning("Could not run facility_id migration for %s: %s", table_name, e)

    # Seed default facility and clinician if none exist
    db = SessionLocal()
    try:
        # Seed facility if none exist
        facility = db.query(Facility).first()
        if not facility:
            facility = Facility(
                name="Pune Diagnostic Center",
                code="PUNE_DIAGNOSTICS",
                address="Pune, Maharashtra"
            )
            db.add(facility)
            db.flush()
            logger.info("Seeded default facility: Pune Diagnostic Center")
        
        # Seed clinician specifically if admin@ojaai.com does not exist
        clinician = db.query(Clinician).filter(Clinician.email == "admin@ojaai.com").first()
        if not clinician:
            clinician = Clinician(
                email="admin@ojaai.com",
                hashed_password=hash_password("admin123"),
                name="Dr. Aditi Sharma",
                role="admin",
                scopes="both"
            )
            # Assign to the Pune Diagnostics facility
            clinician.facilities.append(facility)
            db.add(clinician)
            logger.info("Seeded default clinician: admin@ojaai.com")
            
        # Seed the 11 baseline rules if empty
        seed_baseline_rules(db)
            
        db.commit()
    except Exception as e:
        db.rollback()
        logger.warning("Failed to seed default facility/clinician/rules: %s", e)
    finally:
        db.close()


def seed_baseline_rules(db: Session) -> None:
    """Seed the 11 baseline clinical safety rules if the rules table is empty."""
    if db.query(ClinicalSafetyRule).filter(ClinicalSafetyRule.is_deleted == False).count() > 0:
        return

    logger.info("Seeding 11 baseline clinical safety rules into database...")
    baseline_rules = [
        # Rule 1
        ClinicalSafetyRule(
            rule_name="Metformin Female Creatinine Contraindication",
            drug_inn="metformin",
            analyte_name="CREATININE",
            operator=">",
            threshold_value=1.4,
            gender_specific="female",
            severity="critical",
            description_template="Metformin is contraindicated due to elevated Serum Creatinine ({value} {unit} > {threshold} {unit} limit). Significantly elevated risk of Metformin-induced Lactic Acidosis.",
            management_plan="Discontinue Metformin. Calculate eGFR and consider alternative glycemic agents.",
            is_enabled=True,
            version=1
        ),
        ClinicalSafetyRule(
            rule_name="Metformin Male Creatinine Contraindication",
            drug_inn="metformin",
            analyte_name="CREATININE",
            operator=">",
            threshold_value=1.5,
            gender_specific="male",
            severity="critical",
            description_template="Metformin is contraindicated due to elevated Serum Creatinine ({value} {unit} > {threshold} {unit} limit). Significantly elevated risk of Metformin-induced Lactic Acidosis.",
            management_plan="Discontinue Metformin. Calculate eGFR and consider alternative glycemic agents.",
            is_enabled=True,
            version=1
        ),
        # Rule 2
        ClinicalSafetyRule(
            rule_name="Levothyroxine Under-dosing Warning",
            drug_inn="levothyroxine",
            analyte_name="TSH",
            operator=">",
            threshold_value=4.5,
            gender_specific="both",
            severity="warning",
            description_template="TSH level is elevated ({value} {unit} > {threshold} {unit} range), indicating potential under-dosing. Levothyroxine dose adjustment may be required.",
            management_plan="Re-evaluate thyroid status and consider increasing Levothyroxine dose.",
            is_enabled=True,
            version=1
        ),
        ClinicalSafetyRule(
            rule_name="Levothyroxine Over-dosing Warning",
            drug_inn="levothyroxine",
            analyte_name="TSH",
            operator="<",
            threshold_value=0.4,
            gender_specific="both",
            severity="warning",
            description_template="TSH level is suppressed ({value} {unit} < {threshold} {unit} range), indicating potential over-dosing. Risk of iatrogenic hyperthyroidism.",
            management_plan="Re-evaluate thyroid status and consider reducing Levothyroxine dose.",
            is_enabled=True,
            version=1
        ),
        # Rule 3
        ClinicalSafetyRule(
            rule_name="Aspirin Suppressed Hemoglobin Risk",
            drug_inn="aspirin",
            analyte_name="HEMOGLOBIN",
            operator="<",
            threshold_value=10.0,
            gender_specific="both",
            severity="warning",
            description_template="Severe anemia detected (Hemoglobin {value} {unit} < {threshold} {unit} normal). Concomitant antiplatelet therapy (Aspirin) significantly elevates gastrointestinal bleeding risks.",
            management_plan="Evaluate anemia source. Consider adding PPIs (e.g. Pantoprazole) for gastroprotection if Aspirin is mandatory.",
            is_enabled=True,
            version=1
        ),
        # Rule 4
        ClinicalSafetyRule(
            rule_name="Atorvastatin LDL Efficacy Target",
            drug_inn="atorvastatin",
            analyte_name="LDL",
            operator=">",
            threshold_value=100.0,
            gender_specific="both",
            severity="info",
            description_template="LDL cholesterol is elevated ({value} {unit} > {threshold} {unit} target). Evaluating efficacy of active lipid-lowering therapy (Atorvastatin).",
            management_plan="Continue lipid-lowering statin therapy. Recheck lipid profile in 6-8 weeks.",
            is_enabled=True,
            version=1
        ),
        ClinicalSafetyRule(
            rule_name="Rosuvastatin LDL Efficacy Target",
            drug_inn="rosuvastatin",
            analyte_name="LDL",
            operator=">",
            threshold_value=100.0,
            gender_specific="both",
            severity="info",
            description_template="LDL cholesterol is elevated ({value} {unit} > {threshold} {unit} target). Evaluating efficacy of active lipid-lowering therapy (Rosuvastatin).",
            management_plan="Continue lipid-lowering statin therapy. Recheck lipid profile in 6-8 weeks.",
            is_enabled=True,
            version=1
        ),
        # Rule 5
        ClinicalSafetyRule(
            rule_name="Insulin FBS Hypoglycemia Alert",
            drug_inn="insulin",
            analyte_name="FASTING_BLOOD_SUGAR",
            operator="<",
            threshold_value=70.0,
            gender_specific="both",
            severity="critical",
            description_template="Clinical hypoglycemia detected (Fasting Blood Sugar {value} {unit} < {threshold} {unit}). Concomitant hypoglycemic drug therapy (Insulin) poses severe risk of neuroglycopenia or coma.",
            management_plan="Hold/reduce hypoglycemic drug dosage. Administer fast-acting glucose immediately and counsel patient on hypoglycemia protocols.",
            is_enabled=True,
            version=1
        ),
        ClinicalSafetyRule(
            rule_name="Glimepiride FBS Hypoglycemia Alert",
            drug_inn="glimepiride",
            analyte_name="FASTING_BLOOD_SUGAR",
            operator="<",
            threshold_value=70.0,
            gender_specific="both",
            severity="critical",
            description_template="Clinical hypoglycemia detected (Fasting Blood Sugar {value} {unit} < {threshold} {unit}). Concomitant hypoglycemic drug therapy (Glimepiride) poses severe risk of neuroglycopenia or coma.",
            management_plan="Hold/reduce hypoglycemic drug dosage. Administer fast-acting glucose immediately and counsel patient on hypoglycemia protocols.",
            is_enabled=True,
            version=1
        ),
        # Rule 6
        ClinicalSafetyRule(
            rule_name="Metformin Elevated HbA1c control check",
            drug_inn="metformin",
            analyte_name="HBA1C",
            operator=">",
            threshold_value=8.0,
            gender_specific="both",
            severity="warning",
            description_template="Poor glycemic control detected (HbA1c {value}% > {threshold}%). Patient is under active pharmacotherapy (Metformin). Dose escalation or therapeutic intensification is indicated.",
            management_plan="Evaluate patient adherence, titrate active medications, or consider dual/triple combination oral therapy.",
            is_enabled=True,
            version=1
        ),
        # Rule 7
        ClinicalSafetyRule(
            rule_name="Aspirin Thrombocytopenia Bleeding Check",
            drug_inn="aspirin",
            analyte_name="PLATELET_COUNT",
            operator="<",
            threshold_value=100.0,
            gender_specific="both",
            severity="critical",
            description_template="Significant thrombocytopenia detected (Platelets {value} {unit}). Active antiplatelet therapy (Aspirin) significantly increases severe hemorrhagic risks.",
            management_plan="Hold antiplatelet agent. Investigate etiology of thrombocytopenia. Monitor patient for clinical bleeding signs.",
            is_enabled=True,
            version=1
        ),
        # Rule 8
        ClinicalSafetyRule(
            rule_name="Ramipril Creatinine AKI Check",
            drug_inn="ramipril",
            analyte_name="CREATININE",
            operator=">",
            threshold_value=1.4,
            gender_specific="both",
            severity="warning",
            description_template="Concomitant RAS blocker (Ramipril) with elevated Serum Creatinine ({value} {unit} > {threshold} {unit}). Risk of acute kidney injury (AKI) or severe hyperkalemia.",
            management_plan="Monitor renal functions and serum potassium. Consider temporary discontinuation or dose reduction of RAS inhibitor.",
            is_enabled=True,
            version=1
        ),
        # Rule 9
        ClinicalSafetyRule(
            rule_name="Atorvastatin TG Persistent Dyslipidemia",
            drug_inn="atorvastatin",
            analyte_name="TRIGLYCERIDES",
            operator=">",
            threshold_value=200.0,
            gender_specific="both",
            severity="info",
            description_template="Persistent hypertriglyceridemia detected ({value} {unit} > 150 mg/dL normal) under active lipid-lowering therapy (Atorvastatin).",
            management_plan="Evaluate dietary habits. Consider optimization of statin dose or adding Omega-3 fatty acids/Fibrates if clinically indicated.",
            is_enabled=True,
            version=1
        ),
        # Rule 10
        ClinicalSafetyRule(
            rule_name="Aspirin Mild Anemia Combined Bleeding Risk",
            drug_inn="aspirin",
            analyte_name="HEMOGLOBIN",
            operator="between",
            threshold_value=10.0,
            threshold_value_max=11.5,
            gender_specific="both",
            severity="info",
            description_template="Mild anemia detected (Hemoglobin {value} {unit}). Evaluating bleeding risk for patient taking antiplatelet agent (Aspirin).",
            management_plan="Monitor complete blood count regularly. Assess patient for occult gastrointestinal blood loss.",
            is_enabled=True,
            version=1
        ),
        # Rule 11
        ClinicalSafetyRule(
            rule_name="Methotrexate TLC Myelosuppression Check",
            drug_inn="methotrexate",
            analyte_name="TLC",
            operator="<",
            threshold_value=4.0,
            gender_specific="both",
            severity="critical",
            description_template="Leucopenia detected (Total Leucocyte Count {value} {unit} < {threshold}). Active myelosuppressive agent (Methotrexate) poses severe risk of neutropenic sepsis.",
            management_plan="Hold myelosuppressive agent. Perform urgent differential count. Counsel patient on immediately reporting fever/chills.",
            is_enabled=True,
            version=1
        ),
        ClinicalSafetyRule(
            rule_name="Methotrexate Neutrophils Myelosuppression Check",
            drug_inn="methotrexate",
            analyte_name="NEUTROPHILS",
            operator="<",
            threshold_value=40.0,
            gender_specific="both",
            severity="warning",
            description_template="Relative neutropenia detected (Neutrophils {value}% < {threshold}% range) with active myelosuppressive therapy (Methotrexate).",
            management_plan="Monitor absolute neutrophil count (ANC). Suspend myelosuppressive agent if ANC falls below 1500 cells/uL.",
            is_enabled=True,
            version=1
        ),
    ]

    for rule in baseline_rules:
        exists = db.query(ClinicalSafetyRule).filter(ClinicalSafetyRule.rule_name == rule.rule_name).first()
        if not exists:
            db.add(rule)
    db.flush()
    logger.info("Successfully seeded clinical safety baseline rules.")



def get_db():
    """FastAPI dependency: yields a DB session and ensures it's closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_or_create_patient(db: Session, phone: str) -> Patient:
    """Return existing patient by phone, or create a new one."""
    patient = db.query(Patient).filter(Patient.phone == phone).first()
    if not patient:
        patient = Patient(phone=phone)
        db.add(patient)
        db.flush()   # get the UUID assigned
    return patient


def get_active_medications(db: Session, patient_id: object) -> List[Medication]:
    """
    Return all active medications for a patient.
    Active = prescribed within MAX_ACTIVE_MED_DAYS days OR has no duration (chronic).
    """
    cutoff_date = datetime.utcnow() - timedelta(days=MAX_ACTIVE_MED_DAYS)

    active_meds = (
        db.query(Medication)
        .join(Prescription, Medication.prescription_id == Prescription.id)
        .filter(
            Prescription.patient_id == patient_id,
            Medication.is_active == True,
            # Active if: no duration (chronic) OR prescription is recent
            (
                (Medication.duration_days == None)  # noqa: E711
                | (Prescription.created_at >= cutoff_date)
            ),
        )
        .all()
    )
    return active_meds


def save_prescription_to_db(
    db: Session,
    prescription_id: str,
    patient: Optional[Patient],
    image_path: str,
    raw_ocr_text: str,
    confidence: float,
    doctor_reg: Optional[str],
    patient_age: Optional[str],
    diagnosis: Optional[str],
    prescription_date: Optional[str],
    medications: list,
    interactions: list,
    facility_id: Optional[object] = None,
) -> Prescription:
    """
    Save a high-confidence prescription to the main tables.
    Caller must commit the session.
    """
    from src.models import NormalizedDrug, DrugInteraction

    # Parse prescription date string to date object
    parsed_date: Optional[date] = None
    if prescription_date:
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y"):
            try:
                parsed_date = datetime.strptime(prescription_date, fmt).date()
                break
            except ValueError:
                continue

    rx = Prescription(
        id=prescription_id,
        patient_id=patient.id if patient else None,
        image_path=image_path,
        raw_ocr_text=raw_ocr_text,   # never log this at INFO
        confidence=confidence,
        doctor_reg=doctor_reg,
        patient_age=patient_age,
        diagnosis=diagnosis,
        prescription_date=parsed_date,
        facility_id=facility_id,
    )

    db.add(rx)
    db.flush()

    for med in medications:
        db.add(Medication(
            prescription_id=rx.id,
            raw_drug_name=med.raw_drug_name,
            inn=med.inn,
            rxcui=med.rxcui,
            standard_name=med.standard_name,
            dosage_value=med.dosage_value,
            dosage_unit=med.dosage_unit,
            frequency=med.frequency,
            freq_per_day=med.freq_per_day,
            duration_days=med.duration_days,
            route=med.route,
            is_active=med.is_active,
        ))

    for interaction in interactions:
        db.add(DrugInteractionRecord(
            prescription_id=rx.id,
            patient_id=patient.id if patient else None,
            drug_1=interaction.drug_1,
            drug_2=interaction.drug_2,
            severity=interaction.severity,
            description=interaction.description,
            management=interaction.management,
            source=interaction.source,
        ))

    return rx


def save_to_review_queue(
    db: Session,
    patient: Optional[Patient],
    image_path: str,
    raw_ocr_text: str,
    confidence: float,
    reason: str,
    item_type: str = "prescription",
    facility_id: Optional[object] = None,
) -> ReviewQueue:
    """
    Save a low-confidence prescription or lab report to the review_queue.
    Does NOT write to main prescriptions/medications/labs tables.
    Caller must commit the session.
    """
    item = ReviewQueue(
        patient_id=patient.id if patient else None,
        image_path=image_path,
        raw_ocr_text=raw_ocr_text,
        confidence=confidence,
        reason=reason,
        resolved=False,
        item_type=item_type,
        facility_id=facility_id,
    )
    db.add(item)
    return item



def get_unresolved_review_items(db: Session) -> List[ReviewQueue]:
    """Retrieve all unresolved ReviewQueue items ordered by created_at DESC."""
    return (
        db.query(ReviewQueue)
        .filter(ReviewQueue.resolved == False)
        .order_by(ReviewQueue.created_at.desc())
        .all()
    )


def get_review_item_by_id(db: Session, id: str) -> Optional[ReviewQueue]:
    """Retrieve a single ReviewQueue item by ID (string or UUID)."""
    import uuid
    try:
        val = uuid.UUID(id) if isinstance(id, str) else id
    except ValueError:
        return None
    return db.query(ReviewQueue).filter(ReviewQueue.id == val).first()


def resolve_review_item(db: Session, id: str) -> bool:
    """Mark a ReviewQueue item as resolved. Caller must commit session."""
    item = get_review_item_by_id(db, id)
    if item:
        item.resolved = True
        return True
    return False


def save_lab_report_to_db(
    db: Session,
    patient: Optional[Patient],
    image_path: str,
    lab_name: Optional[str],
    report_date: Optional[str],
    results: list,
    facility_id: Optional[object] = None,
) -> LabReport:
    """Save extracted lab report and its diagnostic results to DB."""
    # Parse date if present
    parsed_date: Optional[date] = None
    if report_date:
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y", "%d-%b-%Y", "%Y/%m/%d"):
            try:
                parsed_date = datetime.strptime(report_date, fmt).date()
                break
            except ValueError:
                continue

    report = LabReport(
        patient_id=patient.id if patient else None,
        image_path=image_path,
        lab_name=lab_name,
        report_date=parsed_date,
        facility_id=facility_id,
    )

    db.add(report)
    db.flush()

    for res in results:
        db.add(LabResult(
            lab_report_id=report.id,
            raw_name=res.raw_name,
            analyte_name=res.analyte_name.upper(),
            value=res.value,
            unit=res.unit,
            ref_range=res.ref_range,
            flag=res.flag,
        ))

    return report


def get_latest_lab_results(db: Session, patient_id: object) -> List[LabResult]:
    """
    Retrieve the latest diagnostic result for each unique analyte/biomarker
    for the patient, sorted by report date / creation time DESC.
    """
    # Get all results for the patient ordered by report date or created_at desc
    all_results = (
        db.query(LabResult)
        .join(LabReport, LabResult.lab_report_id == LabReport.id)
        .filter(LabReport.patient_id == patient_id)
        .order_by(
            func.coalesce(LabReport.report_date, LabReport.created_at).desc(),
            LabResult.analyte_name
        )
        .all()
    )

    # Keep only the latest for each analyte_name
    latest = {}
    for res in all_results:
        analyte = res.analyte_name.upper()
        if analyte not in latest:
            latest[analyte] = res

    return list(latest.values())


# ---------------------------------------------------------------------------
# Clinician Auth & Password Hashing Helpers
# ---------------------------------------------------------------------------

import hashlib
import uuid as uuid_lib

def hash_password(password: str) -> str:
    """Hash password using SHA-256 and a random salt."""
    salt = uuid_lib.uuid4().hex
    hashed = hashlib.sha256((password + salt).encode('utf-8')).hexdigest()
    return f"{salt}:{hashed}"

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against salt and hash."""
    if not hashed_password or ":" not in hashed_password:
        return False
    salt, hashed = hashed_password.split(":", 1)
    test_hash = hashlib.sha256((password + salt).encode('utf-8')).hexdigest()
    return test_hash == hashed

def get_clinician_by_email(db: Session, email: str) -> Optional[Clinician]:
    """Retrieve a clinician by email (case-insensitive)."""
    return db.query(Clinician).filter(func.lower(Clinician.email) == email.lower()).first()

