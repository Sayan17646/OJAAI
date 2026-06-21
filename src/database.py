"""
database.py — SQLAlchemy ORM models and session management for OJAAI.

Tables (per TRD architecture):
  patients, prescriptions, medications, drug_interactions, review_queue

PHG MVP additions (7 new tables, see PHG MVP spec):
  doctors, medication_episodes, medication_dosage_history,
  patient_conditions, drug_condition_signals, phg_events,
  patient_drug_reactions

Rules:
  - Never store image blobs — filepath only.
  - Low-confidence records go to review_queue, NOT main tables.
  - raw_ocr_text stored for audit but never logged at INFO level.
  - PHG writes are inside the same transaction as prescription writes.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from typing import List, Optional

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, ForeignKey,
    Integer, SmallInteger, String, Text, Table,
    CheckConstraint, Index, UniqueConstraint,
    create_engine, func, text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB, ARRAY
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


# ===========================================================================
# PHG MVP ORM Models — Patient History Graph (7 new tables)
# All additive. No existing models modified.
# ===========================================================================

class Doctor(Base):
    """
    Normalized prescribing doctor registry.
    Deduplication via registration_number_normalized (strip non-alphanumeric, uppercase).
    """
    __tablename__ = "doctors"

    id                              = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    registration_number             = Column(String(100), nullable=True)
    registration_number_normalized  = Column(String(50),  nullable=True, index=True)  # unique enforced via DDL index
    name                            = Column(Text,         nullable=True)
    speciality                      = Column(Text,         nullable=True)
    speciality_group                = Column(String(30),   nullable=True)  # 'metabolic'|'cardiovascular'|etc.
    clinic_name                     = Column(Text,         nullable=True)
    facility_id                     = Column(PG_UUID(as_uuid=True), ForeignKey("facilities.id", ondelete="SET NULL"), nullable=True)
    raw_name_variants               = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    first_seen_at                   = Column(DateTime(timezone=True), server_default=func.now())
    updated_at                      = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    medication_episodes = relationship("MedicationEpisode", back_populates="latest_doctor", foreign_keys="MedicationEpisode.latest_doctor_id")


class MedicationEpisode(Base):
    """
    Longitudinal medication episode: one continuous treatment period per patient per drug.
    Partial unique index (enforced via DDL) prevents two active episodes for same patient+INN.
    """
    __tablename__ = "medication_episodes"

    id                  = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    patient_id          = Column(PG_UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    inn                 = Column(Text, nullable=False)
    rxcui               = Column(String(20), nullable=True)
    drug_class          = Column(String(50), nullable=True)

    # FDC support — fdc_components stores the exploded component INNs for safety lookup
    is_fdc              = Column(Boolean, nullable=False, server_default="false")
    fdc_components      = Column(ARRAY(Text), nullable=True)  # ['metformin', 'glibenclamide']

    # Dispensing behavior — drives episode boundary logic
    dispensing_type     = Column(String(20), nullable=False, server_default="scheduled")
    # 'scheduled' | 'acute' | 'prn' | 'periodic'

    status              = Column(String(20), nullable=False, server_default="active")
    # 'active' | 'completed' | 'discontinued' | 'prn_snapshot' | 'unknown'

    start_date          = Column(Date, nullable=False)
    estimated_end_date  = Column(Date, nullable=True)
    actual_end_date     = Column(Date, nullable=True)
    gap_tolerance_days  = Column(Integer, nullable=False, server_default="45")
    prescription_count  = Column(Integer, nullable=False, server_default="1")
    total_duration_days = Column(Integer, nullable=True)

    # Denormalized latest snapshot — avoids join on hot read path
    latest_dosage_value     = Column(Float,       nullable=True)
    latest_dosage_unit      = Column(Text,        nullable=True)
    latest_frequency        = Column(Text,        nullable=True)
    latest_doctor_id        = Column(PG_UUID(as_uuid=True), ForeignKey("doctors.id", ondelete="SET NULL"), nullable=True)
    latest_prescription_id  = Column(PG_UUID(as_uuid=True), ForeignKey("prescriptions.id", ondelete="SET NULL"), nullable=True)

    # Optimistic locking version counter — increment on every UPDATE
    version             = Column(Integer, nullable=False, server_default="1")

    created_at          = Column(DateTime(timezone=True), server_default=func.now())
    updated_at          = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    patient             = relationship("Patient", foreign_keys=[patient_id])
    latest_doctor       = relationship("Doctor", back_populates="medication_episodes", foreign_keys=[latest_doctor_id])
    dosage_history      = relationship(
        "MedicationDosageHistory",
        primaryjoin="MedicationEpisode.id == foreign(MedicationDosageHistory.episode_id)",
        back_populates="episode",
        order_by="MedicationDosageHistory.recorded_date.desc()",
        lazy="dynamic",
    )


class MedicationDosageHistory(Base):
    """
    Append-only audit trail: one row per prescription × drug.
    episode_id FK is app-enforced (no REFERENCES constraint) to allow future partitioning.
    """
    __tablename__ = "medication_dosage_history"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    episode_id      = Column(PG_UUID(as_uuid=True), nullable=False, index=True)  # app-enforced FK
    prescription_id = Column(PG_UUID(as_uuid=True), ForeignKey("prescriptions.id", ondelete="RESTRICT"), nullable=False)
    doctor_id       = Column(PG_UUID(as_uuid=True), ForeignKey("doctors.id",        ondelete="SET NULL"),  nullable=True)

    raw_drug_name   = Column(Text,        nullable=False)
    dosage_value    = Column(Float,       nullable=True)
    dosage_unit     = Column(Text,        nullable=True)
    frequency       = Column(Text,        nullable=True)
    freq_per_day    = Column(Float,       nullable=True)
    duration_days   = Column(Integer,     nullable=True)
    route           = Column(Text,        nullable=True)

    # Outcome tracking fields — zero cost now, critical for Phase 3
    stop_reason     = Column(Text,        nullable=True)  # 'treatment_complete'|'adverse_reaction'|'switched_to'|'patient_request'
    switched_to_inn = Column(Text,        nullable=True)  # populated when stop_reason='switched_to'
    refill_number   = Column(Integer,     nullable=False, server_default="1")

    # Deduplication — prevents double-write on upload retry
    # Value: sha256(prescription_id + '::' + inn)
    idempotency_key = Column(Text,        nullable=True, unique=True)

    recorded_date   = Column(Date,        nullable=False)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    episode         = relationship(
        "MedicationEpisode",
        primaryjoin="foreign(MedicationDosageHistory.episode_id) == MedicationEpisode.id",
        back_populates="dosage_history",
    )


class PatientCondition(Base):
    """
    Inferred clinical condition. Never directly entered.
    Partial unique index (enforced via DDL) allows condition recurrence across episodes.
    """
    __tablename__ = "patient_conditions"

    id              = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    patient_id      = Column(PG_UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)

    condition_code  = Column(String(20), nullable=False)
    condition_name  = Column(Text,       nullable=False)
    condition_group = Column(Text,       nullable=True)

    # episode_number enables condition recurrence without destroying history
    episode_number  = Column(Integer, nullable=False, server_default="1")

    status          = Column(String(20), nullable=False, server_default="probable")
    # 'probable' | 'confirmed' | 'rejected' | 'resolved'

    confidence      = Column(Float, nullable=False)

    # Tracks which inference algorithm version produced this — critical for reproducibility
    inference_engine_version = Column(String(20), nullable=False, server_default="phg_mvp_v1")

    # Structured audit trail: which drugs/episodes/signals triggered inference
    inference_basis = Column(JSONB, nullable=False, server_default=text("'{}' :: jsonb"))

    # Clinician review
    reviewed_by     = Column(PG_UUID(as_uuid=True), ForeignKey("clinicians.id", ondelete="SET NULL"), nullable=True)
    reviewed_at     = Column(DateTime(timezone=True), nullable=True)
    rejection_reason = Column(Text, nullable=True)

    resolved_at     = Column(DateTime(timezone=True), nullable=True)
    resolution_reason = Column(Text, nullable=True)

    first_inferred_at = Column(DateTime(timezone=True), server_default=func.now())
    last_updated_at   = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class DrugConditionSignal(Base):
    """
    Inference rule table. Seeded at startup. Maps drug INN → inferred condition.
    signal_strength = P(condition | drug observed), Noisy-OR combined across drugs.
    """
    __tablename__ = "drug_condition_signals"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    inn                 = Column(Text,         nullable=False)
    condition_code      = Column(String(20),   nullable=False)
    condition_name      = Column(Text,         nullable=False)
    condition_group     = Column(Text,         nullable=False)

    # Probabilistic fields — sensitivity/specificity populated when feedback available (10k+ patients)
    signal_strength     = Column(Float, nullable=False)          # P(condition | drug): 0.0–1.0
    sensitivity         = Column(Float, nullable=False, server_default="0.70")  # P(drug | condition)
    specificity         = Column(Float, nullable=False, server_default="0.50")  # P(drug for THIS condition | drug prescribed)
    condition_prevalence = Column(Float, nullable=True)          # India population base rate

    # Drug class behavior — drives episode gap tolerance
    medication_class    = Column(String(30), nullable=False, server_default="chronic_oral")
    episode_gap_tolerance = Column(Integer, nullable=False, server_default="45")
    is_prn              = Column(Boolean, nullable=False, server_default="false")

    requires_speciality = Column(Text,    nullable=True)   # only infer if this specialist prescribed
    min_prescriptions   = Column(Integer, nullable=False, server_default="1")

    __table_args__ = (UniqueConstraint("inn", "condition_code", name="uq_drug_condition"),)


class PhgEvent(Base):
    """
    Append-only audit log for all PHG state changes.
    Written in the same transaction as the prescription.
    payload_schema_version ensures future replay correctness.
    """
    __tablename__ = "phg_events"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    event_id        = Column(PG_UUID(as_uuid=True), unique=True, nullable=False, server_default=func.gen_random_uuid())
    event_type      = Column(String(60), nullable=False)

    # Schema version — increment when payload shape changes
    payload_schema_version = Column(SmallInteger, nullable=False, server_default="1")

    # Typed FKs (no polymorphic entity_id)
    patient_id      = Column(PG_UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    prescription_id = Column(PG_UUID(as_uuid=True), ForeignKey("prescriptions.id", ondelete="SET NULL"), nullable=True)
    episode_id      = Column(PG_UUID(as_uuid=True), nullable=True)   # app-enforced FK
    condition_id    = Column(PG_UUID(as_uuid=True), nullable=True)   # app-enforced FK
    doctor_id       = Column(PG_UUID(as_uuid=True), ForeignKey("doctors.id", ondelete="SET NULL"), nullable=True)

    # Which clinician triggered this event (NULL = system/pipeline)
    source_clinician_id = Column(PG_UUID(as_uuid=True), ForeignKey("clinicians.id", ondelete="SET NULL"), nullable=True)

    payload         = Column(JSONB, nullable=False, server_default=text("'{}' :: jsonb"))
    created_at      = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class PatientDrugReaction(Base):
    """
    Allergy and adverse drug reaction registry.
    Safety critical — cannot be retroactively inferred, must exist from day one.
    cross_reactive_inns: all INNs that share the allergy (e.g., all penicillins for pen allergy).
    """
    __tablename__ = "patient_drug_reactions"

    id                      = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    patient_id              = Column(PG_UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)

    reaction_type           = Column(String(20), nullable=False)
    # 'allergy' | 'intolerance' | 'adr' | 'contraindication'

    inn                     = Column(Text, nullable=False)
    cross_reactive_inns     = Column(ARRAY(Text), nullable=True)

    severity                = Column(String(20), nullable=False)
    # 'life_threatening' | 'severe' | 'moderate' | 'mild'

    manifestation           = Column(Text, nullable=True)
    source                  = Column(String(20), nullable=False)
    # 'clinician_entered' | 'inferred_from_discontinuation'

    source_episode_id       = Column(PG_UUID(as_uuid=True), nullable=True)  # app-enforced FK
    source_prescription_id  = Column(PG_UUID(as_uuid=True), ForeignKey("prescriptions.id", ondelete="SET NULL"), nullable=True)

    recorded_by             = Column(PG_UUID(as_uuid=True), ForeignKey("clinicians.id", ondelete="SET NULL"), nullable=True)
    recorded_at             = Column(DateTime(timezone=True), server_default=func.now())
    is_active               = Column(Boolean, nullable=False, server_default="true")
    notes                   = Column(Text, nullable=True)


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

        # ── PHG MVP migrations ─────────────────────────────────────────────
        # Add upload_hash and doctor_id columns to prescriptions
        for stmt in (
            "ALTER TABLE prescriptions ADD COLUMN IF NOT EXISTS upload_hash TEXT",
            "ALTER TABLE prescriptions ADD COLUMN IF NOT EXISTS doctor_id UUID REFERENCES doctors(id) ON DELETE SET NULL",
        ):
            try:
                conn.execute(text(stmt))
            except Exception as e:
                logger.debug("PHG migration (already applied?): %s — %s", stmt[:60], e)

        # Create partial unique index: only one active episode per patient+drug
        conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_episode_per_drug
            ON medication_episodes(patient_id, inn)
            WHERE status = 'active'
        """))

        # Create partial unique index: only one active/confirmed condition per patient+code
        conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_condition
            ON patient_conditions(patient_id, condition_code)
            WHERE status IN ('probable', 'confirmed')
        """))

        # Create unique index on doctors.registration_number_normalized
        conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_doctors_reg_normalized
            ON doctors(registration_number_normalized)
            WHERE registration_number_normalized IS NOT NULL
        """))

        # Additional performance indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_episodes_patient_inn    ON medication_episodes(patient_id, inn)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_episodes_patient_status ON medication_episodes(patient_id, status)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_dosage_hist_episode     ON medication_dosage_history(episode_id, recorded_date DESC)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_phg_events_patient      ON phg_events(patient_id, created_at DESC)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_phg_events_episode      ON phg_events(episode_id) WHERE episode_id IS NOT NULL"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_reactions_patient       ON patient_drug_reactions(patient_id, is_active)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_reactions_inn           ON patient_drug_reactions(inn, is_active)"))
        logger.info("PHG MVP DDL migrations applied.")

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

        # Seed drug condition signals for PHG inference
        seed_drug_condition_signals(db)
            
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


# ---------------------------------------------------------------------------
# PHG MVP — Drug Condition Signals Seed
# ---------------------------------------------------------------------------


def seed_drug_condition_signals(db: Session) -> None:
    """
    Seed drug → condition inference signals for the PHG engine.

    Design:
      - Uses INSERT … ON CONFLICT DO NOTHING so it's fully idempotent.
      - Does NOT rollback the parent transaction on duplicates.
      - 70 signals covering the most common Indian outpatient conditions.

    Signal columns (per row tuple):
      inn, condition_code, condition_name, condition_group,
      signal_strength, sensitivity, specificity, condition_prevalence,
      medication_class, episode_gap_tolerance, is_prn,
      min_prescriptions, requires_speciality
    """
    SIGNALS: list[tuple] = [
        # ── METABOLIC ─────────────────────────────────────────────────────────
        # Type 2 Diabetes (ICD-10: E11)  — India prevalence ~11%
        ("metformin",              "E11", "Type 2 Diabetes Mellitus",   "metabolic",       0.93, 0.85, 0.90, 0.11, "chronic_oral",       60, False, 1,  None),
        ("glimepiride",            "E11", "Type 2 Diabetes Mellitus",   "metabolic",       0.90, 0.70, 0.85, 0.11, "chronic_oral",       60, False, 1,  None),
        ("glibenclamide",          "E11", "Type 2 Diabetes Mellitus",   "metabolic",       0.88, 0.65, 0.85, 0.11, "chronic_oral",       60, False, 1,  None),
        ("gliclazide",             "E11", "Type 2 Diabetes Mellitus",   "metabolic",       0.88, 0.62, 0.85, 0.11, "chronic_oral",       60, False, 1,  None),
        ("sitagliptin",            "E11", "Type 2 Diabetes Mellitus",   "metabolic",       0.87, 0.55, 0.88, 0.11, "chronic_oral",       60, False, 2,  None),
        ("vildagliptin",           "E11", "Type 2 Diabetes Mellitus",   "metabolic",       0.87, 0.50, 0.88, 0.11, "chronic_oral",       60, False, 2,  None),
        ("dapagliflozin",          "E11", "Type 2 Diabetes Mellitus",   "metabolic",       0.88, 0.45, 0.88, 0.11, "chronic_oral",       60, False, 2,  None),
        ("empagliflozin",          "E11", "Type 2 Diabetes Mellitus",   "metabolic",       0.88, 0.40, 0.88, 0.11, "chronic_oral",       60, False, 2,  None),
        ("insulin glargine",       "E11", "Type 2 Diabetes Mellitus",   "metabolic",       0.80, 0.40, 0.70, 0.11, "chronic_injectable", 45, False, 1,  None),
        ("insulin aspart",         "E11", "Type 2 Diabetes Mellitus",   "metabolic",       0.78, 0.38, 0.68, 0.11, "chronic_injectable", 45, False, 1,  None),
        ("insulin soluble",        "E11", "Type 2 Diabetes Mellitus",   "metabolic",       0.75, 0.35, 0.65, 0.11, "chronic_injectable", 45, False, 1,  None),
        # Hypothyroidism (E03)  — India prevalence ~5%
        ("levothyroxine",          "E03", "Hypothyroidism",             "metabolic",       0.95, 0.95, 0.97, 0.05, "chronic_oral",       60, False, 1,  None),
        ("thyroxine",              "E03", "Hypothyroidism",             "metabolic",       0.95, 0.95, 0.97, 0.05, "chronic_oral",       60, False, 1,  None),
        # Dyslipidemia (E78)  — India prevalence ~25%
        ("atorvastatin",           "E78", "Dyslipidemia",               "metabolic",       0.85, 0.80, 0.80, 0.25, "chronic_oral",       60, False, 1,  None),
        ("rosuvastatin",           "E78", "Dyslipidemia",               "metabolic",       0.85, 0.75, 0.80, 0.25, "chronic_oral",       60, False, 1,  None),
        ("simvastatin",            "E78", "Dyslipidemia",               "metabolic",       0.82, 0.70, 0.78, 0.25, "chronic_oral",       60, False, 1,  None),
        ("fenofibrate",            "E78", "Dyslipidemia",               "metabolic",       0.75, 0.55, 0.80, 0.25, "chronic_oral",       60, False, 2,  None),
        # Gout / Hyperuricemia (M10)
        ("allopurinol",            "M10", "Gout / Hyperuricemia",       "metabolic",       0.92, 0.85, 0.95, 0.02, "chronic_oral",       60, False, 1,  None),
        ("febuxostat",             "M10", "Gout / Hyperuricemia",       "metabolic",       0.90, 0.70, 0.95, 0.02, "chronic_oral",       60, False, 2,  None),

        # ── CARDIOVASCULAR ────────────────────────────────────────────────────
        # Hypertension (I10)  — India prevalence ~28%
        ("amlodipine",             "I10", "Hypertension",               "cardiovascular",  0.85, 0.65, 0.75, 0.28, "chronic_oral",       60, False, 1,  None),
        ("telmisartan",            "I10", "Hypertension",               "cardiovascular",  0.90, 0.60, 0.85, 0.28, "chronic_oral",       60, False, 1,  None),
        ("ramipril",               "I10", "Hypertension",               "cardiovascular",  0.82, 0.55, 0.70, 0.28, "chronic_oral",       60, False, 1,  None),
        ("losartan",               "I10", "Hypertension",               "cardiovascular",  0.85, 0.50, 0.80, 0.28, "chronic_oral",       60, False, 1,  None),
        ("olmesartan",             "I10", "Hypertension",               "cardiovascular",  0.85, 0.45, 0.80, 0.28, "chronic_oral",       60, False, 1,  None),
        ("metoprolol",             "I10", "Hypertension",               "cardiovascular",  0.72, 0.55, 0.60, 0.28, "chronic_oral",       60, False, 1,  None),
        ("bisoprolol",             "I10", "Hypertension",               "cardiovascular",  0.72, 0.50, 0.60, 0.28, "chronic_oral",       60, False, 1,  None),
        ("hydrochlorothiazide",    "I10", "Hypertension",               "cardiovascular",  0.70, 0.45, 0.55, 0.28, "chronic_oral",       60, False, 1,  None),
        # CAD / post-PTCA (Z95)
        ("clopidogrel",            "Z95", "Coronary Artery Disease",    "cardiovascular",  0.85, 0.75, 0.90, 0.05, "chronic_oral",       45, False, 1,  None),
        ("aspirin",                "Z87", "CAD Prophylaxis",            "cardiovascular",  0.60, 0.80, 0.45, 0.15, "chronic_oral",       60, False, 1,  None),
        # Heart Failure (I50)
        ("furosemide",             "I50", "Heart Failure",              "cardiovascular",  0.72, 0.65, 0.60, 0.02, "chronic_oral",       45, False, 2, "cardiovascular"),
        ("spironolactone",         "I50", "Heart Failure",              "cardiovascular",  0.70, 0.55, 0.65, 0.02, "chronic_oral",       45, False, 2, "cardiovascular"),
        # Atrial fibrillation / DVT (I48)
        ("warfarin",               "I48", "Atrial Fibrillation / DVT",  "cardiovascular",  0.80, 0.70, 0.82, 0.03, "chronic_oral",       45, False, 1,  None),
        ("rivaroxaban",            "I48", "Atrial Fibrillation / DVT",  "cardiovascular",  0.82, 0.65, 0.85, 0.03, "chronic_oral",       45, False, 1,  None),
        ("dabigatran",             "I48", "Atrial Fibrillation / DVT",  "cardiovascular",  0.82, 0.60, 0.85, 0.03, "chronic_oral",       45, False, 1,  None),

        # ── RESPIRATORY ───────────────────────────────────────────────────────
        # Asthma / COPD (J45 / J44)
        ("salbutamol",             "J45", "Asthma / COPD",              "respiratory",     0.75, 0.85, 0.60, 0.06, "prn",                30, True,  1,  None),
        ("montelukast",            "J45", "Asthma / Allergic Rhinitis", "respiratory",     0.78, 0.60, 0.75, 0.06, "chronic_oral",       45, False, 1,  None),
        ("budesonide",             "J45", "Asthma / COPD",              "respiratory",     0.82, 0.65, 0.78, 0.06, "chronic_inhaled",    45, False, 2,  None),
        ("tiotropium",             "J44", "COPD",                       "respiratory",     0.85, 0.70, 0.85, 0.03, "chronic_inhaled",    45, False, 2,  None),
        ("formoterol",             "J45", "Asthma / COPD",              "respiratory",     0.80, 0.62, 0.78, 0.06, "chronic_inhaled",    45, False, 2,  None),

        # ── INFECTIOUS ────────────────────────────────────────────────────────
        # Pulmonary Tuberculosis (A15)  — high signal, specialist context
        ("rifampicin",             "A15", "Pulmonary Tuberculosis",     "infectious",      0.98, 0.98, 0.99, 0.02, "acute_course",       10, False, 1, "pulmonology"),
        ("isoniazid",              "A15", "Pulmonary Tuberculosis",     "infectious",      0.98, 0.98, 0.99, 0.02, "acute_course",       10, False, 1, "pulmonology"),
        ("pyrazinamide",           "A15", "Pulmonary Tuberculosis",     "infectious",      0.97, 0.95, 0.99, 0.02, "acute_course",       10, False, 1, "pulmonology"),
        ("ethambutol",             "A15", "Pulmonary Tuberculosis",     "infectious",      0.97, 0.95, 0.99, 0.02, "acute_course",       10, False, 1, "pulmonology"),
        # Malaria (B54)
        ("chloroquine",            "B54", "Malaria",                    "infectious",      0.85, 0.80, 0.90, 0.01, "acute_course",       10, False, 1,  None),
        ("artemether",             "B54", "Malaria",                    "infectious",      0.90, 0.85, 0.92, 0.01, "acute_course",        7, False, 1,  None),
        ("primaquine",             "B54", "Malaria",                    "infectious",      0.85, 0.80, 0.90, 0.01, "acute_course",        7, False, 1,  None),

        # ── NEUROLOGICAL ──────────────────────────────────────────────────────
        # Epilepsy (G40)
        ("carbamazepine",          "G40", "Epilepsy",                   "neurological",    0.90, 0.55, 0.88, 0.01, "chronic_oral",       45, False, 1, "neurology"),
        ("levetiracetam",          "G40", "Epilepsy",                   "neurological",    0.90, 0.50, 0.90, 0.01, "chronic_oral",       45, False, 1, "neurology"),
        ("valproate",              "G40", "Epilepsy",                   "neurological",    0.88, 0.55, 0.85, 0.01, "chronic_oral",       45, False, 1, "neurology"),
        ("phenytoin",              "G40", "Epilepsy",                   "neurological",    0.88, 0.50, 0.88, 0.01, "chronic_oral",       45, False, 1, "neurology"),
        # Parkinson's (G20)
        ("levodopa",               "G20", "Parkinson's Disease",        "neurological",    0.95, 0.90, 0.98, 0.002,"chronic_oral",       60, False, 1, "neurology"),
        ("pramipexole",            "G20", "Parkinson's Disease",        "neurological",    0.88, 0.70, 0.90, 0.002,"chronic_oral",       60, False, 2, "neurology"),
        # Migraine (G43)
        ("topiramate",             "G43", "Migraine Prophylaxis",       "neurological",    0.82, 0.55, 0.85, 0.01, "chronic_oral",       60, False, 2,  None),
        ("sumatriptan",            "G43", "Migraine",                   "neurological",    0.88, 0.80, 0.90, 0.01, "prn",                30, True,  1,  None),

        # ── PSYCHIATRIC ───────────────────────────────────────────────────────
        # Major Depressive Disorder (F32)
        ("sertraline",             "F32", "Major Depressive Disorder",  "psychiatric",     0.75, 0.45, 0.70, 0.04, "chronic_oral",       60, False, 2, "psychiatry"),
        ("escitalopram",           "F32", "Major Depressive Disorder",  "psychiatric",     0.75, 0.45, 0.72, 0.04, "chronic_oral",       60, False, 2, "psychiatry"),
        ("fluoxetine",             "F32", "Major Depressive Disorder",  "psychiatric",     0.72, 0.42, 0.70, 0.04, "chronic_oral",       60, False, 2, "psychiatry"),
        # Schizophrenia / Bipolar (F20)
        ("olanzapine",             "F20", "Schizophrenia / Bipolar",   "psychiatric",     0.82, 0.60, 0.85, 0.01, "chronic_oral",       60, False, 2, "psychiatry"),
        ("risperidone",            "F20", "Schizophrenia / Bipolar",   "psychiatric",     0.82, 0.58, 0.85, 0.01, "chronic_oral",       60, False, 2, "psychiatry"),
        ("lithium",                "F31", "Bipolar Affective Disorder", "psychiatric",     0.90, 0.75, 0.95, 0.01, "chronic_oral",       60, False, 2, "psychiatry"),
        # Anxiety (F41)
        ("clonazepam",             "F41", "Anxiety Disorder",           "psychiatric",     0.70, 0.55, 0.65, 0.04, "prn",                14, True,  1, "psychiatry"),
        ("alprazolam",             "F41", "Anxiety Disorder",           "psychiatric",     0.65, 0.55, 0.60, 0.04, "prn",                14, True,  1, "psychiatry"),

        # ── RHEUMATOLOGICAL ───────────────────────────────────────────────────
        # Rheumatoid Arthritis (M06 / M05)
        ("methotrexate",           "M06", "Rheumatoid Arthritis",       "rheumatological", 0.88, 0.55, 0.85, 0.01, "periodic",           45, False, 2, "rheumatology"),
        ("hydroxychloroquine",     "M05", "Rheumatoid Arthritis / SLE", "rheumatological", 0.78, 0.60, 0.75, 0.01, "chronic_oral",       60, False, 2, "rheumatology"),
        ("sulfasalazine",          "M06", "Rheumatoid Arthritis",       "rheumatological", 0.82, 0.50, 0.80, 0.01, "chronic_oral",       60, False, 2, "rheumatology"),

        # ── RENAL ─────────────────────────────────────────────────────────────
        # Chronic Kidney Disease (N18)
        ("sevelamer",              "N18", "Chronic Kidney Disease",     "renal",           0.85, 0.65, 0.88, 0.01, "chronic_oral",       60, False, 2, "nephrology"),
        ("erythropoietin",         "N18", "Chronic Kidney Disease",     "renal",           0.90, 0.70, 0.92, 0.01, "periodic",           30, False, 2, "nephrology"),

        # ── MUSCULOSKELETAL ───────────────────────────────────────────────────
        # Musculoskeletal Pain (M79) — PRN signal, low strength
        ("ibuprofen",              "M79", "Musculoskeletal Pain",       "musculoskeletal", 0.45, 0.70, 0.35, 0.20, "prn",                 7, True,  1,  None),
        ("diclofenac",             "M79", "Musculoskeletal Pain",       "musculoskeletal", 0.45, 0.70, 0.35, 0.20, "prn",                 7, True,  1,  None),
        ("tramadol",               "M79", "Musculoskeletal Pain",       "musculoskeletal", 0.50, 0.50, 0.45, 0.20, "acute_course",        7, False, 1,  None),
        # Osteoporosis (M81)
        ("alendronate",            "M81", "Osteoporosis",               "musculoskeletal", 0.88, 0.70, 0.90, 0.01, "periodic",           45, False, 2,  None),

        # ── GASTROINTESTINAL ──────────────────────────────────────────────────
        # GERD / PUD (K21 / K27)
        ("omeprazole",             "K21", "GERD / Peptic Ulcer",        "gastrointestinal",0.75, 0.80, 0.65, 0.08, "chronic_oral",       45, False, 1,  None),
        ("pantoprazole",           "K21", "GERD / Peptic Ulcer",        "gastrointestinal",0.75, 0.80, 0.65, 0.08, "chronic_oral",       45, False, 1,  None),
        ("rabeprazole",            "K21", "GERD / Peptic Ulcer",        "gastrointestinal",0.75, 0.75, 0.65, 0.08, "chronic_oral",       45, False, 1,  None),
        # Inflammatory Bowel (K51)
        ("mesalazine",             "K51", "Inflammatory Bowel Disease", "gastrointestinal",0.90, 0.75, 0.90, 0.003,"chronic_oral",       45, False, 2, "gastroenterology"),
        ("prednisolone",           "K51", "Inflammatory Bowel Disease", "gastrointestinal",0.65, 0.60, 0.40, 0.003,"acute_course",       14, False, 2, "gastroenterology"),
    ]

    # Use raw SQL INSERT ON CONFLICT DO NOTHING — safe to call repeatedly
    insert_sql = text("""
        INSERT INTO drug_condition_signals
            (inn, condition_code, condition_name, condition_group,
             signal_strength, sensitivity, specificity, condition_prevalence,
             medication_class, episode_gap_tolerance, is_prn,
             min_prescriptions, requires_speciality)
        VALUES
            (:inn, :condition_code, :condition_name, :condition_group,
             :signal_strength, :sensitivity, :specificity, :condition_prevalence,
             :medication_class, :episode_gap_tolerance, :is_prn,
             :min_prescriptions, :requires_speciality)
        ON CONFLICT (inn, condition_code) DO NOTHING
    """)

    inserted = 0
    for s in SIGNALS:
        db.execute(insert_sql, {
            "inn":                  s[0],
            "condition_code":       s[1],
            "condition_name":       s[2],
            "condition_group":      s[3],
            "signal_strength":      s[4],
            "sensitivity":          s[5],
            "specificity":          s[6],
            "condition_prevalence": s[7],
            "medication_class":     s[8],
            "episode_gap_tolerance": s[9],
            "is_prn":               s[10],
            "min_prescriptions":    s[11],
            "requires_speciality":  s[12],
        })
        inserted += 1

    logger.info("Drug condition signals seed complete — %d rows processed.", inserted)
