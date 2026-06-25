"""
models.py — Pydantic data contracts for OJAAI.

This is the single source of truth for all data shapes.
FastAPI routes use these for both request validation and response serialization.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Extracted entities (output of medical_ner.py)
# ---------------------------------------------------------------------------

class MedicationExtracted(BaseModel):
    """One medication line as parsed from raw OCR text."""

    raw_drug_name: str
    dosage_value: Optional[float] = None
    dosage_unit: Optional[str] = None        # mg | mcg | ml | g | iu | units
    frequency: Optional[str] = None          # human-readable: "twice daily"
    freq_per_day: Optional[int] = None       # integer: 0 | 1 | 2 | 3 | 4
    duration_days: Optional[int] = None      # null means chronic / ongoing
    route: str = "oral"


class ExtractedPrescription(BaseModel):
    """Full output of the NER module before drug normalisation."""

    medications: List[MedicationExtracted] = Field(default_factory=list)
    diagnosis: Optional[str] = None
    doctor_reg: Optional[str] = None
    patient_age: Optional[str] = None
    prescription_date: Optional[str] = None   # DD/MM/YYYY string
    confidence: float = Field(ge=0.0, le=1.0)
    raw_text: str = ""                         # kept for audit; never logged at INFO


# ---------------------------------------------------------------------------
# Normalised drug (output of drug_normalizer.py)
# ---------------------------------------------------------------------------

class NormalizedDrug(BaseModel):
    """A single drug after INN resolution and RxNorm lookup."""

    raw_drug_name: str
    inn: str                            # always populated; worst case == raw_drug_name
    rxcui: Optional[str] = None         # null if RxNorm lookup fails/times out
    standard_name: str                  # best available name

    dosage_value: Optional[float] = None
    dosage_unit: Optional[str] = None
    frequency: Optional[str] = None
    freq_per_day: Optional[int] = None
    duration_days: Optional[int] = None
    route: str = "oral"
    is_active: bool = True


# ---------------------------------------------------------------------------
# Drug-drug interaction (output of ddi_checker.py)
# ---------------------------------------------------------------------------

class DrugInteraction(BaseModel):
    """A detected drug-drug interaction between two drugs."""

    drug_1: str
    drug_2: str
    severity: str = Field(pattern="^(major|moderate|minor|unknown)$")
    description: str
    management: Optional[str] = None
    source: str                         # "CRITICAL_DDI_DB" | "DRUG_CLASS" | "OPENFDA"


class ClinicalSafetyAlert(BaseModel):
    """A detected drug-laboratory safety conflict warning."""

    drug_name: str
    analyte_name: str
    severity: str = Field(pattern="^(critical|warning|info)$")
    description: str
    val_detected: str
    ref_range: str
    management: Optional[str] = None


# ---------------------------------------------------------------------------
# Final pipeline output (response body for POST /parse)
# ---------------------------------------------------------------------------

class PrescriptionOutput(BaseModel):
    """
    The complete output of the OJAAI pipeline for a single prescription.
    This is the JSON object returned by POST /parse.
    """

    prescription_id: str                        # UUID
    patient_phone: Optional[str] = None

    # Extracted & normalised drugs
    medications: List[NormalizedDrug] = Field(default_factory=list)

    # Clinical context
    diagnosis: Optional[str] = None
    doctor_reg: Optional[str] = None
    patient_age: Optional[str] = None
    prescription_date: Optional[str] = None

    # Interactions across full active medication history
    interactions: List[DrugInteraction] = Field(default_factory=list)

    # Quality flags
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool = False
    has_major_interaction: bool = False

    # Storage metadata
    image_path: str

    # Clinical Safety Alerts (drug-laboratory warnings)
    safety_alerts: List[ClinicalSafetyAlert] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# API response shapes
# ---------------------------------------------------------------------------

class PatientHistory(BaseModel):
    """Response for GET /patient/{phone}/history"""

    phone: str
    total: int
    prescriptions: List[PrescriptionOutput]


class PatientInteractions(BaseModel):
    """Response for GET /patient/{phone}/interactions"""

    phone: str
    active_drugs: List[str]
    interactions: List[DrugInteraction]
    has_major_interaction: bool


class ErrorResponse(BaseModel):
    """Standardised error response — never exposes stack traces."""

    error: str
    detail: Optional[str] = None
    prescription_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Clinical Audit Dashboard Shapes
# ---------------------------------------------------------------------------

class ReviewQueueItemResponse(BaseModel):
    """Simplified item for the pending queue list on the left side of the dashboard."""

    id: str
    patient_id: Optional[str] = None
    image_path: str
    raw_ocr_text: str
    confidence: float
    reason: str
    resolved: bool
    created_at: str
    patient_phone: Optional[str] = None
    item_type: str = "prescription"


class ReviewQueueDetailResponse(BaseModel):
    """Complete detail of a review item, including dynamic non-mutating suggestions."""

    id: str
    image_path: str
    raw_ocr_text: str
    confidence: float
    reason: str
    created_at: str
    patient_phone: Optional[str] = None
    item_type: str = "prescription"
    draft_suggestion: Optional[ExtractedPrescription] = None
    draft_lab_suggestion: Optional[LabReportExtracted] = None


class ResolveMedicationInput(BaseModel):
    """Clinical correction input for a single medication entry."""

    raw_drug_name: str
    dosage_value: Optional[float] = None
    dosage_unit: Optional[str] = None
    frequency: Optional[str] = None
    freq_per_day: Optional[int] = None
    duration_days: Optional[int] = None
    route: str = "oral"


class ResolvePrescriptionInput(BaseModel):
    """Clinical correction input for the entire prescription resolution request."""

    patient_phone: Optional[str] = None
    doctor_reg: Optional[str] = None
    patient_age: Optional[str] = None
    diagnosis: Optional[str] = None
    prescription_date: Optional[str] = None
    medications: List[ResolveMedicationInput] = Field(default_factory=list)
    override_reason: Optional[str] = None


class ResolveLabResultInput(BaseModel):
    """Clinical correction input for a single lab analyte result."""

    raw_name: str
    analyte_name: str
    value: float
    unit: Optional[str] = None
    ref_range: Optional[str] = None
    flag: str = "normal"


class ResolveLabInput(BaseModel):
    """Clinical correction input for the entire lab report resolution request."""

    patient_phone: Optional[str] = None
    lab_name: Optional[str] = None
    report_date: Optional[str] = None
    results: List[ResolveLabResultInput] = Field(default_factory=list)
    override_reason: Optional[str] = None



# ---------------------------------------------------------------------------
# Diagnostic Lab Report Ingestion Shapes
# ---------------------------------------------------------------------------

class LabResultExtracted(BaseModel):
    """A single diagnostic analyte result from a lab report."""

    raw_name: str
    analyte_name: str                  # Normalized (e.g. HBA1C, CREATININE, HEMOGLOBIN, TSH, LDL)
    value: float
    unit: Optional[str] = None
    ref_range: Optional[str] = None
    flag: str = "normal"               # normal | high | low


class LabReportExtracted(BaseModel):
    """Extracted lab report metadata & results before persistence."""

    lab_name: Optional[str] = None
    report_date: Optional[str] = None
    results: List[LabResultExtracted] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    raw_text: str = ""


class LabReportOutput(BaseModel):
    """Complete output of the diagnostic lab parser."""

    lab_report_id: str                 # UUID
    patient_phone: Optional[str] = None
    lab_name: Optional[str] = None
    report_date: Optional[str] = None
    results: List[LabResultExtracted] = Field(default_factory=list)
    confidence: float
    needs_human_review: bool = False
    image_path: str


class ClinicianLoginInput(BaseModel):
    """Input for clinician dashboard authentication."""
    email: str
    password: str


# ---------------------------------------------------------------------------
# Clinical Safety Rules Admin CRUD Shapes
# ---------------------------------------------------------------------------

class ClinicalSafetyRuleSchema(BaseModel):
    id: str
    rule_name: str
    drug_inn: str
    analyte_name: str
    operator: Optional[str] = None
    threshold_value: Optional[float] = None
    threshold_value_max: Optional[float] = None
    flag_match: Optional[str] = None
    gender_specific: str = "both"
    severity: str = "warning"
    description_template: str
    management_plan: Optional[str] = None
    is_enabled: bool = True
    is_deleted: bool = False
    version: int = 1
    created_at: str
    updated_at: str


class CreateClinicalSafetyRuleInput(BaseModel):
    rule_name: str
    drug_inn: str
    analyte_name: str
    operator: Optional[str] = None
    threshold_value: Optional[float] = None
    threshold_value_max: Optional[float] = None
    flag_match: Optional[str] = None
    gender_specific: str = "both"
    severity: str = "warning"
    description_template: str
    management_plan: Optional[str] = None
    is_enabled: bool = True


# Resolve forward references after all models are defined
ReviewQueueDetailResponse.model_rebuild()


# ---------------------------------------------------------------------------
# PHG MVP — Patient History Graph response shapes
# ---------------------------------------------------------------------------

class DoctorSnapshot(BaseModel):
    """Prescribing doctor as seen on an episode or prescription."""
    id: str
    name: Optional[str] = None
    speciality: Optional[str] = None
    speciality_group: Optional[str] = None
    clinic_name: Optional[str] = None
    registration_number: Optional[str] = None


class DosageHistoryEntry(BaseModel):
    """One prescribing event within a medication episode."""
    id: int
    raw_drug_name: str
    dosage_value: Optional[float] = None
    dosage_unit: Optional[str] = None
    frequency: Optional[str] = None
    freq_per_day: Optional[float] = None
    duration_days: Optional[int] = None
    route: Optional[str] = None
    stop_reason: Optional[str] = None
    switched_to_inn: Optional[str] = None
    refill_number: int = 1
    recorded_date: Optional[str] = None


class MedicationEpisodeResponse(BaseModel):
    """One continuous treatment episode for a single drug."""
    id: str
    inn: str
    is_fdc: bool = False
    fdc_components: List[str] = Field(default_factory=list)
    dispensing_type: str
    status: str
    start_date: Optional[str] = None
    estimated_end_date: Optional[str] = None
    actual_end_date: Optional[str] = None
    gap_tolerance_days: int
    prescription_count: int
    latest_dosage: Optional[str] = None
    latest_frequency: Optional[str] = None
    version: int
    dosage_history: List[DosageHistoryEntry] = Field(default_factory=list)
    latest_doctor: Optional[DoctorSnapshot] = None


class ConditionResponse(BaseModel):
    """An inferred condition with confidence and audit trail."""
    id: str
    condition_code: str
    condition_name: str
    condition_group: Optional[str] = None
    episode_number: int = 1
    status: str                         # probable | confirmed | rejected | resolved
    confidence: float
    inference_engine_version: str
    inference_basis: dict = Field(default_factory=dict)
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    rejection_reason: Optional[str] = None
    resolved_at: Optional[str] = None
    first_inferred_at: Optional[str] = None
    last_updated_at: Optional[str] = None


class DrugReactionResponse(BaseModel):
    """Allergy or adverse drug reaction record."""
    id: str
    reaction_type: str
    inn: str
    cross_reactive_inns: List[str] = Field(default_factory=list)
    severity: str
    manifestation: Optional[str] = None
    source: str
    is_active: bool
    recorded_at: Optional[str] = None
    notes: Optional[str] = None


class PatientGraphSummary(BaseModel):
    """Summary counts for the patient graph overview."""
    active_medication_count: int
    probable_conditions: int
    confirmed_conditions: int
    active_reactions: int


class PatientGraphResponse(BaseModel):
    """Full PHG snapshot for a patient. Response for GET /api/patients/{phone}/graph."""
    patient_id: str
    active_medications: List[MedicationEpisodeResponse] = Field(default_factory=list)
    conditions: List[ConditionResponse] = Field(default_factory=list)
    drug_reactions: List[DrugReactionResponse] = Field(default_factory=list)
    prescribing_doctors: List[DoctorSnapshot] = Field(default_factory=list)
    summary: PatientGraphSummary


class ConditionReviewInput(BaseModel):
    """
    Clinician review input for POST /api/patients/{phone}/conditions/{code}/confirm.
    action='confirm' → status becomes 'confirmed'
    action='reject'  → status becomes 'rejected', rejection_reason required
    action='resolve' → status becomes 'resolved', resolution_reason required
    """
    action: str = Field(pattern="^(confirm|reject|resolve)$")
    rejection_reason: Optional[str] = None
    resolution_reason: Optional[str] = None


class RecordAllergyInput(BaseModel):
    """Input for POST /api/patients/{phone}/reactions — record allergy or ADR."""
    reaction_type: str = Field(pattern="^(allergy|intolerance|adr|contraindication)$")
    inn: str
    cross_reactive_inns: List[str] = Field(default_factory=list)
    severity: str = Field(pattern="^(life_threatening|severe|moderate|mild)$")
    manifestation: Optional[str] = None
    notes: Optional[str] = None
