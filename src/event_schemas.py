"""
event_schemas.py — PHG MVP event type constants and payload shape documentation.

Each event type corresponds to a specific schema version.
When the payload shape changes, increment payload_schema_version in PhgEvent
and add a new entry here.

payload_schema_version=1 is the PHG MVP baseline.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

# Medication episode lifecycle
MEDICATION_STARTED    = "MEDICATION_STARTED"      # new episode created
MEDICATION_CONTINUED  = "MEDICATION_CONTINUED"    # same dose refill
MEDICATION_DOSE_CHANGED = "MEDICATION_DOSE_CHANGED"  # dosage change within episode
MEDICATION_COMPLETED  = "MEDICATION_COMPLETED"    # gap exceeded, episode closed
MEDICATION_DISCONTINUED = "MEDICATION_DISCONTINUED"  # explicitly stopped

# Condition lifecycle
CONDITION_INFERRED    = "CONDITION_INFERRED"      # first time inference fires
CONDITION_UPDATED     = "CONDITION_UPDATED"       # confidence recalculated
CONDITION_CONFIRMED   = "CONDITION_CONFIRMED"     # clinician confirmed
CONDITION_REJECTED    = "CONDITION_REJECTED"      # clinician rejected inference
CONDITION_RESOLVED    = "CONDITION_RESOLVED"      # condition resolved after treatment

# Other
DOCTOR_LINKED         = "DOCTOR_LINKED"           # doctor linked to prescription
ALLERGY_RECORDED      = "ALLERGY_RECORDED"        # drug reaction recorded


# ---------------------------------------------------------------------------
# Payload schemas (v1) — documentation only (not enforced at runtime in MVP)
# At 10k patients: add a pydantic validator against these schemas.
# ---------------------------------------------------------------------------

PAYLOAD_SCHEMAS: dict[tuple[str, int], dict] = {
    (MEDICATION_STARTED, 1): {
        "required": ["inn", "start_date", "dosage", "frequency", "duration_days",
                     "is_fdc", "episode_id"],
        "optional": ["rxcui", "doctor_id", "fdc_components"],
    },
    (MEDICATION_CONTINUED, 1): {
        "required": ["inn", "episode_id", "prescription_count", "refill_number"],
        "optional": ["dosage", "frequency"],
    },
    (MEDICATION_DOSE_CHANGED, 1): {
        "required": ["inn", "episode_id", "old_dosage", "new_dosage"],
        "optional": [],
    },
    (MEDICATION_COMPLETED, 1): {
        "required": ["inn", "episode_id", "actual_end_date"],
        "optional": [],
    },
    (MEDICATION_DISCONTINUED, 1): {
        "required": ["inn", "episode_id", "stop_reason"],
        "optional": ["switched_to_inn"],
    },
    (CONDITION_INFERRED, 1): {
        "required": ["condition_code", "condition_name", "confidence",
                     "condition_id", "inference_engine_version"],
        "optional": ["supporting_drugs"],
    },
    (CONDITION_UPDATED, 1): {
        "required": ["condition_code", "condition_id", "old_confidence", "new_confidence"],
        "optional": [],
    },
    (CONDITION_CONFIRMED, 1): {
        "required": ["condition_code", "condition_id", "clinician_id"],
        "optional": [],
    },
    (CONDITION_REJECTED, 1): {
        "required": ["condition_code", "condition_id", "clinician_id", "rejection_reason"],
        "optional": [],
    },
    (CONDITION_RESOLVED, 1): {
        "required": ["condition_code", "condition_id", "resolution_reason"],
        "optional": [],
    },
    (DOCTOR_LINKED, 1): {
        "required": ["doctor_id", "doctor_name", "speciality"],
        "optional": ["registration_number"],
    },
    (ALLERGY_RECORDED, 1): {
        "required": ["inn", "reaction_type", "severity"],
        "optional": ["manifestation", "cross_reactive_inns"],
    },
}
