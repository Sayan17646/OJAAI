"""
doctor_registry.py — PHG MVP doctor deduplication and registry.

Deduplication strategy:
  Pass 1 — Exact match on registration_number_normalized (covers ~90% of cases)
  Pass 2 — Create new record (soundex dedup deferred to 10k patients)

Registration number normalization:
  Strip all non-alphanumeric characters, uppercase.
  'MH - 12345' → 'MH12345'
  'Reg. No. 12345' → '12345'
  'MAHARASHTRA 12345' → 'MAHARASHTRA12345'

Speciality → group mapping:
  Used by the condition inference engine to apply likelihood ratios.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from sqlalchemy.orm import Session

from src.database import Doctor
from src.event_schemas import DOCTOR_LINKED
from src.event_store import EventStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Speciality → canonical group mapping
# Used by ConditionInferenceEngine for likelihood ratio lookup
# ---------------------------------------------------------------------------
_SPECIALITY_GROUP_MAP: dict[str, str] = {
    # Metabolic / endocrine
    "endocrinolog":    "metabolic",
    "diabetolog":      "metabolic",
    "endocrine":       "metabolic",
    # Cardiovascular
    "cardiolog":       "cardiovascular",
    "cardiac":         "cardiovascular",
    "cardio":          "cardiovascular",
    # Respiratory
    "pulmonolog":      "respiratory",
    "respirator":      "respiratory",
    "chest":           "respiratory",
    "pulmo":           "respiratory",
    # Neurological
    "neurolog":        "neurological",
    "neuro":           "neurological",
    # Psychiatric
    "psychiatr":       "psychiatric",
    "psycholog":       "psychiatric",
    "mental health":   "psychiatric",
    # Rheumatological
    "rheumatolog":     "rheumatological",
    # Renal
    "nephrolog":       "renal",
    "kidney":          "renal",
    # General / family
    "general":         "general",
    "family":          "general",
    "internal":        "general",
}


def _normalize_registration_number(raw: Optional[str]) -> Optional[str]:
    """
    Strip all non-alphanumeric characters and uppercase.
    Returns None if the result is empty (no usable digits/letters).

    Examples:
        'MH - 12345'       → 'MH12345'
        'Reg. No. 12345'   → 'REGNO12345'
        'MAHARASHTRA12345' → 'MAHARASHTRA12345'
        None / ''          → None
    """
    if not raw:
        return None
    normalized = re.sub(r"[^A-Z0-9]", "", raw.upper())
    return normalized if normalized else None


def _map_speciality_to_group(speciality: Optional[str]) -> Optional[str]:
    """Map free-text speciality string to a canonical group key."""
    if not speciality:
        return None
    lower = speciality.lower()
    for keyword, group in _SPECIALITY_GROUP_MAP.items():
        if keyword in lower:
            return group
    return "general"


def _update_name_variants(doctor: Doctor, new_name: Optional[str]) -> None:
    """Append a novel OCR name variant to raw_name_variants (JSON array on the ORM model)."""
    if not new_name:
        return
    existing: list = doctor.raw_name_variants or []
    if new_name not in existing:
        doctor.raw_name_variants = existing + [new_name]


class DoctorRegistry:
    """
    Stateless helper class for doctor entity resolution.
    All methods are @staticmethod — no instance needed.
    """

    @staticmethod
    def get_or_create(
        db: Session,
        *,
        reg_number: Optional[str],
        name: Optional[str],
        speciality: Optional[str],
        clinic_name: Optional[str] = None,
        facility_id: Optional[object] = None,
        prescription_id: Optional[object] = None,
        patient_id: Optional[object] = None,
        emit_event: bool = True,
    ) -> Optional[Doctor]:
        """
        Resolve or create a Doctor entity.

        Returns None if both reg_number and name are absent —
        we never create ghost doctor records with no identity at all.

        Deduplication:
          1. Exact match on registration_number_normalized → return existing
          2. No match → create new record

        The caller must flush/commit — this function only flushes.

        Args:
            db:             Active SQLAlchemy session.
            reg_number:     Raw registration number from OCR (may be noisy).
            name:           Doctor name from OCR (may be noisy).
            speciality:     Free-text speciality from OCR.
            clinic_name:    Clinic/hospital name from OCR.
            facility_id:    System facility UUID (optional).
            prescription_id: Needed for event emission.
            patient_id:     Needed for event emission.
            emit_event:     Whether to emit a DOCTOR_LINKED event.

        Returns:
            Doctor ORM instance (existing or newly created), or None.
        """
        if not reg_number and not name:
            logger.debug("Skipping doctor creation — no registration number or name found")
            return None

        reg_norm = _normalize_registration_number(reg_number)
        speciality_group = _map_speciality_to_group(speciality)
        is_new = False

        # ── Pass 1: exact normalized registration number match ──────────────
        if reg_norm:
            doctor = (
                db.query(Doctor)
                .filter(Doctor.registration_number_normalized == reg_norm)
                .first()
            )
            if doctor:
                logger.debug(
                    "Doctor resolved by reg_number_normalized=%s (id=%s)", reg_norm, doctor.id
                )
                _update_name_variants(doctor, name)
                # Update speciality group if we now have better data
                if not doctor.speciality_group and speciality_group:
                    doctor.speciality_group = speciality_group
                db.flush()
                return doctor

        # ── Pass 2: create new record ───────────────────────────────────────
        doctor = Doctor(
            registration_number=reg_number,
            registration_number_normalized=reg_norm,
            name=name,
            speciality=speciality,
            speciality_group=speciality_group,
            clinic_name=clinic_name,
            facility_id=facility_id,
            raw_name_variants=[name] if name else [],
        )
        db.add(doctor)
        db.flush()
        is_new = True
        logger.info(
            "New doctor created: id=%s name=%r reg_norm=%s speciality=%s",
            doctor.id, name, reg_norm, speciality_group,
        )

        # ── Emit event ──────────────────────────────────────────────────────
        if emit_event and patient_id:
            EventStore.emit(
                db,
                DOCTOR_LINKED,
                patient_id=patient_id,
                prescription_id=prescription_id,
                doctor_id=doctor.id,
                payload={
                    "doctor_id": str(doctor.id),
                    "doctor_name": name,
                    "speciality": speciality,
                    "registration_number": reg_number,
                    "is_new_record": is_new,
                },
            )

        return doctor
