"""
patient_history.py — PHG MVP query layer (read-only).

PatientHistoryService provides clean, structured reads from the PHG.
Never modifies data — all writes go through EpisodeManager, DoctorRegistry,
ConditionInferenceEngine, and EventStore.

Used exclusively by api.py PHG endpoints.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.database import (
    Doctor,
    MedicationDosageHistory,
    MedicationEpisode,
    Patient,
    PatientCondition,
    PatientDrugReaction,
    PhgEvent,
    Prescription,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _episode_to_dict(ep: MedicationEpisode) -> Dict:
    return {
        "id": str(ep.id),
        "inn": ep.inn,
        "is_fdc": ep.is_fdc,
        "fdc_components": ep.fdc_components or [],
        "dispensing_type": ep.dispensing_type,
        "status": ep.status,
        "start_date": str(ep.start_date) if ep.start_date else None,
        "estimated_end_date": str(ep.estimated_end_date) if ep.estimated_end_date else None,
        "actual_end_date": str(ep.actual_end_date) if ep.actual_end_date else None,
        "gap_tolerance_days": ep.gap_tolerance_days,
        "prescription_count": ep.prescription_count,
        "latest_dosage": f"{ep.latest_dosage_value or ''} {ep.latest_dosage_unit or ''}".strip() or None,
        "latest_frequency": ep.latest_frequency,
        "version": ep.version,
    }


def _dosage_row_to_dict(row: MedicationDosageHistory) -> Dict:
    return {
        "id": row.id,
        "raw_drug_name": row.raw_drug_name,
        "dosage_value": row.dosage_value,
        "dosage_unit": row.dosage_unit,
        "frequency": row.frequency,
        "freq_per_day": row.freq_per_day,
        "duration_days": row.duration_days,
        "route": row.route,
        "stop_reason": row.stop_reason,
        "switched_to_inn": row.switched_to_inn,
        "refill_number": row.refill_number,
        "recorded_date": str(row.recorded_date) if row.recorded_date else None,
    }


def _condition_to_dict(cond: PatientCondition) -> Dict:
    return {
        "id": str(cond.id),
        "condition_code": cond.condition_code,
        "condition_name": cond.condition_name,
        "condition_group": cond.condition_group,
        "episode_number": cond.episode_number,
        "status": cond.status,
        "confidence": cond.confidence,
        "inference_engine_version": cond.inference_engine_version,
        "inference_basis": cond.inference_basis,
        "reviewed_by": str(cond.reviewed_by) if cond.reviewed_by else None,
        "reviewed_at": cond.reviewed_at.isoformat() if cond.reviewed_at else None,
        "rejection_reason": cond.rejection_reason,
        "resolved_at": cond.resolved_at.isoformat() if cond.resolved_at else None,
        "first_inferred_at": cond.first_inferred_at.isoformat() if cond.first_inferred_at else None,
        "last_updated_at": cond.last_updated_at.isoformat() if cond.last_updated_at else None,
    }


def _doctor_to_dict(doc: Optional[Doctor]) -> Optional[Dict]:
    if not doc:
        return None
    return {
        "id": str(doc.id),
        "name": doc.name,
        "speciality": doc.speciality,
        "speciality_group": doc.speciality_group,
        "clinic_name": doc.clinic_name,
        "registration_number": doc.registration_number,
    }


def _reaction_to_dict(r: PatientDrugReaction) -> Dict:
    return {
        "id": str(r.id),
        "reaction_type": r.reaction_type,
        "inn": r.inn,
        "cross_reactive_inns": r.cross_reactive_inns or [],
        "severity": r.severity,
        "manifestation": r.manifestation,
        "source": r.source,
        "is_active": r.is_active,
        "recorded_at": r.recorded_at.isoformat() if r.recorded_at else None,
        "notes": r.notes,
    }


# ---------------------------------------------------------------------------
# PatientHistoryService
# ---------------------------------------------------------------------------

class PatientHistoryService:
    """Read-only query layer over the PHG. Never modifies data."""

    @staticmethod
    def get_patient_by_phone(db: Session, phone: str) -> Optional[Patient]:
        return db.query(Patient).filter(Patient.phone == phone).first()

    @staticmethod
    def get_full_graph(db: Session, patient_id: Any) -> Dict:
        """
        Full PHG for a patient:
          - active medication episodes (with latest dosage snapshot)
          - all inferred conditions (with confidence)
          - known drug reactions / allergies
          - prescribing doctors seen

        Designed for the GET /api/patients/{phone}/graph endpoint.
        Optimised for fast reads: denormalised snapshot fields avoid joins.
        """
        # Active episodes
        active_episodes = (
            db.query(MedicationEpisode)
            .filter(
                MedicationEpisode.patient_id == patient_id,
                MedicationEpisode.status == "active",
            )
            .order_by(MedicationEpisode.start_date.desc())
            .all()
        )

        # All conditions (any status)
        conditions = (
            db.query(PatientCondition)
            .filter(PatientCondition.patient_id == patient_id)
            .order_by(PatientCondition.confidence.desc())
            .all()
        )

        # Active drug reactions
        reactions = (
            db.query(PatientDrugReaction)
            .filter(
                PatientDrugReaction.patient_id == patient_id,
                PatientDrugReaction.is_active == True,
            )
            .all()
        )

        # Distinct prescribing doctors (from episodes with a doctor linked)
        doctor_ids = list({
            ep.latest_doctor_id
            for ep in active_episodes
            if ep.latest_doctor_id
        })
        doctors = (
            db.query(Doctor).filter(Doctor.id.in_(doctor_ids)).all()
            if doctor_ids else []
        )

        return {
            "patient_id": str(patient_id),
            "active_medications": [_episode_to_dict(ep) for ep in active_episodes],
            "conditions": [_condition_to_dict(c) for c in conditions],
            "drug_reactions": [_reaction_to_dict(r) for r in reactions],
            "prescribing_doctors": [_doctor_to_dict(d) for d in doctors],
            "summary": {
                "active_medication_count": len(active_episodes),
                "probable_conditions": sum(
                    1 for c in conditions if c.status == "probable"
                ),
                "confirmed_conditions": sum(
                    1 for c in conditions if c.status == "confirmed"
                ),
                "active_reactions": len(reactions),
            },
        }

    @staticmethod
    def get_full_timeline(db: Session, patient_id: Any) -> List[Dict]:
        """
        All medication episodes for the patient, sorted by start_date DESC.
        Includes dosage history for each episode.
        For GET /api/patients/{phone}/timeline
        """
        episodes = (
            db.query(MedicationEpisode)
            .filter(MedicationEpisode.patient_id == patient_id)
            .order_by(MedicationEpisode.start_date.desc())
            .all()
        )
        result = []
        for ep in episodes:
            ep_dict = _episode_to_dict(ep)
            # Load dosage history (already ordered by recorded_date DESC via relationship)
            history = (
                db.query(MedicationDosageHistory)
                .filter(MedicationDosageHistory.episode_id == ep.id)
                .order_by(MedicationDosageHistory.recorded_date.desc())
                .all()
            )
            ep_dict["dosage_history"] = [_dosage_row_to_dict(r) for r in history]
            # Include doctor snapshot
            doc = db.query(Doctor).get(ep.latest_doctor_id) if ep.latest_doctor_id else None
            ep_dict["latest_doctor"] = _doctor_to_dict(doc)
            result.append(ep_dict)
        return result

    @staticmethod
    def get_drug_timeline(db: Session, patient_id: Any, inn: str) -> Dict:
        """
        All episodes + full dosage history for a single drug (INN).
        For GET /api/patients/{phone}/timeline/{inn}
        """
        inn_lower = inn.lower().strip()
        episodes = (
            db.query(MedicationEpisode)
            .filter(
                MedicationEpisode.patient_id == patient_id,
                MedicationEpisode.inn == inn_lower,
            )
            .order_by(MedicationEpisode.start_date.desc())
            .all()
        )
        episodes_data = []
        for ep in episodes:
            ep_dict = _episode_to_dict(ep)
            history = (
                db.query(MedicationDosageHistory)
                .filter(MedicationDosageHistory.episode_id == ep.id)
                .order_by(MedicationDosageHistory.recorded_date.desc())
                .all()
            )
            ep_dict["dosage_history"] = [_dosage_row_to_dict(r) for r in history]
            episodes_data.append(ep_dict)

        return {
            "inn": inn_lower,
            "patient_id": str(patient_id),
            "episode_count": len(episodes_data),
            "episodes": episodes_data,
        }

    @staticmethod
    def get_conditions(db: Session, patient_id: Any) -> List[Dict]:
        """
        All inferred conditions for the patient, sorted by confidence DESC.
        For GET /api/patients/{phone}/conditions
        """
        conditions = (
            db.query(PatientCondition)
            .filter(PatientCondition.patient_id == patient_id)
            .order_by(PatientCondition.confidence.desc())
            .all()
        )
        return [_condition_to_dict(c) for c in conditions]

    @staticmethod
    def get_condition_by_code(
        db: Session, patient_id: Any, condition_code: str
    ) -> Optional[PatientCondition]:
        """
        Fetch the active (probable/confirmed) condition record for a patient+code.
        Returns None if not found or already resolved/rejected.
        """
        return (
            db.query(PatientCondition)
            .filter(
                PatientCondition.patient_id == patient_id,
                PatientCondition.condition_code == condition_code,
                PatientCondition.status.in_(["probable", "confirmed"]),
            )
            .first()
        )
