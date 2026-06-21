"""
condition_inference.py — PHG MVP condition inference engine.

Algorithm: Noisy-OR probability combination (not additive heuristic).

For each condition in drug_condition_signals:
  1. Find all active/recent medication episodes matching that drug (including FDC components)
  2. Combine evidence using Noisy-OR: P(cond | drug_1 AND drug_2 ... AND drug_n)
     = 1 - PROD(1 - P_i * specificity_i * activity_weight_i)
  3. Apply temporal decay toward prior if no prescription in > 180 days
  4. Apply speciality likelihood ratio in log-odds space
  5. Apply credibility weighting by prescription count
  6. UPSERT patient_conditions — never deletes existing clinician-confirmed records

Key safety rules:
  - Never overwrites 'confirmed' or 'rejected' conditions.
  - Never sets confidence above 1.0 or below 0.0.
  - Runs synchronously in the same transaction as the prescription write.
    (Deferred to async queue at 10k patients if latency becomes an issue.)
"""

from __future__ import annotations

import logging
import math
import uuid as uuid_lib
from collections import defaultdict
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.database import (
    Doctor,
    DrugConditionSignal,
    MedicationEpisode,
    PatientCondition,
)
from src.event_schemas import CONDITION_INFERRED, CONDITION_UPDATED
from src.event_store import EventStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Speciality likelihood ratios
# Applied in log-odds space to avoid range violations
# ---------------------------------------------------------------------------
_SPECIALITY_LR: Dict[Tuple[str, str], float] = {
    ("metabolic",        "metabolic"):        1.40,
    ("endocrinolog",     "metabolic"):        1.35,
    ("cardiovascular",   "cardiovascular"):   1.30,
    ("respiratory",      "respiratory"):      1.25,
    ("neurological",     "neurological"):     1.25,
    ("psychiatric",      "psychiatric"):      1.20,
    ("rheumatological",  "rheumatological"):  1.30,
    ("renal",            "renal"):            1.25,
}

# Confidence threshold below which we do NOT upsert a condition (noise filter)
_MIN_CONFIDENCE = 0.30


def _noisy_or_confidence(
    drug_signals: List[Tuple[float, float, bool]],
    # each tuple: (signal_strength, specificity, is_episode_active)
    days_since_last_rx: int,
    speciality_group: Optional[str],
    condition_group: str,
    prescription_count: int,
    condition_prevalence: float,
) -> float:
    """
    Proper Noisy-OR confidence computation.

    Args:
        drug_signals:        List of (signal_strength, specificity, is_active) per supporting drug.
        days_since_last_rx:  Days since last prescription in any supporting episode.
        speciality_group:    Canonical speciality group of the prescribing doctor.
        condition_group:     Canonical condition group from drug_condition_signals.
        prescription_count:  Max prescription_count across supporting episodes (credibility proxy).
        condition_prevalence: India-level base rate for this condition (Bayesian prior).

    Returns:
        Float in [0.0, 1.0] — posterior confidence that patient has this condition.
    """
    if not drug_signals:
        return condition_prevalence

    # ── Step 1: Noisy-OR combination ──────────────────────────────────────
    # Active episodes contribute full weight; inactive contribute 40% (temporal discount)
    combined = 1.0 - math.prod(
        1.0 - (s * sp * (1.0 if active else 0.4))
        for s, sp, active in drug_signals
    )
    combined = max(0.0, min(1.0, combined))  # guard against float precision errors

    # ── Step 2: Temporal decay toward prior after 180 days ─────────────────
    # A patient who hasn't been prescribed a drug in 6+ months has decaying confidence
    if days_since_last_rx > 180:
        decay = math.exp(-0.005 * (days_since_last_rx - 180))
        combined = condition_prevalence + (combined - condition_prevalence) * decay
        combined = max(0.0, min(1.0, combined))

    # ── Step 3: Speciality likelihood ratio (log-odds space) ───────────────
    lr = 1.0
    if speciality_group:
        # Try both (speciality_group, condition_group) and (speciality_group, speciality_group)
        lr = _SPECIALITY_LR.get(
            (speciality_group, condition_group),
            _SPECIALITY_LR.get((speciality_group, speciality_group), 1.0),
        )
    if lr != 1.0:
        eps = 1e-9
        # Clamp combined away from 0 and 1 to keep log-odds finite
        clamped = max(eps, min(1.0 - eps, combined))
        log_odds = math.log(clamped / (1.0 - clamped)) + math.log(lr)
        combined = 1.0 / (1.0 + math.exp(-log_odds))
        combined = max(0.0, min(1.0, combined))

    # ── Step 4: Credibility weighting ──────────────────────────────────────
    # More prescriptions → closer to the inferred posterior (vs. prior)
    # log1p(10) ≈ 2.4 — 10 prescriptions = full credibility
    credibility = min(1.0, math.log1p(prescription_count) / math.log1p(10))
    result = credibility * combined + (1.0 - credibility) * condition_prevalence
    return round(max(0.0, min(1.0, result)), 4)


class ConditionInferenceEngine:
    """
    Stateless inference engine.
    Run once per prescription upload, synchronously.
    At < 10k patients typical runtime: < 50ms.
    """

    @staticmethod
    def infer_for_patient(db: Session, patient_id: Any) -> List[PatientCondition]:
        """
        Full inference pass for one patient.

        Steps:
          1. Load all active + recent completed episodes
          2. Load all drug_condition_signals matching these drugs (including FDC components)
          3. Group signals by condition_code
          4. Compute confidence per condition
          5. UPSERT patient_conditions (never touch confirmed/rejected records)
          6. Emit CONDITION_INFERRED or CONDITION_UPDATED events

        Returns the list of upserted PatientCondition records.
        """
        today = date.today()

        # ── 1. Load episodes (active + completed within 365 days) ──────────
        cutoff = today.replace(year=today.year - 1)  # approx 1 year back
        episodes = (
            db.query(MedicationEpisode)
            .filter(
                MedicationEpisode.patient_id == patient_id,
                MedicationEpisode.status.in_(["active", "completed", "prn_snapshot"]),
                MedicationEpisode.start_date >= cutoff,
            )
            .all()
        )
        if not episodes:
            logger.debug("No episodes found for patient=%s — skipping inference", patient_id)
            return []

        # ── 2. Collect all INNs (including FDC components for safety matching) ──
        all_inns: set[str] = set()
        for ep in episodes:
            all_inns.add(ep.inn)
            if ep.fdc_components:
                all_inns.update(ep.fdc_components)

        # ── 3. Load matching signals ────────────────────────────────────────
        signals = (
            db.query(DrugConditionSignal)
            .filter(DrugConditionSignal.inn.in_(list(all_inns)))
            .all()
        )
        if not signals:
            logger.debug("No signals matched for patient=%s INNs=%s", patient_id, all_inns)
            return []

        # ── 4. Group signals by condition_code ─────────────────────────────
        by_condition: Dict[str, List[DrugConditionSignal]] = defaultdict(list)
        for sig in signals:
            by_condition[sig.condition_code].append(sig)

        # ── 5. Get prescribing doctors for speciality boost ─────────────────
        doctor_ids = {ep.latest_doctor_id for ep in episodes if ep.latest_doctor_id}
        doctors: List[Doctor] = []
        if doctor_ids:
            doctors = db.query(Doctor).filter(Doctor.id.in_(list(doctor_ids))).all()

        upserted: List[PatientCondition] = []

        for condition_code, sigs in by_condition.items():
            sig0 = sigs[0]  # representative for condition metadata

            # ── Build drug_tuples for Noisy-OR ──────────────────────────────
            drug_tuples: List[Tuple[float, float, bool]] = []
            max_rx_count = 0
            max_recent_date: Optional[date] = None

            for sig in sigs:
                # Match episodes for this signal's INN (direct or FDC component match)
                matching_eps = [
                    ep for ep in episodes
                    if ep.inn == sig.inn
                    or (ep.fdc_components and sig.inn in ep.fdc_components)
                ]
                # Apply min_prescriptions filter
                if sig.min_prescriptions > 1:
                    matching_eps = [
                        ep for ep in matching_eps
                        if ep.prescription_count >= sig.min_prescriptions
                    ]
                if not matching_eps:
                    continue

                for ep in matching_eps:
                    is_active = ep.status == "active"
                    drug_tuples.append((sig.signal_strength, sig.specificity, is_active))
                    max_rx_count = max(max_rx_count, ep.prescription_count)
                    end = ep.estimated_end_date or ep.actual_end_date or ep.start_date
                    if max_recent_date is None or end > max_recent_date:
                        max_recent_date = end

            if not drug_tuples:
                continue

            # ── Speciality requirement gate ──────────────────────────────────
            if sig0.requires_speciality:
                required = sig0.requires_speciality.lower()
                has_required = any(
                    d.speciality and required in d.speciality.lower()
                    for d in doctors
                )
                if not has_required:
                    logger.debug(
                        "Skipping %s — requires speciality %r not found",
                        condition_code, sig0.requires_speciality,
                    )
                    continue

            # ── Compute confidence ───────────────────────────────────────────
            days_since = (
                (today - max_recent_date).days if max_recent_date else 999
            )
            # Use the speciality_group of the most specialised doctor
            speciality_group = None
            for doc in doctors:
                if doc.speciality_group and doc.speciality_group != "general":
                    speciality_group = doc.speciality_group
                    break
            if not speciality_group and doctors:
                speciality_group = doctors[0].speciality_group

            prevalence = sig0.condition_prevalence or 0.10
            confidence = _noisy_or_confidence(
                drug_signals=drug_tuples,
                days_since_last_rx=days_since,
                speciality_group=speciality_group,
                condition_group=sig0.condition_group,
                prescription_count=max_rx_count,
                condition_prevalence=prevalence,
            )

            # Noise filter — don't create conditions with near-prior confidence
            if confidence < _MIN_CONFIDENCE:
                logger.debug(
                    "Skipping %s — confidence %.3f below threshold %.3f",
                    condition_code, confidence, _MIN_CONFIDENCE,
                )
                continue

            # ── Build inference_basis for audit trail ────────────────────────
            inference_basis = {
                "supporting_drugs": [
                    {
                        "inn": s.inn,
                        "signal_strength": s.signal_strength,
                        "specificity": s.specificity,
                    }
                    for s in sigs
                    if any(
                        ep.inn == s.inn or (ep.fdc_components and s.inn in ep.fdc_components)
                        for ep in episodes
                    )
                ],
                "prescription_count": max_rx_count,
                "days_since_last_rx": days_since,
                "speciality_group": speciality_group,
                "condition_group": sig0.condition_group,
                "noisy_or_inputs": len(drug_tuples),
                "computed_at": datetime.utcnow().isoformat(),
            }

            # ── UPSERT patient_conditions ────────────────────────────────────
            existing = (
                db.query(PatientCondition)
                .filter(
                    PatientCondition.patient_id == patient_id,
                    PatientCondition.condition_code == condition_code,
                    PatientCondition.status.in_(["probable", "confirmed"]),
                )
                .first()
            )

            if existing:
                if existing.status == "confirmed":
                    # Never downgrade a clinician-confirmed condition
                    logger.debug(
                        "Skipping update for confirmed condition %s (patient=%s)",
                        condition_code, patient_id,
                    )
                    continue

                old_confidence = existing.confidence
                existing.confidence = confidence
                existing.inference_basis = inference_basis
                existing.last_updated_at = func.now()
                db.flush()

                EventStore.emit(
                    db,
                    CONDITION_UPDATED,
                    patient_id=patient_id,
                    condition_id=existing.id,
                    payload={
                        "condition_code": condition_code,
                        "condition_id": str(existing.id),
                        "old_confidence": old_confidence,
                        "new_confidence": confidence,
                    },
                )
                upserted.append(existing)
                logger.debug(
                    "Condition updated: %s patient=%s confidence %.3f→%.3f",
                    condition_code, patient_id, old_confidence, confidence,
                )

            else:
                cond = PatientCondition(
                    patient_id=patient_id,
                    condition_code=condition_code,
                    condition_name=sig0.condition_name,
                    condition_group=sig0.condition_group,
                    confidence=confidence,
                    inference_engine_version="phg_mvp_v1",
                    inference_basis=inference_basis,
                    status="probable",
                )
                db.add(cond)
                db.flush()

                EventStore.emit(
                    db,
                    CONDITION_INFERRED,
                    patient_id=patient_id,
                    condition_id=cond.id,
                    payload={
                        "condition_code": condition_code,
                        "condition_name": sig0.condition_name,
                        "condition_id": str(cond.id),
                        "confidence": confidence,
                        "inference_engine_version": "phg_mvp_v1",
                        "supporting_drugs": [s.inn for s in sigs],
                    },
                )
                upserted.append(cond)
                logger.info(
                    "Condition inferred: %s (%s) patient=%s confidence=%.3f",
                    condition_code, sig0.condition_name, patient_id, confidence,
                )

        return upserted
