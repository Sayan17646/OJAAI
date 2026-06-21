"""
episode_manager.py — PHG MVP medication episode lifecycle management.

Core logic:
  EpisodeManager.process_drug() is the single entry point.
  It runs the four-branch decision tree to determine episode state,
  writes dosage history, and emits the correct PHG event.

All writes are flush-only — the caller (pipeline.py) owns the transaction.

Four-branch decision tree:
  Branch 1: Active episode exists, within gap window → CONTINUE (or DOSE_CHANGED)
  Branch 2: Active episode exists, gap exceeded      → CLOSE old, START new
  Branch 3: No active episode                        → START new
  PRN branch: Drug is PRN (is_prn=True)              → prn_snapshot (never 'active')
"""

from __future__ import annotations

import hashlib
import logging
import uuid as uuid_lib
from datetime import date, timedelta
from typing import Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.database import (
    Doctor,
    DrugConditionSignal,
    MedicationDosageHistory,
    MedicationEpisode,
)
from src.models import NormalizedDrug
from src.event_schemas import (
    MEDICATION_COMPLETED,
    MEDICATION_CONTINUED,
    MEDICATION_DISCONTINUED,
    MEDICATION_DOSE_CHANGED,
    MEDICATION_STARTED,
)
from src.event_store import EventStore

logger = logging.getLogger(__name__)


def _make_idempotency_key(prescription_id: str, inn: str) -> str:
    """
    sha256(prescription_id + '::' + inn) — prevents double-write on upload retry.
    Stored in medication_dosage_history.idempotency_key.
    """
    raw = f"{prescription_id}::{inn}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _get_signal_defaults(
    db: Session, inn: str
) -> Tuple[int, str, bool]:
    """
    Fetch gap_tolerance_days, medication_class, is_prn from drug_condition_signals.
    Returns defaults (45, 'chronic_oral', False) if no signal found.
    Only needs the first matching signal row — class attributes are per-drug, not per-condition.
    """
    signal = (
        db.query(DrugConditionSignal)
        .filter(DrugConditionSignal.inn == inn)
        .first()
    )
    if signal:
        return signal.episode_gap_tolerance, signal.medication_class, signal.is_prn
    return 45, "chronic_oral", False


class EpisodeManager:
    """
    Stateless episode lifecycle manager.
    All methods are @staticmethod — no instance needed.
    """

    @staticmethod
    def process_drug(
        db: Session,
        *,
        patient_id: object,
        prescription_id: str,
        doctor: Optional[Doctor],
        drug: NormalizedDrug,
        rx_date: date,
    ) -> Optional[MedicationEpisode]:
        """
        Main entry point: one call per drug per prescription.

        Resolves or creates a MedicationEpisode, writes one MedicationDosageHistory row,
        and emits the appropriate PHG event.

        Returns the affected MedicationEpisode, or None if inn is missing.
        """
        inn = (drug.inn or "").strip().lower()
        if not inn:
            logger.debug("Skipping episode for drug with no INN: %r", drug.raw_drug_name)
            return None

        gap_tolerance_days, medication_class, is_prn = _get_signal_defaults(db, inn)

        # ── PRN branch ─────────────────────────────────────────────────────
        if is_prn:
            return EpisodeManager._handle_prn(
                db, patient_id=patient_id, prescription_id=prescription_id,
                doctor=doctor, drug=drug, inn=inn, rx_date=rx_date,
                gap_tolerance_days=gap_tolerance_days, medication_class=medication_class,
            )

        # ── Resolve episode ────────────────────────────────────────────────
        episode, event_type = EpisodeManager._resolve_episode(
            db, patient_id=patient_id, inn=inn, rx_date=rx_date, drug=drug,
            gap_tolerance_days=gap_tolerance_days, medication_class=medication_class,
        )

        # ── Write dosage history ───────────────────────────────────────────
        EpisodeManager._write_dosage_history(
            db, episode=episode, prescription_id=prescription_id,
            doctor=doctor, drug=drug, rx_date=rx_date,
        )

        # ── Emit event ─────────────────────────────────────────────────────
        EventStore.emit(
            db,
            event_type,
            patient_id=patient_id,
            prescription_id=prescription_id,
            episode_id=episode.id,
            doctor_id=doctor.id if doctor else None,
            payload={
                "inn": inn,
                "dosage": f"{drug.dosage_value or ''} {drug.dosage_unit or ''}".strip(),
                "frequency": drug.frequency,
                "duration_days": drug.duration_days,
                "episode_status": episode.status,
                "prescription_count": episode.prescription_count,
            },
        )
        return episode

    # ── Four-branch decision tree ──────────────────────────────────────────

    @staticmethod
    def _resolve_episode(
        db: Session,
        *,
        patient_id: object,
        inn: str,
        rx_date: date,
        drug: NormalizedDrug,
        gap_tolerance_days: int,
        medication_class: str,
    ) -> Tuple[MedicationEpisode, str]:
        """
        Returns (episode, event_type). All writes are flush-only.

        Note: SELECT FOR UPDATE is deferred to 10k patients.
        At MVP scale (< 10k patients), concurrent uploads for the same
        patient+drug are rare enough that plain SELECT is acceptable.
        """
        # Compute new estimated end date from this prescription
        new_end: Optional[date] = None
        if drug.duration_days:
            new_end = rx_date + timedelta(days=drug.duration_days)

        # Query active episode
        active = (
            db.query(MedicationEpisode)
            .filter(
                MedicationEpisode.patient_id == patient_id,
                MedicationEpisode.inn == inn,
                MedicationEpisode.status == "active",
            )
            .first()
        )

        if active:
            gap = timedelta(days=active.gap_tolerance_days)
            within_gap = (
                active.estimated_end_date is None
                or rx_date <= active.estimated_end_date + gap
            )

            if within_gap:
                # Branch 1: continue existing episode
                dosage_changed = (
                    active.latest_dosage_value != drug.dosage_value
                    or active.latest_dosage_unit != drug.dosage_unit
                )
                event_type = (
                    MEDICATION_DOSE_CHANGED if dosage_changed else MEDICATION_CONTINUED
                )

                if dosage_changed:
                    logger.info(
                        "Dosage change detected for patient=%s inn=%s: %s→%s %s",
                        patient_id, inn,
                        active.latest_dosage_value, drug.dosage_value, drug.dosage_unit,
                    )

                # Update snapshot on episode
                active.prescription_count += 1
                active.version += 1
                active.latest_dosage_value = drug.dosage_value
                active.latest_dosage_unit = drug.dosage_unit
                active.latest_frequency = drug.frequency
                active.updated_at = func.now()
                if new_end and (
                    active.estimated_end_date is None
                    or new_end > active.estimated_end_date
                ):
                    active.estimated_end_date = new_end
                db.flush()
                return active, event_type

            else:
                # Branch 2: gap exceeded — close old, open new
                logger.info(
                    "Episode gap exceeded for patient=%s inn=%s "
                    "(estimated_end=%s rx_date=%s gap=%d days) — closing.",
                    patient_id, inn,
                    active.estimated_end_date, rx_date, active.gap_tolerance_days,
                )
                active.status = "completed"
                active.actual_end_date = active.estimated_end_date
                active.version += 1
                db.flush()

                # Emit closure event before creating new episode
                EventStore.emit(
                    db,
                    MEDICATION_COMPLETED,
                    patient_id=patient_id,
                    episode_id=active.id,
                    payload={
                        "inn": inn,
                        "actual_end_date": str(active.actual_end_date or ""),
                        "prescription_count": active.prescription_count,
                    },
                )

                # Fall through to create new episode
                new_episode = EpisodeManager._create_episode(
                    db,
                    patient_id=patient_id,
                    inn=inn,
                    start_date=rx_date,
                    estimated_end_date=new_end,
                    drug=drug,
                    gap_tolerance_days=gap_tolerance_days,
                    medication_class=medication_class,
                    dispensing_type="scheduled",
                )
                return new_episode, MEDICATION_STARTED

        else:
            # Branch 3: no active episode — start new (first or restart)
            new_episode = EpisodeManager._create_episode(
                db,
                patient_id=patient_id,
                inn=inn,
                start_date=rx_date,
                estimated_end_date=new_end,
                drug=drug,
                gap_tolerance_days=gap_tolerance_days,
                medication_class=medication_class,
                dispensing_type="scheduled",
            )
            return new_episode, MEDICATION_STARTED

    @staticmethod
    def _handle_prn(
        db: Session,
        *,
        patient_id: object,
        prescription_id: str,
        doctor: Optional[Doctor],
        drug: NormalizedDrug,
        inn: str,
        rx_date: date,
        gap_tolerance_days: int,
        medication_class: str,
    ) -> MedicationEpisode:
        """
        PRN drugs never become 'active'.
        Always create a prn_snapshot episode for historical record.
        Multiple prn_snapshot rows for same patient+drug are allowed.
        """
        episode = EpisodeManager._create_episode(
            db,
            patient_id=patient_id,
            inn=inn,
            start_date=rx_date,
            estimated_end_date=rx_date + timedelta(days=drug.duration_days)
                               if drug.duration_days else None,
            drug=drug,
            gap_tolerance_days=gap_tolerance_days,
            medication_class=medication_class,
            dispensing_type="prn",
        )

        EpisodeManager._write_dosage_history(
            db, episode=episode, prescription_id=prescription_id,
            doctor=doctor, drug=drug, rx_date=rx_date,
        )
        EventStore.emit(
            db,
            MEDICATION_STARTED,
            patient_id=patient_id,
            prescription_id=prescription_id,
            episode_id=episode.id,
            doctor_id=doctor.id if doctor else None,
            payload={
                "inn": inn,
                "dispensing_type": "prn",
                "dosage": f"{drug.dosage_value or ''} {drug.dosage_unit or ''}".strip(),
            },
        )
        return episode

    @staticmethod
    def _create_episode(
        db: Session,
        *,
        patient_id: object,
        inn: str,
        start_date: date,
        estimated_end_date: Optional[date],
        drug: NormalizedDrug,
        gap_tolerance_days: int,
        medication_class: str,
        dispensing_type: str,
    ) -> MedicationEpisode:
        """Create and flush a new MedicationEpisode row."""
        # Detect FDC: INN contains '+' (e.g., 'metformin + glibenclamide')
        is_fdc = "+" in inn
        fdc_components: Optional[list] = None
        if is_fdc:
            fdc_components = [c.strip() for c in inn.split("+")]

        status = "prn_snapshot" if dispensing_type == "prn" else "active"

        episode = MedicationEpisode(
            patient_id=patient_id,
            inn=inn,
            is_fdc=is_fdc,
            fdc_components=fdc_components,
            dispensing_type=dispensing_type,
            status=status,
            start_date=start_date,
            estimated_end_date=estimated_end_date,
            gap_tolerance_days=gap_tolerance_days,
            latest_dosage_value=drug.dosage_value,
            latest_dosage_unit=drug.dosage_unit,
            latest_frequency=drug.frequency,
            version=1,
        )
        db.add(episode)
        db.flush()
        logger.info(
            "Episode created: id=%s patient=%s inn=%s status=%s dispensing=%s",
            episode.id, patient_id, inn, status, dispensing_type,
        )
        return episode

    @staticmethod
    def _write_dosage_history(
        db: Session,
        *,
        episode: MedicationEpisode,
        prescription_id: str,
        doctor: Optional[Doctor],
        drug: NormalizedDrug,
        rx_date: date,
    ) -> Optional[MedicationDosageHistory]:
        """
        Write one MedicationDosageHistory row for this prescription × drug.
        Idempotent via idempotency_key unique constraint — silently skips duplicates.
        """
        idempotency_key = _make_idempotency_key(prescription_id, drug.inn or "")

        # Check idempotency — prevents double-write on upload retry
        existing = (
            db.query(MedicationDosageHistory)
            .filter(MedicationDosageHistory.idempotency_key == idempotency_key)
            .first()
        )
        if existing:
            logger.debug(
                "Skipping duplicate dosage history write (idempotency_key=%s)", idempotency_key
            )
            return existing

        # Count existing rows for this episode to compute refill_number
        refill_number = (
            db.query(MedicationDosageHistory)
            .filter(MedicationDosageHistory.episode_id == episode.id)
            .count()
        ) + 1

        row = MedicationDosageHistory(
            episode_id=episode.id,
            prescription_id=prescription_id,
            doctor_id=doctor.id if doctor else None,
            raw_drug_name=drug.raw_drug_name,
            dosage_value=drug.dosage_value,
            dosage_unit=drug.dosage_unit,
            frequency=drug.frequency,
            freq_per_day=drug.freq_per_day,
            duration_days=drug.duration_days,
            route=drug.route,
            refill_number=refill_number,
            idempotency_key=idempotency_key,
            recorded_date=rx_date,
        )
        db.add(row)
        db.flush()
        return row

    @staticmethod
    def discontinue_episode(
        db: Session,
        *,
        episode_id: object,
        patient_id: object,
        stop_reason: str,
        switched_to_inn: Optional[str] = None,
        source_clinician_id: Optional[object] = None,
    ) -> Optional[MedicationEpisode]:
        """
        Mark an episode as discontinued. Called from API when clinician records stop reason.
        Emits MEDICATION_DISCONTINUED event.
        """
        episode = (
            db.query(MedicationEpisode)
            .filter(
                MedicationEpisode.id == episode_id,
                MedicationEpisode.patient_id == patient_id,
            )
            .first()
        )
        if not episode:
            return None

        from datetime import date as _date
        episode.status = "discontinued"
        episode.actual_end_date = _date.today()
        episode.version += 1
        db.flush()

        EventStore.emit(
            db,
            MEDICATION_DISCONTINUED,
            patient_id=patient_id,
            episode_id=episode_id,
            source_clinician_id=source_clinician_id,
            payload={
                "inn": episode.inn,
                "stop_reason": stop_reason,
                "switched_to_inn": switched_to_inn,
            },
        )
        return episode
