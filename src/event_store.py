"""
event_store.py — Append-only PHG event log writer.

Design rules:
  - EventStore.emit() is the ONLY function that writes to phg_events.
  - Never call directly from API layer — only from service layer.
  - db.flush() (not commit) — caller owns the transaction.
  - payload_schema_version=1 for PHG MVP baseline.
"""

from __future__ import annotations

import logging
import uuid as uuid_lib
from typing import Any, Optional

from sqlalchemy.orm import Session

from src.database import PhgEvent

logger = logging.getLogger(__name__)


class EventStore:
    """Append-only writer for phg_events. All writes are flush-only (caller commits)."""

    @staticmethod
    def emit(
        db: Session,
        event_type: str,
        *,
        patient_id: Any,
        payload: dict,
        prescription_id: Optional[Any] = None,
        episode_id: Optional[Any] = None,
        condition_id: Optional[Any] = None,
        doctor_id: Optional[Any] = None,
        source_clinician_id: Optional[Any] = None,
        schema_version: int = 1,
    ) -> PhgEvent:
        """
        Write a single event to phg_events.
        Flushes to the session (does NOT commit).
        The caller is responsible for committing or rolling back.

        Args:
            db:                 Active SQLAlchemy session.
            event_type:         One of the constants in event_schemas.py.
            patient_id:         UUID of the patient this event belongs to.
            payload:            JSON-serialisable dict (event data).
            prescription_id:    Optional — prescription that triggered this event.
            episode_id:         Optional — medication episode affected.
            condition_id:       Optional — patient condition affected.
            doctor_id:          Optional — doctor involved.
            source_clinician_id: Optional — clinician who triggered this event.
            schema_version:     Payload schema version (default 1 = PHG MVP).

        Returns:
            The flushed PhgEvent ORM instance.
        """
        event = PhgEvent(
            event_type=event_type,
            payload_schema_version=schema_version,
            patient_id=patient_id,
            prescription_id=prescription_id,
            episode_id=episode_id,
            condition_id=condition_id,
            doctor_id=doctor_id,
            source_clinician_id=source_clinician_id,
            payload=payload,
        )
        db.add(event)
        db.flush()
        logger.debug(
            "PHG event emitted: type=%s patient=%s episode=%s condition=%s",
            event_type, patient_id, episode_id, condition_id,
        )
        return event
