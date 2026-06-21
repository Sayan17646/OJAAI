"""
tests/test_doctor_registry.py — Unit tests for DoctorRegistry.

Run: pytest tests/test_doctor_registry.py -v
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.doctor_registry import (
    DoctorRegistry,
    _normalize_registration_number,
    _map_speciality_to_group,
)


# ---------------------------------------------------------------------------
# Pure function tests — no DB needed
# ---------------------------------------------------------------------------

class TestNormalizeRegistrationNumber:
    def test_strips_spaces_and_dashes(self):
        assert _normalize_registration_number("MH - 12345") == "MH12345"

    def test_uppercase(self):
        assert _normalize_registration_number("mh12345") == "MH12345"

    def test_strips_punctuation(self):
        assert _normalize_registration_number("Reg. No. 12345") == "REGNO12345"

    def test_none_input(self):
        assert _normalize_registration_number(None) is None

    def test_empty_string(self):
        assert _normalize_registration_number("") is None

    def test_whitespace_only(self):
        assert _normalize_registration_number("   ") is None

    def test_numbers_only(self):
        assert _normalize_registration_number("12345") == "12345"

    def test_complex_format(self):
        assert _normalize_registration_number("MAHARASHTRA / 12345 - A") == "MAHARASHTRA12345A"


class TestMapSpecialityToGroup:
    def test_endocrinologist(self):
        assert _map_speciality_to_group("Endocrinologist") == "metabolic"

    def test_cardiologist(self):
        assert _map_speciality_to_group("Cardiologist") == "cardiovascular"

    def test_general_physician(self):
        assert _map_speciality_to_group("General Physician") == "general"

    def test_none_input(self):
        assert _map_speciality_to_group(None) is None

    def test_unknown_speciality(self):
        assert _map_speciality_to_group("Dermatologist") == "general"

    def test_case_insensitive(self):
        assert _map_speciality_to_group("NEUROLOGIST") == "neurological"

    def test_partial_match(self):
        assert _map_speciality_to_group("Pulmonology Expert") == "respiratory"


# ---------------------------------------------------------------------------
# DoctorRegistry.get_or_create() — with mocked DB
# ---------------------------------------------------------------------------

class TestDoctorRegistryGetOrCreate:
    def _make_db(self):
        """Return a MagicMock that behaves like a SQLAlchemy session."""
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        return db

    def test_returns_none_when_no_identity(self):
        db = self._make_db()
        result = DoctorRegistry.get_or_create(
            db, reg_number=None, name=None, speciality=None,
        )
        assert result is None
        db.add.assert_not_called()

    def test_creates_new_doctor_when_no_match(self):
        db = self._make_db()
        with patch("src.doctor_registry.EventStore.emit"):
            DoctorRegistry.get_or_create(
                db,
                reg_number="MH12345",
                name="Dr. Sharma",
                speciality="Endocrinologist",
                emit_event=False,
            )
        db.add.assert_called_once()
        db.flush.assert_called()

    def test_returns_existing_doctor_on_reg_match(self):
        db = self._make_db()
        existing = MagicMock()
        existing.speciality_group = "metabolic"
        existing.raw_name_variants = []
        db.query.return_value.filter.return_value.first.return_value = existing

        result = DoctorRegistry.get_or_create(
            db,
            reg_number="MH12345",
            name="Dr. Sharma",
            speciality="Endocrinologist",
            emit_event=False,
        )
        assert result is existing
        db.add.assert_not_called()

    def test_appends_name_variant_on_match(self):
        db = self._make_db()
        existing = MagicMock()
        existing.speciality_group = "metabolic"
        existing.raw_name_variants = ["Dr. S. Sharma"]
        db.query.return_value.filter.return_value.first.return_value = existing

        DoctorRegistry.get_or_create(
            db,
            reg_number="MH12345",
            name="Dr. Suresh Sharma",
            speciality=None,
            emit_event=False,
        )
        assert "Dr. Suresh Sharma" in existing.raw_name_variants

    def test_no_duplicate_name_variant(self):
        db = self._make_db()
        existing = MagicMock()
        existing.speciality_group = None
        existing.raw_name_variants = ["Dr. Sharma"]
        db.query.return_value.filter.return_value.first.return_value = existing

        DoctorRegistry.get_or_create(
            db,
            reg_number="MH12345",
            name="Dr. Sharma",
            speciality=None,
            emit_event=False,
        )
        # Should not duplicate "Dr. Sharma"
        assert existing.raw_name_variants.count("Dr. Sharma") == 1

    def test_creates_doctor_from_name_only(self):
        db = self._make_db()
        with patch("src.doctor_registry.EventStore.emit"):
            DoctorRegistry.get_or_create(
                db,
                reg_number=None,
                name="Dr. Patel",
                speciality="Cardiologist",
                emit_event=False,
            )
        db.add.assert_called_once()

    def test_emits_event_when_patient_id_provided(self):
        db = self._make_db()
        with patch("src.doctor_registry.EventStore.emit") as mock_emit:
            DoctorRegistry.get_or_create(
                db,
                reg_number="MH99999",
                name="Dr. Test",
                speciality=None,
                patient_id="some-uuid",
                prescription_id="rx-uuid",
                emit_event=True,
            )
            mock_emit.assert_called_once()
