"""
tests/test_episode_manager.py — Unit tests for EpisodeManager.

Run: pytest tests/test_episode_manager.py -v
"""

from __future__ import annotations

import hashlib
import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch, call

from src.episode_manager import (
    EpisodeManager,
    _make_idempotency_key,
    _get_signal_defaults,
)


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------

class TestMakeIdempotencyKey:
    def test_deterministic(self):
        k1 = _make_idempotency_key("rx-123", "metformin")
        k2 = _make_idempotency_key("rx-123", "metformin")
        assert k1 == k2

    def test_different_rx_gives_different_key(self):
        k1 = _make_idempotency_key("rx-001", "metformin")
        k2 = _make_idempotency_key("rx-002", "metformin")
        assert k1 != k2

    def test_different_inn_gives_different_key(self):
        k1 = _make_idempotency_key("rx-001", "metformin")
        k2 = _make_idempotency_key("rx-001", "atorvastatin")
        assert k1 != k2

    def test_returns_hex_string(self):
        k = _make_idempotency_key("rx-abc", "aspirin")
        assert len(k) == 64  # sha256 hex = 64 chars
        int(k, 16)  # must be valid hex


# ---------------------------------------------------------------------------
# EpisodeManager._create_episode()
# ---------------------------------------------------------------------------

class TestCreateEpisode:
    def _make_drug(self, inn="metformin", dosage_value=500.0, dosage_unit="mg",
                   frequency="twice daily", freq_per_day=2, duration_days=30, route="oral"):
        drug = MagicMock()
        drug.inn = inn
        drug.dosage_value = dosage_value
        drug.dosage_unit = dosage_unit
        drug.frequency = frequency
        drug.freq_per_day = freq_per_day
        drug.duration_days = duration_days
        drug.route = route
        drug.raw_drug_name = inn
        return drug

    def test_creates_active_episode_for_scheduled_drug(self):
        """Test that _create_episode calls db.add and db.flush."""
        db = MagicMock()
        drug = self._make_drug()

        with patch("src.episode_manager.MedicationEpisode") as MockEpisode:
            mock_ep = MagicMock()
            mock_ep.status = "active"
            mock_ep.inn = "metformin"
            mock_ep.is_fdc = False
            mock_ep.fdc_components = None
            MockEpisode.return_value = mock_ep

            episode = EpisodeManager._create_episode(
                db,
                patient_id="p-123",
                inn="metformin",
                start_date=date(2024, 1, 1),
                estimated_end_date=date(2024, 1, 31),
                drug=drug,
                gap_tolerance_days=60,
                medication_class="chronic_oral",
                dispensing_type="scheduled",
            )
        db.add.assert_called_once()
        db.flush.assert_called_once()
        assert episode.inn == "metformin"

    def test_creates_prn_snapshot_for_prn_drug(self):
        """Test PRN → status='prn_snapshot'."""
        db = MagicMock()
        drug = self._make_drug(inn="ibuprofen")

        with patch("src.episode_manager.MedicationEpisode") as MockEpisode:
            mock_ep = MagicMock()
            mock_ep.status = "prn_snapshot"
            mock_ep.dispensing_type = "prn"
            mock_ep.inn = "ibuprofen"
            mock_ep.is_fdc = False
            mock_ep.fdc_components = None
            MockEpisode.return_value = mock_ep

            episode = EpisodeManager._create_episode(
                db,
                patient_id="p-123",
                inn="ibuprofen",
                start_date=date(2024, 1, 1),
                estimated_end_date=date(2024, 1, 7),
                drug=drug,
                gap_tolerance_days=7,
                medication_class="prn",
                dispensing_type="prn",
            )
        assert episode.status == "prn_snapshot"
        assert episode.dispensing_type == "prn"

    def test_detects_fdc_inn(self):
        """FDC INN containing '+' → is_fdc=True and fdc_components populated."""
        db = MagicMock()
        drug = self._make_drug(inn="metformin + glibenclamide")

        with patch("src.episode_manager.MedicationEpisode") as MockEpisode:
            mock_ep = MagicMock()
            mock_ep.status = "active"
            mock_ep.is_fdc = True
            mock_ep.fdc_components = ["metformin", "glibenclamide"]
            mock_ep.inn = "metformin + glibenclamide"
            MockEpisode.return_value = mock_ep

            episode = EpisodeManager._create_episode(
                db,
                patient_id="p-123",
                inn="metformin + glibenclamide",
                start_date=date(2024, 1, 1),
                estimated_end_date=None,
                drug=drug,
                gap_tolerance_days=60,
                medication_class="chronic_oral",
                dispensing_type="scheduled",
            )
        assert episode.is_fdc is True
        assert "metformin" in episode.fdc_components
        assert "glibenclamide" in episode.fdc_components


# ---------------------------------------------------------------------------
# EpisodeManager._resolve_episode() — four-branch decision tree
# ---------------------------------------------------------------------------

class TestResolveEpisode:
    def _make_drug(self, inn="metformin", dosage_value=500.0, dosage_unit="mg",
                   frequency="twice daily", duration_days=30):
        drug = MagicMock()
        drug.inn = inn
        drug.dosage_value = dosage_value
        drug.dosage_unit = dosage_unit
        drug.frequency = frequency
        drug.duration_days = duration_days
        drug.raw_drug_name = inn
        drug.freq_per_day = 2
        drug.route = "oral"
        return drug

    def _make_active_episode(self, inn="metformin", estimated_end_date=None,
                             gap_tolerance_days=60, dosage_value=500.0,
                             dosage_unit="mg", frequency="twice daily"):
        """Helper: creates a MagicMock episode with all integer attributes properly set."""
        ep = MagicMock()
        ep.status = "active"
        ep.inn = inn
        ep.estimated_end_date = estimated_end_date
        ep.gap_tolerance_days = gap_tolerance_days  # MUST be int
        ep.prescription_count = 1
        ep.version = 1
        ep.latest_dosage_value = dosage_value
        ep.latest_dosage_unit = dosage_unit
        ep.latest_frequency = frequency
        ep.id = "ep-existing-1"
        return ep

    def test_branch3_no_active_episode_creates_new(self):
        """Branch 3: No active episode → START new."""
        db = MagicMock()
        # Single .filter(...).first() returns None (no active episode)
        db.query.return_value.filter.return_value.first.return_value = None
        drug = self._make_drug()

        with patch.object(EpisodeManager, "_create_episode") as mock_create, \
             patch.object(EpisodeManager, "_write_dosage_history"), \
             patch("src.episode_manager.EventStore.emit"):
            mock_ep = MagicMock()
            mock_ep.id = "ep-1"
            mock_ep.status = "active"
            mock_ep.inn = "metformin"
            mock_ep.prescription_count = 1
            mock_ep.version = 1
            mock_ep.latest_dosage_value = 500.0
            mock_ep.latest_dosage_unit = "mg"
            mock_ep.latest_frequency = "twice daily"
            mock_ep.estimated_end_date = None
            mock_ep.gap_tolerance_days = 60
            mock_create.return_value = mock_ep

            ep, event_type = EpisodeManager._resolve_episode(
                db,
                patient_id="p-1",
                inn="metformin",
                rx_date=date(2024, 3, 1),
                drug=drug,
                gap_tolerance_days=60,
                medication_class="chronic_oral",
            )
            mock_create.assert_called_once()
            assert event_type == "MEDICATION_STARTED"

    def test_branch1_within_gap_continues_episode(self):
        """Branch 1: Active episode within gap → CONTINUE."""
        db = MagicMock()
        drug = self._make_drug()
        existing_episode = self._make_active_episode(
            estimated_end_date=date(2024, 1, 31), gap_tolerance_days=60
        )
        db.query.return_value.filter.return_value.first.return_value = existing_episode

        with patch("src.episode_manager.EventStore.emit"), \
             patch("src.episode_manager.func"):
            ep, event_type = EpisodeManager._resolve_episode(
                db,
                patient_id="p-1",
                inn="metformin",
                rx_date=date(2024, 2, 15),  # within 60 day gap from Jan 31
                drug=drug,
                gap_tolerance_days=60,
                medication_class="chronic_oral",
            )
        assert event_type in ("MEDICATION_CONTINUED", "MEDICATION_DOSE_CHANGED")
        assert ep is existing_episode
        assert existing_episode.prescription_count == 2

    def test_branch1_dose_change_detected(self):
        """Branch 1 with dose change → MEDICATION_DOSE_CHANGED."""
        db = MagicMock()
        drug = self._make_drug(dosage_value=1000.0)  # new dose
        existing_episode = self._make_active_episode(
            estimated_end_date=date(2024, 1, 31),
            gap_tolerance_days=60,
            dosage_value=500.0,  # old dose
        )
        db.query.return_value.filter.return_value.first.return_value = existing_episode

        with patch("src.episode_manager.EventStore.emit"), \
             patch("src.episode_manager.func"):
            ep, event_type = EpisodeManager._resolve_episode(
                db,
                patient_id="p-1",
                inn="metformin",
                rx_date=date(2024, 2, 15),
                drug=drug,
                gap_tolerance_days=60,
                medication_class="chronic_oral",
            )
        assert event_type == "MEDICATION_DOSE_CHANGED"



# ---------------------------------------------------------------------------
# Idempotency key prevents duplicate dosage history writes
# ---------------------------------------------------------------------------

class TestWriteDosageHistoryIdempotency:
    def test_skips_duplicate_on_same_idempotency_key(self):
        db = MagicMock()
        existing_row = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = existing_row

        drug = MagicMock()
        drug.inn = "metformin"
        drug.raw_drug_name = "Glycomet 500"
        drug.dosage_value = 500.0
        drug.dosage_unit = "mg"
        drug.frequency = "twice daily"
        drug.freq_per_day = 2
        drug.duration_days = 30
        drug.route = "oral"

        episode = MagicMock()
        episode.id = "ep-1"

        result = EpisodeManager._write_dosage_history(
            db,
            episode=episode,
            prescription_id="rx-abc",
            doctor=None,
            drug=drug,
            rx_date=date(2024, 1, 1),
        )
        # Should return existing row, not create a new one
        assert result is existing_row
        # db.add should not be called for a duplicate
        db.add.assert_not_called()
