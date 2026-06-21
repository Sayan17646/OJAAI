"""
tests/test_condition_inference.py — Unit tests for ConditionInferenceEngine.

Tests the Noisy-OR algorithm, temporal decay, speciality LR, and UPSERT logic.

Run: pytest tests/test_condition_inference.py -v
"""

from __future__ import annotations

import math
import pytest
from datetime import date, datetime
from unittest.mock import MagicMock, patch

from src.condition_inference import (
    ConditionInferenceEngine,
    _noisy_or_confidence,
    _MIN_CONFIDENCE,
)


# ---------------------------------------------------------------------------
# _noisy_or_confidence() — pure function tests
# ---------------------------------------------------------------------------

class TestNoisyOrConfidence:

    def test_single_strong_signal(self):
        """Single strong signal (metformin → T2DM) should give high confidence."""
        conf = _noisy_or_confidence(
            drug_signals=[(0.93, 0.90, True)],
            days_since_last_rx=10,
            speciality_group=None,
            condition_group="metabolic",
            prescription_count=5,
            condition_prevalence=0.11,
        )
        assert conf > 0.6

    def test_two_independent_signals_gives_higher_confidence(self):
        """Noisy-OR: two independent signals > either alone."""
        single = _noisy_or_confidence(
            drug_signals=[(0.85, 0.80, True)],
            days_since_last_rx=5,
            speciality_group=None,
            condition_group="cardiovascular",
            prescription_count=3,
            condition_prevalence=0.25,
        )
        double = _noisy_or_confidence(
            drug_signals=[(0.85, 0.80, True), (0.82, 0.75, True)],
            days_since_last_rx=5,
            speciality_group=None,
            condition_group="cardiovascular",
            prescription_count=3,
            condition_prevalence=0.25,
        )
        assert double > single

    def test_inactive_signal_weighted_lower(self):
        """Inactive episode should contribute only 0.4x the signal strength."""
        active = _noisy_or_confidence(
            drug_signals=[(0.90, 0.85, True)],
            days_since_last_rx=10,
            speciality_group=None,
            condition_group="metabolic",
            prescription_count=5,
            condition_prevalence=0.10,
        )
        inactive = _noisy_or_confidence(
            drug_signals=[(0.90, 0.85, False)],
            days_since_last_rx=10,
            speciality_group=None,
            condition_group="metabolic",
            prescription_count=5,
            condition_prevalence=0.10,
        )
        assert active > inactive

    def test_temporal_decay_reduces_confidence_after_180_days(self):
        """Confidence should decay toward prevalence after > 180 days without prescription."""
        recent = _noisy_or_confidence(
            drug_signals=[(0.93, 0.90, False)],
            days_since_last_rx=10,
            speciality_group=None,
            condition_group="metabolic",
            prescription_count=10,
            condition_prevalence=0.11,
        )
        old = _noisy_or_confidence(
            drug_signals=[(0.93, 0.90, False)],
            days_since_last_rx=360,
            speciality_group=None,
            condition_group="metabolic",
            prescription_count=10,
            condition_prevalence=0.11,
        )
        assert old < recent

    def test_speciality_match_boosts_confidence(self):
        """Speciality match should boost confidence via LR."""
        no_speciality = _noisy_or_confidence(
            drug_signals=[(0.85, 0.80, True)],
            days_since_last_rx=5,
            speciality_group=None,
            condition_group="cardiovascular",
            prescription_count=3,
            condition_prevalence=0.25,
        )
        with_speciality = _noisy_or_confidence(
            drug_signals=[(0.85, 0.80, True)],
            days_since_last_rx=5,
            speciality_group="cardiovascular",
            condition_group="cardiovascular",
            prescription_count=3,
            condition_prevalence=0.25,
        )
        assert with_speciality > no_speciality

    def test_output_always_in_valid_range(self):
        """Result must always be in [0.0, 1.0]."""
        for signals in [
            [],
            [(1.0, 1.0, True)],
            [(0.0, 0.0, True)],
            [(0.99, 0.99, True), (0.99, 0.99, True), (0.99, 0.99, True)],
        ]:
            conf = _noisy_or_confidence(
                drug_signals=signals,
                days_since_last_rx=0,
                speciality_group="metabolic",
                condition_group="metabolic",
                prescription_count=10,
                condition_prevalence=0.10,
            )
            assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf} for signals {signals}"

    def test_single_prescription_has_lower_credibility(self):
        """First prescription → lower credibility → confidence closer to prevalence."""
        one_rx = _noisy_or_confidence(
            drug_signals=[(0.93, 0.90, True)],
            days_since_last_rx=5,
            speciality_group=None,
            condition_group="metabolic",
            prescription_count=1,
            condition_prevalence=0.11,
        )
        ten_rx = _noisy_or_confidence(
            drug_signals=[(0.93, 0.90, True)],
            days_since_last_rx=5,
            speciality_group=None,
            condition_group="metabolic",
            prescription_count=10,
            condition_prevalence=0.11,
        )
        # More prescriptions → higher confidence (farther from prevalence baseline)
        assert ten_rx > one_rx

    def test_no_signals_returns_prevalence(self):
        """With no evidence, result should equal the prior (prevalence)."""
        conf = _noisy_or_confidence(
            drug_signals=[],
            days_since_last_rx=0,
            speciality_group=None,
            condition_group="metabolic",
            prescription_count=0,
            condition_prevalence=0.11,
        )
        assert conf == 0.11


# ---------------------------------------------------------------------------
# ConditionInferenceEngine.infer_for_patient() — integration-style tests
# with mocked DB
# ---------------------------------------------------------------------------

class TestInferForPatient:

    def _mock_episode(self, inn, status="active", start_date=None,
                      estimated_end_date=None, prescription_count=3,
                      fdc_components=None):
        ep = MagicMock()
        ep.inn = inn
        ep.status = status
        # Use today as start_date so temporal decay does NOT apply (days_since < 180)
        ep.start_date = start_date or date.today()
        ep.estimated_end_date = estimated_end_date
        ep.actual_end_date = None
        ep.prescription_count = prescription_count
        ep.latest_doctor_id = None
        ep.fdc_components = fdc_components
        return ep

    def _mock_signal(self, inn, condition_code="E11",
                     condition_name="Type 2 Diabetes", condition_group="metabolic",
                     signal_strength=0.93, sensitivity=0.85, specificity=0.90,
                     condition_prevalence=0.11, is_prn=False, min_prescriptions=1,
                     requires_speciality=None):
        sig = MagicMock()
        sig.inn = inn
        sig.condition_code = condition_code
        sig.condition_name = condition_name
        sig.condition_group = condition_group
        sig.signal_strength = signal_strength
        sig.sensitivity = sensitivity
        sig.specificity = specificity
        sig.condition_prevalence = condition_prevalence
        sig.is_prn = is_prn
        sig.min_prescriptions = min_prescriptions
        sig.requires_speciality = requires_speciality
        return sig

    def _setup_db(self, episodes, signals, existing_condition=None):
        """Set up a fully mocked DB with proper query chain for infer_for_patient."""
        db = MagicMock()

        # We need to distinguish between queries for different ORM classes.
        # infer_for_patient does:
        #   1. db.query(MedicationEpisode).filter(...).filter(...).filter(...).all()
        #   2. db.query(DrugConditionSignal).filter(...).all()
        #   3. db.query(Doctor).filter(...).all()
        #   4. db.query(PatientCondition).filter(...).filter(...).first()
        # Use side_effect to dispatch based on the queried class

        from src.database import MedicationEpisode, DrugConditionSignal, Doctor, PatientCondition

        def query_side_effect(cls):
            mock_q = MagicMock()
            if cls is MedicationEpisode:
                # ONE .filter(a, b, c).all() call
                mock_q.filter.return_value.all.return_value = episodes
            elif cls is DrugConditionSignal:
                mock_q.filter.return_value.all.return_value = signals
            elif cls is Doctor:
                mock_q.filter.return_value.all.return_value = []
            elif cls is PatientCondition:
                # ONE .filter(a, b, c).first() call
                mock_q.filter.return_value.first.return_value = existing_condition
            return mock_q

        db.query.side_effect = query_side_effect
        return db

    def test_no_episodes_returns_empty(self):
        db = MagicMock()
        db.query.return_value.filter.return_value.filter.return_value.filter.return_value.all.return_value = []
        result = ConditionInferenceEngine.infer_for_patient(db, "patient-1")
        assert result == []

    def test_infers_condition_from_strong_signal(self):
        """Strong metformin signal → a new condition record is created."""
        episodes = [self._mock_episode("metformin", prescription_count=5)]
        signals = [self._mock_signal("metformin")]

        # Use a flat MagicMock so all query chains return our data
        db = MagicMock()
        # ALL filter chains return correct data via the chain we patch
        db.query.return_value.filter.return_value.all.return_value = episodes

        # Override specific sub-chains as needed
        sig_q = MagicMock()
        sig_q.filter.return_value.all.return_value = signals
        doc_q = MagicMock()
        doc_q.filter.return_value.all.return_value = []
        # PatientCondition lookup returns None (new condition)
        pc_q = MagicMock()
        # .filter(a, b, c).first() — ONE filter call
        pc_q.filter.return_value.first.return_value = None

        call_order = ["MedicationEpisode", "DrugConditionSignal", "Doctor", "PatientCondition"]
        query_index = [0]
        def side_effect(cls):
            name = getattr(cls, '__name__', '') if not isinstance(cls, MagicMock) else 'Mock'
            if name == 'DrugConditionSignal' or query_index[0] == 1:
                query_index[0] += 1
                return sig_q
            if name == 'Doctor' or query_index[0] == 2:
                query_index[0] += 1
                return doc_q
            if query_index[0] >= 3:
                return pc_q
            query_index[0] += 1
            m = MagicMock()
            m.filter.return_value.all.return_value = episodes
            return m

        with patch("src.condition_inference.EventStore.emit"), \
             patch("src.condition_inference.PatientCondition") as MockPC:
            mock_cond = MagicMock()
            mock_cond.id = "new-cond-1"
            MockPC.return_value = mock_cond

            # Direct mock: make db.query dispatch by class name
            from src.database import (MedicationEpisode as ME, DrugConditionSignal as DCS,
                                       Doctor as D)
            def dispatched_query(cls):
                if cls is ME:
                    m = MagicMock()
                    m.filter.return_value.all.return_value = episodes
                    return m
                if cls is DCS:
                    return sig_q
                if cls is D:
                    return doc_q
                # PatientCondition (patched or real)
                return pc_q

            db.query.side_effect = dispatched_query
            ConditionInferenceEngine.infer_for_patient(db, "p-1")

        # Should have called db.add for a new condition
        db.add.assert_called()

    def test_skips_condition_below_min_confidence(self):
        """Very low signal strength should be filtered by _MIN_CONFIDENCE."""
        episodes = [self._mock_episode("ibuprofen", status="prn_snapshot", prescription_count=1)]
        signals = [self._mock_signal(
            "ibuprofen", condition_code="M79",
            signal_strength=0.05,  # very weak
            specificity=0.10,
            condition_prevalence=0.20
        )]
        db = self._setup_db(episodes, signals)
        with patch("src.condition_inference.EventStore.emit"):
            result = ConditionInferenceEngine.infer_for_patient(db, "p-1")
        # With such low signal, confidence < _MIN_CONFIDENCE → no condition created
        db.add.assert_not_called()

    def test_never_overwrites_confirmed_condition(self):
        """A clinician-confirmed condition must not be updated by inference."""
        confirmed = MagicMock()
        confirmed.status = "confirmed"
        confirmed.confidence = 0.95

        episodes = [self._mock_episode("metformin", prescription_count=5)]
        signals = [self._mock_signal("metformin")]

        db = self._setup_db(episodes, signals, existing_condition=confirmed)

        with patch("src.condition_inference.EventStore.emit"):
            ConditionInferenceEngine.infer_for_patient(db, "p-1")

        # status must not be changed on confirmed condition
        assert confirmed.status == "confirmed"
        assert confirmed.confidence == 0.95

    def test_updates_existing_probable_condition(self):
        """A probable condition should be updated with new confidence."""
        class ProbableCondition:
            status = "probable"
            confidence = 0.50
            id = "cond-1"
            inference_basis = {}
            last_updated_at = None

        probable = ProbableCondition()

        episodes = [self._mock_episode("metformin", prescription_count=5)]
        signals = [self._mock_signal("metformin")]

        sig_q = MagicMock()
        sig_q.filter.return_value.all.return_value = signals
        doc_q = MagicMock()
        doc_q.filter.return_value.all.return_value = []
        pc_q = MagicMock()
        # .filter(a, b, c).first() — ONE filter call
        pc_q.filter.return_value.first.return_value = probable

        from src.database import (MedicationEpisode as ME, DrugConditionSignal as DCS,
                                   Doctor as D)
        db = MagicMock()
        def dispatched_query(cls):
            if cls is ME:
                m = MagicMock()
                m.filter.return_value.all.return_value = episodes
                return m
            if cls is DCS:
                return sig_q
            if cls is D:
                return doc_q
            # PatientCondition (real class) — return existing probable
            return pc_q

        db.query.side_effect = dispatched_query

        with patch("src.condition_inference.EventStore.emit"), \
             patch("src.condition_inference.func"):
            ConditionInferenceEngine.infer_for_patient(db, "p-1")

        # Confidence should be updated (not 0.50 anymore)
        assert probable.confidence != 0.50

    def test_min_prescriptions_filter(self):
        """Signal with min_prescriptions=3 should be skipped if only 1 prescription."""
        episodes = [self._mock_episode("warfarin", prescription_count=1)]
        signals = [self._mock_signal(
            "warfarin", condition_code="I48",
            min_prescriptions=3,  # requires 3+ prescriptions
        )]
        db = self._setup_db(episodes, signals)
        with patch("src.condition_inference.EventStore.emit"):
            ConditionInferenceEngine.infer_for_patient(db, "p-1")
        db.add.assert_not_called()
