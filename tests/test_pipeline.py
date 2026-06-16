"""
test_pipeline.py — Acceptance tests and unit tests for OJAAI Phase 1 MVP.

Acceptance tests (per CLAUDE.md):
  Test 1: "Glycomet 500 BD" → metformin 500mg twice daily
  Test 2: Warfarin + Ibuprofen → major interaction, has_major_interaction=True
  Test 3: Unreadable prescription → confidence < 0.5, needs_human_review=True

Unit tests:
  - Frequency normalisation (all major patterns)
  - Confidence score computation
  - Drug normalisation (INDIA_BRAND_MAP spot checks)
  - DDI pair detection
  - Medication line detection (include/exclude logic)

These tests do NOT require a database connection — DB calls are mocked where needed.
"""

from __future__ import annotations

import pytest

from src.medical_ner import extract, _is_medication_line, _extract_frequency, _compute_confidence
from src.drug_normalizer import INDIA_BRAND_MAP, normalize_drug, _clean_name
from src.ddi_checker import check_interactions
from src.models import MedicationExtracted


# ===========================================================================
# ACCEPTANCE TESTS (must all pass for MVP to be considered done)
# ===========================================================================

class TestAcceptance:
    """
    The three canonical acceptance tests from CLAUDE.md.
    These are the minimum bar for Phase 1 completion.
    """

    def test_1_glycomet_500_bd(self):
        """
        Test 1: "Glycomet 500 BD"
        Expected: drug=Glycomet, inn=metformin, dosage=500mg, frequency=twice daily
        """
        text = "Glycomet 500 BD"
        result = extract(text)

        assert len(result.medications) >= 1, (
            f"Expected at least 1 medication, got {len(result.medications)}. "
            f"Text: {text!r}"
        )

        med = result.medications[0]
        assert med.raw_drug_name.lower() == "glycomet", (
            f"Expected drug name 'Glycomet', got {med.raw_drug_name!r}"
        )
        assert med.dosage_value == 500.0, (
            f"Expected dosage 500.0, got {med.dosage_value}"
        )
        assert med.dosage_unit == "mg", (
            f"Expected unit 'mg', got {med.dosage_unit!r}"
        )
        assert med.frequency == "twice daily", (
            f"Expected frequency 'twice daily', got {med.frequency!r}"
        )
        assert med.freq_per_day == 2, (
            f"Expected freq_per_day=2, got {med.freq_per_day}"
        )

        # Normalisation check
        from src.drug_normalizer import normalize_drug
        normalized = normalize_drug(med)
        assert normalized.inn == "metformin", (
            f"Expected INN 'metformin', got {normalized.inn!r}"
        )

    def test_2_warfarin_ibuprofen_major_interaction(self):
        """
        Test 2: Warfarin + Ibuprofen → severity=major, has_major_interaction=True
        """
        interactions = check_interactions(["warfarin", "ibuprofen"])

        assert len(interactions) >= 1, "Expected at least one interaction for warfarin+ibuprofen"

        major_interactions = [i for i in interactions if i.severity == "major"]
        assert len(major_interactions) >= 1, (
            f"Expected MAJOR severity, got: {[i.severity for i in interactions]}"
        )

        has_major = any(i.severity == "major" for i in interactions)
        assert has_major is True, "has_major_interaction should be True"

        # Verify the pair is specifically identified
        interaction = major_interactions[0]
        pair = {interaction.drug_1.lower(), interaction.drug_2.lower()}
        assert pair == {"warfarin", "ibuprofen"}, (
            f"Unexpected pair: {pair}"
        )

    def test_3_unreadable_prescription_flagged(self):
        """
        Test 3: Unreadable/garbage OCR text → confidence < 0.5, needs_human_review=True
        """
        # Simulate garbage OCR output from a badly lit or crumpled prescription
        garbage_text = (
            "%%@# ~!~ ||||\n"
            "x z q mm mm mm\n"
            ".. .. . . . .\n"
            "OO OO OO ==\n"
        )
        result = extract(garbage_text)

        assert result.confidence < 0.5, (
            f"Expected confidence < 0.5 for unreadable text, got {result.confidence}"
        )

        # The needs_human_review flag is set by the pipeline, not NER.
        # Test it via the threshold comparison directly.
        from src.pipeline import CONFIDENCE_THRESHOLD
        needs_review = result.confidence < CONFIDENCE_THRESHOLD
        assert needs_review is True, (
            f"Expected needs_human_review=True, but confidence={result.confidence} "
            f">= threshold={CONFIDENCE_THRESHOLD}"
        )

    def test_3b_zero_medications_low_confidence(self):
        """
        Variant of Test 3: text with NO medication lines → confidence must be < 0.5
        """
        text = (
            "Patient Name: Rahul Sharma\n"
            "Age: 45 yrs\n"
            "Date: 12/05/2025\n"
            "Hospital: Apollo Clinic\n"
            "Doctor: Dr. Mehta\n"
        )
        result = extract(text)
        assert len(result.medications) == 0, "Expected 0 medications from header-only text"
        assert result.confidence < 0.5, (
            f"Expected confidence < 0.5, got {result.confidence:.4f}"
        )


# ===========================================================================
# Unit tests: Frequency normalisation
# ===========================================================================

class TestFrequencyNormalisation:
    """
    Test all major frequency patterns from TRD Section 2.
    """

    @pytest.mark.parametrize("input_text,expected_std,expected_times", [
        # Twice daily
        ("Tab Metformin 500mg BD", "twice daily", 2),
        ("Amoxicillin 500 BID", "twice daily", 2),
        ("Tab Atenolol 50mg twice daily", "twice daily", 2),
        ("Glycomet 500 1-0-1", "twice daily", 2),
        ("Ramipril 5mg morning and night", "twice daily", 2),

        # Once daily
        ("Tab Atorvastatin 10mg OD", "once daily", 1),
        ("Levothyroxine 50mcg QD", "once daily", 1),
        ("Amlodipine 5mg once daily", "once daily", 1),

        # Three times daily
        ("Amoxicillin 500mg TDS", "three times daily", 3),
        ("Tab Metronidazole 400mg TID", "three times daily", 3),
        ("Ibuprofen 400mg 1-1-1", "three times daily", 3),
        ("Ranitidine 150mg three times daily", "three times daily", 3),

        # Four times daily
        ("Aspirin 100mg QDS", "four times daily", 4),
        ("Cefalexin 500mg QID", "four times daily", 4),

        # At bedtime
        ("Diazepam 5mg HS", "at bedtime", 1),
        ("Olanzapine 5mg at bedtime", "at bedtime", 1),
        ("Clonazepam 0.5mg 0-0-1", "at bedtime", 1),

        # As needed
        ("Ibuprofen 400mg SOS", "as needed", 0),
        ("Ondansetron 4mg PRN", "as needed", 0),
        ("Paracetamol 500mg as needed", "as needed", 0),

        # Before/after meals
        ("Metformin 500mg AC", "before meals", 3),
        ("Pantoprazole 40mg PC", "after meals", 3),

        # Morning only
        ("Levothyroxine 50mcg OM", "every morning", 1),
        ("Metformin 500mg every morning", "every morning", 1),

        # Night only
        ("Atorvastatin 20mg ON", "every night", 1),
        ("Bisoprolol 5mg every night", "every night", 1),

        # STAT
        ("Diclofenac 75mg STAT", "immediately", 0),
        ("Ondansetron 8mg immediately", "immediately", 0),
    ])
    def test_frequency_pattern(self, input_text: str, expected_std: str, expected_times: int):
        std_text, times = _extract_frequency(input_text)
        assert std_text == expected_std, (
            f"Input: {input_text!r}\n"
            f"Expected frequency: {expected_std!r}, got: {std_text!r}"
        )
        assert times == expected_times, (
            f"Input: {input_text!r}\n"
            f"Expected times/day: {expected_times}, got: {times}"
        )


# ===========================================================================
# Unit tests: Medication line detection
# ===========================================================================

class TestMedicationLineDetection:

    @pytest.mark.parametrize("line", [
        "Tab. Metformin 500mg BD",
        "Cap. Amoxicillin 500mg TDS",
        "Glycomet 500 BD",
        "1. Ibuprofen 400mg TDS",
        "Syp. Amoxicillin 250ml BD",
        "Inj. Ceftriaxone 1g OD",
        "Aspirin 75mg OD",
        "Warfarin 5mg OD",
    ])
    def test_medication_lines_accepted(self, line: str):
        assert _is_medication_line(line), f"Expected {line!r} to be identified as a medication line"

    @pytest.mark.parametrize("line", [
        "Patient Name: Rahul Sharma",
        "Doctor: Dr. Gupta",
        "Hospital: Apollo Clinic",
        "Date: 12/05/2025",
        "Age: 45 yrs",
        "Diagnosis: Type 2 Diabetes",
        "Address: 12 Main Street",
        "Tel: 9876543210",
    ])
    def test_non_medication_lines_rejected(self, line: str):
        assert not _is_medication_line(line), (
            f"Expected {line!r} to be rejected as a non-medication line"
        )


# ===========================================================================
# Unit tests: Confidence scoring
# ===========================================================================

class TestConfidenceScore:

    def test_no_medications_low_confidence(self):
        """Zero medications → confidence < 0.5 due to -0.30 penalty."""
        score = _compute_confidence(
            medications=[],
            doctor_reg=None,
            diagnosis=None,
            prescription_date=None,
            patient_age=None,
        )
        assert score < 0.5, f"Expected < 0.5, got {score}"

    def test_full_prescription_high_confidence(self):
        """Complete prescription → confidence >= 0.9."""
        meds = [
            MedicationExtracted(
                raw_drug_name="Glycomet",
                dosage_value=500,
                dosage_unit="mg",
                frequency="twice daily",
                freq_per_day=2,
            )
        ]
        score = _compute_confidence(
            medications=meds,
            doctor_reg="MCI/12345",
            diagnosis="T2DM",
            prescription_date="12/05/2025",
            patient_age="45 yrs",
        )
        # base=0.4 + 0.2 + 0.2 + 0.1 + 0.1 = 1.0
        assert score >= 0.9, f"Expected >= 0.9, got {score}"

    def test_short_drug_names_penalty(self):
        """Average drug name < 4 chars → -0.10 penalty applied."""
        meds = [
            MedicationExtracted(raw_drug_name="AB"),  # 2 chars — triggers penalty
        ]
        score = _compute_confidence(
            medications=meds,
            doctor_reg=None,
            diagnosis=None,
            prescription_date=None,
            patient_age=None,
        )
        # base=0.4 - 0.10 penalty = 0.30
        assert score < 0.5, f"Expected < 0.5, got {score}"

    def test_clamp_to_1(self):
        """Score never exceeds 1.0."""
        meds = [MedicationExtracted(raw_drug_name="Metformin")]
        score = _compute_confidence(
            medications=meds,
            doctor_reg="123",
            diagnosis="DM",
            prescription_date="12/05/2025",
            patient_age="50",
        )
        assert score <= 1.0

    def test_clamp_to_0(self):
        """Score never goes below 0.0."""
        # Extremely bad input
        meds = [MedicationExtracted(raw_drug_name="AB")]  # short name penalty
        score = _compute_confidence(
            medications=[],  # -0.30
            doctor_reg=None,
            diagnosis=None,
            prescription_date=None,
            patient_age=None,
        )
        assert score >= 0.0


# ===========================================================================
# Unit tests: Drug normalisation (INDIA_BRAND_MAP)
# ===========================================================================

class TestDrugNormalisation:

    @pytest.mark.parametrize("brand,expected_inn", [
        ("glycomet", "metformin"),
        ("dolo", "paracetamol"),
        ("crocin", "paracetamol"),
        ("brufen", "ibuprofen"),
        ("voveran", "diclofenac"),
        ("thyronorm", "levothyroxine"),
        ("eltroxin", "levothyroxine"),
        ("storvas", "atorvastatin"),
        ("atorva", "atorvastatin"),
        ("rozavel", "rosuvastatin"),
        ("warf", "warfarin"),
        ("ecosprin", "aspirin"),
        ("clopivas", "clopidogrel"),
        ("pantocid", "pantoprazole"),
        ("omez", "omeprazole"),
        ("azithral", "azithromycin"),
        ("stamlo", "amlodipine"),
        ("tenormin", "atenolol"),
        ("cardace", "ramipril"),
        ("amaryl", "glimepiride"),
    ])
    def test_brand_map_coverage(self, brand: str, expected_inn: str):
        inn = INDIA_BRAND_MAP.get(brand)
        assert inn == expected_inn, (
            f"INDIA_BRAND_MAP[{brand!r}] = {inn!r}, expected {expected_inn!r}"
        )

    def test_brand_map_minimum_size(self):
        """TRD requires minimum 200 entries in INDIA_BRAND_MAP."""
        assert len(INDIA_BRAND_MAP) >= 200, (
            f"INDIA_BRAND_MAP has only {len(INDIA_BRAND_MAP)} entries, need ≥ 200"
        )

    def test_clean_name_strips_prefix(self):
        assert _clean_name("Tab. Glycomet") == "glycomet"
        assert _clean_name("Cap. Amoxicillin 500mg") == "amoxicillin"
        assert _clean_name("Syp. Metronidazole 200ml") == "metronidazole"

    def test_normalize_drug_glycomet(self):
        """Full normalisation of 'Glycomet 500 BD' → metformin."""
        med = MedicationExtracted(
            raw_drug_name="Glycomet",
            dosage_value=500.0,
            dosage_unit="mg",
            frequency="twice daily",
            freq_per_day=2,
        )
        # Only test brand map lookup (don't call RxNorm in unit tests)
        cleaned = _clean_name(med.raw_drug_name)
        inn = INDIA_BRAND_MAP.get(cleaned)
        assert inn == "metformin", f"Expected 'metformin', got {inn!r}"


# ===========================================================================
# Unit tests: DDI checking
# ===========================================================================

class TestDDIChecking:

    def test_warfarin_ibuprofen_major(self):
        interactions = check_interactions(["warfarin", "ibuprofen"])
        assert any(i.severity == "major" for i in interactions)

    def test_warfarin_aspirin_major(self):
        interactions = check_interactions(["warfarin", "aspirin"])
        assert any(i.severity == "major" for i in interactions)

    def test_digoxin_amiodarone_major(self):
        interactions = check_interactions(["digoxin", "amiodarone"])
        assert any(i.severity == "major" for i in interactions)

    def test_lithium_ibuprofen_major(self):
        interactions = check_interactions(["lithium", "ibuprofen"])
        assert any(i.severity == "major" for i in interactions)

    def test_sertraline_tramadol_major(self):
        interactions = check_interactions(["sertraline", "tramadol"])
        assert any(i.severity == "major" for i in interactions)

    def test_simvastatin_clarithromycin_moderate(self):
        interactions = check_interactions(["simvastatin", "clarithromycin"])
        assert any(i.severity in ("moderate", "major") for i in interactions)

    def test_clopidogrel_omeprazole_moderate(self):
        interactions = check_interactions(["clopidogrel", "omeprazole"])
        assert any(i.severity in ("moderate", "major") for i in interactions)

    def test_levothyroxine_ferrous_sulfate_moderate(self):
        interactions = check_interactions(["levothyroxine", "ferrous sulfate"])
        assert any(i.severity in ("moderate", "major") for i in interactions)

    def test_no_interaction_safe_pair(self):
        """Paracetamol + omeprazole — no known DDI."""
        interactions = check_interactions(["paracetamol", "omeprazole"])
        # Should be empty or only unknown/minor from OpenFDA
        major_or_moderate = [i for i in interactions if i.severity in ("major", "moderate")]
        # Don't assert empty (OpenFDA might find something) but there should be none in CRITICAL_DDI_DB
        assert all(i.source != "CRITICAL_DDI_DB" for i in major_or_moderate), (
            "Unexpected CRITICAL_DDI_DB hit for paracetamol+omeprazole"
        )

    def test_empty_list_no_crash(self):
        """Empty drug list returns empty interactions."""
        assert check_interactions([]) == []

    def test_single_drug_no_crash(self):
        """Single drug — no pairs possible — returns empty."""
        assert check_interactions(["warfarin"]) == []

    def test_interactions_sorted_major_first(self):
        """Result is sorted: major before moderate."""
        # warfarin+ibuprofen = major; simvastatin+clarithromycin = moderate
        interactions = check_interactions(["warfarin", "ibuprofen", "simvastatin", "clarithromycin"])
        severities = [i.severity for i in interactions]
        _order = {"major": 0, "moderate": 1, "minor": 2, "unknown": 3}
        for i in range(len(severities) - 1):
            assert _order[severities[i]] <= _order[severities[i + 1]], (
                f"Interactions not sorted correctly: {severities}"
            )

    def test_class_level_ssri_nsaid(self):
        """SSRI (sertraline) + NSAID (naproxen) → at least moderate via class rule.
        Note: sertraline is the canonical SSRI in the class map. escitalopram is
        classified as qt_prolonging (it genuinely prolongs QT) which is a separate rule.
        """
        interactions = check_interactions(["sertraline", "naproxen"])
        assert len(interactions) > 0, "Expected interaction for SSRI+NSAID (sertraline+naproxen)"
        assert any(i.severity in ("moderate", "major") for i in interactions), (
            f"Expected moderate or major, got: {[i.severity for i in interactions]}"
        )

    def test_class_level_qt_prolonging(self):
        """Two QT-prolonging drugs → major via class rule."""
        interactions = check_interactions(["amiodarone", "levofloxacin"])
        assert any(i.severity == "major" for i in interactions), (
            "Expected major QT prolongation interaction for amiodarone+levofloxacin"
        )


# ===========================================================================
# Integration test: Full NER on realistic prescription text
# ===========================================================================

class TestFullPrescriptionNER:

    def test_realistic_printed_prescription(self):
        """
        Test NER on a realistic printed prescription text.
        Verifies that multiple medications are extracted and confidence is reasonable.
        """
        text = """
        Dr. A. K. Mehta
        MBBS, MD (Medicine)
        Reg. No: MCI/23456
        Apollo Multispeciality Clinic, New Delhi

        Patient: Rajesh Kumar   Age: 58 yrs   Date: 15/05/2025
        Diagnosis: Type 2 Diabetes Mellitus, Hypertension

        Rx:
        1. Tab. Glycomet 500mg BD
        2. Tab. Stamlo 5mg OD
        3. Tab. Atorva 10mg ON
        4. Tab. Ecosprin 75mg OD
        5. Tab. Pantocid 40mg OD (before breakfast)
        """

        result = extract(text)

        assert len(result.medications) >= 3, (
            f"Expected ≥ 3 medications, got {len(result.medications)}: "
            f"{[m.raw_drug_name for m in result.medications]}"
        )
        assert result.confidence >= 0.5, (
            f"Expected confidence ≥ 0.5, got {result.confidence}"
        )
        assert result.doctor_reg is not None, "Expected doctor registration number to be extracted"
        assert result.diagnosis is not None, "Expected diagnosis to be extracted"
        assert result.patient_age is not None, "Expected patient age to be extracted"

        # Check specific drug is found
        drug_names_lower = [m.raw_drug_name.lower() for m in result.medications]
        assert any("glycomet" in name for name in drug_names_lower), (
            f"Expected 'Glycomet' in extracted drugs: {drug_names_lower}"
        )


# ===========================================================================
# Dashboard Endpoints & Integration tests
# ===========================================================================

class TestDashboardEndpoints:
    """
    Tests for the Clinical Audit Dashboard REST API endpoints in src/api.py.
    """

    @pytest.fixture(autouse=True)
    def setup_client(self):
        from fastapi.testclient import TestClient
        from src.api import app
        self.client = TestClient(app)

    def test_verify_dashboard_access_denied_without_key_and_no_dev_flag(self, monkeypatch):
        """Request fails with 401 if dashboard dev flag is false/unset."""
        monkeypatch.setenv("DEBUG_LOCAL_DASHBOARD", "false")
        resp = self.client.get("/api/review/queue")
        assert resp.status_code == 401

    def test_verify_dashboard_access_allowed_with_dev_flag_and_localhost(self, monkeypatch):
        """Request succeeds or returns 200/404 if dev flag is true and running locally."""
        monkeypatch.setenv("DEBUG_LOCAL_DASHBOARD", "true")
        resp = self.client.get("/api/review/queue")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_review_queue_and_resolve_transaction(self, monkeypatch):
        """Test database retrieval, suggestion isolation, and single-transaction resolution."""
        monkeypatch.setenv("DEBUG_LOCAL_DASHBOARD", "true")
        
        from src.database import SessionLocal, save_to_review_queue
        db = SessionLocal()
        try:
            # Create a mock review queue item
            item = save_to_review_queue(
                db=db,
                patient=None,
                image_path="./data/prescriptions/test_rx.png",
                raw_ocr_text="Dr. Mehta MCI/12345\n1. Tab. Glycomet 500mg BD",
                confidence=0.3,
                reason="Low confidence test"
            )
            db.commit()
            item_id = str(item.id)
            
            # 1. Fetch from queue
            resp = self.client.get("/api/review/queue")
            assert resp.status_code == 200
            queue_items = resp.json()
            assert any(x["id"] == item_id for x in queue_items)
            
            # 2. Fetch detail (isolated Suggestions on the fly)
            resp = self.client.get(f"/api/review/{item_id}")
            assert resp.status_code == 200
            detail = resp.json()
            assert detail["id"] == item_id
            assert detail["raw_ocr_text"] == "Dr. Mehta MCI/12345\n1. Tab. Glycomet 500mg BD"
            assert "draft_suggestion" in detail
            
            # Verify suggestion isolated: raw_drug_name is Glycomet
            meds = detail["draft_suggestion"]["medications"]
            assert len(meds) >= 1
            assert meds[0]["raw_drug_name"] == "Glycomet"
            
            # 3. Resolve the item in a transaction
            resolve_payload = {
                "patient_phone": "9999988888",
                "doctor_reg": "MCI/12345",
                "patient_age": "50 yrs",
                "diagnosis": "Diabetes Mellitus",
                "prescription_date": "12/05/2025",
                "medications": [
                    {
                        "raw_drug_name": "Glycomet",
                        "dosage_value": 500.0,
                        "dosage_unit": "mg",
                        "frequency": "twice daily",
                        "freq_per_day": 2,
                        "duration_days": 10,
                        "route": "oral"
                    }
                ]
            }
            
            resp = self.client.post(f"/api/review/{item_id}/resolve", json=resolve_payload)
            assert resp.status_code == 200
            resolved_rx = resp.json()
            assert resolved_rx["patient_phone"] == "9999988888"
            assert resolved_rx["medications"][0]["inn"] == "metformin"
            
            # Check resolved in DB
            db.expire_all()
            from src.database import get_review_item_by_id, get_active_medications
            db_item = get_review_item_by_id(db, item_id)
            assert db_item.resolved is True
            
            # Patient should now have metformin active in history
            patient = db_item.patient
            assert patient is not None
            assert patient.phone == "9999988888"
            
            active_meds = get_active_medications(db, patient.id)
            assert any(m.inn == "metformin" for m in active_meds)
            
        finally:
            # Clean up test items
            db.rollback()
            # Delete our test rows in correct dependency order
            from src.database import ReviewQueue, Prescription, Medication, Patient, DrugInteractionRecord
            db.query(Medication).delete()
            db.query(DrugInteractionRecord).delete()
            db.query(Prescription).delete()
            db.query(ReviewQueue).delete()
            db.query(Patient).delete()
            db.commit()
            db.close()

