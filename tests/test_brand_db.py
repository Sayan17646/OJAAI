"""
test_brand_db.py — Unit tests for the SQLite fuzzy brand dictionary.

Tests the two-pass lookup strategy:
  Pass 1: Exact match (case-insensitive)
  Pass 2: Levenshtein fuzzy match for OCR typos (edit distance ≤ 2)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

# Minimal test brand map — avoids importing the full 300+ entry INDIA_BRAND_MAP
_TEST_BRANDS: dict[str, str] = {
    "glycomet": "metformin",
    "glycomet sr": "metformin",
    "acetaminophen": "paracetamol",
    "paracetamol": "paracetamol",
    "dolo": "paracetamol",
    "warfarin": "warfarin",
    "ibuprofen": "ibuprofen",
    "atorvastatin": "atorvastatin",
    "rosuvastatin": "rosuvastatin",
    "amoxicillin": "amoxicillin",
    "augmentin": "amoxicillin + clavulanate",
    "telma": "telmisartan",
    "levothyroxine": "levothyroxine",
    "thyronorm": "levothyroxine",
}


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Build a fresh SQLite DB from the test brand map in a temp dir."""
    from src.brand_db import build_db

    db_path = tmp_path / "test_brand_dict.db"
    build_db(_TEST_BRANDS, db_path)
    return db_path


# ---------------------------------------------------------------------------
# Pass 1: Exact match tests
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_exact_lowercase(self, tmp_db: Path) -> None:
        """Exact lowercase key should resolve instantly."""
        from src.brand_db import lookup_brand

        assert lookup_brand("glycomet", db_path=tmp_db) == "metformin"

    def test_exact_mixed_case(self, tmp_db: Path) -> None:
        """Lookup is case-insensitive — 'Glycomet' should resolve to 'metformin'."""
        from src.brand_db import lookup_brand

        assert lookup_brand("Glycomet", db_path=tmp_db) == "metformin"

    def test_exact_multi_word(self, tmp_db: Path) -> None:
        """Multi-word brand names with exact spacing should match."""
        from src.brand_db import lookup_brand

        assert lookup_brand("glycomet sr", db_path=tmp_db) == "metformin"
        assert lookup_brand("Glycomet SR", db_path=tmp_db) == "metformin"

    def test_exact_generic_name(self, tmp_db: Path) -> None:
        """Generic INN names stored as-is should match themselves."""
        from src.brand_db import lookup_brand

        assert lookup_brand("acetaminophen", db_path=tmp_db) == "paracetamol"

    def test_exact_fdc(self, tmp_db: Path) -> None:
        """Fixed-dose combination brand should map correctly."""
        from src.brand_db import lookup_brand

        assert lookup_brand("augmentin", db_path=tmp_db) == "amoxicillin + clavulanate"

    def test_unknown_brand_returns_none(self, tmp_db: Path) -> None:
        """Unknown brand not in DB should return None."""
        from src.brand_db import lookup_brand

        assert lookup_brand("unknownbrand99", db_path=tmp_db) is None

    def test_empty_string_returns_none(self, tmp_db: Path) -> None:
        """Empty string should return None immediately."""
        from src.brand_db import lookup_brand

        assert lookup_brand("", db_path=tmp_db) is None

    def test_whitespace_only_returns_none(self, tmp_db: Path) -> None:
        """Whitespace-only input should return None."""
        from src.brand_db import lookup_brand

        assert lookup_brand("   ", db_path=tmp_db) is None


# ---------------------------------------------------------------------------
# Pass 2: Fuzzy match tests (OCR typo recovery)
# ---------------------------------------------------------------------------

class TestFuzzyMatch:
    def test_single_char_substitution(self, tmp_db: Path) -> None:
        """
        OCR often substitutes single characters.
        'Acetaminophin' (o→i typo) should fuzzy-match 'acetaminophen' → 'paracetamol'.
        This is the canonical example from the Phase 1 evaluation report.
        """
        from src.brand_db import lookup_brand

        result = lookup_brand("Acetaminophin", db_path=tmp_db)
        assert result == "paracetamol", (
            f"Expected 'paracetamol' for 'Acetaminophin', got {result!r}"
        )

    def test_single_char_deletion(self, tmp_db: Path) -> None:
        """'Glycomt' (missing 'e') should fuzzy-match 'glycomet' → 'metformin'."""
        from src.brand_db import lookup_brand

        result = lookup_brand("Glycomt", db_path=tmp_db)
        assert result == "metformin", (
            f"Expected 'metformin' for 'Glycomt', got {result!r}"
        )

    def test_single_char_insertion(self, tmp_db: Path) -> None:
        """'Wvarfarin' (extra 'v') should fuzzy-match 'warfarin'."""
        from src.brand_db import lookup_brand

        result = lookup_brand("Wvarfarin", db_path=tmp_db)
        assert result == "warfarin", (
            f"Expected 'warfarin' for 'Wvarfarin', got {result!r}"
        )

    def test_trailing_digit_noise(self, tmp_db: Path) -> None:
        """
        OCR sometimes appends a stray digit to the end of a word.
        'Thyronorm1' should still fuzzy-match 'thyronorm' → 'levothyroxine'.
        """
        from src.brand_db import lookup_brand

        result = lookup_brand("Thyronorm1", db_path=tmp_db)
        assert result == "levothyroxine", (
            f"Expected 'levothyroxine' for 'Thyronorm1', got {result!r}"
        )

    def test_completely_different_name_no_match(self, tmp_db: Path) -> None:
        """A name that is completely different should not fuzzy-match."""
        from src.brand_db import lookup_brand

        result = lookup_brand("xyz123abc", db_path=tmp_db)
        assert result is None

    def test_very_short_string_not_fuzzy_matched(self, tmp_db: Path) -> None:
        """
        Strings shorter than _MIN_MATCH_LEN (3 chars) should not attempt fuzzy
        matching — too many false positives at that length.
        """
        from src.brand_db import lookup_brand

        # 'do' is not in DB and is only 2 chars — should return None even though
        # 'dolo' is similar
        result = lookup_brand("do", db_path=tmp_db)
        assert result is None


# ---------------------------------------------------------------------------
# DB build tests
# ---------------------------------------------------------------------------

class TestBuildDb:
    def test_build_creates_file(self, tmp_path: Path) -> None:
        """build_db should create the SQLite file."""
        from src.brand_db import build_db

        db_path = tmp_path / "brand.db"
        assert not db_path.exists()
        build_db({"glycomet": "metformin"}, db_path)
        assert db_path.exists()

    def test_build_is_idempotent(self, tmp_db: Path) -> None:
        """Calling build_db twice should not raise or corrupt data."""
        from src.brand_db import build_db, lookup_brand

        # Build again over the existing DB
        build_db(_TEST_BRANDS, tmp_db)
        assert lookup_brand("glycomet", db_path=tmp_db) == "metformin"

    def test_entry_count(self, tmp_db: Path) -> None:
        """DB should contain exactly as many rows as the input brand map."""
        import sqlite3

        conn = sqlite3.connect(str(tmp_db))
        count = conn.execute("SELECT COUNT(*) FROM brand_names").fetchone()[0]
        conn.close()
        assert count == len(_TEST_BRANDS)


# ---------------------------------------------------------------------------
# Edit distance helper tests
# ---------------------------------------------------------------------------

class TestEditDistance:
    def test_identical_strings(self) -> None:
        from src.brand_db import _edit_distance
        assert _edit_distance("abc", "abc") == 0

    def test_empty_string(self) -> None:
        from src.brand_db import _edit_distance
        assert _edit_distance("", "abc") == 3
        assert _edit_distance("abc", "") == 3

    def test_single_substitution(self) -> None:
        from src.brand_db import _edit_distance
        assert _edit_distance("glycomet", "glysomet") == 1

    def test_single_insertion(self) -> None:
        from src.brand_db import _edit_distance
        assert _edit_distance("glycomet", "glycoomet") == 1

    def test_single_deletion(self) -> None:
        from src.brand_db import _edit_distance
        assert _edit_distance("glycomet", "glycmet") == 1

    def test_two_edits(self) -> None:
        from src.brand_db import _edit_distance
        # acetaminophen → acetaminophin: 1 substitution (e→i at pos 12)
        assert _edit_distance("acetaminophen", "acetaminophin") == 1

    def test_completely_different(self) -> None:
        from src.brand_db import _edit_distance
        dist = _edit_distance("abc", "xyz")
        assert dist == 3
