"""
brand_db.py — SQLite-backed fuzzy brand-to-INN dictionary for OJAAI.

Replaces the plain Python dict lookup in drug_normalizer with a two-pass strategy:
  Pass 1: Exact match (case-insensitive) — instant, covers ~80% of cases.
  Pass 2: Levenshtein fuzzy match (edit distance ≤ 2) — recovers OCR typos
          e.g. "Acetaminophin" → "acetaminophen" → "paracetamol"
               "Glycomet5" → "glycomet" → "metformin"

Design decisions:
  - Uses Python stdlib only: sqlite3 + difflib. Zero new pip dependencies.
  - DB file is auto-created at runtime from INDIA_BRAND_MAP if missing.
  - DB path is configurable via BRAND_DB_PATH env var.
  - Thread-safe: each call opens its own connection (cheap for SQLite).
  - Fuzzy cutoff uses difflib.get_close_matches (n-gram based, fast).
    For very short strings (≤ 4 chars) we add an explicit edit-distance guard.

Never raises — all errors degrade gracefully to None.
"""

from __future__ import annotations

import difflib
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = Path(os.getenv("BRAND_DB_PATH", "./data/brand_dict.db"))

# Fuzzy matching thresholds
_FUZZY_CUTOFF = 0.82       # difflib ratio threshold (0.0–1.0); 0.82 ≈ 1-2 char typos
_MAX_EDIT_DIST = 2          # hard cap: never accept more than 2 character edits
_MIN_MATCH_LEN = 3          # don't fuzzy-match strings shorter than 3 chars


# ---------------------------------------------------------------------------
# Edit distance (stdlib only — no python-Levenshtein needed)
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.
    Classic DP implementation — O(len(a) * len(b)).
    Fast enough for short drug names (< 40 chars).
    """
    m, n = len(a), len(b)
    # Early exit for identical strings or empty inputs
    if a == b:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    # Use two rows to save memory
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,       # insertion
                prev[j] + 1,           # deletion
                prev[j - 1] + cost,    # substitution
            )
        prev, curr = curr, prev

    return prev[n]


# ---------------------------------------------------------------------------
# DB builder
# ---------------------------------------------------------------------------

def build_db(brand_map: dict[str, str], db_path: Path = _DEFAULT_DB_PATH) -> None:
    """
    Build (or rebuild) the SQLite brand dictionary from a Python dict.

    Schema:
        brand_name  TEXT PRIMARY KEY   — lowercase brand name (key)
        inn         TEXT NOT NULL      — INN / generic name (value)

    Safe to call multiple times — uses INSERT OR REPLACE.
    Creates parent directories if they don't exist.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS brand_names (
                brand_name TEXT PRIMARY KEY,
                inn        TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_brand ON brand_names(brand_name)")

        rows = [(k.lower().strip(), v.lower().strip()) for k, v in brand_map.items()]
        conn.executemany("INSERT OR REPLACE INTO brand_names VALUES (?, ?)", rows)
        conn.commit()
        logger.info(
            "Brand DB built at %s with %d entries.", db_path, len(rows)
        )
    finally:
        conn.close()


def _ensure_db(db_path: Path) -> None:
    """Auto-build the DB if it doesn't exist yet."""
    if not db_path.exists():
        logger.info("Brand DB not found at %s — building from INDIA_BRAND_MAP.", db_path)
        # Import here to avoid circular imports
        from src.drug_normalizer import INDIA_BRAND_MAP
        build_db(INDIA_BRAND_MAP, db_path)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def lookup_brand(
    raw_name: str,
    db_path: Path = _DEFAULT_DB_PATH,
) -> Optional[str]:
    """
    Look up a brand name and return its INN, or None if no match found.

    Two-pass strategy:
      Pass 1 — Exact match (after lowercase + strip): O(1) indexed SQL lookup.
      Pass 2 — Fuzzy match: fetches all brand names, uses difflib.get_close_matches
               then validates with Levenshtein edit distance ≤ MAX_EDIT_DIST.

    Returns:
        str: the INN (lowercase) if a match is found
        None: if no acceptable match (caller should fall back to RxNorm)

    Never raises.
    """
    if not raw_name or not raw_name.strip():
        return None

    cleaned = raw_name.lower().strip()

    try:
        _ensure_db(db_path)

        conn = sqlite3.connect(str(db_path))
        try:
            # ── Pass 1: Exact match ──────────────────────────────────────────
            row = conn.execute(
                "SELECT inn FROM brand_names WHERE brand_name = ?",
                (cleaned,),
            ).fetchone()
            if row:
                logger.debug("Brand DB exact match: %r → %r", cleaned, row[0])
                return row[0]

            # ── Pass 2: Fuzzy match ──────────────────────────────────────────
            if len(cleaned) < _MIN_MATCH_LEN:
                return None  # too short to fuzzy-match reliably

            all_brands = [r[0] for r in conn.execute("SELECT brand_name FROM brand_names")]
            candidates = difflib.get_close_matches(
                cleaned,
                all_brands,
                n=3,
                cutoff=_FUZZY_CUTOFF,
            )

            best_brand: Optional[str] = None
            best_dist = _MAX_EDIT_DIST + 1  # start above threshold

            for candidate in candidates:
                dist = _edit_distance(cleaned, candidate)
                if dist <= _MAX_EDIT_DIST and dist < best_dist:
                    best_dist = dist
                    best_brand = candidate

            if best_brand:
                inn_row = conn.execute(
                    "SELECT inn FROM brand_names WHERE brand_name = ?",
                    (best_brand,),
                ).fetchone()
                if inn_row:
                    logger.info(
                        "Brand DB fuzzy match: %r → %r (edit_dist=%d, inn=%r)",
                        cleaned, best_brand, best_dist, inn_row[0],
                    )
                    return inn_row[0]

        finally:
            conn.close()

    except Exception as exc:
        logger.warning("Brand DB lookup error for %r: %s — returning None.", raw_name, exc)

    return None


# ---------------------------------------------------------------------------
# CLI helper — rebuild DB on demand
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_DB_PATH
    from src.drug_normalizer import INDIA_BRAND_MAP  # noqa: E402
    build_db(INDIA_BRAND_MAP, db_path)
    print(f"Brand DB built at {db_path} with {len(INDIA_BRAND_MAP)} entries.")
