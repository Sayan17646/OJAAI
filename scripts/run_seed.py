"""
scripts/run_seed.py — Run seed_drug_condition_signals against the live DB.

Usage:
    python3.11 scripts/run_seed.py

Run from the project root (OJAAI/).
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

from src.database import (
    SessionLocal, seed_drug_condition_signals,
    DrugConditionSignal, create_all_tables,
)

def main() -> None:
    # Ensure all tables + PHG DDL migrations exist (idempotent)
    print("[INFO] Running create_all_tables() to create/verify schema…")
    create_all_tables()
    print("[INFO] Tables ready.")

    db = SessionLocal()
    try:
        # Check if the new columns exist (live DB may predate them)
        from sqlalchemy import text, inspect
        inspector = inspect(db.bind)
        cols = {c["name"] for c in inspector.get_columns("drug_condition_signals")}

        if "min_prescriptions" not in cols:
            print("[WARN] Column 'min_prescriptions' missing — running ALTER TABLE…")
            db.execute(text("ALTER TABLE drug_condition_signals ADD COLUMN IF NOT EXISTS min_prescriptions INTEGER NOT NULL DEFAULT 1"))
            db.commit()

        if "requires_speciality" not in cols:
            print("[WARN] Column 'requires_speciality' missing — running ALTER TABLE…")
            db.execute(text("ALTER TABLE drug_condition_signals ADD COLUMN IF NOT EXISTS requires_speciality TEXT"))
            db.commit()

        before = db.query(DrugConditionSignal).count()
        print(f"[INFO] Rows before seed: {before}")

        seed_drug_condition_signals(db)
        db.commit()

        after = db.query(DrugConditionSignal).count()
        print(f"[INFO] Rows after  seed: {after}  (+{after - before} new)")

        # Print a summary by condition group
        from sqlalchemy import func
        rows = (
            db.query(
                DrugConditionSignal.condition_group,
                func.count().label("n")
            )
            .group_by(DrugConditionSignal.condition_group)
            .order_by(func.count().desc())
            .all()
        )
        print("\n  Signals by condition group:")
        for group, n in rows:
            print(f"    {group:<20}  {n:>3} signals")

    except Exception as exc:
        db.rollback()
        print(f"[ERROR] Seed failed: {exc}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    main()
