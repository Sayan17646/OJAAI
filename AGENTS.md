# OJAAI — Codex Project Context

## What This Project Is
OJAAI is a prescription intelligence engine for India. It parses prescription images (handwritten or printed), extracts drug information, normalises drug names to INN/RxNorm, and detects drug-drug interactions across the patient's medication history.

## Current Phase
**Phase 1 MVP** — Backend pipeline only. No patient-facing UI. No mobile app.

## Stack
- Python 3.11 (native Windows)
- OpenCV for image preprocessing
- Tesseract 5 for OCR
- Rule-based regex NER (no ML in Phase 1)
- FastAPI + Pydantic for the API layer
- SQLAlchemy + PostgreSQL 15 for persistence
- RxNorm API (NIH, free) for drug normalisation
- OpenFDA API (free) for DDI fallback

## Module Map
```
src/
├── models.py          # Pydantic data contracts — source of truth
├── preprocessor.py    # Image → clean grayscale numpy array
├── medical_ner.py     # Raw OCR text → ExtractedPrescription
├── drug_normalizer.py # Drug name → INN + RxCUI (INDIA_BRAND_MAP first)
├── ddi_checker.py     # Drug list → DrugInteraction list
├── pipeline.py        # Orchestrates all modules end-to-end
├── database.py        # SQLAlchemy ORM + session management
└── api.py             # FastAPI routes
```

---

# OJAAI Agent Rules

## Phase 1 Philosophy
- Keep the system simple and debuggable.
- Prefer rule-based systems over ML unless explicitly requested.
- Avoid premature optimization.
- Avoid unnecessary abstractions.

## Architecture Constraints
- Do NOT introduce microservices.
- Do NOT introduce Docker.
- Do NOT introduce Redis/Kafka/RabbitMQ.
- Do NOT introduce async distributed workflows.
- PostgreSQL only.
- Local filesystem storage only.

## Coding Rules
- Prefer readable code over clever code.
- Every external API call must have timeout=5.
- Never silently swallow exceptions.
- Use Pydantic models for all API I/O.
- Type hints required everywhere.

## Medical Safety Rules
- Never auto-delete low-confidence records.
- Never auto-resolve DDIs.
- Never make treatment recommendations.
- Flag uncertainty explicitly.

## AI/ML Rules
- Rule-based extraction first.
- BioBERT only after baseline metrics are collected.
- Do not introduce training pipelines in Phase 1.

## Performance Constraints
- /parse endpoint target < 8 seconds.
- Avoid unnecessary API calls.
- Cache RxNorm lookups.

## File Rules
- Never store images in DB blobs.
- Never commit real patient data.
- Use synthetic or anonymized examples only.

---

# Acceptance Tests

## Test 1
Input:
"Glycomet 500 BD"

Expected:
drug_name = Glycomet
inn = metformin
dosage = 500mg
frequency = twice daily

---

## Test 2
Input:
Warfarin + Ibuprofen

Expected:
severity = major
has_major_interaction = true

---

## Test 3
Unreadable prescription image

Expected:
confidence < 0.5
needs_human_review = true
