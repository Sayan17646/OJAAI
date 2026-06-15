# OJAAI

> Prescription intelligence engine for India. Phase 1 MVP.

Parses prescription images → structured JSON with drug info + drug-drug interaction alerts.

---

## Prerequisites

- Python 3.11
- Tesseract 5 (installed to `C:\Program Files\Tesseract-OCR\`)
- PostgreSQL 15

---

## Setup

### 1. Clone and enter the project
```bash
cd OJAAI
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
copy .env.example .env
# Edit .env with your database credentials
```

### 5. Create the PostgreSQL database
```bash
# Run in psql or pgAdmin:
CREATE USER ojaai WITH PASSWORD 'password';
CREATE DATABASE ojaai_dev OWNER ojaai;
```

### 6. Start the server
```bash
uvicorn src.api:app --reload --port 8000
```

### 7. Open API docs
```
http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/parse` | Upload prescription image → structured JSON |
| GET | `/patient/{phone}/history` | All prescriptions for a patient |
| GET | `/patient/{phone}/interactions` | All active DDIs for a patient |

All endpoints require header: `X-API-Key: test-key-ojaai-1234`

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
ojaai/
├── src/
│   ├── models.py          # Pydantic data contracts
│   ├── preprocessor.py    # Image cleaning pipeline
│   ├── medical_ner.py     # Rule-based drug extraction
│   ├── drug_normalizer.py # Brand → INN + RxNorm
│   ├── ddi_checker.py     # Drug interaction detection
│   ├── pipeline.py        # End-to-end orchestrator
│   ├── database.py        # SQLAlchemy ORM
│   └── api.py             # FastAPI routes
├── tests/
│   └── test_pipeline.py
├── data/
│   ├── prescriptions/     # Test images (never commit real data)
│   └── lab_reports/       # Phase 2
├── .env                   # Never commit
├── .env.example
├── requirements.txt
├── CLAUDE.md
└── README.md
```
