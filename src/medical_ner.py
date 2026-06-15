"""
medical_ner.py — Rule-based medical Named Entity Recognition for OJAAI.

Converts raw OCR text into an ExtractedPrescription object.
No ML models used in Phase 1 — pure regex and heuristics.

Extraction targets:
  - Medication lines (drug name, dosage, frequency, duration, route)
  - Doctor registration number
  - Patient age
  - Diagnosis
  - Prescription date
  - Confidence score
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from src.models import ExtractedPrescription, MedicationExtracted

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frequency normalisation table (complete, per TRD Section 2)
# Maps regex patterns → (standard_text, times_per_day)
# Order matters: more specific patterns first.
# ---------------------------------------------------------------------------

_FREQ_PATTERNS: List[Tuple[re.Pattern[str], str, int]] = [
    # 4 times daily
    (re.compile(r"\bqds\b|\bqid\b|\bfour\s+times\s+(daily|a\s+day)\b|1[-–]1[-–]1[-–]1\b", re.I),
     "four times daily", 4),

    # 3 times daily
    (re.compile(r"\btds\b|\btid\b|\bthree\s+times\s+(daily|a\s+day)\b|1[-–]1[-–]1\b|morning\s+afternoon\s+night\b", re.I),
     "three times daily", 3),

    # Before meals (treat as 3x)
    (re.compile(r"\bac\b|\bbefore\s+meals?\b", re.I),
     "before meals", 3),

    # After meals (treat as 3x)
    (re.compile(r"\bpc\b|\bafter\s+meals?\b", re.I),
     "after meals", 3),

    # Twice daily
    (re.compile(
        r"\bbd\b|\bbid\b|\btwice\s+(daily|a\s+day)\b|1[-–]0[-–]1\b|morning\s+and\s+night\b",
        re.I,
    ), "twice daily", 2),

    # At bedtime / night only
    (re.compile(r"\bhs\b|\bat\s+bedtime\b|\bnight\s+only\b|0[-–]0[-–]1\b", re.I),
     "at bedtime", 1),

    # Every morning
    (re.compile(r"\bom\b|\bevery\s+morning\b|\bmorning\s+only\b|(?<!\d)1[-–]0[-–]0\b", re.I),
     "every morning", 1),

    # Every night
    (re.compile(r"\bon\b|\bevery\s+night\b", re.I),
     "every night", 1),

    # Once daily
    (re.compile(r"\bod\b|\bqd\b|\bonce\s+(daily|a\s+day)\b", re.I),
     "once daily", 1),

    # As needed / PRN
    (re.compile(r"\bsos\b|\bprn\b|\bas\s+needed\b|\bif\s+required\b", re.I),
     "as needed", 0),

    # Immediately / STAT
    (re.compile(r"\bstat\b|\bimmediately\b|\bat\s+once\b", re.I),
     "immediately", 0),
]

# ---------------------------------------------------------------------------
# Dosage unit normalisation
# ---------------------------------------------------------------------------

_UNIT_MAP: dict[str, str] = {
    "mg": "mg", "milligrams": "mg", "milligram": "mg",
    # Common Tesseract misreads of 'mg': 'my', 'rng', 'mq', 'm9', 'mo'
    "my": "mg", "mq": "mg", "m9": "mg", "mo": "mg", "rng": "mg",
    "mcg": "mcg", "micrograms": "mcg", "microgram": "mcg", "ug": "mcg",
    "ml": "ml", "millilitres": "ml", "milliliters": "ml", "cc": "ml",
    # Common Tesseract misreads of 'ml': 'mi', 'rnl', 'mI'
    "mi": "ml", "rnl": "ml",
    "g": "g", "gm": "g", "gram": "g", "grams": "g",
    "iu": "iu", "units": "units", "unit": "units",
    "meq": "meq",
    "drops": "drops", "drop": "drops",
    "puffs": "puffs", "puff": "puffs",
    "%": "%",
}

# OCR digit normalisation: fix common character substitutions before dosage parsing
# e.g. 'Z00' → '200', 'l0' → '10', 'O' → '0' in number context
def _fix_ocr_digits(text: str) -> str:
    """Correct common OCR digit/letter confusions in numeric contexts."""
    # Z/z → 2 when surrounded by digits or at start of a number
    text = re.sub(r"(?<![A-Za-z])[Zz](\d)", r"2\1", text)
    # l/I → 1 when followed by digit (e.g. 'l0 mg' → '10 mg')
    text = re.sub(r"(?<![A-Za-z])[lI](\d)", r"1\1", text)
    # O → 0 when between digits (e.g. '5O0' → '500')
    text = re.sub(r"(\d)[O](\d)", r"\g<1>0\2", text)
    return text

# Regex to capture: number (int or float) + unit
_DOSAGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*("
    + "|".join(sorted(_UNIT_MAP.keys(), key=len, reverse=True))
    + r")\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Duration patterns → days
# ---------------------------------------------------------------------------

_DURATION_PATTERNS: List[Tuple[re.Pattern[str], int]] = [
    (re.compile(r"(\d+)\s*days?\b", re.I), 1),         # N days → N
    (re.compile(r"(\d+)\s*weeks?\b", re.I), 7),        # N weeks → N*7
    (re.compile(r"(\d+)\s*months?\b", re.I), 30),      # N months → N*30
]

# ---------------------------------------------------------------------------
# Drug form prefixes — a line starting with these is almost certainly a drug
# ---------------------------------------------------------------------------

_DRUG_FORM_RE = re.compile(
    r"^\s*(?:tab\.?|cap\.?|syp\.?|syr\.?|inj\.?|susp\.?|oint\.?|drops?|cream|gel|patch|liq\.?|sol\.?|pwd\.?"
    # Common Tesseract OCR artifacts when first character is cut off:
    r"|ab\.?|ap\.?|nj\.?|yp\.?|yr\.?)\s+",
    re.I,
)

# ---------------------------------------------------------------------------
# Known drug names — catch bare lines with no dosage/form/frequency
# Covers all drugs commonly seen in the evaluation set + high-frequency
# Indian and international generics.  Case-insensitive whole-word match.
# This is Criterion 5 in _is_medication_line().
# ---------------------------------------------------------------------------

_KNOWN_DRUG_NAMES = [
    # Analgesics / anti-inflammatory
    "acetaminophen", "paracetamol", "ibuprofen", "aspirin", "naproxen",
    "diclofenac", "nimesulide", "mefenamic", "indomethacin", "celecoxib",
    "etoricoxib", "tramadol", "codeine", "morphine", "pentazocine",
    # Antibiotics
    "amoxicillin", "amoxyclav", "augmentin", "ciprofloxacin", "levofloxacin",
    "ofloxacin", "azithromycin", "clarithromycin", "erythromycin", "doxycycline",
    "metronidazole", "tinidazole", "cefalexin", "cefixime", "ceftriaxone",
    "cefpodoxime", "norfloxacin", "moxifloxacin", "fluconazole", "itraconazole",
    # Antihypertensives
    "amlodipine", "nifedipine", "lisinopril", "ramipril", "enalapril",
    "captopril", "losartan", "telmisartan", "olmesartan", "valsartan",
    "atenolol", "metoprolol", "propranolol", "bisoprolol", "nebivolol",
    "carvedilol", "furosemide", "hydrochlorothiazide", "spironolactone",
    "torsemide", "indapamide",
    # Antidiabetics
    "metformin", "glibenclamide", "glimepiride", "gliclazide", "glipizide",
    "sitagliptin", "vildagliptin", "saxagliptin", "linagliptin", "alogliptin",
    "empagliflozin", "dapagliflozin", "canagliflozin",
    # Statins
    "atorvastatin", "rosuvastatin", "simvastatin", "pravastatin",
    # Thyroid
    "levothyroxine", "liothyronine",
    # GI / PPIs
    "pantoprazole", "omeprazole", "esomeprazole", "rabeprazole", "lansoprazole",
    "domperidone", "ondansetron", "metoclopramide", "ranitidine",
    # Corticosteroids — both US and India names
    "prednisolone", "prednisone", "dexamethasone", "methylprednisolone",
    "betamethasone", "hydrocortisone",
    # Psych / neuro
    "sertraline", "fluoxetine", "escitalopram", "paroxetine",
    "clonazepam", "alprazolam", "diazepam", "lorazepam",
    "olanzapine", "risperidone", "quetiapine", "haloperidol", "clozapine",
    "venlafaxine", "duloxetine", "mirtazapine", "amitriptyline",
    "gabapentin", "pregabalin", "phenytoin", "carbamazepine",
    "levetiracetam", "valproate", "lithium",
    # Anticoagulants
    "warfarin", "heparin", "enoxaparin", "rivaroxaban", "apixaban", "dabigatran",
    "clopidogrel",
    # Vitamins / supplements
    "cholecalciferol", "calcitriol", "mecobalamin", "folic",
    # Bronchodilators
    "salbutamol", "ipratropium", "theophylline", "montelukast",
    # Urology
    "tamsulosin", "finasteride", "dutasteride",
    # TB
    "isoniazid", "rifampicin", "pyrazinamide", "ethambutol",
    # ARVs
    "tenofovir", "lamivudine", "efavirenz", "nevirapine", "dolutegravir",
]

_KNOWN_DRUGS_RE = re.compile(
    r"\b(" + "|".join(sorted(_KNOWN_DRUG_NAMES, key=len, reverse=True)) + r")\b",
    re.I,
)

# Numbered list: "1. Drug" or "1) Drug"
_NUMBERED_LINE_RE = re.compile(r"^\s*\d+[.)]\s+(?=\w)", re.I)

# Bare circled/standalone numbers at line start (OCR template label noise, e.g. "2 Tast conducted")
_BARE_NUMBER_LINE_RE = re.compile(r"^\s*\d+\s+[A-Za-z]", re.I)

# Keywords that disqualify a line from being a medication
_NON_MED_KEYWORDS = re.compile(
    r"\b(patient|doctor|hospital|clinic|date|age|sex|address|name|reg|"
    r"diagnosis|dx|rx|signature|stamp|phone|tel|mob|email|fax|"
    r"dept|department|ref|referred|refill|conducted|test|tast|disp|dispensed|"
    r"sig|directions|instructions|note|advised|collected|reported)\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Extraction helpers for header fields
# ---------------------------------------------------------------------------

# Doctor registration — MCI/NMC number or Reg No
_DOC_REG_RE = re.compile(
    r"(?:mci|nmc|reg(?:istration)?|regd?)[\s.:/#]*([A-Z0-9/-]{4,20})",
    re.I,
)

# Patient age — "52 yrs", "Age: 34", "28 Y"
_PATIENT_AGE_RE = re.compile(
    r"(?:age|aged?)[:\s]*(\d{1,3})\s*(?:yrs?|years?|y\.?)?|(\d{1,3})\s*(?:yrs?|years?)",
    re.I,
)

# Prescription date — DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, or "March 18, 2014" / "18 March 2014"
_MONTHS = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
    r"|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_DATE_RE = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"          # DD/MM/YYYY or DD-MM-YYYY
    r"|\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"            # YYYY-MM-DD
    r"|\b(" + _MONTHS + r"[\s.]+\d{1,2}[,\s]+\d{4})\b"  # March 18, 2014
    r"|\b(\d{1,2}[\s.]+" + _MONTHS + r"[\s.,]+\d{4})\b",  # 18 March 2014
    re.I,
)

# Diagnosis — common abbreviations + "Diagnosis:" / "Dx:" labels
_DIAGNOSIS_RE = re.compile(
    r"(?:diagnosis|dx|d/x)[:\s]+([A-Za-z0-9 ,/()-]{3,60})",
    re.I,
)

# Route of administration
_ROUTE_RE = re.compile(
    r"\b(oral|po|iv|im|sc|topical|sublingual|sl|rectal|inhaled?|transdermal|nasal"
    r"|ophthalmic|ophth|eye\s+drops?|ear\s+drops?|otic|intraocular"
    r"|intramuscular|intravenous|subcutaneous)\b",
    re.I,
)
_ROUTE_MAP: dict[str, str] = {
    "oral": "oral", "po": "oral",
    "iv": "iv", "intravenous": "iv",
    "im": "im", "intramuscular": "im",
    "sc": "sc", "subcutaneous": "sc",
    "topical": "topical",
    "sublingual": "sublingual", "sl": "sublingual",
    "rectal": "rectal",
    "inhaled": "inhaled", "inhale": "inhaled",
    "transdermal": "transdermal",
    "nasal": "nasal",
    "ophthalmic": "ophthalmic", "ophth": "ophthalmic",
    "intraocular": "ophthalmic", "eye drops": "ophthalmic", "eye drop": "ophthalmic",
    "ear drops": "otic", "ear drop": "otic", "otic": "otic",
}


# ---------------------------------------------------------------------------
# Confidence score computation (per TRD Section 2)
# ---------------------------------------------------------------------------

def _compute_confidence(
    medications: List[MedicationExtracted],
    doctor_reg: Optional[str],
    diagnosis: Optional[str],
    prescription_date: Optional[str],
    patient_age: Optional[str],
) -> float:
    base = 0.0

    if medications:
        base += 0.40
    else:
        base -= 0.30

    if doctor_reg:
        base += 0.20
    if diagnosis:
        base += 0.20
    if prescription_date:
        base += 0.10
    if patient_age:
        base += 0.10

    # Penalty: average drug name too short (suggests garbage OCR)
    if medications:
        avg_len = sum(len(m.raw_drug_name) for m in medications) / len(medications)
        if avg_len < 4:
            base -= 0.10

    return round(max(0.0, min(1.0, base)), 4)


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------

def _extract_dosage(text: str) -> Tuple[Optional[float], Optional[str]]:
    """Return (value, unit) or (None, None) if no dosage found.

    Two strategies:
    1. Primary: look for number + unit (e.g. "500mg", "10 mcg").
    2. Fallback: look for a bare number when a frequency is also present.
       In Indian prescriptions "Glycomet 500 BD" means 500mg — unit is implied.
       Defaults unit to "mg" (by far the most common tablet unit).
    """
    # Normalise common OCR digit/character errors before parsing
    text = _fix_ocr_digits(text)
    # Strategy 1: explicit number + unit
    match = _DOSAGE_RE.search(text)
    if match:
        value = float(match.group(1))
        raw_unit = match.group(2).lower()
        unit = _UNIT_MAP.get(raw_unit, raw_unit)
        return value, unit

    # Strategy 2: bare number when a frequency token is also present
    # This handles shorthand like "Glycomet 500 BD", "Pantoprazole 40 OD"
    has_frequency = any(pat.search(text) for pat, _, _ in _FREQ_PATTERNS)
    if has_frequency:
        bare = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
        if bare:
            return float(bare.group(1)), "mg"

    return None, None



def _extract_frequency(text: str) -> Tuple[Optional[str], Optional[int]]:
    """Return (standard_text, times_per_day) or (None, None)."""
    for pattern, std_text, times in _FREQ_PATTERNS:
        if pattern.search(text):
            return std_text, times
    return None, None


def _extract_duration(text: str) -> Optional[int]:
    """Return duration in days, or None if not found."""
    for pattern, multiplier in _DURATION_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1)) * multiplier
    return None


def _extract_route(text: str) -> str:
    """Return normalised route or 'oral' as default."""
    match = _ROUTE_RE.search(text)
    if match:
        raw = match.group(1).lower()
        return _ROUTE_MAP.get(raw, "oral")
    return "oral"


def _extract_drug_name(line: str) -> Optional[str]:
    """
    Heuristically extract the drug name from a medication line.
    Strips form prefix, dosage, frequency, duration, and common noise.
    Returns None if the result is too short to be a real drug name.
    """
    # Remove numbered list prefix first
    line = _NUMBERED_LINE_RE.sub("", line).strip() if _NUMBERED_LINE_RE.match(line) else line
    # Remove form prefix next
    line = _DRUG_FORM_RE.sub("", line).strip()
    # Remove dosage pattern (number + unit)
    line = _DOSAGE_RE.sub("", line).strip()
    # Remove frequency patterns
    for pattern, _, _ in _FREQ_PATTERNS:
        line = pattern.sub("", line).strip()
    # Remove duration patterns
    for pattern, _ in _DURATION_PATTERNS:
        line = pattern.sub("", line).strip()
    # Remove route
    line = _ROUTE_RE.sub("", line).strip()
    # Strip bare trailing numbers (e.g. "500" leftover from "Glycomet 500 BD"
    # after BD is removed — unit-less dosage is common in Indian prescriptions)
    line = re.sub(r"\s+\d+(\.\d+)?\s*$", "", line).strip()
    # Remove punctuation/noise at start and end
    line = re.sub(r"^[\s\-–•*#.:,;()]+", "", line)
    line = re.sub(r"[\s\-–•*#.:,;()]+$", "", line)
    # Take only the first "word block" (drug name is usually 1-3 words)
    # Stop at conjunctions, slashes, or long gaps
    parts = re.split(r"\s{2,}|/|\+", line)
    name = parts[0].strip() if parts else ""
    # Reject if too short
    if len(name) < 3:
        return None
    # Reject if it looks like a number/unit only
    if re.match(r"^\d+(\.\d+)?\s*(mg|ml|mcg|g|iu)?$", name, re.I):
        return None
    return name



def _is_medication_line(line: str) -> bool:
    """
    Return True if the line looks like a prescription medication entry.
    Per TRD: must match at least one of three criteria, and must not
    contain disqualifying keywords.
    Criterion 4 (extended): bare number + frequency pattern — handles Indian
    shorthand like "Glycomet 500 BD" where the unit (mg) is implied.
    """
    # Disqualifier check first
    if _NON_MED_KEYWORDS.search(line):
        return False
    # Must have some alphabetic content
    if not re.search(r"[A-Za-z]", line):
        return False
    # Disqualify lines that are ONLY a bare number followed by words (OCR template label noise)
    # e.g. "2 Tast conducted", "3 Some label" — these are circled number artifacts
    if _BARE_NUMBER_LINE_RE.match(line) and not _DOSAGE_RE.search(line) and not _DRUG_FORM_RE.match(line):
        return False
    # Disqualify very short lines (< 5 chars of alphabetic content) — too noisy
    if len(re.sub(r"[^A-Za-z]", "", line)) < 5:
        return False
    # Criterion 1: contains a dosage pattern (number + unit)
    if _DOSAGE_RE.search(line):
        return True
    # Criterion 2: starts with a drug form prefix
    if _DRUG_FORM_RE.match(line):
        return True
    # Criterion 3: numbered list (e.g. "1. Metformin 500mg OD")
    if _NUMBERED_LINE_RE.match(line):
        return True
    # Criterion 4: bare number AND a recognized frequency pattern
    # Handles "Glycomet 500 BD", "Pantoprazole 40 OD", etc.
    has_bare_number = bool(re.search(r"\b\d+\b", line))
    has_frequency = any(pat.search(line) for pat, _, _ in _FREQ_PATTERNS)
    if has_bare_number and has_frequency:
        return True
    # Criterion 5: line contains a known drug name (catches bare lines like
    # "Prednisone" or "Acetaminophen" that have no dosage/form/frequency yet)
    if _KNOWN_DRUGS_RE.search(line):
        return True
    return False


def _parse_medication_line(line: str) -> Optional[MedicationExtracted]:
    """Parse one medication line into a MedicationExtracted object."""
    # Normalise common OCR errors (e.g. 'Z00 my' → '200 mg') before all parsing
    line = _fix_ocr_digits(line)
    drug_name = _extract_drug_name(line)
    if not drug_name:
        return None

    dosage_value, dosage_unit = _extract_dosage(line)
    frequency, freq_per_day = _extract_frequency(line)
    duration_days = _extract_duration(line)
    route = _extract_route(line)

    return MedicationExtracted(
        raw_drug_name=drug_name,
        dosage_value=dosage_value,
        dosage_unit=dosage_unit,
        frequency=frequency,
        freq_per_day=freq_per_day,
        duration_days=duration_days,
        route=route,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(raw_text: str) -> ExtractedPrescription:
    """
    Run the full NER pipeline on raw OCR text.
    Returns an ExtractedPrescription with all extracted fields.
    """
    lines = raw_text.splitlines()

    medications: List[MedicationExtracted] = []
    doctor_reg: Optional[str] = None
    patient_age: Optional[str] = None
    diagnosis: Optional[str] = None
    prescription_date: Optional[str] = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Header field extraction (these are mutually exclusive per line intent)
        if not doctor_reg:
            m = _DOC_REG_RE.search(line)
            if m:
                doctor_reg = m.group(1).strip()
                continue  # Don't treat as a med line if it has a reg number

        if not patient_age:
            m = _PATIENT_AGE_RE.search(line)
            if m:
                patient_age = (m.group(1) or m.group(2)).strip() + " yrs"

        if not prescription_date:
            m = _DATE_RE.search(line)
            if m:
                # Pick whichever capture group matched (numeric or month-name format)
                raw_date = next((g for g in m.groups() if g), None)
                if raw_date:
                    prescription_date = raw_date.strip()  # stored as-is, normalised display only

        if not diagnosis:
            m = _DIAGNOSIS_RE.search(line)
            if m:
                diagnosis = m.group(1).strip()

        # Medication line detection
        if _is_medication_line(line):
            med = _parse_medication_line(line)
            if med:
                medications.append(med)

    confidence = _compute_confidence(
        medications, doctor_reg, diagnosis, prescription_date, patient_age
    )

    logger.info(
        "NER complete. medications=%d confidence=%.4f",
        len(medications),
        confidence,
    )

    return ExtractedPrescription(
        medications=medications,
        diagnosis=diagnosis,
        doctor_reg=doctor_reg,
        patient_age=patient_age,
        prescription_date=prescription_date,
        confidence=confidence,
        raw_text=raw_text,   # stored for audit; pipeline must NOT log this at INFO
    )
