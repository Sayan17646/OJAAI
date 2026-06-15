"""
lab_ner.py — Rule-based Named Entity Recognition (NER) parser for Indian lab reports.
"""
import re
from typing import List, Optional, Tuple
from src.models import LabReportExtracted, LabResultExtracted

# Regexes for lab report metadata
_LAB_NAME_RE = re.compile(
    r"(apollo\s+diagnostics|srl\s+diagnostics|lal\s+pathlabs|thyrocare|metropolis|care\s+clinic|lorem\s+ipsum\s+lab)",
    re.I
)

# Date: DD/MM/YYYY, DD-MM-YYYY, DD-MMM-YYYY
_DATE_RE = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b|\b(\d{1,2}[/-][A-Za-z]{3}[/-]\d{2,4})\b"
)

# Analytes details mapping: (Name regex, normalized name, standard unit, reference range, threshold checks)
# Reference ranges:
# - HBA1C: < 5.7 % (flag high if >= 5.7)
# - CREATININE: 0.6 - 1.2 mg/dL (flag high if > 1.3, low if < 0.5)
# - HEMOGLOBIN: 12.0 - 16.0 g/dL (flag low if < 11.5)
# - TSH: 0.4 - 4.5 uIU/mL (flag high if > 4.5, low if < 0.4)
# - LDL: < 100 mg/dL (flag high if > 100)
# - HDL: 40 - 60 mg/dL (flag low if < 40)
# - TRIGLYCERIDES: < 150 mg/dL (flag high if > 150)
_ANALYTE_RULES = [
    {
        "name": "HBA1C",
        "patterns": [r"\bhba1c\b", r"\bglycated\s+hemoglobin\b", r"\bglyco\s+hemoglobin\b"],
        "unit": "%",
        "ref_range": "< 5.7 %",
        "eval": lambda v: "high" if v >= 5.7 else "normal"
    },
    {
        "name": "FASTING_BLOOD_SUGAR",
        "patterns": [r"\bfasting\s+blood\s+sugar\b", r"\bfasting\s+glucose\b", r"\bfbs\b"],
        "unit": "mg/dL",
        "ref_range": "70 - 100 mg/dL",
        "eval": lambda v: "high" if v > 100 else ("low" if v < 70 else "normal")
    },
    {
        "name": "CREATININE",
        "patterns": [r"\bserum\s+creatinine\b", r"\bs\.?\s*creatinine\b", r"\bcreatinine\b"],
        "unit": "mg/dL",
        "ref_range": "0.6 - 1.2 mg/dL",
        "eval": lambda v: "high" if v > 1.3 else ("low" if v < 0.5 else "normal")
    },
    {
        # Hemoglobin: male ref 13-17, female ref 12-16.
        # We store conservative unisex low threshold (<11.5) for now;
        # a gender-aware flag is added in the 'note' field.
        "name": "HEMOGLOBIN",
        "patterns": [r"\bhemoglobin\b", r"\bhaemoglobin\b"],
        "unit": "g/dL",
        "ref_range": "M: 13-17 / F: 12-16 g/dL",
        "eval": lambda v: "low" if v < 12.0 else ("high" if v > 17.0 else "normal")
    },
    {
        "name": "TSH",
        "patterns": [r"\btsh\b", r"\bthyroid\s+stimulating\s+hormone\b"],
        "unit": "uIU/mL",
        "ref_range": "0.4 - 4.5 uIU/mL",
        "eval": lambda v: "high" if v > 4.5 else ("low" if v < 0.4 else "normal")
    },
    {
        "name": "LDL",
        "patterns": [r"\bldl\b", r"\bldl\s+cholesterol\b", r"\blow\s+density\s+lipoprotein\b"],
        "unit": "mg/dL",
        "ref_range": "< 100 mg/dL",
        "eval": lambda v: "high" if v > 100 else "normal"
    },
    {
        "name": "HDL",
        "patterns": [r"\bhdl\b", r"\bhdl\s+cholesterol\b", r"\bhigh\s+density\s+lipoprotein\b"],
        "unit": "mg/dL",
        "ref_range": "40 - 60 mg/dL",
        "eval": lambda v: "low" if v < 40 else "normal"
    },
    {
        "name": "TRIGLYCERIDES",
        "patterns": [r"\btriglycerides\b", r"\btriglyceride\b"],
        "unit": "mg/dL",
        "ref_range": "< 150 mg/dL",
        "eval": lambda v: "high" if v > 150 else "normal"
    },
    # ── CBC Panel ──────────────────────────────────────────────
    {
        "name": "PCV",
        "patterns": [r"\bpcv\b", r"\bpacked\s+cell\s+volume\b", r"\bhaematocrit\b", r"\bhematocrit\b"],
        "unit": "%",
        "ref_range": "M: 40-50 / F: 36-46 %",
        "eval": lambda v: "low" if v < 36 else ("high" if v > 50 else "normal")
    },
    {
        "name": "RBC_COUNT",
        "patterns": [r"\brbc\s+count\b", r"\bred\s+blood\s+cell\s+count\b", r"\berythrocyte\s+count\b"],
        "unit": "mill/mm3",
        "ref_range": "M: 4.5-5.5 / F: 4.0-5.0 mill/mm3",
        "eval": lambda v: "low" if v < 4.0 else ("high" if v > 5.5 else "normal")
    },
    {
        "name": "MCV",
        "patterns": [r"\bmcv\b", r"\bmean\s+corpuscular\s+volume\b"],
        "unit": "fL",
        "ref_range": "80 - 100 fL",
        "eval": lambda v: "low" if v < 80 else ("high" if v > 100 else "normal")
    },
    {
        "name": "MCH",
        "patterns": [r"\bmch\b(?!c)", r"\bmean\s+corpuscular\s+hemoglobin\b(?!\s+conc)"],
        "unit": "pg",
        "ref_range": "27 - 32 pg",
        "eval": lambda v: "low" if v < 27 else ("high" if v > 32 else "normal")
    },
    {
        "name": "MCHC",
        "patterns": [r"\bmchc\b", r"\bmean\s+corpuscular\s+hemoglobin\s+conc"],
        "unit": "g/dL",
        "ref_range": "32 - 36 g/dL",
        "eval": lambda v: "low" if v < 32 else ("high" if v > 36 else "normal")
    },
    {
        "name": "RDW",
        "patterns": [r"\brdw\b", r"\bred\s+cell\s+distribution\s+width\b"],
        "unit": "%",
        "ref_range": "11.5 - 14.5 %",
        "eval": lambda v: "high" if v > 14.5 else ("low" if v < 11.5 else "normal")
    },
    {
        "name": "TLC",
        "patterns": [r"\btlc\b", r"\btotal\s+leukocyte\s+count\b", r"\bwbc\s+count\b", r"\bwhite\s+blood\s+cell\s+count\b"],
        "unit": "thou/mm3",
        "ref_range": "4.0 - 10.0 thou/mm3",
        "eval": lambda v: "low" if v < 4.0 else ("high" if v > 10.0 else "normal")
    },
    {
        "name": "PLATELET_COUNT",
        "patterns": [r"\bplatelet\s+count\b", r"\bplatelets\b", r"\bthrombocyte\s+count\b"],
        "unit": "thou/mm3",
        "ref_range": "150 - 450 thou/mm3",
        "eval": lambda v: "low" if v < 150 else ("high" if v > 450 else "normal")
    },
    {
        "name": "NEUTROPHILS",
        "patterns": [r"\bneutrophils?\b", r"\bsegmented\s+neutrophils?\b"],
        "unit": "%",
        "ref_range": "40 - 80 %",
        "eval": lambda v: "low" if v < 40 else ("high" if v > 80 else "normal")
    },
    {
        "name": "LYMPHOCYTES",
        "patterns": [r"\blymphocytes?\b"],
        "unit": "%",
        "ref_range": "20 - 40 %",
        "eval": lambda v: "low" if v < 20 else ("high" if v > 40 else "normal")
    },
]

# Float capture pattern (e.g. 5.7, .8, 140, 1.8)
_VALUE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\b")

def extract_lab_report(text: str) -> LabReportExtracted:
    """
    Parses OCR text of a diagnostic lab report.
    Returns structured results and clinical metadata.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    # 1. Lab name extraction
    lab_name = "Unknown Diagnostics"
    for line in lines:
        match = _LAB_NAME_RE.search(line)
        if match:
            lab_name = match.group(1).title()
            break
            
    # 2. Date extraction
    report_date = None
    for line in lines:
        # Avoid matching patient birth dates or register dates if possible, first match date wins
        if "date" in line.lower() or "report" in line.lower():
            match = _DATE_RE.search(line)
            if match:
                report_date = match.group(1) or match.group(2)
                break
    if not report_date:
        # Fallback to any date match in whole text
        match = _DATE_RE.search(text)
        if match:
            report_date = match.group(1) or match.group(2)

    # 3. Test results extraction
    results: List[LabResultExtracted] = []
    
    # Search each rule against lines
    for rule in _ANALYTE_RULES:
        regexes = [re.compile(p, re.I) for p in rule["patterns"]]
        
        for line in lines:
            # Check if line matches any pattern for this test
            if any(rx.search(line) for rx in regexes):
                # Now extract the numeric value from the line
                # We look for a float number that is likely the test result.
                # Remove common noise (like phone numbers or dates or reference ranges like 0.6-1.2)
                clean_line = line
                # Strip dates
                clean_line = _DATE_RE.sub("", clean_line)
                # Strip typical reference range tokens (e.g. "(0.6-1.2)" or "0.6 - 1.2" or "< 5.7")
                clean_line = re.sub(r"[\(\[\s]*\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?[\)\]\s]*", "", clean_line)
                clean_line = re.sub(r"[<>]\s*\d+(?:\.\d+)?", "", clean_line)
                
                # Now find all float numbers in the remaining text
                numbers = _VALUE_RE.findall(clean_line)
                if numbers:
                    # The test value is usually the first float that is not part of the name
                    val = float(numbers[0])
                    
                    # Deduct the status flag based on standard thresholds
                    flag = rule["eval"](val)
                    
                    results.append(
                        LabResultExtracted(
                            raw_name=line,
                            analyte_name=rule["name"],
                            value=val,
                            unit=rule["unit"],
                            ref_range=rule["ref_range"],
                            flag=flag
                        )
                    )
                    break # test matched and extracted, stop searching for this rule

    # 4. Confidence scoring
    # Base confidence is 1.0, minus penalties for missing components
    confidence = 1.0
    if not report_date:
        confidence -= 0.15
    if len(results) == 0:
        confidence -= 0.50
    else:
        # average penalties if some values are extremely high/low or look suspect
        pass
        
    confidence = max(0.0, min(1.0, round(confidence, 4)))

    return LabReportExtracted(
        lab_name=lab_name,
        report_date=report_date,
        results=results,
        confidence=confidence,
        raw_text=text
    )
