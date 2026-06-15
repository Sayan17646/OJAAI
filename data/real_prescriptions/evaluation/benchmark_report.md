# OJAAI Pipeline Benchmark Evaluation Report

This report presents a comparative evaluation of the **Tesseract OCR + Regex NER Pipeline** against the **Donut ML Model** on the real-world evaluation dataset.

## Summary Metrics

| Metric | Tesseract OCR Pipeline | Donut ML Model |
| :--- | :---: | :---: |
| **Total Prescriptions Evaluated** | 1 | 1 |
| **Average Latency per Image** | 49.42s | 43.55s |
| **Medication Precision** | 0.00% | 0.00% |
| **Medication Recall** | 0.00% | 0.00% |
| **Medication F1-Score** | 0.00% | 0.00% |

---

## Field-by-Field Clinical Extraction Accuracy

### Tesseract OCR Pipeline
| Clinical Field | Total Present | Strict Accuracy | Fuzzy Accuracy (Levenshtein >= 80%) |
| :--- | :---: | :---: | :---: |
| **doctor_name** | 1 | 0.00% | 0.00% |
| **doctor_reg** | 1 | 0.00% | 0.00% |
| **patient_name** | 1 | 0.00% | 0.00% |
| **patient_age** | 1 | 0.00% | 0.00% |
| **prescription_date** | 0 | 0.00% | 0.00% |

### Donut ML Model
| Clinical Field | Total Present | Strict Accuracy | Fuzzy Accuracy (Levenshtein >= 80%) |
| :--- | :---: | :---: | :---: |
| **doctor_name** | 1 | 0.00% | 0.00% |
| **doctor_reg** | 1 | 0.00% | 0.00% |
| **patient_name** | 1 | 0.00% | 0.00% |
| **patient_age** | 1 | 0.00% | 0.00% |
| **prescription_date** | 0 | 0.00% | 0.00% |

---

## Deployment Recommendation

Recommendation: Donut ML must remain as Tier 3 Fallback. The Tesseract OCR + Regex pipeline maintains higher reliability/F1-score, or Donut's improvement margin does not yet justify the latency cost.

*Report generated at: 2026-06-10 16:27:47*
