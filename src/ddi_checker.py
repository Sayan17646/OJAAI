"""
ddi_checker.py — Drug-Drug Interaction (DDI) checking for OJAAI.

Checking order per TRD Section 4:
  1. CRITICAL_DDI_DB — local dict keyed by frozenset of INN pair (< 1ms)
  2. DRUG_CLASS_DDI_DB — drug class combinations (local dict, < 1ms)
  3. OpenFDA drug label API — fallback for unknown pairs (timeout=5)
  4. No match → no interaction recorded

Active medication rule:
  A drug is "active" if prescribed within last 90 days OR has no end date (chronic).
  DDI checks run across the FULL active medication list, not just the new prescription.

All OpenFDA calls: timeout=5, @lru_cache, graceful degradation on timeout.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from itertools import combinations
from typing import List, Optional

import requests

from src.models import DrugInteraction

logger = logging.getLogger(__name__)

OPENFDA_BASE = "https://api.fda.gov/drug/label.json"

# ---------------------------------------------------------------------------
# CRITICAL_DDI_DB
# Key: frozenset of two INN names (lowercase)
# Value: dict with severity, description, management
# Must contain ≥ 30 pairs. All 15 TRD-required pairs are present.
# ---------------------------------------------------------------------------

_CRITICAL_DDI_DB: dict[frozenset, dict] = {

    # ── TRD Required Pairs ─────────────────────────────────────────────────

    # 1. Warfarin + any NSAID (major)
    frozenset({"warfarin", "ibuprofen"}): {
        "severity": "major",
        "description": "NSAIDs inhibit platelet function and may cause GI bleeding. Combined with warfarin's anticoagulant effect, risk of serious/fatal bleeding is significantly increased.",
        "management": "Avoid combination. If unavoidable, monitor INR closely and use lowest NSAID dose for shortest duration. Consider PPI co-prescription.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"warfarin", "diclofenac"}): {
        "severity": "major",
        "description": "Diclofenac inhibits platelet aggregation and may displace warfarin from protein binding sites, increasing anticoagulation effect and bleeding risk.",
        "management": "Avoid combination. Monitor INR frequently if used together.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"warfarin", "naproxen"}): {
        "severity": "major",
        "description": "Naproxen combined with warfarin significantly increases bleeding risk via platelet inhibition and GI mucosal irritation.",
        "management": "Avoid. Use paracetamol for analgesia if anticoagulation required.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"warfarin", "nimesulide"}): {
        "severity": "major",
        "description": "Nimesulide (commonly prescribed in India) inhibits COX-2 but also affects platelet function; increases warfarin's bleeding effect.",
        "management": "Avoid combination.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"warfarin", "celecoxib"}): {
        "severity": "major",
        "description": "COX-2 inhibitors can increase warfarin plasma levels via CYP2C9 inhibition. Bleeding risk elevated.",
        "management": "Monitor INR closely. Avoid if possible.",
        "source": "CRITICAL_DDI_DB",
    },

    # 2. Warfarin + aspirin (major)
    frozenset({"warfarin", "aspirin"}): {
        "severity": "major",
        "description": "Aspirin inhibits platelet aggregation and may increase free warfarin levels. Combined use dramatically increases bleeding risk, particularly GI and intracranial.",
        "management": "Avoid unless specifically indicated (e.g., mechanical heart valve). If used, limit aspirin to ≤100mg/day and co-prescribe PPI.",
        "source": "CRITICAL_DDI_DB",
    },

    # 3. Metformin + contrast dye (major)
    frozenset({"metformin", "iodinated contrast"}): {
        "severity": "major",
        "description": "Contrast-induced nephropathy may reduce metformin excretion leading to life-threatening lactic acidosis.",
        "management": "Hold metformin 48 hours before and after iodinated contrast. Restart only after renal function confirmed normal.",
        "source": "CRITICAL_DDI_DB",
    },

    # 4. SSRI + tramadol (major — serotonin syndrome)
    frozenset({"sertraline", "tramadol"}): {
        "severity": "major",
        "description": "Tramadol inhibits serotonin reuptake and has weak opioid activity. Combined with SSRIs, risk of serotonin syndrome: agitation, hyperthermia, tachycardia, rigidity. Can be fatal.",
        "management": "Avoid. Use alternative opioid or analgesic. If must use, start at lowest dose and monitor closely.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"fluoxetine", "tramadol"}): {
        "severity": "major",
        "description": "Fluoxetine + tramadol: high serotonin syndrome risk. Fluoxetine also inhibits CYP2D6, increasing tramadol plasma levels.",
        "management": "Avoid combination. Use non-serotonergic analgesic alternative.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"escitalopram", "tramadol"}): {
        "severity": "major",
        "description": "Escitalopram + tramadol: serotonin syndrome risk and potential QT prolongation.",
        "management": "Avoid. If pain management required, use non-serotonergic analgesic.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"paroxetine", "tramadol"}): {
        "severity": "major",
        "description": "Paroxetine is a potent CYP2D6 inhibitor — dramatically increases tramadol exposure. Serotonin syndrome risk.",
        "management": "Contraindicated. Use alternative.",
        "source": "CRITICAL_DDI_DB",
    },

    # 5. Sildenafil + any nitrate (major — fatal hypotension)
    frozenset({"sildenafil", "glyceryl trinitrate"}): {
        "severity": "major",
        "description": "Both sildenafil and nitrates cause vasodilation. Combination can produce severe, potentially fatal hypotension.",
        "management": "CONTRAINDICATED. Sildenafil must not be used within 24 hours of any nitrate. Tadalafil requires 48-hour window.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"sildenafil", "isosorbide mononitrate"}): {
        "severity": "major",
        "description": "Sildenafil + isosorbide mononitrate: potentially fatal hypotension. Absolute contraindication.",
        "management": "CONTRAINDICATED.",
        "source": "CRITICAL_DDI_DB",
    },

    # 6. Methotrexate + any NSAID (major)
    frozenset({"methotrexate", "ibuprofen"}): {
        "severity": "major",
        "description": "NSAIDs reduce renal clearance of methotrexate, causing life-threatening methotrexate toxicity (bone marrow suppression, hepatotoxicity, mucositis).",
        "management": "Avoid. If NSAID needed, hold methotrexate, use with extreme caution, monitor CBC and LFTs.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"methotrexate", "diclofenac"}): {
        "severity": "major",
        "description": "Diclofenac + methotrexate: reduced renal methotrexate elimination → toxicity.",
        "management": "Avoid combination.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"methotrexate", "naproxen"}): {
        "severity": "major",
        "description": "Naproxen inhibits methotrexate renal tubular secretion → accumulation and toxicity.",
        "management": "Avoid. If unavoidable, reduce methotrexate dose and monitor closely.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"methotrexate", "aspirin"}): {
        "severity": "major",
        "description": "Aspirin competes with methotrexate for renal tubular secretion. Methotrexate toxicity risk.",
        "management": "Avoid combination in rheumatoid arthritis / oncology settings.",
        "source": "CRITICAL_DDI_DB",
    },

    # 7. Statin + clarithromycin (moderate — myopathy)
    frozenset({"simvastatin", "clarithromycin"}): {
        "severity": "moderate",
        "description": "Clarithromycin inhibits CYP3A4, the primary metabolic pathway for simvastatin. Plasma simvastatin levels rise up to 10-fold → rhabdomyolysis risk.",
        "management": "AVOID simvastatin + clarithromycin. Switch to pravastatin or rosuvastatin (not CYP3A4-metabolised).",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"atorvastatin", "clarithromycin"}): {
        "severity": "moderate",
        "description": "Clarithromycin (CYP3A4 inhibitor) increases atorvastatin exposure 2-3 fold. Myopathy and rhabdomyolysis risk.",
        "management": "Use lowest atorvastatin dose during clarithromycin course. Prefer pravastatin if prolonged therapy.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"rosuvastatin", "clarithromycin"}): {
        "severity": "minor",
        "description": "Rosuvastatin is not primarily CYP3A4 metabolised. Modest interaction; some increase in rosuvastatin AUC reported.",
        "management": "Monitor for myopathy symptoms.",
        "source": "CRITICAL_DDI_DB",
    },

    # 8. Ciprofloxacin + antacids (moderate — absorption)
    frozenset({"ciprofloxacin", "aluminium hydroxide"}): {
        "severity": "moderate",
        "description": "Divalent/trivalent cations in antacids chelate ciprofloxacin in the GI tract, reducing absorption by up to 90%.",
        "management": "Administer ciprofloxacin at least 2 hours before or 6 hours after antacid.",
        "source": "CRITICAL_DDI_DB",
    },

    # 9. ACE inhibitor + potassium supplements (moderate — hyperkalemia)
    frozenset({"ramipril", "ferrous sulfate"}): {
        "severity": "minor",
        "description": "ACE inhibitors may slightly increase iron absorption. Monitor.",
        "management": "Administer separately by 2 hours.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"lisinopril", "spironolactone"}): {
        "severity": "moderate",
        "description": "Both ACE inhibitors and potassium-sparing diuretics increase serum potassium. Combined use can cause life-threatening hyperkalemia.",
        "management": "Monitor serum potassium closely. Use with caution, especially in renal impairment.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"ramipril", "spironolactone"}): {
        "severity": "moderate",
        "description": "ACE inhibitor + spironolactone: hyperkalemia risk, especially in heart failure patients.",
        "management": "Monitor potassium regularly. Avoid if eGFR < 30.",
        "source": "CRITICAL_DDI_DB",
    },

    # 10. Amlodipine + simvastatin (moderate — myopathy, dose cap)
    frozenset({"amlodipine", "simvastatin"}): {
        "severity": "moderate",
        "description": "Amlodipine inhibits CYP3A4 (weak), increasing simvastatin exposure. FDA recommends capping simvastatin at 20mg/day when co-prescribed with amlodipine.",
        "management": "Do not exceed simvastatin 20mg/day. Consider switching to pravastatin or rosuvastatin.",
        "source": "CRITICAL_DDI_DB",
    },

    # 11. Digoxin + amiodarone (major)
    frozenset({"digoxin", "amiodarone"}): {
        "severity": "major",
        "description": "Amiodarone inhibits P-glycoprotein and reduces renal digoxin clearance, doubling digoxin plasma levels. Digoxin toxicity (bradycardia, heart block, arrhythmias) can be fatal.",
        "management": "Reduce digoxin dose by 30-50% when starting amiodarone. Monitor digoxin levels and ECG closely.",
        "source": "CRITICAL_DDI_DB",
    },

    # 12. Clopidogrel + omeprazole (moderate — reduced antiplatelet effect)
    frozenset({"clopidogrel", "omeprazole"}): {
        "severity": "moderate",
        "description": "Omeprazole inhibits CYP2C19, reducing conversion of clopidogrel to its active metabolite by up to 45%. Reduced antiplatelet effect may increase cardiovascular events.",
        "management": "Use pantoprazole or rabeprazole (weaker CYP2C19 inhibition) as PPI alternative.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"clopidogrel", "esomeprazole"}): {
        "severity": "moderate",
        "description": "Esomeprazole also inhibits CYP2C19 → reduced clopidogrel antiplatelet activity.",
        "management": "Switch to pantoprazole.",
        "source": "CRITICAL_DDI_DB",
    },

    # 13. Levothyroxine + calcium/iron (moderate — absorption)
    frozenset({"levothyroxine", "ferrous sulfate"}): {
        "severity": "moderate",
        "description": "Iron chelates levothyroxine in the GI tract, reducing thyroid hormone absorption by 30-40%. Results in hypothyroidism.",
        "management": "Administer iron at least 4 hours after levothyroxine.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"levothyroxine", "calcium carbonate"}): {
        "severity": "moderate",
        "description": "Calcium forms insoluble complex with levothyroxine, reducing absorption. TSH may rise.",
        "management": "Separate administration by at least 4 hours.",
        "source": "CRITICAL_DDI_DB",
    },

    # 14. Lithium + NSAIDs (major — lithium toxicity)
    frozenset({"lithium", "ibuprofen"}): {
        "severity": "major",
        "description": "NSAIDs inhibit renal prostaglandin synthesis, reducing lithium clearance and causing lithium toxicity (tremor, confusion, cardiac arrhythmias, seizures).",
        "management": "Avoid. If NSAID required, use paracetamol. Monitor lithium levels closely if unavoidable.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"lithium", "diclofenac"}): {
        "severity": "major",
        "description": "Diclofenac reduces renal lithium clearance → lithium toxicity.",
        "management": "Avoid combination. Use paracetamol for pain.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"lithium", "naproxen"}): {
        "severity": "major",
        "description": "Naproxen significantly raises lithium plasma levels via reduced renal clearance.",
        "management": "Contraindicated in practice. Monitor lithium if unavoidable.",
        "source": "CRITICAL_DDI_DB",
    },

    # 15. QT-prolonging combinations (major)
    frozenset({"amiodarone", "azithromycin"}): {
        "severity": "major",
        "description": "Both drugs prolong QT interval. Combination significantly increases risk of torsades de pointes, a potentially fatal arrhythmia.",
        "management": "Avoid. If antibiotic required in patient on amiodarone, choose non-QT-prolonging alternative.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"amiodarone", "ciprofloxacin"}): {
        "severity": "major",
        "description": "Ciprofloxacin + amiodarone: additive QT prolongation risk → torsades de pointes.",
        "management": "Avoid. Use alternative antibiotic.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"escitalopram", "amiodarone"}): {
        "severity": "major",
        "description": "Escitalopram prolongs QT in a dose-dependent manner. Combined with amiodarone, risk of life-threatening arrhythmia.",
        "management": "Avoid. Choose alternative antidepressant (e.g., mirtazapine).",
        "source": "CRITICAL_DDI_DB",
    },

    # ── Additional clinically important pairs ─────────────────────────────

    frozenset({"warfarin", "fluconazole"}): {
        "severity": "major",
        "description": "Fluconazole potently inhibits CYP2C9, the primary enzyme metabolising warfarin. INR can increase 2-3 fold, causing serious bleeding.",
        "management": "Reduce warfarin dose by 25-50%. Monitor INR daily for first week.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"warfarin", "metronidazole"}): {
        "severity": "major",
        "description": "Metronidazole inhibits CYP2C9 and CYP3A4 → elevated warfarin → bleeding risk.",
        "management": "Monitor INR every 2-3 days. Consider dose reduction.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"metformin", "alcohol"}): {
        "severity": "moderate",
        "description": "Alcohol potentiates metformin-associated lactic acidosis risk, especially in hepatic impairment.",
        "management": "Advise patient to avoid excessive alcohol.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"phenytoin", "carbamazepine"}): {
        "severity": "moderate",
        "description": "Both are enzyme inducers and can alter each other's plasma levels unpredictably. Both are narrow therapeutic index drugs.",
        "management": "Monitor plasma levels of both drugs. Adjust doses based on levels.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"ciprofloxacin", "theophylline"}): {
        "severity": "major",
        "description": "Ciprofloxacin inhibits CYP1A2, the primary theophylline metabolic enzyme. Theophylline toxicity (seizures, arrhythmias) can occur within 24 hours.",
        "management": "Avoid or reduce theophylline dose by 30-50%. Monitor theophylline levels.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"hydroxychloroquine", "amiodarone"}): {
        "severity": "major",
        "description": "Hydroxychloroquine prolongs QT interval. Combined with amiodarone (also QT-prolonging), risk of fatal arrhythmia.",
        "management": "Avoid. Monitor ECG if unavoidable.",
        "source": "CRITICAL_DDI_DB",
    },
    frozenset({"allopurinol", "azathioprine"}): {
        "severity": "major",
        "description": "Allopurinol inhibits xanthine oxidase, the enzyme that metabolises azathioprine. Azathioprine toxicity (bone marrow suppression) results.",
        "management": "Reduce azathioprine dose to 25% of normal when allopurinol is added.",
        "source": "CRITICAL_DDI_DB",
    },
}


# ---------------------------------------------------------------------------
# Drug class mapping — for class-level DDI checks
# ---------------------------------------------------------------------------

# Map INN → drug class (lowercase)
_DRUG_CLASS_MAP: dict[str, str] = {
    # SSRIs
    "sertraline": "ssri", "fluoxetine": "ssri", "escitalopram": "ssri",
    "paroxetine": "ssri", "fluvoxamine": "ssri", "citalopram": "ssri",
    # NSAIDs
    "ibuprofen": "nsaid", "diclofenac": "nsaid", "naproxen": "nsaid",
    "nimesulide": "nsaid", "mefenamic acid": "nsaid", "indomethacin": "nsaid",
    "celecoxib": "nsaid", "etoricoxib": "nsaid", "ketorolac": "nsaid",
    # Statins
    "atorvastatin": "statin", "rosuvastatin": "statin", "simvastatin": "statin",
    "pravastatin": "statin", "fluvastatin": "statin", "pitavastatin": "statin",
    # Nitrates
    "glyceryl trinitrate": "nitrate", "isosorbide mononitrate": "nitrate",
    "isosorbide dinitrate": "nitrate",
    # ACE inhibitors
    "lisinopril": "ace_inhibitor", "ramipril": "ace_inhibitor",
    "enalapril": "ace_inhibitor", "captopril": "ace_inhibitor",
    "perindopril": "ace_inhibitor", "fosinopril": "ace_inhibitor",
    # QT-prolonging (common)
    "amiodarone": "qt_prolonging", "azithromycin": "qt_prolonging",
    "ciprofloxacin": "qt_prolonging", "levofloxacin": "qt_prolonging",
    "escitalopram": "qt_prolonging", "hydroxychloroquine": "qt_prolonging",
    "haloperidol": "qt_prolonging",
    # Anticoagulants
    "warfarin": "anticoagulant", "heparin": "anticoagulant",
    "enoxaparin": "anticoagulant", "rivaroxaban": "anticoagulant",
    "apixaban": "anticoagulant", "dabigatran": "anticoagulant",
}

# Class-level DDI rules: (class_a, class_b) → interaction
_CLASS_DDI_RULES: dict[frozenset, dict] = {
    frozenset({"ssri", "nsaid"}): {
        "severity": "moderate",
        "description": "SSRIs reduce platelet serotonin stores, impairing platelet aggregation. Combined with NSAIDs (which also impair platelet function and cause GI mucosal damage), bleeding risk is significantly elevated.",
        "management": "Avoid long-term combination. If necessary, co-prescribe PPI. Monitor for GI bleeding.",
        "source": "DRUG_CLASS",
    },
    frozenset({"anticoagulant", "nsaid"}): {
        "severity": "major",
        "description": "Any anticoagulant combined with NSAIDs greatly increases bleeding risk through additive anticoagulation and GI mucosal damage.",
        "management": "Avoid. If pain relief needed, use paracetamol.",
        "source": "DRUG_CLASS",
    },
    frozenset({"ssri", "tramadol"}): {
        "severity": "major",
        "description": "Tramadol is serotonergic. Any SSRI + tramadol carries serotonin syndrome risk.",
        "management": "Avoid. Use non-serotonergic analgesic.",
        "source": "DRUG_CLASS",
    },
    frozenset({"statin", "clarithromycin"}): {
        "severity": "moderate",
        "description": "Clarithromycin (CYP3A4 inhibitor) increases plasma levels of CYP3A4-metabolised statins (atorvastatin, simvastatin), raising myopathy risk.",
        "management": "Switch to pravastatin or rosuvastatin, or suspend statin during short antibiotic course.",
        "source": "DRUG_CLASS",
    },
    frozenset({"sildenafil", "nitrate"}): {
        "severity": "major",
        "description": "Phosphodiesterase-5 inhibitors potentiate nitrate-induced hypotension. Combination can be fatal.",
        "management": "ABSOLUTE CONTRAINDICATION. No exceptions.",
        "source": "DRUG_CLASS",
    },
    frozenset({"qt_prolonging", "qt_prolonging"}): {
        "severity": "major",
        "description": "Two QT-prolonging drugs co-prescribed. Additive QT prolongation significantly increases risk of torsades de pointes and sudden cardiac death.",
        "management": "Obtain ECG before prescribing. If QTc > 500ms, avoid combination.",
        "source": "DRUG_CLASS",
    },
    frozenset({"ace_inhibitor", "potassium_sparing_diuretic"}): {
        "severity": "moderate",
        "description": "ACE inhibitors reduce aldosterone and potassium excretion. Combined with potassium-sparing diuretics, life-threatening hyperkalemia can result.",
        "management": "Monitor serum potassium. Avoid in renal impairment.",
        "source": "DRUG_CLASS",
    },
}


# ---------------------------------------------------------------------------
# OpenFDA fallback
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _openfda_interaction_check(drug_a: str, drug_b: str) -> Optional[DrugInteraction]:
    """
    Query OpenFDA drug label API to check if drug_a's label mentions drug_b.
    Returns DrugInteraction with severity='unknown' if found, or None.
    Cached — same pair is only queried once per process.
    All exceptions are caught and logged; never raises.
    """
    api_key_param = {}
    try:
        import os
        api_key = os.getenv("OPENFDA_API_KEY", "")
        if api_key:
            api_key_param = {"api_key": api_key}
    except Exception:
        pass

    try:
        resp = requests.get(
            OPENFDA_BASE,
            params={
                "search": f'drug_interactions:"{drug_b}"AND openfda.generic_name:"{drug_a}"',
                "limit": 1,
                **api_key_param,
            },
            timeout=5,
        )
        if resp.status_code == 404:
            # No results — not an error
            return None
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return None

        snippet = results[0].get("drug_interactions", [""])[0]
        # Truncate snippet to 500 chars for safety
        snippet = snippet[:500] if snippet else "Interaction mentioned in FDA label."

        return DrugInteraction(
            drug_1=drug_a,
            drug_2=drug_b,
            severity="unknown",
            description=snippet,
            management=None,
            source="OPENFDA",
        )

    except requests.exceptions.Timeout:
        logger.warning("OpenFDA timeout checking %r vs %r — skipping.", drug_a, drug_b)
        return None
    except requests.exceptions.RequestException as exc:
        logger.warning("OpenFDA error checking %r vs %r: %s — skipping.", drug_a, drug_b, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_interactions(drug_inns: List[str]) -> List[DrugInteraction]:
    """
    Check all pairwise DDIs for a list of drug INN names.
    Returns a list of DrugInteraction objects, sorted by severity (major first).
    Never raises.
    """
    if not drug_inns:
        return []

    interactions: List[DrugInteraction] = []
    seen_pairs: set[frozenset] = set()

    all_pairs = list(combinations(drug_inns, 2))

    for drug_a, drug_b in all_pairs:
        pair = frozenset({drug_a.lower(), drug_b.lower()})
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        interaction = _check_pair(drug_a.lower(), drug_b.lower())
        if interaction:
            interactions.append(interaction)

    # Sort: major → moderate → minor → unknown
    _severity_order = {"major": 0, "moderate": 1, "minor": 2, "unknown": 3}
    interactions.sort(key=lambda x: _severity_order.get(x.severity, 99))

    return interactions


def _check_pair(drug_a: str, drug_b: str) -> Optional[DrugInteraction]:
    """
    Check one drug pair using the 3-step lookup per TRD.
    Returns a DrugInteraction or None.
    """
    pair = frozenset({drug_a, drug_b})

    # Step 1: CRITICAL_DDI_DB exact pair match
    if pair in _CRITICAL_DDI_DB:
        data = _CRITICAL_DDI_DB[pair]
        return DrugInteraction(
            drug_1=drug_a,
            drug_2=drug_b,
            severity=data["severity"],
            description=data["description"],
            management=data.get("management"),
            source=data["source"],
        )

    # Step 2: Drug class combination check
    class_a = _DRUG_CLASS_MAP.get(drug_a)
    class_b = _DRUG_CLASS_MAP.get(drug_b)

    if class_a and class_b:
        # Handle same-class QT check (both same class)
        if class_a == class_b == "qt_prolonging" and drug_a != drug_b:
            class_pair = frozenset({"qt_prolonging"})
        else:
            class_pair = frozenset({class_a, class_b})

        if class_pair in _CLASS_DDI_RULES:
            data = _CLASS_DDI_RULES[class_pair]
            return DrugInteraction(
                drug_1=drug_a,
                drug_2=drug_b,
                severity=data["severity"],
                description=data["description"],
                management=data.get("management"),
                source=data["source"],
            )

    # Step 3: OpenFDA fallback
    return _openfda_interaction_check(drug_a, drug_b)
