"""
drug_normalizer.py — Drug name normalisation for OJAAI.

Pipeline (Phase 2 upgrade):
  1. Lowercase + strip
  2. brand_db.lookup_brand() — SQLite + Levenshtein fuzzy match (India-first)
     - Pass 1: exact match (O(1) indexed)
     - Pass 2: fuzzy match (edit distance ≤ 2) — recovers OCR typos
  3. Query RxNorm API (NIH, free, ~200ms)
  4. Degrade gracefully on RxNorm failure

INDIA_BRAND_MAP is the source-of-truth dict used to seed the SQLite DB.
The DB is auto-built at ./data/brand_dict.db on first startup.
RxNorm lookups are cached with @lru_cache.
All HTTP calls have timeout=5.

Coverage: 300+ entries across 14 therapeutic categories:
  Antidiabetics, Antihypertensives, Statins, Antibiotics, Thyroid,
  PPIs/GI, Analgesics/NSAIDs, Anticoagulants/Antiplatelets, Vitamins,
  Bronchodilators, Corticosteroids, Antidepressants/Psych,
  Cardiac, TB/Anti-Mycobacterial, ARVs, and common FDCs.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import List, Optional

import requests

from src.brand_db import lookup_brand
from src.models import MedicationExtracted, NormalizedDrug

logger = logging.getLogger(__name__)

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"

# ---------------------------------------------------------------------------
# INDIA_BRAND_MAP
# Brand name → INN (International Nonproprietary Name)
# Coverage: ≥200 entries across all 9 required therapeutic categories.
# Keys are lowercase. Values are lowercase INN.
# ---------------------------------------------------------------------------

INDIA_BRAND_MAP: dict[str, str] = {

    # ── Antidiabetics ──────────────────────────────────────────────────────
    # Metformin brands
    "glycomet": "metformin", "glycomet sr": "metformin", "glycomet gp": "metformin + glipizide",
    "glucophage": "metformin", "obimet": "metformin", "carbophage": "metformin",
    "cetapin": "metformin", "diabon": "metformin", "bigomet": "metformin",
    "glucored": "metformin + glibenclamide", "gluconorm g": "metformin + glipizide",

    # Sulfonylureas
    "daonil": "glibenclamide", "glynase": "glibenclamide", "semi-daonil": "glibenclamide",
    "glipizide": "glipizide", "minodiab": "glipizide",
    "amaryl": "glimepiride", "glimir": "glimepiride", "glimy": "glimepiride",
    "glycoday": "glimepiride", "glimisave": "glimepiride",
    "reclide": "gliclazide", "diamicron": "gliclazide", "glycinorm": "gliclazide",

    # DPP-4 inhibitors
    "januvia": "sitagliptin", "istavel": "sitagliptin",
    "galvus": "vildagliptin", "jalra": "vildagliptin",
    "onglyza": "saxagliptin", "kombiglyze": "saxagliptin + metformin",
    "tradjenta": "linagliptin", "trajenta": "linagliptin",
    "nesina": "alogliptin",

    # SGLT2 inhibitors
    "jardiance": "empagliflozin", "forxiga": "dapagliflozin",
    "farxiga": "dapagliflozin", "invokana": "canagliflozin", "sugarfree": "dapagliflozin",

    # Insulins
    "mixtard": "insulin (biphasic isophane)", "insulatard": "insulin isophane",
    "actrapid": "insulin soluble", "monotard": "insulin zinc",
    "lantus": "insulin glargine", "levemir": "insulin detemir",
    "novorapid": "insulin aspart", "humalog": "insulin lispro",
    "toujeo": "insulin glargine", "tresiba": "insulin degludec",
    "basalin": "insulin glargine", "wosulin": "insulin",

    # ── Antihypertensives ─────────────────────────────────────────────────
    # CCBs
    "stamlo": "amlodipine", "amlokind": "amlodipine", "amlovas": "amlodipine",
    "amlong": "amlodipine", "amlip": "amlodipine", "norvasc": "amlodipine",
    "nicardia": "nifedipine", "adalat": "nifedipine",
    "felodac": "felodipine", "plendil": "felodipine",
    "tazloc": "telmisartan + amlodipine",

    # ACE inhibitors
    "aceten": "captopril", "capoten": "captopril",
    "enalapril": "enalapril", "envas": "enalapril", "enam": "enalapril",
    "lisinopril": "lisinopril", "listril": "lisinopril", "zestril": "lisinopril",
    "ramipril": "ramipril", "cardace": "ramipril", "ramace": "ramipril",
    "perindopril": "perindopril", "coversyl": "perindopril",
    "fosinopril": "fosinopril", "quinapril": "quinapril",

    # ARBs
    "losartan": "losartan", "losium": "losartan", "tozaar": "losartan",
    "repace": "losartan", "covance": "losartan",
    "telmisartan": "telmisartan", "telma": "telmisartan", "telsartan": "telmisartan",
    "olmesartan": "olmesartan", "olsar": "olmesartan", "benicar": "olmesartan",
    "valsartan": "valsartan", "valtan": "valsartan", "diovan": "valsartan",
    "irbesartan": "irbesartan", "avapro": "irbesartan",
    "candesartan": "candesartan", "blopress": "candesartan",

    # Beta-blockers
    "atenolol": "atenolol", "tenormin": "atenolol", "aten": "atenolol",
    "metoprolol": "metoprolol", "metolar": "metoprolol", "betaloc": "metoprolol",
    "propranolol": "propranolol", "inderal": "propranolol",
    "bisoprolol": "bisoprolol", "concor": "bisoprolol", "corbis": "bisoprolol",
    "nebivolol": "nebivolol", "nebicard": "nebivolol", "nodon": "nebivolol",
    "carvedilol": "carvedilol", "cardivas": "carvedilol",

    # Diuretics
    "furosemide": "furosemide", "lasix": "furosemide", "frusenex": "furosemide",
    "hydrochlorothiazide": "hydrochlorothiazide", "hctz": "hydrochlorothiazide",
    "spironolactone": "spironolactone", "aldactone": "spironolactone",
    "torsemide": "torsemide", "dytor": "torsemide",
    "indapamide": "indapamide", "lorvas": "indapamide",
    "chlorthalidone": "chlorthalidone",

    # Alpha blockers
    "prazosin": "prazosin", "minipress": "prazosin",
    "doxazosin": "doxazosin", "doxacard": "doxazosin",

    # ── Statins ───────────────────────────────────────────────────────────
    "atorvastatin": "atorvastatin", "storvas": "atorvastatin", "atorva": "atorvastatin",
    "lipitor": "atorvastatin", "lipicure": "atorvastatin", "aztor": "atorvastatin",
    "rosuvastatin": "rosuvastatin", "rozavel": "rosuvastatin", "rosuvas": "rosuvastatin",
    "crestor": "rosuvastatin", "rosulip": "rosuvastatin",
    "simvastatin": "simvastatin", "zocor": "simvastatin", "simcard": "simvastatin",
    "pravastatin": "pravastatin", "pravachol": "pravastatin",
    "lovastatin": "lovastatin", "mevacor": "lovastatin",
    "pitavastatin": "pitavastatin",
    "fluvastatin": "fluvastatin",

    # ── Antibiotics ───────────────────────────────────────────────────────
    # Penicillins
    "amoxicillin": "amoxicillin", "mox": "amoxicillin", "novamox": "amoxicillin",
    "amoxyclav": "amoxicillin + clavulanate", "augmentin": "amoxicillin + clavulanate",
    "clavam": "amoxicillin + clavulanate", "ciplox": "ciprofloxacin",

    # Cephalosporins
    "cefalexin": "cefalexin", "sporidex": "cefalexin", "reflin": "cefalexin",
    "cefixime": "cefixime", "zifi": "cefixime", "taxim": "cefixime",
    "cefpodoxime": "cefpodoxime", "cepodem": "cefpodoxime",
    "ceftriaxone": "ceftriaxone", "monocef": "ceftriaxone", "oframax": "ceftriaxone",

    # Fluoroquinolones
    "ciprofloxacin": "ciprofloxacin", "cifran": "ciprofloxacin",
    "levofloxacin": "levofloxacin", "levomac": "levofloxacin", "lox": "levofloxacin",
    "ofloxacin": "ofloxacin", "zanocin": "ofloxacin",
    "norfloxacin": "norfloxacin", "norflox": "norfloxacin",
    "moxifloxacin": "moxifloxacin", "avelox": "moxifloxacin",

    # Macrolides
    "azithromycin": "azithromycin", "azithral": "azithromycin",
    "zithromax": "azithromycin", "azee": "azithromycin",
    "clarithromycin": "clarithromycin", "claribid": "clarithromycin",
    "erythromycin": "erythromycin",

    # Tetracyclines
    "doxycycline": "doxycycline", "doxinate": "doxycycline", "biodoxi": "doxycycline",
    "tetracycline": "tetracycline",

    # Nitroimidazoles
    "metronidazole": "metronidazole", "flagyl": "metronidazole", "metrogyl": "metronidazole",
    "tinidazole": "tinidazole", "fasigyn": "tinidazole",

    # Antivirals
    "acyclovir": "acyclovir", "zovirax": "acyclovir",
    "oseltamivir": "oseltamivir", "tamiflu": "oseltamivir",

    # Antifungals
    "fluconazole": "fluconazole", "forcan": "fluconazole", "zocon": "fluconazole",
    "itraconazole": "itraconazole", "canditral": "itraconazole",
    "clotrimazole": "clotrimazole",

    # ── Thyroid ───────────────────────────────────────────────────────────
    "levothyroxine": "levothyroxine", "eltroxin": "levothyroxine",
    "thyronorm": "levothyroxine", "thyrox": "levothyroxine",
    "liothyronine": "liothyronine", "cytomel": "liothyronine",

    # ── PPIs / GI ─────────────────────────────────────────────────────────
    "pantoprazole": "pantoprazole", "pan": "pantoprazole", "pantocid": "pantoprazole",
    "protonix": "pantoprazole", "pantodac": "pantoprazole",
    "omeprazole": "omeprazole", "omez": "omeprazole", "prilosec": "omeprazole",
    "esomeprazole": "esomeprazole", "nexium": "esomeprazole", "nexpro": "esomeprazole",
    "rabeprazole": "rabeprazole", "razo": "rabeprazole", "rablet": "rabeprazole",
    "lansoprazole": "lansoprazole", "lanzol": "lansoprazole",
    "domperidone": "domperidone", "domstal": "domperidone", "vomitab": "domperidone",
    "ondansetron": "ondansetron", "emeset": "ondansetron", "ondem": "ondansetron",
    "metoclopramide": "metoclopramide", "perinorm": "metoclopramide",
    "ranitidine": "ranitidine", "rantac": "ranitidine", "aciloc": "ranitidine",

    # ── Analgesics / NSAIDs ───────────────────────────────────────────────
    "paracetamol": "paracetamol", "crocin": "paracetamol", "dolo": "paracetamol",
    "calpol": "paracetamol", "tylenol": "paracetamol", "metacin": "paracetamol",
    # US name — alias to paracetamol INN
    "acetaminophen": "paracetamol",
    "ibuprofen": "ibuprofen", "brufen": "ibuprofen", "combiflam": "ibuprofen + paracetamol",
    "advil": "ibuprofen", "nurofen": "ibuprofen",
    "diclofenac": "diclofenac", "voveran": "diclofenac", "volini": "diclofenac",
    "reactine": "diclofenac",
    "naproxen": "naproxen", "naprosyn": "naproxen", "proxen": "naproxen",
    "nimesulide": "nimesulide", "nimulid": "nimesulide", "nise": "nimesulide",
    "mefenamic acid": "mefenamic acid", "meftal": "mefenamic acid",
    "indomethacin": "indomethacin", "indocap": "indomethacin",
    "celecoxib": "celecoxib", "celebrex": "celecoxib",
    "etoricoxib": "etoricoxib", "arcoxia": "etoricoxib",
    "tramadol": "tramadol", "ultracet": "tramadol + paracetamol",
    "contramal": "tramadol", "tramazac": "tramadol",
    "codeine": "codeine", "morphine": "morphine",
    "pentazocine": "pentazocine", "fortwin": "pentazocine",

    # ── Anticoagulants / Antiplatelets ────────────────────────────────────
    "warfarin": "warfarin", "warf": "warfarin", "coumadin": "warfarin",
    "aspirin": "aspirin", "ecosprin": "aspirin", "disprin": "aspirin",
    "cardiprin": "aspirin", "loprin": "aspirin",
    "clopidogrel": "clopidogrel", "clopivas": "clopidogrel", "plavix": "clopidogrel",
    "heparin": "heparin", "clexane": "enoxaparin", "lovenox": "enoxaparin",
    "rivaroxaban": "rivaroxaban", "xarelto": "rivaroxaban",
    "apixaban": "apixaban", "eliquis": "apixaban",
    "dabigatran": "dabigatran", "pradaxa": "dabigatran",

    # ── Vitamins / Supplements ────────────────────────────────────────────
    "calcirol": "cholecalciferol", "arachitol": "cholecalciferol",
    "shelcal": "calcium + cholecalciferol",
    "mecobalamin": "mecobalamin", "methylcobalamin": "mecobalamin",
    "neurobion": "vitamin b complex", "becadexamin": "vitamin b complex",
    "folic acid": "folic acid", "folvite": "folic acid",
    "vitamin c": "ascorbic acid", "celin": "ascorbic acid",

    # ── Bronchodilators / Respiratory ─────────────────────────────────────
    "salbutamol": "salbutamol", "asthalin": "salbutamol", "ventolin": "salbutamol",
    "levolin": "levosalbutamol",
    "ipratropium": "ipratropium", "ipravent": "ipratropium",
    "theophylline": "theophylline", "deriphyllin": "theophylline",
    "montelukast": "montelukast", "montair": "montelukast", "singulair": "montelukast",

    # ── Corticosteroids ───────────────────────────────────────────────────
    "prednisolone": "prednisolone", "omnacortil": "prednisolone",
    # US name — alias to prednisolone INN
    "prednisone": "prednisolone",
    "dexamethasone": "dexamethasone", "dexona": "dexamethasone",
    "betamethasone": "betamethasone", "betnesol": "betamethasone",
    "methylprednisolone": "methylprednisolone", "medrol": "methylprednisolone",
    "hydrocortisone": "hydrocortisone", "cortef": "hydrocortisone",

    # ── Antidepressants / Psych ───────────────────────────────────────────
    "sertraline": "sertraline", "serlift": "sertraline", "zoloft": "sertraline",
    "fluoxetine": "fluoxetine", "fludep": "fluoxetine", "prozac": "fluoxetine",
    "escitalopram": "escitalopram", "nexito": "escitalopram", "stalopam": "escitalopram",
    "paroxetine": "paroxetine", "paxidep": "paroxetine",
    "clonazepam": "clonazepam", "petril": "clonazepam", "rivotril": "clonazepam",
    "alprazolam": "alprazolam", "alprax": "alprazolam", "restyl": "alprazolam",
    "diazepam": "diazepam", "valium": "diazepam",
    "lithium": "lithium", "licab": "lithium", "eskalith": "lithium",
    "olanzapine": "olanzapine", "oleanz": "olanzapine", "zyprexa": "olanzapine",

    # ── Cardiac ───────────────────────────────────────────────────────────
    "digoxin": "digoxin", "lanoxin": "digoxin",
    "amiodarone": "amiodarone", "cordarone": "amiodarone",
    "nitroglycerine": "glyceryl trinitrate", "nitrostat": "glyceryl trinitrate",
    "isosorbide mononitrate": "isosorbide mononitrate", "ismo": "isosorbide mononitrate",
    "sildenafil": "sildenafil", "viagra": "sildenafil", "penegra": "sildenafil",
    "ivabradine": "ivabradine", "coralan": "ivabradine",

    # ── Other commonly prescribed ─────────────────────────────────────────
    "methotrexate": "methotrexate", "folitrax": "methotrexate",
    "hydroxychloroquine": "hydroxychloroquine", "hcqs": "hydroxychloroquine",
    "colchicine": "colchicine",
    "allopurinol": "allopurinol", "zyloric": "allopurinol",
    "febuxostat": "febuxostat", "febustat": "febuxostat",
    "gabapentin": "gabapentin", "gabapin": "gabapentin",
    "pregabalin": "pregabalin", "lyrica": "pregabalin", "pregeb": "pregabalin",
    "phenytoin": "phenytoin", "eptoin": "phenytoin",
    "carbamazepine": "carbamazepine", "tegretol": "carbamazepine",
    "levetiracetam": "levetiracetam", "levepsy": "levetiracetam",
    "valproate": "valproate", "valparin": "valproate", "encorate": "valproate",
    "cetirizine": "cetirizine", "cetzine": "cetirizine", "alerid": "cetirizine",
    "loratadine": "loratadine", "loratab": "loratadine",
    "levocetirizine": "levocetirizine", "levorid": "levocetirizine",
    "hydroxyzine": "hydroxyzine", "atarax": "hydroxyzine",
    "iron": "ferrous sulfate", "fersolate": "ferrous sulfate", "feosol": "ferrous sulfate",
    "ferrous sulfate": "ferrous sulfate", "fercayl": "ferrous sulfate",

    # ── TB / Anti-Mycobacterial ───────────────────────────────────────────
    # First-line single agents
    "isoniazid": "isoniazid", "inh": "isoniazid", "isonex": "isoniazid",
    "rifampicin": "rifampicin", "rimactane": "rifampicin", "rifacin": "rifampicin",
    "pyrazinamide": "pyrazinamide", "pza": "pyrazinamide", "pyrafat": "pyrazinamide",
    "ethambutol": "ethambutol", "combutol": "ethambutol", "myambutol": "ethambutol",
    "streptomycin": "streptomycin",

    # First-line fixed-dose combinations (DOTS)
    "rcinex": "rifampicin + isoniazid",
    "rimactazid": "rifampicin + isoniazid",
    "akurit": "rifampicin + isoniazid + pyrazinamide + ethambutol",
    "forecox": "rifampicin + isoniazid + pyrazinamide + ethambutol",
    "myrin": "ethambutol + isoniazid + rifampicin + pyrazinamide",
    "myrin p": "ethambutol + isoniazid + rifampicin",
    "rifinah": "rifampicin + isoniazid",

    # Second-line / MDR-TB
    "kanamycin": "kanamycin",
    "amikacin": "amikacin", "amicin": "amikacin",
    "capreomycin": "capreomycin",
    "cycloserine": "cycloserine",
    "ethionamide": "ethionamide",
    "para-aminosalicylic acid": "para-aminosalicylic acid", "pas": "para-aminosalicylic acid",
    "bedaquiline": "bedaquiline", "sirturo": "bedaquiline",
    "delamanid": "delamanid", "deltyba": "delamanid",
    "linezolid": "linezolid", "lizolid": "linezolid",
    "clofazimine": "clofazimine", "lamprene": "clofazimine",
    "rifabutin": "rifabutin", "mycobutin": "rifabutin",

    # ── ARVs — Antiretrovirals ──────────────────────────────────────────
    # NRTIs
    "tenofovir": "tenofovir", "tenvir": "tenofovir", "viread": "tenofovir",
    "lamivudine": "lamivudine", "hepitec": "lamivudine", "3tc": "lamivudine",
    "zidovudine": "zidovudine", "retrovir": "zidovudine", "azt": "zidovudine",
    "stavudine": "stavudine", "zerit": "stavudine", "d4t": "stavudine",
    "abacavir": "abacavir", "ziagen": "abacavir",
    "emtricitabine": "emtricitabine", "emtriva": "emtricitabine", "ftc": "emtricitabine",
    "didanosine": "didanosine", "videx": "didanosine",

    # NNRTIs
    "efavirenz": "efavirenz", "efavir": "efavirenz", "stocrin": "efavirenz",
    "nevirapine": "nevirapine", "nevimune": "nevirapine", "viramune": "nevirapine",
    "etravirine": "etravirine", "intelence": "etravirine",
    "rilpivirine": "rilpivirine", "edurant": "rilpivirine",

    # PIs
    "lopinavir": "lopinavir",
    "ritonavir": "ritonavir", "norvir": "ritonavir",
    "aluvia": "lopinavir + ritonavir", "kaletra": "lopinavir + ritonavir",
    "atazanavir": "atazanavir", "reyataz": "atazanavir",
    "darunavir": "darunavir", "prezista": "darunavir",
    "saquinavir": "saquinavir",

    # INSTIs
    "dolutegravir": "dolutegravir", "tivicay": "dolutegravir",
    "raltegravir": "raltegravir", "isentress": "raltegravir",
    "elvitegravir": "elvitegravir",
    "bictegravir": "bictegravir",

    # ARV FDCs
    "tenvir em": "tenofovir + emtricitabine",
    "truvada": "tenofovir + emtricitabine",
    "combivir": "zidovudine + lamivudine",
    "duovir": "zidovudine + lamivudine",
    "duovir n": "zidovudine + lamivudine + nevirapine",
    "triomune": "stavudine + lamivudine + nevirapine",
    "atripla": "efavirenz + emtricitabine + tenofovir",
    "symfi": "efavirenz + lamivudine + tenofovir",
    "triumeq": "abacavir + dolutegravir + lamivudine",

    # ── Extended Psychiatry / Neurology ──────────────────────────────
    # Atypical antipsychotics
    "risperidone": "risperidone", "risnia": "risperidone", "risperdal": "risperidone",
    "quetiapine": "quetiapine", "qutipin": "quetiapine", "seroquil": "quetiapine",
    "haloperidol": "haloperidol", "serenace": "haloperidol", "haldol": "haloperidol",
    "clozapine": "clozapine", "sizopin": "clozapine", "clozaril": "clozapine",
    "aripiprazole": "aripiprazole", "arip": "aripiprazole", "abilify": "aripiprazole",
    "ziprasidone": "ziprasidone",
    "paliperidone": "paliperidone", "invega": "paliperidone",
    "amisulpride": "amisulpride", "amipride": "amisulpride",

    # SNRIs
    "venlafaxine": "venlafaxine", "venlor": "venlafaxine", "veniz": "venlafaxine",
    "duloxetine": "duloxetine", "duvanta": "duloxetine", "cymbalta": "duloxetine",
    "desvenlafaxine": "desvenlafaxine", "dvance": "desvenlafaxine",

    # Other antidepressants
    "mirtazapine": "mirtazapine", "mirtaz": "mirtazapine", "remeron": "mirtazapine",
    "bupropion": "bupropion", "bupron": "bupropion", "wellbutrin": "bupropion",
    "trazodone": "trazodone", "trazonil": "trazodone",
    "amitriptyline": "amitriptyline", "amitril": "amitriptyline", "elavil": "amitriptyline",
    "imipramine": "imipramine", "tofranil": "imipramine",
    "nortriptyline": "nortriptyline",

    # Hypnotics / anxiolytics
    "zolpidem": "zolpidem", "nitrazepam": "nitrazepam", "nitrosun": "nitrazepam",
    "zopiclone": "zopiclone", "zopicon": "zopiclone",
    "lorazepam": "lorazepam", "ativan": "lorazepam", "larpose": "lorazepam",
    "modafinil": "modafinil", "modalert": "modafinil",
    "melatonin": "melatonin",

    # ── SGLT2 Fixed-Dose Combinations ─────────────────────────────────
    "synjardy": "empagliflozin + metformin",
    "synjardy xr": "empagliflozin + metformin",
    "xigduo": "dapagliflozin + metformin",
    "xigduo xr": "dapagliflozin + metformin",
    "qtern": "dapagliflozin + saxagliptin",
    "glyxambi": "empagliflozin + linagliptin",
    "steglujan": "ertugliflozin + sitagliptin",
    "segluromet": "ertugliflozin + metformin",
    "ertugliflozin": "ertugliflozin", "steglatro": "ertugliflozin",
    "sotagliflozin": "sotagliflozin",

    # ── Common Indian FDCs (previously missing) ────────────────────────
    # GI combinations
    "pan d": "pantoprazole + domperidone",
    "pantocid d": "pantoprazole + domperidone",
    "pantodac d": "pantoprazole + domperidone",
    "nexpro rd": "esomeprazole + domperidone",
    "razo d": "rabeprazole + domperidone",
    "rablet d": "rabeprazole + domperidone",
    "omez d": "omeprazole + domperidone",
    "ontime": "pantoprazole + domperidone",

    # Antibiotic FDCs
    "augmentin duo": "amoxicillin + clavulanate",
    "clavam 625": "amoxicillin + clavulanate",
    "taxim o": "cefixime",
    "monocef o": "cefpodoxime",
    "zifi o": "cefixime + ofloxacin",

    # Analgesic FDCs
    "dolo 650": "paracetamol",
    "dolo 500": "paracetamol",
    "combiflam plus": "ibuprofen + paracetamol + caffeine",
    "flexon": "ibuprofen + paracetamol",

    # Anti-inflammatory / enzyme
    "chymoral forte": "trypsin + chymotrypsin",
    "chymotrypsin": "chymotrypsin",
    "serratiopeptidase": "serratiopeptidase", "serratia": "serratiopeptidase",

    # Steroid brands
    "wysolone": "prednisolone",
    "decdan": "dexamethasone",

    # Urological
    "tamsulosin": "tamsulosin", "urimax": "tamsulosin", "veltam": "tamsulosin",
    "silodosin": "silodosin", "sylodix": "silodosin",
    "alfuzosin": "alfuzosin", "alfred": "alfuzosin",
    "finasteride": "finasteride", "finpecia": "finasteride",
    "dutasteride": "dutasteride", "dutas": "dutasteride",

    # Ophthalmic (common oral dosing)
    "acetazolamide": "acetazolamide", "diamox": "acetazolamide",

    # Bone health
    "alendronate": "alendronate", "alenost": "alendronate", "osteofos": "alendronate",
    "risedronate": "risedronate", "actonel": "risedronate",
    "zoledronic acid": "zoledronic acid",
    "denosumab": "denosumab", "prolia": "denosumab",
    "calcitriol": "calcitriol", "rocaltrol": "calcitriol",
    "alphacalcidol": "alfacalcidol", "alpha d3": "alfacalcidol",
}


def _clean_name(raw: str) -> str:
    """Lowercase, strip whitespace, remove tablet/capsule prefix clutter."""
    name = raw.lower().strip()
    # Remove form prefix: "tab.", "cap.", etc.
    name = re.sub(r"^(tab\.?|cap\.?|syp\.?|inj\.?|susp\.?)\s*", "", name)
    # Remove trailing dosage like "500mg", "10 mg"
    name = re.sub(r"\s*\d+(\.\d+)?\s*(mg|mcg|ml|g|iu|units?)\b.*$", "", name)
    return name.strip()


@lru_cache(maxsize=512)
def _rxnorm_lookup(inn: str) -> tuple[Optional[str], Optional[str]]:
    """
    Query RxNorm API for a given INN name.
    Returns (rxcui, standard_name) or (None, None) on failure.
    Cached so each unique INN is only queried once per process.
    """
    if not inn or inn.strip() == "":
        return None, None

    url = f"{RXNORM_BASE}/rxcui.json"
    try:
        resp = requests.get(url, params={"name": inn}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        rxcui = data.get("idGroup", {}).get("rxnormId", [None])[0]
        if not rxcui:
            return None, None

        # Fetch the display name for the rxcui
        name_resp = requests.get(
            f"{RXNORM_BASE}/rxcui/{rxcui}/property.json",
            params={"propName": "RxNorm Name"},
            timeout=5,
        )
        name_resp.raise_for_status()
        name_data = name_resp.json()
        props = name_data.get("propConceptGroup", {}).get("propConcept", [])
        standard_name = props[0].get("propValue", inn) if props else inn
        return rxcui, standard_name

    except requests.exceptions.Timeout:
        logger.warning("RxNorm API timed out for %r — degrading gracefully.", inn)
        return None, None
    except requests.exceptions.RequestException as exc:
        logger.warning("RxNorm API error for %r: %s — degrading gracefully.", inn, exc)
        return None, None


def normalize_drug(med: MedicationExtracted) -> NormalizedDrug:
    """
    Normalise a single extracted medication.
    Returns a NormalizedDrug with inn, rxcui, and standard_name populated.
    Never raises — degrades gracefully on API failure.
    """
    cleaned = _clean_name(med.raw_drug_name)

    # Step 1: brand_db two-pass lookup (exact → Levenshtein fuzzy)
    # Pass 1: exact match against SQLite brand_names table (indexed, O(1))
    # Pass 2: fuzzy match with edit distance ≤ 2 — recovers OCR typos
    inn = lookup_brand(cleaned)
    if inn is None:
        # Try partial match on first word (e.g. "Glycomet SR 500" → "glycomet sr" → "glycomet")
        first_word = cleaned.split()[0] if cleaned else cleaned
        if first_word != cleaned:
            inn = lookup_brand(first_word)
    if inn is None:
        inn = cleaned  # worst-case fallback — let RxNorm try

    # Step 2: RxNorm lookup
    rxcui, standard_name = _rxnorm_lookup(inn)
    if standard_name is None:
        standard_name = inn

    return NormalizedDrug(
        raw_drug_name=med.raw_drug_name,
        inn=inn,
        rxcui=rxcui,
        standard_name=standard_name,
        dosage_value=med.dosage_value,
        dosage_unit=med.dosage_unit,
        frequency=med.frequency,
        freq_per_day=med.freq_per_day,
        duration_days=med.duration_days,
        route=med.route,
        is_active=True,
    )


def normalize_all(medications: List[MedicationExtracted]) -> List[NormalizedDrug]:
    """Normalise a list of extracted medications. Returns list of NormalizedDrug."""
    return [normalize_drug(med) for med in medications]
