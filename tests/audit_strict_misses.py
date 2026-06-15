import json, re, sys, difflib, logging
from pathlib import Path
from unittest.mock import MagicMock

# Suppress logs
logging.disable(logging.CRITICAL)

sys.modules['src.database'] = MagicMock()
import src.database as db
db.SessionLocal = MagicMock()
db.get_or_create_patient = MagicMock()
db.get_active_medications = MagicMock(return_value=[])
db.save_prescription_to_db = MagicMock()
db.save_to_review_queue = MagicMock()

from src.pipeline import process_prescription
from src.drug_normalizer import _clean_name

EVAL_DIR = Path('c:/Users/USER/Desktop/OJAAI/data/evaluation')
ANN_DIR = EVAL_DIR / 'annotations'
IMG_DIR = EVAL_DIR / 'images'

def parse_gt_meds(gt_str):
    meds_match = re.search(
        r'medications:\s*(.*?)\s*(?:signature:|date:|doctor_name:|patient_name:|patient_age:|clinic_name:|</s>)',
        gt_str, re.S)
    if not meds_match:
        return []
    meds_block = meds_match.group(1)
    items = [i.strip() for i in meds_block.split('- ') if i.strip()]
    meds = []
    for item in items:
        m = re.search(r'(\d+(?:\.\d+)?)\s*(mg|mcg|ml|g|iu|units?|puffs?|drops?)\b', item, re.I)
        if m:
            drug_name = item[:m.start()].strip()
            if drug_name:
                meds.append(drug_name)
    return meds

print('=== STRICT MISS AUDIT ===\n')
misses = []

for ann_file in sorted(ANN_DIR.glob('*.json'))[:10]:
    img_file = IMG_DIR / (ann_file.stem + '.png')
    if not img_file.exists():
        continue
    with open(ann_file) as f:
        data = json.load(f)
    gt_meds = parse_gt_meds(data.get('ground_truth', ''))
    if not gt_meds:
        continue

    img_bytes = img_file.read_bytes()
    out = process_prescription(img_bytes, img_file.name)
    ext_cleaned = [_clean_name(m.raw_drug_name) for m in out.medications]
    ext_raw = [m.raw_drug_name for m in out.medications]

    for gt_name in gt_meds:
        gt_c = _clean_name(gt_name)
        strict_hit = any(gt_c in e or e in gt_c for e in ext_cleaned)
        if not strict_hit:
            fuzzy_scores = [(difflib.SequenceMatcher(None, gt_c, e).ratio(), e, r)
                            for e, r in zip(ext_cleaned, ext_raw)]
            best = max(fuzzy_scores, key=lambda x: x[0], default=(0, '', ''))
            fuzzy_hit = best[0] >= 0.5
            status = 'FUZZY_ONLY' if fuzzy_hit else 'TOTAL_MISS'
            misses.append({
                'status': status,
                'file': ann_file.stem,
                'gt': gt_name,
                'gt_clean': gt_c,
                'extracted_raw': ext_raw,
                'best_fuzzy_raw': best[2],
                'best_fuzzy_ratio': best[0],
            })

for m in misses:
    print(f"[{m['status']}] {m['file']}")
    print(f"  GT expected :  {m['gt']!r}  (cleaned: {m['gt_clean']!r})")
    print(f"  Extracted   :  {m['extracted_raw']}")
    if m['best_fuzzy_ratio'] > 0:
        print(f"  Best fuzzy  :  {m['best_fuzzy_raw']!r}  (ratio={m['best_fuzzy_ratio']:.2f})")
    print()

print(f"Total misses: {len(misses)}")
