"""Download OPUS-100 en-hi, clean, and write fixed train/val/test subsets (identical for all four models)."""
import json, random
from datasets import load_dataset

SEED = 42
N_TRAIN, N_VAL, N_TEST = 20000, 500, 300

def ok(en, hi):
    a, b = en.split(), hi.split()
    return 5 <= len(a) <= 30 and 5 <= len(b) <= 40 and en != hi

def clean(split):
    seen, out = set(), []
    for ex in split:
        en, hi = ex["translation"]["en"].strip(), ex["translation"]["hi"].strip()
        if ok(en, hi) and en not in seen:
            seen.add(en); out.append({"en": en, "hi": hi})
    return out

d = load_dataset("Helsinki-NLP/opus-100", "en-hi")
rng = random.Random(SEED)
tr, va, te = clean(d["train"]), clean(d["validation"]), clean(d["test"])
rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
sets = {"train": tr[:N_TRAIN], "val": va[:N_VAL], "test": te[:N_TEST]}
print({k: len(v) for k, v in sets.items()}, "cleaned pool sizes:", len(tr), len(va), len(te))
for k, v in sets.items():
    with open(f"data/{k}.jsonl", "w") as f:
        for r in v: f.write(json.dumps(r, ensure_ascii=False) + "\n")
