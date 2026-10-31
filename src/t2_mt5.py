"""Fine-tune mT5-small (encoder-decoder) for en->hi."""
import time, random, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from common import *

torch.manual_seed(0); random.seed(0); dev = device()
N_TRAIN, BS, LR = 8000, 16, 3e-4
train, test = load("train")[:N_TRAIN], load("test")
tok = AutoTokenizer.from_pretrained("google/mt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small").to(dev); n = nparams(model)
PFX = "translate English to Hindi: "

def gen(rows, temp=0.0, bs=50):
    model.eval(); outs = []
    for i in range(0, len(rows), bs):
        x = tok([PFX + r["en"] for r in rows[i:i+bs]], return_tensors="pt", padding=True, truncation=True, max_length=64).to(dev)
        kw = dict(do_sample=True, temperature=temp, top_k=0) if temp > 0 else dict(do_sample=False)
        if temp > 0: torch.manual_seed(0)
        with torch.no_grad(): y = model.generate(**x, max_new_tokens=64, num_beams=1, **kw)
        outs += tok.batch_decode(y, skip_special_tokens=True)
    return outs

opt = torch.optim.AdamW(model.parameters(), lr=LR); t0 = time.time(); log = []
model.train(); random.shuffle(train)
for i in range(0, len(train), BS):
    b = train[i:i+BS]
    x = tok([PFX + r["en"] for r in b], return_tensors="pt", padding=True, truncation=True, max_length=64).to(dev)
    y = tok([r["hi"] for r in b], return_tensors="pt", padding=True, truncation=True, max_length=64).input_ids.to(dev)
    y[y == tok.pad_token_id] = -100
    loss = model(**x, labels=y).loss; opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if (i // BS) % 50 == 0: print(i // BS, round(loss.item(), 3), round(time.time() - t0), flush=True); log.append((i // BS, round(loss.item(), 3)))
train_time = time.time() - t0
t0 = time.time(); hyps = gen(test); inf = time.time() - t0
res = {"model": "mT5-small (fine-tuned)", "params": n, "train_examples": len(train), "train_time_s": round(train_time),
       "infer_time_s": round(inf, 1), "infer_s_per_sent": round(inf / len(test), 3), **score(hyps, [r["hi"] for r in test]), "log": log}
sub = test[:100]; res["temperature"] = {}
for T in [0.3, 0.7, 1.0, 1.5]:
    h = gen(sub, T); res["temperature"][T] = {**score(h, [r["hi"] for r in sub]), "samples": h[:3]}
save("mt5", res); save("mt5_hyps", hyps); print({k: v for k, v in res.items() if k != "log"})
