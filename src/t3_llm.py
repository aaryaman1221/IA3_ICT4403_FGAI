"""LoRA fine-tune a decoder-only LLM for en->hi. usage: t3_llm.py <name> <hf_id>"""
import sys, time, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from common import *

name, hf = sys.argv[1], sys.argv[2]
torch.manual_seed(0); random.seed(0); dev = device()
N_TRAIN, BS, LR, MAXTOK = 2000, 8, 2e-4, 192
train, test = load("train")[:N_TRAIN], load("test")
tok = AutoTokenizer.from_pretrained(hf); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16).to(dev); n = nparams(model)
P = lambda en: f"Translate English to Hindi.\nEnglish: {en}\nHindi:"

def gen(rows, temp=0.0, bs=25, m=None):
    m = m or model; m.eval(); outs = []
    for i in range(0, len(rows), bs):
        x = tok([P(r["en"]) for r in rows[i:i+bs]], return_tensors="pt", padding=True).to(dev)
        kw = dict(do_sample=True, temperature=temp, top_k=0, top_p=1.0) if temp > 0 else dict(do_sample=False)
        if temp > 0: torch.manual_seed(0)
        with torch.no_grad(): y = m.generate(**x, max_new_tokens=100, pad_token_id=tok.pad_token_id, **kw)
        outs += [o.split("\n")[0].strip() for o in tok.batch_decode(y[:, x.input_ids.size(1):], skip_special_tokens=True)]
    return outs

refs = [r["hi"] for r in test]
# zero-shot prompting baseline (before fine-tuning)
t0 = time.time(); zs = gen(test); zs_inf = time.time() - t0; zs_sc = score(zs, refs); print("zero-shot", zs_sc, flush=True)

model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
for p in model.parameters():
    if p.requires_grad: p.data = p.data.float()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR); t0 = time.time(); log = []
random.shuffle(train); model.train()
for i in range(0, len(train), BS):
    b = train[i:i+BS]; ids, labs = [], []
    for r in b:
        p = tok(P(r["en"]), add_special_tokens=True).input_ids; t = tok(" " + r["hi"] + "\n", add_special_tokens=False).input_ids + [tok.eos_token_id]
        s = (p + t)[:MAXTOK]; l = ([-100] * len(p) + t)[:MAXTOK]; ids.append(s); labs.append(l)
    L = max(map(len, ids))
    x = torch.tensor([s + [tok.pad_token_id] * (L - len(s)) for s in ids]).to(dev)
    y = torch.tensor([l + [-100] * (L - len(l)) for l in labs]).to(dev)
    a = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in ids]).to(dev)
    with torch.autocast("mps", dtype=torch.bfloat16):
        loss = model(input_ids=x, attention_mask=a, labels=y).loss
    opt.zero_grad(); loss.backward(); opt.step()
    if (i // BS) % 25 == 0: print(i // BS, round(loss.item(), 3), round(time.time() - t0), flush=True); log.append((i // BS, round(loss.item(), 3)))
train_time = time.time() - t0
t0 = time.time(); hyps = gen(test); inf = time.time() - t0
res = {"model": f"{name} (LoRA fine-tuned)", "hf_id": hf, "params": n, "trainable_params": trainable, "train_examples": len(train),
       "train_time_s": round(train_time), "infer_time_s": round(inf, 1), "infer_s_per_sent": round(inf / len(test), 3),
       **score(hyps, refs), "zero_shot": {**zs_sc, "infer_time_s": round(zs_inf, 1), "samples": zs[:3]}, "log": log}
sub = test[:100]; res["temperature"] = {}
for T in [0.3, 0.7, 1.0, 1.5]:
    h = gen(sub, T); res["temperature"][T] = {**score(h, [r["hi"] for r in sub]), "samples": h[:3]}
save(name, res); save(name + "_hyps", hyps); print({k: v for k, v in res.items() if k != "log"})
