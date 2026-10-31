"""Basic encoder-decoder Transformer trained from scratch (joint SentencePiece BPE)."""
import math, time, random, sys, torch, torch.nn as nn, sentencepiece as spm
from common import *

torch.manual_seed(0); random.seed(0)
dev = device()
train, val, test = load("train"), load("val"), load("test")
with open("data/spm_train.txt", "w") as f:
    for r in train: f.write(r["en"] + "\n" + r["hi"] + "\n")
spm.SentencePieceTrainer.train(input="data/spm_train.txt", model_prefix="data/spm", vocab_size=8000,
    model_type="bpe", character_coverage=1.0, pad_id=0, unk_id=1, bos_id=2, eos_id=3, minloglevel=2)
sp = spm.SentencePieceProcessor(model_file="data/spm.model")
PAD, BOS, EOS, V, MAXLEN = 0, 2, 3, 8000, 64

class TM(nn.Module):
    def __init__(s, d=384, h=6, L=4, ff=1536, p=0.2):
        super().__init__()
        s.emb = nn.Embedding(V, d, padding_idx=PAD); s.pos = nn.Embedding(128, d); s.d = d
        s.tf = nn.Transformer(d, h, L, L, ff, p, batch_first=True, norm_first=True)
        s.out = nn.Linear(d, V, bias=False); s.out.weight = s.emb.weight  # tied
        s.drop = nn.Dropout(p)
        nn.init.normal_(s.emb.weight, 0, d ** -0.5); nn.init.normal_(s.pos.weight, 0, 0.02)
    def embed(s, x):
        return s.drop(s.emb(x) * math.sqrt(s.d) + s.pos(torch.arange(x.size(1), device=x.device)))
    def encode(s, src):
        return s.tf.encoder(s.embed(src), src_key_padding_mask=(src == PAD))
    def decode(s, mem, src, tgt):
        m = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        h = s.tf.decoder(s.embed(tgt), mem, tgt_mask=m, tgt_is_causal=True,
                         tgt_key_padding_mask=(tgt == PAD), memory_key_padding_mask=(src == PAD))
        return s.out(h)

def pad(seqs):
    n = max(map(len, seqs)); return torch.tensor([x + [PAD] * (n - len(x)) for x in seqs])

def enc(rows):
    return [(sp.encode(r["en"])[:MAXLEN - 1] + [EOS], [BOS] + sp.encode(r["hi"])[:MAXLEN - 2] + [EOS]) for r in rows]

@torch.no_grad()
def translate(model, rows, temp=0.0, bs=50, seed=0):
    model.eval(); g = torch.Generator().manual_seed(seed); outs = []
    for i in range(0, len(rows), bs):
        src = pad([sp.encode(r["en"])[:MAXLEN - 1] + [EOS] for r in rows[i:i + bs]]).to(dev)
        mem = model.encode(src); ys = torch.full((src.size(0), 1), BOS, device=dev)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=dev)
        for _ in range(MAXLEN):
            logits = model.decode(mem, src, ys)[:, -1].float()
            if temp == 0: nxt = logits.argmax(-1)
            else: nxt = torch.multinomial(torch.softmax(logits / temp, -1).cpu(), 1, generator=g).squeeze(1).to(dev)
            nxt = nxt.masked_fill(done, PAD); ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all(): break
        for y in ys[:, 1:].tolist():
            y = [t for t in y if t not in (PAD,)]; y = y[:y.index(EOS)] if EOS in y else y
            outs.append(sp.decode(y))
    return outs

model = TM().to(dev); n = nparams(model); print("params", n, flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=0.01)
data = enc(train); EPOCHS, BS = 15, 64; steps = EPOCHS * math.ceil(len(data) / BS); warm = 800
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, math.sqrt(warm / (s + 1))) )
lossf = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
t0 = time.time(); log = []
for ep in range(EPOCHS):
    model.train(); random.shuffle(data); tot = 0
    for i in range(0, len(data), BS):
        b = data[i:i + BS]; src = pad([x for x, _ in b]).to(dev); tgt = pad([y for _, y in b]).to(dev)
        logits = model.decode(model.encode(src), src, tgt[:, :-1])
        loss = lossf(logits.reshape(-1, V), tgt[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        tot += loss.item() * len(b)
    msg = {"epoch": ep + 1, "train_loss": round(tot / len(data), 4), "elapsed_s": round(time.time() - t0)}
    if (ep + 1) % 5 == 0:
        vh = translate(model, val[:200]); msg["val_bleu"] = score(vh, [r["hi"] for r in val[:200]])["bleu"]
    print(msg, flush=True); log.append(msg)
train_time = time.time() - t0

t0 = time.time(); hyps = translate(model, test); inf = time.time() - t0
refs = [r["hi"] for r in test]
res = {"model": "Transformer (scratch)", "params": n, "train_time_s": round(train_time), "infer_time_s": round(inf, 1),
       "infer_s_per_sent": round(inf / len(test), 3), **score(hyps, refs), "log": log}
temps = {}
sub = test[:100]
for T in [0.3, 0.7, 1.0, 1.5]:
    h = translate(model, sub, temp=T); temps[T] = {**score(h, [r["hi"] for r in sub]), "samples": h[:3]}
res["temperature"] = temps
save("transformer", res); save("transformer_hyps", hyps); print({k: v for k, v in res.items() if k != "log"})
