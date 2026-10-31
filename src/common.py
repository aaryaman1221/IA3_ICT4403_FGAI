import json, time, sacrebleu, torch

def load(split):
    return [json.loads(l) for l in open(f"data/{split}.jsonl")]

def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def score(hyps, refs):
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="none" if False else "13a")
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    return {"bleu": round(bleu.score, 2), "chrf": round(chrf.score, 2)}

def nparams(model):
    return sum(p.numel() for p in model.parameters())

def save(name, obj):
    json.dump(obj, open(f"results/{name}.json", "w"), ensure_ascii=False, indent=1)
