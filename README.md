# English→Hindi Machine Translation: Encoder–Decoder vs Decoder-Only Models

Foundations of Generative AI (ICT4403) — Assignment 3.

> **Status: work in progress.** Results, comparison and conclusion are filled in as each experiment finishes.

## Problem statement and objective
Build English→Hindi translation systems with four architectures under comparable settings and compare quality and cost:
Basic Transformer (scratch), mT5-small (fine-tuned), LLaMA 3.2 1B (LoRA), Gemma 3 1B (LoRA).

## Dataset
OPUS-100 English–Hindi — https://huggingface.co/datasets/Helsinki-NLP/opus-100 (`en-hi` config).
Cleaned (5–30 English words, 5–40 Hindi words, deduplicated) and subsampled with seed 42:
20,000 train / 500 validation / 300 test pairs (`data/*.jsonl`, identical for all models).
Note: OPUS-100 en-hi is noisy (many software-UI strings, loose alignments), which lowers absolute scores.

## Repository layout
- `src/prep_data.py` — dataset download, cleaning, splits
- `src/t1_transformer.py` — Transformer trained from scratch
- `src/t2_mt5.py` — mT5-small fine-tuning
- `src/t3_llm.py` — LoRA fine-tuning for LLaMA / Gemma (+ zero-shot baseline)
- `results/` — metrics (JSON), generated translations, training logs

## Execution steps
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/prep_data.py
PYTHONPATH=src python src/t1_transformer.py
PYTHONPATH=src python src/t2_mt5.py
PYTHONPATH=src python src/t3_llm.py llama unsloth/Llama-3.2-1B-Instruct
PYTHONPATH=src python src/t3_llm.py gemma unsloth/gemma-3-1b-it
```
All runs were done on an Apple M4 (16 GB, PyTorch MPS). `unsloth/*` are ungated mirrors of the official Meta/Google weights.

## Results
_TBD_

## Model comparison and evaluation
_TBD_

## Sampling-temperature analysis
_TBD_

## Responsible AI discussion
_TBD_

## Conclusion
_TBD_
