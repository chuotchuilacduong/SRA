# skill_probe_hidden — Per-query skill-probing + hidden-state analysis

Probe **Qwen3-4B** (HuggingFace, local) on 10 test queries/dataset under
`{no_skill, gold, top-10 candidates}` for two retrievers (**M4**, **CE-HYRR**),
record correctness + per-candidate utility, and capture pooled hidden states for
separability analysis.

> Hidden states need a **local HF causal LM on GPU** — Ollama/vLLM can't return
> them. Run probing on **H100/CUDA**. Candidate-loading and analysis are CPU-only.

## Output layout (`results/skill_probe_hidden/`)
```
<ds>/<qid>__M4.json          # per (query, method): candidates+utility+probe refs
<ds>/<qid>__CE-HYRR.json
<ds>/hidden_states/<qid>__<skilltag>.npy   # pooled [37, 2560] float16, deduped
<ds>/probe_cache.jsonl        # resumable probe outcomes (raw text lives here)
<ds>/index.json               # picked qids + seed
analysis/<ds>/*.png, summary.json
analysis/SUMMARY.md
```
`.npy` are referenced by **relative path** in the JSON, never inlined.

## Run (H100)
```bash
# 0. deps already in requirements.txt (transformers 4.46.3, torch, sklearn, matplotlib, seaborn).
#    Model auto-downloads from HF hub on first run (Qwen/Qwen3-4B).

# 1. CPU checks first (no model) — also runs on the mac:
python -m experiments.skill_probe_hidden.smoke_test
python -m experiments.skill_probe_hidden.run_probe --dry-run

# 2. one real probe end-to-end (GPU):
python -m experiments.skill_probe_hidden.smoke_test --real

# 3. full probing:
python -m experiments.skill_probe_hidden.run_probe \
  --datasets logicbench medcalcbench theoremqa champ \
  --methods M4 CE-HYRR --n-queries 10 --seed 42 \
  --max-new-tokens 2048 --layer-policy all

# 4. analysis (CPU, after probing):
python -m experiments.skill_probe_hidden.analyze        # add --umap for UMAP (needs umap-learn)
```

## Key choices (see plan)
- Prompt reproduced exactly from `sragents.prompts.build_prompt` with skill
  **`content`** only (same as `DirectEngine`). Greedy decode, thinking OFF.
- Pooling: **last generated token, all 37 layers**, float16. `--pool span_mean`
  and `--layer-policy every4` are optional.
- Probes deduped by `(instance_id, sorted skill_ids)` → shared `no_skill`/`gold`/
  overlapping candidates run once; cache makes runs resumable.
- Datasets limited to deterministic-verifier single-shot tasks (no toolqa/bigcodebench).

## Caveat
Qwen3-4B (HF fp16) ≠ the Ollama Qwen2.5:7B-Q4 used in `probehyrr_validation`, so
absolute correctness/utility numbers here are **self-contained** — don't
cross-compare with those runs. Retrieval metrics (gold-rank, hit@k) are
model-independent and comparable.
