# Plan — Add bge_ft (fine-tuned retriever) signal as features for L6-final

## 0. Context / motivation
In §21 (Standalone CE, query_gen-test), **bge_ft (retriever-only)** is the strongest single row:
R@1≈57, R@10≈87, **R@100≈99.3**, **nDCG@10≈76.9** — above L6-final (67.4), M7-CE@500 (69.4),
BM25/BGE-base/M4. The hypothesis here: feed the bge_ft signal into **L6-final** (LightGBM
LambdaRank) as new features and see if it makes L6 the best no-CE supervised reranker.

### Why bge_ft is strong + how it's produced
- `bge_ft` = `BAAI/bge-base-en-v1.5` **fine-tuned on query_gen-train** in Phase B
  (`src/kmeans/scripts/ceraw_train_retriever.py`, MultipleNegativesRankingLoss = InfoNCE in-batch
  + full-corpus-mined hard negatives), saved to `results/models/sr-emb-bge-v1/`.
- At eval (`ceraw_eval.py` `first_stage("bge_ft")`): encode the full 26,262-skill corpus with
  `sr-emb-bge-v1` (cached at `results/models/sr-emb-bge-v1/corpus_emb.npy`, normalized, row-aligned
  to `results/bge/corpus_ids.json`), encode the query, take cosine top-D.
- It wins because it directly learned the **query→gold-skill** mapping (supervised), unlike the
  zero-shot/unsupervised first stages — and unlike the CE rerankers it isn't hurt by a
  distribution-mismatched rerank. It is **held-out** on query_gen-test (trained on train only) → fair.

### Why it should help L6
L6's 45 features (`src/kmeans/ltr_features.py`, groups: retrieval/m4/a7/qsc/confidence) are derived
from **BM25, BGE-base, M4 clustering, QSC, PRF** — *none fine-tuned*. A `bge_ft` cosine column adds a
signal L6 has no proxy for. LightGBM can combine it with the existing signals (BM25 lexical, M4
cluster priors, QSC, confidence). Expectation: L6+bge_ft nDCG@10 should rise well above 67.4, likely
toward bge_ft-alone (~77) within L6's candidate-pool recall ceiling.

### Honest caveat (the ceiling)
L6 ranks the **M4 Stage-1 pool** (`e["skill_ids"]`, ~500/query, gold-in-pool **97.6%**), whereas
bge_ft-alone retrieves from the **full corpus** (R@100 99.3%). So L6+bge_ft can approach bge_ft's
*ordering* but is bounded by the M4 pool's 97.6% recall — it won't beat bge_ft-alone on recall unless
we also widen the pool (Extension §6). Net: L6+bge_ft is expected to be the **best no-CE supervised
reranker on the M4 pool**, and a fair, leakage-free comparison vs bge_ft-alone and M7-CE@500.

## 1. Feature design — new group `bge_ft`
For each (query q, candidate skill s in the L6 pool), with normalized embeddings
`u=qft(q)`, `v=sft(s)` from `sr-emb-bge-v1`:

| feature | definition |
|---|---|
| `bgeft_cosine` | `u·v` (raw cosine; LightGBM handles raw scale) |
| `bgeft_rank` | rank of s by cosine within the query's pool (1=best) |
| `bgeft_inv_rank` | `1/(bgeft_rank)` |
| `bgeft_rank_norm` | `bgeft_rank / pool_size` |
| `bgeft_is_top1` | 1 if s is the pool's argmax cosine else 0 |
| `bgeft_margin_top1` | `bgeft_cosine − max_pool_cosine` (≤0; 0 for the top) |
| `bgeft_z` | within-query z-score of cosine (`(cos−mean)/std`) |

7 features → new `FEATURE_GROUPS["bge_ft"]`. (Ranks/z are within-pool, so they're comparable across
queries — the same design idea as the existing rrf/bm25/bge rank features.)

## 2. Implementation (edits + new script) — all reproducible on H100
1. **`src/kmeans/ltr_features.py`** — register the group (additive, backward-compatible):
   - add the 7 names to a `_BGEFT_FEATS` list; `FEATURE_GROUPS["bge_ft"] = _BGEFT_FEATS`;
   - extend the `ALL_FEATURES` group list from `["retrieval","m4","a7","qsc","confidence"]` to
     append `"bge_ft"` **last** (so existing 45-col caches map unchanged for the first 45 indices,
     and `column_indices([...])` still works; selecting without "bge_ft" reproduces baseline L6).
2. **NEW `src/kmeans/scripts/add_bgeft_features.py`** — augment the cached feature tables (no base-table
   rebuild; least invasive, easy A/B):
   - load `sr-emb-bge-v1` corpus embeddings: `np.load(results/models/sr-emb-bge-v1/corpus_emb.npy)`
     + `results/bge/corpus_ids.json` → `skill_id → v` (already normalized);
   - encode all query_gen queries (train+dev+test, from `io.load_instances`) with
     `SentenceTransformer("results/models/sr-emb-bge-v1", normalize_embeddings=True)` →
     `instance_id → u`; cache to `results/m4_v2/cache/query_emb_ft/{ds}.npy` (+ `_ids.json`) for reuse;
   - for each ds: `tbl = ltr_features.load_features(results/m4_v2/cache/ltr_features/{ds}.npz)`; walk its
     `group_sizes`/`instance_ids`/`skill_ids`, compute the 7 features per row, **append columns** to `X`
     in `FEATURE_GROUPS["bge_ft"]` order, set `feature_names = ALL_FEATURES`, and
     `save_features(results/m4_v2/cache/ltr_features_bgeft/{ds}.npz, tbl)`.
   - GPU only for the query encode (~5,400 short queries → seconds); rest is numpy. Idempotent.
3. **`src/kmeans/scripts/run_fair_eval.py`** — add a knob to fit L6 with the extra group + alt cache:
   - `--feature-cache {default|bgeft}` → read `cache/ltr_features` or `cache/ltr_features_bgeft`;
   - `--l6-groups` default `retrieval m4 a7 qsc qsc confidence` **+** optional `bge_ft`;
   - run **two** final fits for the ablation: `L6-final` (45) and `L6-final+bgeft` (52), both on
     query_gen-train(+dev), evaluate on query_gen-test; keep the existing §20 outputs and **append a
     new comparison block** (don't overwrite §20). Save `results/models/l6_ltr_final_bgeft.txt` +
     `.features.json` (with LightGBM gain importances — to see how high `bgeft_*` ranks).
4. **Reporting** — append a section (e.g. `results/comparisons/l6_bgeft.{json,md}` and a `§23` in
   `FULL_M4_V2_RESULTS.md`) comparing, on query_gen-test (macro + per-dataset nDCG@10 / R@1 / R@10 /
   R@100): `L6-final`, **`L6-final+bgeft`**, `bge_ft (retriever-only)`, `M7-CE@500`, with paired
   bootstrap `L6+bgeft vs L6` and `L6+bgeft vs bge_ft-alone`. Reuse `cv.paired_bootstrap`,
   `evaluate.eval_variant`.

## 3. Fairness / leakage (must hold)
- `sr-emb-bge-v1` is trained on **query_gen-train only**; the `bgeft_*` features are computed the
  **same way at train and test** time → no test-label leakage (a learned feature, like rrf/bge already are).
- L6+bge_ft is trained on query_gen-train(+dev), evaluated on the **same held-out query_gen-test (1,079)**
  as §20/§22 → directly comparable.
- Note the category nuance (for the writeup): L6+bge_ft ranks the **M4 pool**; bge_ft-alone ranks the
  **full corpus**. The comparison isolates "bge_ft as a *signal* inside the pool" — report retriever-only
  R@100 next to it so the pool-ceiling is explicit.

## 4. H100 run steps (after `git pull`)
```bash
cd /home/quynhtl/projects/SRA && git pull && export PYTHONPATH=src
# prereq: results/models/sr-emb-bge-v1 (+corpus_emb.npy), results/bge/corpus_ids.json,
#         results/m4_v2/cache/ltr_features/{ds}.npz   (from the §20 run_fair_eval)
python src/kmeans/scripts/add_bgeft_features.py            # -> cache/ltr_features_bgeft/{ds}.npz (+ query_emb_ft)
python src/kmeans/scripts/run_fair_eval.py --grid moderate --feature-cache bgeft --l6-groups retrieval m4 a7 qsc confidence bge_ft   # L6+bgeft
python src/kmeans/scripts/run_fair_eval.py --grid moderate                                                                          # baseline L6 (45) for the A/B
# -> compare in results/comparisons/l6_bgeft.md (+ §23 in FULL_M4_V2_RESULTS.md)
```
(CPU is fine except the one-time query encode in `add_bgeft_features.py`, which uses the GPU if available.)

## 5. Success criteria
- **Primary:** `L6-final+bgeft` macro nDCG@10 on query_gen-test **> L6-final (67.4)**, with paired
  bootstrap p<0.05. Stretch: ≥ M7-CE@500 (69.4), approaching bge_ft-alone (~77).
- **Diagnostic:** `bgeft_cosine` / `bgeft_rank` rank high in LightGBM gain importance (confirms the
  signal is used). Per-dataset: expect biggest gains where bge_ft dominates (toolqa, theoremqa).
- If L6+bge_ft does **not** beat bge_ft-alone, that's the M4-pool ceiling → motivates §6.

## 6. Extension (optional follow-up, bigger change)
Lift the candidate-pool ceiling: **union the M4 Stage-1 pool with bge_ft top-k** (e.g. top-100) per
query before building features, so gold-in-pool rises 97.6% → ~99%. Then L6+bge_ft can also recover
gold that the M4 pool misses. Requires editing `base_table.iter_base_table` pool construction (and
recomputing M4/QSC features for the added candidates) — heavier; do only if §5 shows the ceiling binds.

## 7. Risks / notes
- **Embedding alignment:** `sr-emb-bge-v1/corpus_emb.npy` rows are aligned to `results/bge/corpus_ids.json`
  (that's the order `ceraw_eval` encoded). The augment script must map by that id list, not assume corpus order.
- **Pool ceiling (97.6%)** bounds recall — primary metric is nDCG@10/ordering, not recall.
- **Backward compatibility:** appending `bge_ft` LAST in `ALL_FEATURES` keeps the existing 45-col caches
  and all prior `column_indices` selections valid; the §20 baseline is reproducible.
- **Cost:** tiny — one query encode (~5,400 queries) + numpy; L6 fit is seconds–minutes.

## Files
| File | Change |
|---|---|
| `src/kmeans/ltr_features.py` | add `FEATURE_GROUPS["bge_ft"]` (7 feats), append to `ALL_FEATURES` |
| `src/kmeans/scripts/add_bgeft_features.py` | NEW — encode query_ft, augment `ltr_features_bgeft/{ds}.npz` |
| `src/kmeans/scripts/run_fair_eval.py` | `--feature-cache`, `--l6-groups`; fit L6+bgeft; append §23 comparison |
| `src/kmeans/base_table.py` | (only for §6 extension) widen pool with bge_ft top-k |
| `FULL_M4_V2_RESULTS.md` | new §23 (L6 vs L6+bgeft vs bge_ft-alone vs M7-CE@500, query_gen-test) |
