#!/usr/bin/env python3
"""Compare all retrieval methods across all 6 datasets, @1 through @50."""
import json
import os
from pathlib import Path

DATASETS = ['champ', 'theoremqa', 'logicbench', 'toolqa', 'medcalcbench', 'bigcodebench']
KS = [1, 5, 10, 50]

METHODS = [
    # (display_name, path_template)
    ("LinearRAG (alone)",       "results/retrieval_all/{ds}-linearrag.json"),
    ("BM25 (alone)",            "results/retrieval_bm25/{ds}-bm25.json"),
    ("Dense BGE (alone)",       "results/retrieval_dense/{ds}-dense-bge.json"),
    ("Hybrid (official.py)",    "results/retrieval_hybrid_official/{ds}-hybrid-official.json"),
    ("3-way RRF (BM25+TFIDF+LR)","results/retrieval_enhanced/{ds}-rrf3way.json"),
    ("⭐ RRF Complementary (BM25+Dense+LR)",  "results/retrieval_complementary/{ds}-rrf-complementary.json"),
]


def load_metrics(path):
    if not Path(path).exists():
        return None
    data = json.load(open(path))
    return data.get('metrics', {})


def compute_bm25_metrics_if_missing():
    """BM25 alone results were saved without metrics by run_hybrid_official.
    Compute them inline."""
    import math
    for ds in DATASETS:
        bm25_path = f"results/retrieval_bm25/{ds}-bm25.json"
        if not Path(bm25_path).exists():
            continue
        data = json.load(open(bm25_path))
        if data.get('metrics'):
            continue  # already computed

        metrics = {}
        for k in KS:
            rec_sum = 0
            ndcg_sum = 0
            valid = 0
            for r in data['results']:
                gold = r.get('gold_skill_ids', [])
                if not gold:
                    continue
                retrieved_ids = [x['skill_id'] for x in r['retrieved'][:k]]
                gold_set = set(gold)
                rec = len(set(retrieved_ids) & gold_set) / len(gold_set)
                rec_sum += rec
                dcg = sum(1.0/math.log2(i+2) for i, sid in enumerate(retrieved_ids) if sid in gold_set)
                ideal = sum(1.0/math.log2(i+2) for i in range(min(len(gold_set), k)))
                ndcg = dcg/ideal if ideal > 0 else 0
                ndcg_sum += ndcg
                valid += 1
            if valid > 0:
                metrics[f'Recall@{k}'] = rec_sum / valid
                metrics[f'nDCG@{k}'] = ndcg_sum / valid

        data['metrics'] = metrics
        with open(bm25_path, 'w') as f:
            json.dump(data, f, indent=2)


def main():
    compute_bm25_metrics_if_missing()

    # ============================================================
    # PER-DATASET TABLES
    # ============================================================
    for ds in DATASETS:
        print(f"\n{'=' * 100}")
        print(f"  Dataset: {ds}")
        print('=' * 100)

        # Header
        cols = []
        for k in KS:
            cols.append(f"R@{k}")
            cols.append(f"nDCG@{k}")
        header = f"{'Method':<42}" + " | ".join(f"{c:>8}" for c in cols)
        print(header)
        print('-' * len(header))

        for name, tmpl in METHODS:
            path = tmpl.format(ds=ds)
            m = load_metrics(path)
            if m is None:
                row = f"{name:<42}" + " | ".join(f"{'  -   ':>8}" for _ in cols)
            else:
                vals = []
                for k in KS:
                    r = m.get(f'Recall@{k}', None)
                    n = m.get(f'nDCG@{k}', None)
                    vals.append(f"{r:.4f}" if r is not None else "  -   ")
                    vals.append(f"{n:.4f}" if n is not None else "  -   ")
                row = f"{name:<42}" + " | ".join(f"{v:>8}" for v in vals)
            print(row)

    # ============================================================
    # CROSS-METHOD SUMMARY: R@10 across datasets
    # ============================================================
    print(f"\n\n{'=' * 100}")
    print(f"  CROSS-METHOD SUMMARY: Recall@10 (higher is better)")
    print('=' * 100)
    header = f"{'Method':<42}" + " | ".join(f"{ds[:9]:>9}" for ds in DATASETS) + f" | {'Mean':>8}"
    print(header)
    print('-' * len(header))
    for name, tmpl in METHODS:
        vals = []
        nums = []
        for ds in DATASETS:
            m = load_metrics(tmpl.format(ds=ds))
            if m is None:
                vals.append("  -   ")
            else:
                r = m.get('Recall@10', 0)
                vals.append(f"{r:.4f}")
                nums.append(r)
        mean = sum(nums)/len(nums) if nums else 0
        row = f"{name:<42}" + " | ".join(f"{v:>9}" for v in vals) + f" | {mean:>8.4f}"
        print(row)

    print(f"\n{'=' * 100}")
    print(f"  CROSS-METHOD SUMMARY: Recall@50 (higher is better)")
    print('=' * 100)
    print(header)
    print('-' * len(header))
    for name, tmpl in METHODS:
        vals = []
        nums = []
        for ds in DATASETS:
            m = load_metrics(tmpl.format(ds=ds))
            if m is None:
                vals.append("  -   ")
            else:
                r = m.get('Recall@50', 0)
                vals.append(f"{r:.4f}")
                nums.append(r)
        mean = sum(nums)/len(nums) if nums else 0
        row = f"{name:<42}" + " | ".join(f"{v:>9}" for v in vals) + f" | {mean:>8.4f}"
        print(row)

    print(f"\n{'=' * 100}")
    print(f"  CROSS-METHOD SUMMARY: nDCG@10 (higher is better)")
    print('=' * 100)
    print(header)
    print('-' * len(header))
    for name, tmpl in METHODS:
        vals = []
        nums = []
        for ds in DATASETS:
            m = load_metrics(tmpl.format(ds=ds))
            if m is None:
                vals.append("  -   ")
            else:
                r = m.get('nDCG@10', 0)
                vals.append(f"{r:.4f}")
                nums.append(r)
        mean = sum(nums)/len(nums) if nums else 0
        row = f"{name:<42}" + " | ".join(f"{v:>9}" for v in vals) + f" | {mean:>8.4f}"
        print(row)


if __name__ == "__main__":
    main()
