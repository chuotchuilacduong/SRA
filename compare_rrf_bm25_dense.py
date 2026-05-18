#!/usr/bin/env python3
"""Compare retrieval methods across 6 SRA-Bench datasets.

Computes Recall@k and nDCG@k for k in {1, 10, 50, 100} by re-reading
retrieved skill_ids from each method's saved JSON. Existing cache files
only have metrics up to @50, so we recompute @100 from the ranked lists.

Methods compared:
  - BM25
  - Dense (BGE-base-en-v1.5)
  - LinearRAG
  - Hybrid (BM25 + LinearRAG, round-robin merge)
  - RRF 3-way (BM25 + Dense + LinearRAG)
  - RRF Comp (BM25 + Dense)  <-- new
"""
import json
import math
from pathlib import Path

DATASETS = ["champ", "theoremqa", "logicbench", "toolqa", "medcalcbench", "bigcodebench"]
KS = [1, 10, 50, 100]

METHODS = [
    ("BM25",                "results/retrieval_bm25/{ds}-bm25.json"),
    ("Dense (BGE)",         "results/retrieval_dense/{ds}-dense-bge.json"),
    ("LinearRAG",           "results/retrieval_all/{ds}-linearrag.json"),
    ("Hybrid (BM25+LR)",    "results/retrieval_hybrid/{ds}-hybrid-bm25-linearrag.json"),
    ("RRF 3-way (BM25+Dense+LR)", "results/retrieval_complementary/{ds}-rrf-complementary.json"),
    ("RRF Comp (BM25+Dense)",     "results/retrieval_rrf_bm25_dense/{ds}-rrf-bm25-dense.json"),
]


def load_gold(ds):
    instances = json.load(open(f"data/bench/instances/{ds}.json"))
    return {
        inst.get("instance_id") or inst.get("id"):
        inst.get("skill_annotations") or inst.get("gold_skill_ids") or []
        for inst in instances
    }


def compute(results, gold_lookup, k):
    rec_sum, ndcg_sum, valid = 0.0, 0.0, 0
    for r in results:
        gold = gold_lookup.get(r["instance_id"], [])
        if not gold:
            continue
        retrieved_ids = [x["skill_id"] for x in r["retrieved"][:k]]
        gold_set = set(gold)
        rec = len(set(retrieved_ids) & gold_set) / len(gold_set)
        rec_sum += rec
        dcg = sum(
            1.0 / math.log2(i + 2)
            for i, sid in enumerate(retrieved_ids)
            if sid in gold_set
        )
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_set), k)))
        ndcg = dcg / ideal if ideal > 0 else 0
        ndcg_sum += ndcg
        valid += 1
    if not valid:
        return None, None
    return rec_sum / valid, ndcg_sum / valid


def main():
    table = {}  # method -> ds -> {k: (recall, ndcg)}
    coverage_note = {}  # method -> max top_k available

    for ds in DATASETS:
        gold = load_gold(ds)
        for method, pat in METHODS:
            path = pat.format(ds=ds)
            if not Path(path).exists():
                table.setdefault(method, {})[ds] = None
                continue
            d = json.load(open(path))
            results = d["results"]
            max_k = max(len(r["retrieved"]) for r in results) if results else 0
            coverage_note[method] = min(coverage_note.get(method, 1e9), max_k)
            row = {}
            for k in KS:
                rec, ndcg = compute(results, gold, k)
                row[k] = (rec, ndcg, max_k >= k)
            table.setdefault(method, {})[ds] = row

    # ------ Per-k tables ------
    for k in KS:
        print(f"\n{'=' * 96}")
        print(f"Recall@{k}  /  nDCG@{k}")
        print("=" * 96)
        header = f"{'Method':<28}" + "".join(f"{ds[:11]:>13}" for ds in DATASETS) + f"{'mean':>10}"
        print(header)
        print("-" * len(header))
        for method, _ in METHODS:
            row = table.get(method, {})
            cells = []
            recalls = []
            for ds in DATASETS:
                r = row.get(ds)
                if not r or r.get(k) is None or r[k][0] is None:
                    cells.append("    n/a    ")
                    continue
                rec, ndcg, ok = r[k]
                marker = "" if ok else "*"
                cells.append(f"{rec:.4f}/{ndcg:.3f}{marker}")
                if ok:
                    recalls.append(rec)
            mean_rec = sum(recalls) / len(recalls) if recalls else 0
            print(f"{method:<28}" + "".join(f"{c:>13}" for c in cells) + f"{mean_rec:>10.4f}")
        print("(* = method's saved top-k < requested k; numbers truncated to available list)")

    # ------ Mean recall per method per k ------
    print(f"\n{'=' * 96}")
    print("MEAN Recall@k across 6 datasets")
    print("=" * 96)
    header = f"{'Method':<28}" + "".join(f"{'R@' + str(k):>10}" for k in KS) + "".join(f"{'nDCG@' + str(k):>10}" for k in KS)
    print(header)
    print("-" * len(header))
    for method, _ in METHODS:
        row = table.get(method, {})
        means_r = []
        means_n = []
        for k in KS:
            recs = [row[ds][k][0] for ds in DATASETS if row.get(ds) and row[ds].get(k) and row[ds][k][2]]
            ndcgs = [row[ds][k][1] for ds in DATASETS if row.get(ds) and row[ds].get(k) and row[ds][k][2]]
            means_r.append(sum(recs) / len(recs) if recs else float("nan"))
            means_n.append(sum(ndcgs) / len(ndcgs) if ndcgs else float("nan"))
        line = f"{method:<28}"
        for v in means_r:
            line += f"{v:>10.4f}"
        for v in means_n:
            line += f"{v:>10.4f}"
        print(line)

    print("\nNotes:")
    print("  - BM25/LinearRAG/Hybrid/RRF-3way cache only top-50; @100 columns truncated to 50 (marked *).")
    print("  - Dense and RRF-Comp (BM25+Dense) have full top-100 coverage.")


if __name__ == "__main__":
    main()
