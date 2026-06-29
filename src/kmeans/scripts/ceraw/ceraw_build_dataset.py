"""CE-Raw / SkillRouter Phase 1 — build a *raw* training dataset (no stage-1 pool).

Unlike the existing CE (ce-joint-v3) whose negatives are mined from the M4 stage-1
`extended-{ds}.json` top-100 pool, here every negative is mined from the FULL 26,262-skill
corpus via independent BM25 + BGE retrieval + cluster + random, then passed through the
SkillRouter false-negative filter (name dedup / body trigram-Jaccard>0.6 / BGE cosine>0.92).

Trains on query_gen-train; dev pool (for the trainer's nDCG early-stop) is the full-corpus
BGE@100 ranking over query_gen-dev. Outputs everything under data/ce_raw/ :

  ce_pairs_train.json   listwise CE pairs  {instance_id,dataset,question,skill,label,gold_skill_ids}
  dev_pool.json         {"results":[{instance_id,gold_skill_ids,retrieved:[{skill_id}]}]}
  instances_all.json    concatenated instances (for the trainer --instances)
  de_triples_train.jsonl {query, positive_id, negative_ids}  (for Phase B retriever)
  stats.json / README.md

CPU-only; no GPU needed. Run anywhere (locally to inspect, or on H100).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from sragents.config import PROJECT_ROOT
from sragents.corpus import load_corpus_dict
from kmeans.qsc_ltr_runner import load_ext
from kmeans import io

OUT = PROJECT_ROOT / "data" / "ce_raw"
SEED = 42


def _trigrams(text: str) -> set:
    t = " ".join(text.lower().split())
    return {t[i:i + 3] for i in range(len(t) - 2)} if len(t) >= 3 else {t}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _skill_text(s: dict, cap: int = 2000) -> str:
    return f"{s.get('name','')} {s.get('description','')} {str(s.get('content',''))[:cap]}"


def _read_list(path: Path) -> dict:
    """retrieval json -> {instance_id: [skill_id ranked]}"""
    d = json.loads(path.read_text())
    return {r["instance_id"]: [c["skill_id"] for c in r["retrieved"]] for r in d["results"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build raw (no stage-1) CE training dataset")
    ap.add_argument("--bm25", type=int, default=4)
    ap.add_argument("--bge", type=int, default=3)
    ap.add_argument("--cluster", type=int, default=2)
    ap.add_argument("--random", type=int, default=1)
    ap.add_argument("--jaccard", type=float, default=0.6)
    ap.add_argument("--cos", type=float, default=0.92)
    args = ap.parse_args()
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    ext, cfg = load_ext()
    datasets = ext["datasets"]
    corpus = load_corpus_dict()
    clusters = json.loads((PROJECT_ROOT / "results/clusters.json").read_text())  # skill_id -> cluster_id
    # cluster -> [skill_id]
    cl_members: dict = {}
    for sid, cid in clusters.items():
        cl_members.setdefault(cid, []).append(sid)
    # corpus_emb (normalised) for the cosine false-negative filter
    emb = np.load(PROJECT_ROOT / "results/bge/corpus_emb.npy").astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    row = {sid: i for i, sid in enumerate(json.loads((PROJECT_ROOT / "results/bge/corpus_ids.json").read_text()))}
    all_ids = list(row.keys())

    splits = {ds: json.loads((PROJECT_ROOT / f"results/splits/{ds}-query_gen.json").read_text()) for ds in datasets}
    train_ids = {q for ds in datasets for q in splits[ds]["train"]}
    dev_ids = {q for ds in datasets for q in splits[ds].get("dev", [])}

    # instances (question text + gold) and concat for trainer
    inst_by_id, all_instances = {}, []
    for ds in datasets:
        recs = io.load_instances(cfg, ds)
        for r in recs:
            inst_by_id[r["instance_id"]] = {"question": r["query"], "dataset": ds,
                                            "gold_skill_ids": r.get("gold_skill_ids") or []}
        all_instances += [{"instance_id": r["instance_id"], "question": r["query"],
                           "dataset": ds, "gold_skill_ids": r.get("gold_skill_ids") or []} for r in recs]

    bm25 = {ds: _read_list(PROJECT_ROOT / f"results/retrieval_bm25/{ds}-bm25.json") for ds in datasets}
    bge = {ds: _read_list(PROJECT_ROOT / f"results/retrieval_dense/{ds}-dense-bge.json") for ds in datasets}

    def is_false_neg(nid: str, gold_set: list) -> bool:
        if nid not in corpus:
            return True
        ns = corpus[nid]
        ntri = _trigrams(_skill_text(ns))
        nname = (ns.get("name") or "").strip().lower()
        ni = row.get(nid)
        for g in gold_set:
            gs = corpus.get(g)
            if gs is None:
                continue
            if nname and nname == (gs.get("name") or "").strip().lower():
                return True
            if _jaccard(ntri, _trigrams(_skill_text(gs))) > args.jaccard:
                return True
            gi = row.get(g)
            if ni is not None and gi is not None and float(emb[ni] @ emb[gi]) > args.cos:
                return True
        return False

    pairs, triples = [], []
    n_filtered = 0
    src_counts: dict = {}
    for ds in datasets:
        for iid in splits[ds]["train"]:
            inst = inst_by_id.get(iid)
            if inst is None:
                continue
            gold = [g for g in inst["gold_skill_ids"] if g in corpus]
            if not gold:
                continue
            q = inst["question"]
            gold_set = set(gold)
            # candidate negative sources (full corpus), gold-excluded
            def take(cands, n, label):
                out = []
                for sid in cands:
                    if len(out) >= n:
                        break
                    if sid in gold_set or sid in out:
                        continue
                    if is_false_neg(sid, gold):
                        nonlocal_filtered()
                        continue
                    out.append(sid)
                return out
            filtered_box = {"n": 0}

            def nonlocal_filtered():
                filtered_box["n"] += 1

            negs = []
            negs += [(s, "bm25_hard") for s in take(bm25[ds].get(iid, []), args.bm25, "bm25")]
            negs += [(s, "bge_hard") for s in take(bge[ds].get(iid, []), args.bge, "bge")]
            # cluster-hard: same cluster as any gold
            cl_pool = []
            for g in gold:
                cl_pool += cl_members.get(clusters.get(g, -999), [])
            rng.shuffle(cl_pool)
            negs += [(s, "cluster_hard") for s in take(cl_pool, args.cluster, "cluster")]
            # random: different cluster
            gold_clusters = {clusters.get(g) for g in gold}
            rand_pool = [s for s in (rng.sample(all_ids, min(len(all_ids), 200)))
                         if clusters.get(s) not in gold_clusters]
            negs += [(s, "random") for s in take(rand_pool, args.random, "random")]
            n_filtered += filtered_box["n"]

            taken = {sid for sid, _ in negs}
            # emit pairs
            for g in gold:
                pairs.append({"instance_id": iid, "dataset": ds, "question": q,
                              "skill": corpus[g], "label": 1.0, "gold_skill_ids": gold,
                              "negative_source": "positive"})
                src_counts["positive"] = src_counts.get("positive", 0) + 1
            for sid, src in negs:
                pairs.append({"instance_id": iid, "dataset": ds, "question": q,
                              "skill": corpus[sid], "label": 0.0, "gold_skill_ids": gold,
                              "negative_source": src})
                src_counts[src] = src_counts.get(src, 0) + 1
            triples.append({"query": q, "positive_id": gold[0], "negative_ids": list(taken)})

    # dev pool = full-corpus BGE@100 over query_gen-dev (trainer nDCG early-stop signal)
    dev_results = []
    for ds in datasets:
        for iid in splits[ds].get("dev", []):
            inst = inst_by_id.get(iid)
            if inst is None:
                continue
            dev_results.append({"instance_id": iid, "gold_skill_ids": inst["gold_skill_ids"],
                                "retrieved": [{"skill_id": s} for s in bge[ds].get(iid, [])[:100]]})

    (OUT / "ce_pairs_train.json").write_text(json.dumps(pairs))
    (OUT / "dev_pool.json").write_text(json.dumps({"results": dev_results}))
    (OUT / "instances_all.json").write_text(json.dumps(all_instances))
    with (OUT / "de_triples_train.jsonl").open("w") as f:
        for t in triples:
            f.write(json.dumps(t) + "\n")
    stats = {"train_queries": len(triples), "dev_queries": len(dev_results),
             "pairs": len(pairs), "source_counts": src_counts,
             "false_negatives_removed": n_filtered,
             "mix": {"bm25": args.bm25, "bge": args.bge, "cluster": args.cluster, "random": args.random},
             "filter": {"jaccard": args.jaccard, "cos": args.cos}}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    (OUT / "README.md").write_text(
        "# CE-Raw training dataset (no stage-1 pool)\n\n"
        f"{stats['pairs']} listwise pairs over {stats['train_queries']} query_gen-train queries; "
        f"negatives mined from the FULL 26,262-skill corpus (bm25/bge/cluster/random) with "
        f"SkillRouter false-negative filtering ({n_filtered} removed). dev_pool = full-corpus "
        f"BGE@100 over {stats['dev_queries']} query_gen-dev queries.\n\n"
        "Feed to the trainer: ce_pairs_train.json (--train-pairs), dev_pool.json (--dev-pool), "
        "instances_all.json (--instances). de_triples_train.jsonl feeds the Phase-B retriever.\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
