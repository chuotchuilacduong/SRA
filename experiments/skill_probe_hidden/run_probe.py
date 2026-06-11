"""CLI orchestration: probe Qwen3-4B over {no_skill, gold, top-10} for M4 & CE-HYRR.

Writes one JSON per (query, method) + pooled hidden-state .npy per unique probe,
de-duplicated/cached so shared probes (no_skill, gold, overlapping candidates)
run the model once. Resumable.

Output layout (under results/skill_probe_hidden/):
    <ds>/<qid>__<method>.json
    <ds>/hidden_states/<qid>__<skilltag>.npy
    <ds>/probe_cache.jsonl
    <ds>/index.json

Dry-run (no torch/model): prints the probe plan and validates candidate loading.
    python -m experiments.skill_probe_hidden.run_probe --dry-run
"""

from __future__ import annotations

import argparse
import json

from experiments.skill_probe_hidden import config
from experiments.skill_probe_hidden.data_io import (
    build_probe_set,
    gold_skill_ids,
    load_instances,
    load_method_candidates,
    pick_queries,
)


def _rel_hs(dataset: str, hs_name: str) -> str:
    return f"{dataset}/hidden_states/{hs_name}"


def _gold_rank(cand_meta: list[dict]) -> int | None:
    ranks = [c["rank"] for c in cand_meta if c["is_gold"]]
    return min(ranks) if ranks else None


def _assemble_json(dataset, method, instance, cand_meta, job_recs) -> dict:
    """job_recs: list of (probe_tag, skill_ids, rec) for this (query, method)."""
    by_tag = {}
    cand_recs = []
    for tag, sids, rec in job_recs:
        if tag == "cand":
            cand_recs.append(rec)
        else:
            by_tag[tag] = rec

    v_no = by_tag["no_skill"]["v"]
    gold_rec = by_tag.get("gold")
    v_gold = gold_rec["v"] if gold_rec else None

    candidates_out = []
    num_pos = 0
    best_pos_rank = None
    for meta, rec in zip(cand_meta, cand_recs):
        util = rec["v"] - v_no
        if util > 0:
            num_pos += 1
            if best_pos_rank is None:
                best_pos_rank = meta["rank"]
        candidates_out.append(
            {**meta, "v": rec["v"], "utility": util,
             "hidden_state_path": _rel_hs(dataset, rec["hs_name"])}
        )

    probes_out = {
        "no_skill": {
            "skill_ids": [], "v": v_no, "extracted": by_tag["no_skill"]["extracted"],
            "raw_output_len": by_tag["no_skill"]["raw_output_len"],
            "thinking_leaked": by_tag["no_skill"]["thinking_leaked"],
            "hidden_state_path": _rel_hs(dataset, by_tag["no_skill"]["hs_name"]),
        }
    }
    if gold_rec:
        probes_out["gold_skill"] = {
            "skill_ids": gold_skill_ids(instance), "v": v_gold,
            "utility": (v_gold - v_no) if v_gold is not None else None,
            "extracted": gold_rec["extracted"], "raw_output_len": gold_rec["raw_output_len"],
            "thinking_leaked": gold_rec["thinking_leaked"],
            "hidden_state_path": _rel_hs(dataset, gold_rec["hs_name"]),
        }

    return {
        "qid": instance["instance_id"], "dataset": dataset, "method": method,
        "generator_model": config.MODEL_ID, "generator_backend": "hf-transformers-4.46.3",
        "thinking_mode": False, "query": instance.get("question"),
        "gold_skill_ids": gold_skill_ids(instance),
        "candidates": candidates_out, "probes": probes_out,
        "metrics": {
            "v_no": v_no, "v_gold": v_gold,
            "gold_rank_in_candidates": _gold_rank(cand_meta),
            "gold_in_top10": any(c["is_gold"] for c in cand_meta),
            "num_positive_utility": num_pos, "best_candidate_rank": best_pos_rank,
            "score_field_meaning": config.SCORE_MEANING[method],
            "hidden_dim": by_tag["no_skill"]["hidden_state_shape"][-1],
            "num_layers": by_tag["no_skill"]["hidden_state_shape"][0],
            "pool": "last_token", "dtype": "float16",
        },
    }


def run_dry(args) -> None:
    for ds in args.datasets:
        qids = pick_queries(ds, args.n_queries, args.seed)
        instances = load_instances(ds)
        cand = {m: load_method_candidates(ds, m) for m in args.methods}
        print(f"\n=== {ds}: {len(qids)} queries ===")
        for qid in qids:
            inst = instances[qid]
            line = [f"  {qid}  gold={gold_skill_ids(inst)}"]
            for m in args.methods:
                assert qid in cand[m], f"{qid} missing in {m} candidates"
                jobs, meta = build_probe_set(inst, cand[m][qid], config.TOP_K)
                line.append(f"{m}:{len(jobs)}probes(top{len(meta)})")
            print("  " + " | ".join(line))
    print("\nDRY-RUN OK (no model loaded).")


def run_real(args) -> None:
    from sragents.corpus import load_corpus_dict

    from experiments.skill_probe_hidden.hf_generator import HFGenerator
    from experiments.skill_probe_hidden.probe_runner import HFProber

    corpus = load_corpus_dict()
    gen = HFGenerator(max_new_tokens=args.max_new_tokens)
    print(f"loaded {gen.model_id} on {gen.device} (layers+1={gen.num_layers + 1}, hidden={gen.hidden_size})")
    keep_layers = config.layer_indices(gen.num_layers + 1, args.layer_policy)

    for ds in args.datasets:
        out_dir = config.OUTPUT_ROOT / ds
        prober = HFProber(
            generator=gen, corpus=corpus,
            cache_path=out_dir / "probe_cache.jsonl",
            hs_dir=out_dir / "hidden_states",
            pool=("span_mean" if args.pool == "span_mean" else "last_token"),
            keep_layers=(None if args.layer_policy == "all" else keep_layers),
        )
        qids = pick_queries(ds, args.n_queries, args.seed)
        instances = load_instances(ds)
        cand = {m: load_method_candidates(ds, m) for m in args.methods}
        index = {"dataset": ds, "seed": args.seed, "qids": qids, "files": []}

        for qi, qid in enumerate(qids, 1):
            inst = instances[qid]
            for method in args.methods:
                jpath = out_dir / f"{qid}__{method}.json"
                if jpath.exists() and not args.force:
                    index["files"].append(str(jpath.relative_to(config.OUTPUT_ROOT)))
                    continue
                jobs, meta = build_probe_set(inst, cand[method].get(qid, []), config.TOP_K)
                job_recs = [(tag, sids, prober.probe(inst, sids)) for (tag, sids) in jobs]
                obj = _assemble_json(ds, method, inst, meta, job_recs)
                out_dir.mkdir(parents=True, exist_ok=True)
                jpath.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
                index["files"].append(str(jpath.relative_to(config.OUTPUT_ROOT)))
            print(f"  [{ds}] {qi}/{len(qids)} {qid} done", flush=True)

        (out_dir / "index.json").write_text(json.dumps(index, indent=2))
        print(f"[{ds}] wrote {len(index['files'])} method-JSONs -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Skill-probe + hidden-state capture (Qwen3-4B)")
    ap.add_argument("--datasets", nargs="*", default=config.DATASETS)
    ap.add_argument("--methods", nargs="*", default=config.METHODS)
    ap.add_argument("--n-queries", type=int, default=config.N_QUERIES)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    ap.add_argument("--layer-policy", default=config.LAYER_POLICY, choices=["all", "every4"])
    ap.add_argument("--pool", default="last_token", choices=["last_token", "span_mean"])
    ap.add_argument("--force", action="store_true", help="overwrite existing method JSONs")
    ap.add_argument("--dry-run", action="store_true", help="no model; validate candidate loading")
    args = ap.parse_args()

    if args.dry_run:
        run_dry(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
