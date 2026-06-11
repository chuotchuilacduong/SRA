"""Smoke tests.

Default (CPU, no model): validates candidate loading (dry-run) + pooling logic
with fake tensors. Use --real to additionally load Qwen3-4B and run ONE probe
end-to-end (requires GPU + transformers; H100).

    python -m experiments.skill_probe_hidden.smoke_test            # CPU checks
    python -m experiments.skill_probe_hidden.smoke_test --real     # + 1 real probe
"""

from __future__ import annotations

import argparse

import numpy as np

from experiments.skill_probe_hidden import config
from experiments.skill_probe_hidden.data_io import (
    build_probe_set,
    gold_skill_ids,
    load_instances,
    load_method_candidates,
    load_test_ids,
    pick_queries,
)


def test_dataio() -> None:
    ds = config.DATASETS[0]
    qids = pick_queries(ds, config.N_QUERIES, config.SEED)
    assert len(qids) == config.N_QUERIES, qids
    test_ids = set(load_test_ids(ds))
    assert set(qids) <= test_ids, "picked ids not in CE-HYRR test set"
    instances = load_instances(ds)
    m4 = load_method_candidates(ds, "M4")
    ce = load_method_candidates(ds, "CE-HYRR")
    for qid in qids:
        assert qid in m4 and qid in ce, f"{qid} missing in a method pool"
        jobs, meta = build_probe_set(instances[qid], ce[qid], config.TOP_K)
        n_cand = len(meta)
        has_gold = bool(gold_skill_ids(instances[qid]))
        assert len(jobs) == 1 + (1 if has_gold else 0) + n_cand, (qid, len(jobs), n_cand)
        assert jobs[0] == ("no_skill", [])
    print(f"  dataio OK: {ds} {len(qids)} queries, probe sets well-formed")


def test_pooling() -> None:
    import torch

    from experiments.skill_probe_hidden import pooling

    # fake hidden_states: 3 generation steps, 5 layers, hidden=8, batch=1
    n_layers, hidden = 5, 8
    def step(seq):
        return tuple(torch.randn(1, seq, hidden) for _ in range(n_layers))
    hs = (step(7), step(1), step(1))          # prefill(7) + 2 decode(1)
    pooled = pooling.last_token_pool(hs)
    assert pooled.shape == (n_layers, hidden), pooled.shape
    assert pooled.dtype == np.float16, pooled.dtype
    # last-token == last layer-stack of final step's last position
    expected = np.stack([hs[-1][L][0, -1, :].to(torch.float16).numpy() for L in range(n_layers)])
    assert np.allclose(pooled, expected), "last_token_pool mismatch"
    span = pooling.answer_span_mean_pool_lastlayer(hs)
    assert span.shape == (hidden,), span.shape
    sub = pooling.select_layers(pooled, [0, 2, 4])
    assert sub.shape == (3, hidden)
    print("  pooling OK: last_token + span_mean + select_layers shapes/values correct")


def test_real_probe() -> None:
    from sragents.corpus import load_corpus_dict

    from experiments.skill_probe_hidden.hf_generator import HFGenerator
    from experiments.skill_probe_hidden.probe_runner import HFProber

    ds = config.DATASETS[0]
    qid = pick_queries(ds, config.N_QUERIES, config.SEED)[0]
    inst = load_instances(ds)[qid]
    gen = HFGenerator(max_new_tokens=256)
    corpus = load_corpus_dict()
    out_dir = config.OUTPUT_ROOT / "_smoke"
    prober = HFProber(gen, corpus, out_dir / "cache.jsonl", out_dir / "hs")

    rec = prober.probe(inst, [])                       # no_skill probe
    assert isinstance(rec["v"], int) and rec["v"] in (0, 1)
    assert rec["raw_output_len"] >= 0
    arr = np.load(out_dir / "hs" / rec["hs_name"])
    assert arr.shape == (gen.num_layers + 1, gen.hidden_size), arr.shape
    assert arr.dtype == np.float16
    assert "<think>" not in rec["raw_output"], "thinking leaked despite enable_thinking=False"
    rec2 = prober.probe(inst, [])                      # must hit cache (no re-gen)
    assert rec2["_key"] == rec.get("_key", rec2["_key"])
    print(f"  real OK: probe v={rec['v']} hs_shape={arr.shape} dtype={arr.dtype} (cache hit on re-run)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="also run 1 real probe (needs GPU)")
    args = ap.parse_args()
    print("== smoke: dataio =="); test_dataio()
    print("== smoke: pooling =="); test_pooling()
    if args.real:
        print("== smoke: real probe =="); test_real_probe()
    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
