"""``sragents mine-negatives`` — produce train_pairs.json from a pool JSON."""

from __future__ import annotations

import json
from pathlib import Path

from sragents.cli._common import require_exists
from sragents.config import PROJECT_ROOT
from sragents.corpus import load_corpus_dict
from sragents.data_config import load_data_config
from sragents.train.negative_sampler import NegativeMix, mine_from_pool

_HELP = (
    "Mine HYRR-style hybrid hard negatives from a candidate pool JSON. "
    "Default mix: 4 BM25-hard + 4 BGE/RRF-hard + 2 random per query."
)


def add_parser(subparsers) -> None:
    p = subparsers.add_parser("mine-negatives", help="Mine hybrid hard negatives", description=_HELP)
    p.add_argument("--pool", type=Path, required=True,
                   help="Pool JSON (output of `sragents build-pool`)")
    p.add_argument("--instances", type=Path, required=True,
                   help="Instances JSON for the same dataset")
    p.add_argument("--corpus", type=Path, default=None,
                   help="Optional corpus override; defaults to configs/data.yaml")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSON path (array of train pairs)")
    p.add_argument("--ratio", default="4:4:2",
                   help="Negatives ratio bm25:bge:random (default 4:4:2). "
                        "Append a 4th to add cluster-hard negatives, e.g. 4:4:2:2.")
    p.add_argument("--clusters", type=Path, default=None,
                   help="Optional clusters.json (skill_id -> cluster_id)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--restrict-instance-ids", type=Path, default=None,
                   help="Optional JSON with {\"train\":[...]} or a flat list "
                        "of instance_ids; restrict mining to those.")
    p.add_argument("--data-config", type=Path,
                   default=PROJECT_ROOT / "configs" / "data.yaml")
    p.set_defaults(func=run)


def _parse_ratio(s: str) -> NegativeMix:
    parts = [int(x) for x in s.split(":")]
    if len(parts) == 3:
        bm, bg, rn = parts
        return NegativeMix(bm25_hard=bm, bge_hard=bg, random=rn)
    if len(parts) == 4:
        bm, bg, rn, cl = parts
        return NegativeMix(bm25_hard=bm, bge_hard=bg, random=rn, cluster_hard=cl)
    raise SystemExit(f"--ratio must have 3 or 4 ints, got {s!r}")


def _load_restrict(p: Path | None) -> set[str] | None:
    if p is None:
        return None
    doc = json.loads(p.read_text())
    if isinstance(doc, dict):
        ids: list[str] = []
        for v in doc.values():
            if isinstance(v, list):
                ids.extend(v)
        return set(ids)
    if isinstance(doc, list):
        return set(doc)
    raise SystemExit(f"unrecognized restrict file shape: {p}")


def run(args) -> None:
    require_exists(args.pool, "pool")
    require_exists(args.instances, "instances")
    data_cfg = load_data_config(args.data_config)

    corpus_dict = load_corpus_dict(args.corpus) if args.corpus else load_corpus_dict()
    instances = json.loads(args.instances.read_text())
    instances_by_id = {data_cfg.instance_query_id(i): i for i in instances}

    restrict = _load_restrict(args.restrict_instance_ids)
    if restrict is not None:
        instances_by_id = {k: v for k, v in instances_by_id.items() if k in restrict}

    mix = _parse_ratio(args.ratio)
    clusters = None
    if args.clusters:
        require_exists(args.clusters, "clusters")
        clusters = json.loads(args.clusters.read_text())

    pairs = mine_from_pool(
        args.pool, corpus_dict, instances_by_id,
        question_field=data_cfg.f_query,
        mix=mix, seed=args.seed, clusters=clusters,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([p.to_dict() for p in pairs], ensure_ascii=False, indent=2))

    n_pos = sum(1 for p in pairs if p.label == 1)
    counts: dict[str, int] = {}
    for p in pairs:
        counts[p.negative_source] = counts.get(p.negative_source, 0) + 1
    print(f"Wrote {len(pairs)} pairs ({n_pos} positives) -> {args.out}")
    print("  by source: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
