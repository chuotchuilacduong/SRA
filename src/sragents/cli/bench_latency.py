"""``sragents bench-latency`` — measure CE reranker latency & throughput."""

from __future__ import annotations

from pathlib import Path

from sragents.cli._common import require_exists
from sragents.config import PROJECT_ROOT
from sragents.data_config import load_data_config
from sragents.eval.latency_benchmark import run_full_benchmark


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "bench-latency", help="Measure CE reranker latency and throughput",
        description="Run a warmup + measured loop on a pool JSON and write a "
                    "report with p50/p95/p99 latency and QPS.",
    )
    p.add_argument("--model", required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--instances", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    p.add_argument("--batch", type=int, nargs="+", default=[1, 32])
    p.add_argument("--top-k", type=int, nargs="+", default=[50, 100])
    p.add_argument("--packing", default="field_tagged")
    p.add_argument("--maxp", action="store_true")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--measure-steps", type=int, default=50)
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--data-config", type=Path,
                   default=PROJECT_ROOT / "configs" / "data.yaml")
    p.set_defaults(func=run)


def run(args) -> None:
    require_exists(args.pool, "pool")
    require_exists(args.instances, "instances")
    data_cfg = load_data_config(args.data_config)
    run_full_benchmark(
        model=args.model,
        pool_path=args.pool,
        instances_path=args.instances,
        output=args.output,
        device=args.device,
        packing=args.packing,
        use_maxp=args.maxp,
        batch_sizes=args.batch,
        top_ks=args.top_k,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        max_length=args.max_length,
        corpus_path=args.corpus,
        question_field=data_cfg.f_query,
        instance_id_field=data_cfg.f_query_id,
    )
