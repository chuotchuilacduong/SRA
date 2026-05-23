"""Latency / throughput benchmark for the cross-encoder reranker.

* batch=1 : p50 / p95 — online single-query latency.
* batch=32: throughput — queries-per-second under bulk reranking.

Measured numbers go straight to a JSON report — never hardcode them.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sragents.corpus import load_corpus_dict
from sragents.retrieve.chunking import MaxPChunker
from sragents.retrieve.cross_rerank import CrossEncoderReranker
from sragents.retrieve.skill_packer import SkillPacker


@dataclass
class LatencyReport:
    model: str
    device: str
    top_k: int
    batch_size: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    throughput_qps: float
    n_queries: int


def benchmark_reranker(
    reranker: CrossEncoderReranker,
    samples: list[dict],
    *,
    top_k: int,
    batch_size: int,
    warmup_steps: int = 5,
    measure_steps: int = 50,
) -> LatencyReport:
    """Run a latency loop.

    ``samples`` is a list of ``{question, candidates}`` dicts. The loop
    cycles through them (modulo) until ``measure_steps`` queries have run.
    """
    if not samples:
        raise ValueError("no samples to benchmark")

    # Warmup (not measured) — important for JIT/kernel selection.
    for i in range(warmup_steps):
        s = samples[i % len(samples)]
        reranker.rerank(s["question"], s["candidates"], top_k=top_k, batch_size=batch_size)

    times: list[float] = []
    start = time.time()
    for i in range(measure_steps):
        s = samples[i % len(samples)]
        t0 = time.perf_counter()
        reranker.rerank(s["question"], s["candidates"], top_k=top_k, batch_size=batch_size)
        times.append((time.perf_counter() - t0) * 1000.0)
    wall = time.time() - start

    times_sorted = sorted(times)
    n = len(times_sorted)
    p50 = times_sorted[max(0, n // 2 - 1)]
    p95 = times_sorted[max(0, int(n * 0.95) - 1)]
    p99 = times_sorted[max(0, int(n * 0.99) - 1)]
    return LatencyReport(
        model=getattr(reranker.model, "model_name_or_path", "unknown"),
        device=reranker.device,
        top_k=top_k,
        batch_size=batch_size,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        mean_ms=statistics.mean(times),
        throughput_qps=measure_steps / wall if wall > 0 else 0.0,
        n_queries=measure_steps,
    )


def build_samples_from_pool(
    pool_records: Sequence[dict],
    instances_by_id: dict[str, dict],
    question_field: str = "question",
    top_k: int = 100,
) -> list[dict]:
    """Convert pool records to ``{question, candidates}`` dicts."""
    out: list[dict] = []
    for r in pool_records:
        inst = instances_by_id.get(r["instance_id"])
        if inst is None:
            continue
        out.append({
            "question": inst.get(question_field) or "",
            "candidates": r.get("retrieved", [])[:top_k],
        })
    return out


def run_full_benchmark(
    *,
    model: str,
    pool_path: Path,
    instances_path: Path,
    output: Path,
    device: str = "cpu",
    packing: str = "field_tagged",
    include_tools: bool = False,
    max_content_chars: int = 1800,
    use_maxp: bool = False,
    batch_sizes: Sequence[int] = (1, 32),
    top_ks: Sequence[int] = (50, 100),
    warmup_steps: int = 5,
    measure_steps: int = 50,
    max_length: int = 256,
    corpus_path: Path | None = None,
    question_field: str = "question",
    instance_id_field: str = "instance_id",
) -> dict:
    """Driver used by ``sragents bench-latency``.

    Builds one shared :class:`CrossEncoderReranker` and iterates the cross
    product of (batch_size, top_k). Returns the report dict that's written
    to ``output``.
    """
    pool = json.loads(Path(pool_path).read_text())
    instances = json.loads(Path(instances_path).read_text())
    instances_by_id = {i[instance_id_field]: i for i in instances}
    corpus = load_corpus_dict(corpus_path) if corpus_path else load_corpus_dict()

    packer = SkillPacker(
        mode=packing,
        include_tools=include_tools,
        max_content_chars=max_content_chars,
    )
    chunker = MaxPChunker() if use_maxp else None
    reranker = CrossEncoderReranker(
        model_name=model, device=device, max_length=max_length,
        packer=packer, chunker=chunker, corpus=corpus,
    )

    runs: list[dict] = []
    for top_k in top_ks:
        samples = build_samples_from_pool(
            pool["results"], instances_by_id,
            question_field=question_field, top_k=top_k,
        )
        for batch_size in batch_sizes:
            report = benchmark_reranker(
                reranker, samples,
                top_k=top_k, batch_size=batch_size,
                warmup_steps=warmup_steps, measure_steps=measure_steps,
            )
            row = report.__dict__ | {"model": model}
            print(f"  top_k={top_k:>3} batch={batch_size:>2}  "
                  f"p50={report.p50_ms:.1f}ms  p95={report.p95_ms:.1f}ms  "
                  f"throughput={report.throughput_qps:.2f} q/s")
            runs.append(row)

    out_doc = {
        "model": model, "device": device, "packing": packing,
        "use_maxp": use_maxp, "pool": str(pool_path),
        "instances": str(instances_path),
        "n_pool_records": len(pool["results"]),
        "runs": runs,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out_doc, indent=2))
    return out_doc
