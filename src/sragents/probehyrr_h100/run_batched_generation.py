"""Stage 2/6 — vLLM offline batched generation over a probe task manifest.

Reads a ``*_probe_tasks_{split}.jsonl`` manifest (anchor or UCB), applies the
Qwen chat template (``enable_thinking=False``), sorts/buckets prompts for
throughput, and batch-generates with vLLM offline inference. Append-only JSONL
output, resumable by ``task_id``.

GPU path runs on the H100 in the dedicated ``vllm`` conda env. ``--dry-run``
needs no GPU / no model / no vLLM (uses a char-length proxy for bucketing) so the
batching plan can be validated on a CPU dev node.

Examples
--------
    # dry-run on a CPU node (prints the plan, writes nothing)
    python -m sragents.probehyrr_h100.run_batched_generation \
        --tasks results/ce_probehyrr_h100/tasks/anchor_probe_tasks_train.jsonl \
        --out  results/ce_probehyrr_h100/generations/anchor_outputs_train.jsonl \
        --dry-run --limit-queries 500

    # real run on H100
    python -m sragents.probehyrr_h100.run_batched_generation \
        --tasks ... --out ... --model Qwen/Qwen3-8B --dtype bfloat16 \
        --gpu-memory-utilization 0.90 --max-model-len 8192 \
        --enable-prefix-caching --limit-queries 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from sragents.probehyrr_h100 import config as C
from sragents.probehyrr_h100.io_utils import iter_jsonl, write_jsonl
from sragents.probehyrr_h100.prompt_templates import render_chat_template

# Input-token length buckets (spec §2.4): short/medium/long/xlong.
_BUCKETS = ((1024, "short"), (2048, "medium"), (4096, "long"))


def _bucket(n: int) -> str:
    for hi, name in _BUCKETS:
        if n <= hi:
            return name
    return "xlong"


def _done_task_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    return {r["task_id"] for r in iter_jsonl(out_path)}


def _limit_queries(tasks: list[dict], n: int | None) -> list[dict]:
    """Keep tasks for a deterministic stratified sample of ``n`` distinct qids.

    Split the budget evenly across datasets (so the pilot exercises every
    verifier, incl. the theoremqa 1024-token path), taking the lowest-id qids
    per dataset for reproducibility.
    """
    if not n:
        return tasks
    from collections import defaultdict
    qids_by_ds: dict[str, set[str]] = defaultdict(set)
    for t in tasks:
        qids_by_ds[t["dataset"]].add(t["qid"])
    datasets = sorted(qids_by_ds)
    per, rem = divmod(n, len(datasets))
    keep: set[str] = set()
    for i, ds in enumerate(datasets):
        k = per + (1 if i < rem else 0)
        keep |= set(sorted(qids_by_ds[ds])[:k])
    return [t for t in tasks if t["qid"] in keep]


def annotate_lengths(tasks: list[dict], length_fn) -> None:
    for t in tasks:
        t["_input_len"] = length_fn(t)
        t["_bucket"] = _bucket(t["_input_len"])


def sort_for_throughput(tasks: list[dict]) -> list[dict]:
    """Sort by (dataset, probe_kind, max_new_tokens, input-length bucket).

    Same dataset/probe_kind → similar output length + more prefix sharing;
    similar length → less padding waste.
    """
    return sorted(
        tasks,
        key=lambda t: (
            t.get("dataset", ""),
            t.get("probe_kind", ""),
            t["generation_config"].get("max_new_tokens", C.DEFAULT_MAX_NEW_TOKENS),
            t["_input_len"],
        ),
    )


def chunk_by_sampling_params(tasks: list[dict], max_per_chunk: int):
    """Yield contiguous chunks sharing one (max_new_tokens, temperature, top_p)."""
    chunk: list[dict] = []
    last_key = None
    for t in tasks:
        cfg = t["generation_config"]
        key = (cfg.get("max_new_tokens"), cfg.get("temperature"), cfg.get("top_p"))
        if (last_key is not None and key != last_key) or len(chunk) >= max_per_chunk:
            if chunk:
                yield chunk
            chunk, last_key = [], None
        chunk.append(t)
        last_key = key
    if chunk:
        yield chunk


def _row_from_output(task: dict, text: str, finish_reason: str, n_out: int) -> dict:
    return {
        "task_id": task["task_id"],
        "qid": task["qid"],
        "dataset": task["dataset"],
        "split": task.get("split"),
        "probe_kind": task["probe_kind"],
        "skill_ids": task.get("skill_ids", []),
        "prompt_hash": task.get("prompt_hash"),
        "cache_key": task.get("cache_key"),
        "raw_output": text,
        "finish_reason": finish_reason,
        "num_output_tokens": n_out,
        "input_tokens": task.get("_input_len"),
        "generation_config": task["generation_config"],
        "metadata": task.get("metadata", {}),
    }


def _plan_summary(chunks: list[list[dict]], tasks: list[dict]) -> str:
    from collections import Counter
    bks = Counter(t["_bucket"] for t in tasks)
    mnt = Counter(t["generation_config"]["max_new_tokens"] for t in tasks)
    lines = [
        f"  tasks to generate: {len(tasks)}",
        f"  chunks: {len(chunks)} (max_per_chunk applied)",
        f"  length buckets: {dict(bks)}",
        f"  max_new_tokens histogram: {dict(sorted(mnt.items()))}",
        f"  est. output tokens (upper bound): "
        f"{sum(t['generation_config']['max_new_tokens'] for t in tasks):,}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="vLLM offline batched probe generation")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=C.MODEL_ID)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--enable-prefix-caching", action="store_true")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--generation-config", default="vllm",
                    help="vLLM generation_config source ('vllm'|'auto'); pass 'auto' on older vLLM")
    ap.add_argument("--max-tasks-per-chunk", type=int, default=2048)
    ap.add_argument("--limit-queries", type=int, default=None,
                    help="cap to the first N distinct qids (pilot)")
    ap.add_argument("--limit", type=int, default=None, help="cap total tasks (debug)")
    ap.add_argument("--force", action="store_true", help="ignore existing outputs (no resume)")
    ap.add_argument("--dry-run", action="store_true", help="plan only; no GPU/model/vLLM")
    args = ap.parse_args()

    out_path = Path(args.out)
    tasks = list(iter_jsonl(args.tasks))
    tasks = _limit_queries(tasks, args.limit_queries)
    if args.limit:
        tasks = tasks[: args.limit]

    # Resume: drop already-generated task_ids.
    done = set() if args.force else _done_task_ids(out_path)
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"[gen] manifest={len(tasks)} done={len(done)} todo={len(todo)} -> {out_path}")
    if not todo:
        print("[gen] nothing to do.")
        return

    if args.dry_run:
        # Char-length proxy (no tokenizer needed) for the batching plan.
        annotate_lengths(todo, lambda t: len(t.get("system", "")) + len(t.get("user", "")))
        todo = sort_for_throughput(todo)
        chunks = list(chunk_by_sampling_params(todo, args.max_tasks_per_chunk))
        print("[gen] DRY-RUN plan (char-length proxy):")
        print(_plan_summary(chunks, todo))
        return

    # --- real GPU path ---
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def tok_len(t: dict) -> int:
        prompt = render_chat_template(t.get("system", ""), t["user"], tok)
        t["_prompt_text"] = prompt
        return len(tok(prompt).input_ids)

    print("[gen] templating + token-length annotation ...")
    annotate_lengths(todo, tok_len)

    # Drop prompts that cannot fit in the model window; otherwise a single
    # oversized prompt aborts the whole llm.generate() chunk with
    # "decoder prompt ... is longer than the maximum model length".
    fit, dropped = [], []
    for t in todo:
        mnt = int(t["generation_config"].get("max_new_tokens", C.DEFAULT_MAX_NEW_TOKENS))
        if t["_input_len"] + mnt > args.max_model_len:
            dropped.append((t["task_id"], t["_input_len"], mnt))
        else:
            fit.append(t)
    if dropped:
        ex = ", ".join(f"{tid}({n}+{m})" for tid, n, m in dropped[:5])
        print(f"[gen] WARNING: skipping {len(dropped)}/{len(todo)} task(s) with "
              f"input+max_new_tokens > max_model_len={args.max_model_len}: {ex}"
              f"{' ...' if len(dropped) > 5 else ''}", flush=True)
    todo = fit
    if not todo:
        print("[gen] nothing to do after length filtering.")
        return

    todo = sort_for_throughput(todo)
    chunks = list(chunk_by_sampling_params(todo, args.max_tasks_per_chunk))
    print(_plan_summary(chunks, todo))

    llm_kwargs = dict(
        model=args.model,
        dtype=args.dtype,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=args.enable_prefix_caching,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    if args.generation_config:
        llm_kwargs["generation_config"] = args.generation_config
    llm = LLM(**llm_kwargs)

    t0 = time.time()
    n_done = n_out_tokens = 0
    for i, chunk in enumerate(chunks, 1):
        cfg = chunk[0]["generation_config"]
        sp = SamplingParams(
            temperature=float(cfg.get("temperature", 0.0)),
            top_p=float(cfg.get("top_p", 1.0)),
            max_tokens=int(cfg.get("max_new_tokens", C.DEFAULT_MAX_NEW_TOKENS)),
        )
        prompts = [t["_prompt_text"] for t in chunk]
        outputs = llm.generate(prompts, sp)
        rows = []
        for task, out in zip(chunk, outputs):
            o = out.outputs[0]
            rows.append(_row_from_output(task, o.text, o.finish_reason, len(o.token_ids)))
            n_out_tokens += len(o.token_ids)
        write_jsonl(out_path, rows, append=True)
        n_done += len(rows)
        dt = time.time() - t0
        print(f"[gen] chunk {i}/{len(chunks)}  +{len(rows)}  total={n_done}  "
              f"{n_done/dt:.1f} gen/s  {n_out_tokens/dt:.0f} out_tok/s", flush=True)

    print(f"[gen] done: {n_done} generations in {time.time()-t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
