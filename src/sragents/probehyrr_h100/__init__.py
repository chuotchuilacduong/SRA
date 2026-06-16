"""High-throughput H100 + Qwen3-8B dataset-generation pipeline for CE-ProbeHYRR.

Implements the Option-A cold-start build described in
``docs/H100_Qwen3_8B_CE_ProbeHYRR_Dataset_Pipeline.md`` and the decisions in
``docs/H100_CE_ProbeHYRR_Implementation_Plan.md``.

Stage 0 (this package, CPU-only) prepares the inputs the spec assumes already
exist: merged train/dev/test query splits, a JSONL skill corpus, the
``m4_top50`` candidate cache (sliced from the existing RRF+KMeans pools), and the
anchor probe task manifest (``no_skill`` / ``gold_skill`` / ``m4_top1``).

Later stages (vLLM offline generation, CPU verification, micro-batch UCB,
label/pairs/groups builders) consume that manifest.
"""
