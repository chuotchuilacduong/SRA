"""Per-query skill-probing + hidden-state analysis (Qwen3-4B HF).

For 10 test queries per dataset, probe a local HuggingFace causal LM (Qwen3-4B)
under {no_skill, gold, top-10 candidate} skill injections for two retrieval
methods (M4, CE-HYRR); record output + correctness + per-candidate utility, and
capture the model's pooled hidden states per probe for separability analysis.

Hidden states REQUIRE a local HF model — the Ollama/vLLM path used elsewhere in
this repo cannot return them. Runs on CUDA (H100); the candidate-loading
(`data_io`) and analysis (`analyze`) parts run CPU-only.

Modules
-------
config        : constants (model id, datasets, paths, layer policy).
data_io       : no-LLM candidate/query loading + probe-set assembly.
pooling       : hidden-state pooling (last-token / answer-span).
hf_generator  : the ONLY module that loads torch/transformers + the model.
probe_runner  : HFProber — cache + build_prompt + generate + evaluate + utility + .npy.
run_probe     : CLI orchestration.
analyze       : CLI analysis/visualization (CPU).
smoke_test    : 1-query end-to-end + no-LLM dry-run.
"""
