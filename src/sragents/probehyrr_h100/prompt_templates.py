"""Prompt construction for probes — thin wrapper over sragents.prompts.build_prompt.

Decision D1: reuse the repo's per-dataset prompt builders + content-only skill
injection so the existing deterministic verifiers stay valid. We do NOT use the
spec §10 generic templates.

The manifest stores ``(system, user)`` + a ``prompt_hash``; the vLLM generator
applies the Qwen chat template (``enable_thinking=False``) at generation time.
``render_chat_template`` is provided for the generator / for length probing but
needs the Qwen tokenizer, so it is NOT called during Stage 0.
"""

from __future__ import annotations

from sragents.prompts import build_prompt
from sragents.probehyrr_h100 import config as C


def build_probe_prompt(instance: dict, skill_contents: list[str] | None = None) -> tuple[str, str]:
    """Return ``(system, user)`` for a probe.

    ``instance`` must carry ``dataset`` and ``question`` (a split-jsonl row or a
    bench instance both satisfy this once ``question`` is present). ``skill_contents``
    is the list of raw skill ``content`` strings to inject (empty/None = no_skill).
    """
    inst = instance
    if "question" not in inst:
        raise KeyError("instance needs a 'question' field for prompt building")
    return build_prompt(inst, skills=skill_contents or None)


def skill_contents_for(skill_ids: list[str], corpus: dict[str, dict]) -> list[str]:
    """Resolve skill_ids -> content strings (content only, in order), skipping empties."""
    out = []
    for sid in skill_ids:
        s = corpus.get(sid)
        if s and s.get("content"):
            out.append(s["content"])
    return out


def render_chat_template(system: str, user: str, tokenizer) -> str:
    """Apply the Qwen chat template for vLLM offline generation (gen-time only).

    Empty system messages are omitted (matches the prior-art HFGenerator).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=C.ENABLE_THINKING,
    )
