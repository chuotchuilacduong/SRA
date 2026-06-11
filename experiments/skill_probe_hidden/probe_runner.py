"""HFProber — run one counterfactual probe through the HF backend, cached.

Mirrors `experiments/probehyrr_validation/probe_once.py` (Prober/ProbeCache) but
(a) generates via the local HF model and (b) saves pooled hidden states to .npy.
Reuses the EXACT skill-injection + prompt logic of `DirectEngine`/`build_prompt`
and the deterministic per-dataset `evaluate`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sragents.evaluate import evaluate
from sragents.evaluate.common import strip_think_tags
from sragents.prompts import build_prompt

from experiments.skill_probe_hidden.data_io import skill_tag


def cache_key(instance_id: str, skill_ids: list[str], model: str) -> str:
    return f"{model}\t{instance_id}\t{','.join(sorted(skill_ids))}"


class ProbeCache:
    """Append-only JSONL cache of probe outcomes (single-thread; batch=1)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._mem: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._mem[rec["_key"]] = rec
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, key: str):
        return self._mem.get(key)

    def put(self, key: str, rec: dict):
        rec = {**rec, "_key": key}
        self._mem[key] = rec
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class HFProber:
    def __init__(self, generator, corpus: dict, cache_path: Path, hs_dir: Path,
                 pool: str = "last_token", keep_layers: list[int] | None = None):
        self.gen = generator
        self.corpus = corpus
        self.cache = ProbeCache(cache_path)
        self.hs_dir = Path(hs_dir)
        self.hs_dir.mkdir(parents=True, exist_ok=True)
        self.pool = pool
        self.keep_layers = keep_layers  # None -> keep all; else subset [num_layers+1, h] rows

    def _skill_texts(self, skill_ids: list[str]) -> list[str]:
        # EXACT DirectEngine logic: only the `content` field, in order.
        out = []
        for sid in skill_ids:
            s = self.corpus.get(sid)
            if s and s.get("content"):
                out.append(s["content"])
        return out

    def probe(self, instance: dict, skill_ids: list[str]) -> dict:
        """Run (or fetch cached) one probe. Returns the record dict (with v, paths)."""
        key = cache_key(instance["instance_id"], skill_ids, self.gen.model_id)
        hit = self.cache.get(key)
        if hit is not None:
            return hit

        skill_texts = self._skill_texts(skill_ids)
        system, user = build_prompt(instance, skills=skill_texts)
        text, pooled = self.gen.generate(system, user, pool=self.pool)
        ev = evaluate(text, instance)

        if self.keep_layers is not None and pooled.ndim == 2:
            from experiments.skill_probe_hidden.pooling import select_layers

            pooled = select_layers(pooled, self.keep_layers)

        hs_name = f"{instance['instance_id']}__{skill_tag(skill_ids)}.npy"
        hs_path = self.hs_dir / hs_name
        if not hs_path.exists():
            np.save(hs_path, pooled)

        stripped = strip_think_tags(text)
        rec = {
            "instance_id": instance["instance_id"],
            "dataset": instance.get("dataset"),
            "skill_ids": skill_ids,
            "generator_model": self.gen.model_id,
            "v": int(bool(ev.get("correct"))),
            "extracted": ev.get("extracted_answer"),
            "raw_output": text,
            "raw_output_len": len(text),
            "stripped_output": stripped,
            "thinking_leaked": "<think>" in text,
            "hs_name": hs_name,
            "hidden_state_shape": list(pooled.shape),
        }
        self.cache.put(key, rec)
        return rec
