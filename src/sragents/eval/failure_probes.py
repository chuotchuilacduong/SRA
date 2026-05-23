"""Heuristic per-query failure probes.

A query gets a list of probe labels — each is a coarse hypothesis about
*why* retrieval might fail or *what kind of query it is*. Used for slicing
the eval report by failure mode (formula-heavy, code-heavy, multi-label,
long-skill, lexical/semantic confounder).

These are **heuristics**, intentionally simple, and the regex thresholds
live in ``configs/eval.yaml`` so paper-numbers stay reproducible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

PROBE_FORMULA_HEAVY = "formula_heavy"
PROBE_CODE_HEAVY = "code_heavy"
PROBE_MULTI_LABEL = "multi_label"
PROBE_LONG_SKILL = "long_skill"
PROBE_LEXICAL_CONFOUNDER = "lexical_confounder"     # BM25 top1 != gold, BGE has gold
PROBE_SEMANTIC_CONFOUNDER = "semantic_confounder"   # BGE top1 != gold, BM25 has gold


@dataclass
class ProbeConfig:
    long_skill_chars: int = 2000
    formula_regexes: tuple[str, ...] = (r"\$.+?\$", r"\\frac", r"\\sum", r"\\int")
    code_regex: str = r"```"
    code_import_regex: str = r"(?m)^\s*import\b|(?m)^\s*from\s+\S+\s+import\b"

    @classmethod
    def from_yaml(cls, path: Path) -> "ProbeConfig":
        cfg = yaml.safe_load(Path(path).read_text())
        pr = cfg.get("probes", {})
        return cls(
            long_skill_chars=pr.get("long_skill_chars", 2000),
            formula_regexes=tuple(pr.get("formula_regexes", [])),
            code_regex=pr.get("code_regex", "```"),
            code_import_regex=pr.get("code_import_regex", ""),
        )


def label_query(
    instance: dict,
    pool_entry: dict,
    corpus: dict[str, dict],
    cfg: ProbeConfig,
    *,
    question_field: str = "question",
    gold_field: str = "skill_annotations",
) -> list[str]:
    """Return the list of probe labels that apply to this query."""
    labels: list[str] = []
    question = (instance.get(question_field) or "")
    gold = instance.get(gold_field) or []

    if any(re.search(rgx, question) for rgx in cfg.formula_regexes):
        labels.append(PROBE_FORMULA_HEAVY)
    if re.search(cfg.code_regex, question) and (
        not cfg.code_import_regex or re.search(cfg.code_import_regex, question)
    ):
        labels.append(PROBE_CODE_HEAVY)
    if len(gold) > 1:
        labels.append(PROBE_MULTI_LABEL)

    # Long-skill: any gold whose content exceeds threshold.
    if any(
        len((corpus.get(g) or {}).get("content") or "") > cfg.long_skill_chars
        for g in gold
    ):
        labels.append(PROBE_LONG_SKILL)

    # Lexical vs semantic confounder requires per-retriever provenance.
    gold_set = set(gold)
    bm25 = sorted(
        [c for c in pool_entry.get("retrieved", []) if "bm25_rank" in c],
        key=lambda x: x["bm25_rank"],
    )
    bge = sorted(
        [c for c in pool_entry.get("retrieved", []) if "bge_rank" in c],
        key=lambda x: x["bge_rank"],
    )
    bm25_top1 = bm25[0]["skill_id"] if bm25 else None
    bge_top1 = bge[0]["skill_id"] if bge else None
    bm25_has_gold = any(c["skill_id"] in gold_set for c in bm25[:10])
    bge_has_gold = any(c["skill_id"] in gold_set for c in bge[:10])
    if bm25_top1 and bm25_top1 not in gold_set and bge_has_gold:
        labels.append(PROBE_LEXICAL_CONFOUNDER)
    if bge_top1 and bge_top1 not in gold_set and bm25_has_gold:
        labels.append(PROBE_SEMANTIC_CONFOUNDER)

    return labels


def annotate_pool(
    pool_records: list[dict],
    instances_by_id: dict[str, dict],
    corpus: dict[str, dict],
    cfg: ProbeConfig,
    *,
    question_field: str = "question",
    gold_field: str = "skill_annotations",
) -> dict[str, list[str]]:
    """Return ``{instance_id: [probe_labels]}`` for every query in the pool."""
    out: dict[str, list[str]] = {}
    for r in pool_records:
        inst = instances_by_id.get(r["instance_id"])
        if inst is None:
            continue
        out[r["instance_id"]] = label_query(
            inst, r, corpus, cfg,
            question_field=question_field, gold_field=gold_field,
        )
    return out


def write_annotations(annotations: dict[str, list[str]], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(annotations, indent=2))
