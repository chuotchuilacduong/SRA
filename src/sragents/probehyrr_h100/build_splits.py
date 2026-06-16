"""Stage 0a — materialize merged query splits + a JSONL skill corpus.

The H100 spec assumes ``data/splits/{train,dev,test}.jsonl`` and
``data/corpus/skills.jsonl`` exist. They don't. This builds them from:

  * per-dataset id-lists at ``results/splits/{ds}-query_gen.json``
  * full query rows at ``data/bench/instances/{ds}.json``
  * the 26,262-skill array at ``data/bench/corpus/corpus.json``

Each ``data/splits/{split}.jsonl`` row is one query:
    {qid, dataset, split, question, gold_skill_ids, eval_data}

Each ``data/corpus/skills.jsonl`` row is one skill (native fields kept;
``title`` aliased from ``name`` for spec compatibility — ``skill_id`` is kept
but must never be shown to the model).
"""

from __future__ import annotations

import argparse

from sragents.config import INSTANCES_DIR, PROJECT_ROOT
from sragents.corpus import load_corpus
from sragents.probehyrr_h100 import config as C
from sragents.probehyrr_h100.io_utils import load_json, write_jsonl


def _load_split_ids(dataset: str) -> dict[str, list[str]]:
    rel = C.SPLIT_FILE_TMPL.format(ds=dataset, protocol=C.SPLIT_PROTOCOL)
    d = load_json(PROJECT_ROOT / rel)
    return {s: list(d.get(s, [])) for s in C.SPLITS}


def _load_instances(dataset: str) -> dict[str, dict]:
    rows = load_json(INSTANCES_DIR / f"{dataset}.json")
    return {r["instance_id"]: r for r in rows}


def build_query_splits(datasets: list[str]) -> dict[str, int]:
    """Write data/splits/{split}.jsonl merged across datasets. Returns counts."""
    per_split: dict[str, list[dict]] = {s: [] for s in C.SPLITS}
    for ds in datasets:
        ids = _load_split_ids(ds)
        instances = _load_instances(ds)
        for split in C.SPLITS:
            for qid in ids[split]:
                inst = instances[qid]
                per_split[split].append(
                    {
                        "qid": qid,
                        "dataset": ds,
                        "split": split,
                        "question": inst["question"],
                        "gold_skill_ids": list(inst.get("skill_annotations") or []),
                        "eval_data": inst.get("eval_data", {}),
                    }
                )
    counts = {}
    for split in C.SPLITS:
        n = write_jsonl(C.split_jsonl_path(split), per_split[split])
        counts[split] = n
    return counts


def build_corpus_jsonl() -> int:
    """Write data/corpus/skills.jsonl (one skill per line, title aliased from name)."""
    skills = load_corpus()
    rows = (
        {
            "skill_id": s["skill_id"],
            "title": s.get("name", ""),
            "name": s.get("name", ""),
            "description": s.get("description", ""),
            "content": s.get("content", ""),
            **({"tools": s["tools"]} if "tools" in s else {}),
        }
        for s in skills
    )
    return write_jsonl(C.OUT_CORPUS_JSONL, rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build merged query splits + JSONL corpus")
    ap.add_argument("--datasets", nargs="*", default=C.VALID_4)
    ap.add_argument("--skip-corpus", action="store_true", help="don't rebuild skills.jsonl")
    args = ap.parse_args()

    counts = build_query_splits(args.datasets)
    print(f"splits -> {C.OUT_SPLITS_DIR}")
    for split, n in counts.items():
        print(f"  {split}: {n} queries")
    print(f"  total: {sum(counts.values())}")

    if not args.skip_corpus:
        n = build_corpus_jsonl()
        print(f"corpus -> {C.OUT_CORPUS_JSONL}: {n} skills")


if __name__ == "__main__":
    main()
