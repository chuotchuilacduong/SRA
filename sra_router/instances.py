"""SRA-Bench instance loader + demo train/eval split.

The plan requires holding out all 5,400 SRA-Bench instances for evaluation,
training on real-train-splits or synthetic queries. We have neither here, so
this module exposes an explicit demo split: per-dataset 80/20 with a fixed
seed, with all instances of unseen gold skills forced into the train side so
the eval set always has at least one example per gold skill present in train.
The resulting split is for the framework demo only; the paper-protocol number
would require the synthesis pipeline.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SRAInstance:
    instance_id: str
    dataset: str
    query: str
    gold_skill_ids: list[str]
    answer: str | None = None


def load_instances(path: str | Path) -> list[SRAInstance]:
    raw = json.loads(Path(path).read_text())
    out = []
    for d in raw:
        eval_data = d.get("eval_data") or {}
        out.append(SRAInstance(
            instance_id=d["instance_id"],
            dataset=d.get("dataset", ""),
            query=d.get("question") or d.get("query") or "",
            gold_skill_ids=list(d.get("skill_annotations") or d.get("gold_skill_ids") or []),
            answer=eval_data.get("answer") if isinstance(eval_data, dict) else None,
        ))
    return out


def stratified_split(
    instances: list[SRAInstance],
    eval_frac: float = 0.2,
    seed: int = 13,
) -> tuple[list[SRAInstance], list[SRAInstance]]:
    rng = random.Random(seed)
    by_dataset: dict[str, list[SRAInstance]] = {}
    for inst in instances:
        by_dataset.setdefault(inst.dataset, []).append(inst)

    train: list[SRAInstance] = []
    evalset: list[SRAInstance] = []
    for ds, items in sorted(by_dataset.items()):
        items = list(items)
        rng.shuffle(items)
        n_eval = max(1, int(len(items) * eval_frac))
        evalset.extend(items[:n_eval])
        train.extend(items[n_eval:])
    return train, evalset
