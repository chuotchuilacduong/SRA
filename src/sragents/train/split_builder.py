"""Train / dev / test splits for cross-encoder fine-tuning.

Three protocols, each tests a different generalization axis:

* ``query_gen``  — random split on ``instance_id``. Skill overlap is fine;
                   we just want a held-out query distribution.
* ``skill_gen``  — partition skills disjointly, then assign each query to
                   the split that contains *all* of its gold skills. Queries
                   whose gold ids span splits are moved to ``test`` to
                   guarantee zero leakage (the "safe sink" rule).
* ``ldo``        — leave-domain-out: a single dataset is held out as test;
                   the rest are train+dev.

All protocols emit lists of ``instance_id`` strings, never raw instances —
the caller re-joins to the bench files. This keeps the artifact small,
diff-able, and deterministic given a seed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Splits:
    train: list[str]
    dev: list[str]
    test: list[str]

    def to_dict(self) -> dict:
        return {"train": self.train, "dev": self.dev, "test": self.test}

    def dump(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def query_generalization_split(
    instances: list[dict],
    *,
    instance_id_field: str = "instance_id",
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Splits:
    ids = [i[instance_id_field] for i in instances]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = int(n * test_ratio)
    n_dev = int(n * dev_ratio)
    test = sorted(ids[:n_test])
    dev = sorted(ids[n_test : n_test + n_dev])
    train = sorted(ids[n_test + n_dev :])
    return Splits(train=train, dev=dev, test=test)


def skill_generalization_split(
    instances: list[dict],
    *,
    instance_id_field: str = "instance_id",
    gold_field: str = "skill_annotations",
    dev_skill_ratio: float = 0.1,
    test_skill_ratio: float = 0.1,
    seed: int = 42,
) -> Splits:
    """Disjoint skill sets per split. Cross-split queries -> test.

    Multi-label safety: any query whose gold spans two split-skill-sets is
    re-homed to ``test``. This is conservative but eliminates leakage.
    """
    all_skills: set[str] = set()
    for inst in instances:
        all_skills.update(inst.get(gold_field) or [])
    skills_sorted = sorted(all_skills)
    rng = random.Random(seed)
    rng.shuffle(skills_sorted)
    n = len(skills_sorted)
    n_test = int(n * test_skill_ratio)
    n_dev = int(n * dev_skill_ratio)
    test_skills = set(skills_sorted[:n_test])
    dev_skills = set(skills_sorted[n_test : n_test + n_dev])
    train_skills = set(skills_sorted[n_test + n_dev :])

    train, dev, test = [], [], []
    for inst in instances:
        gold = set(inst.get(gold_field) or [])
        qid = inst[instance_id_field]
        if not gold:
            # No supervision; park in test to avoid contaminating train.
            test.append(qid)
            continue
        in_train = gold.issubset(train_skills)
        in_dev = gold.issubset(dev_skills)
        in_test = gold.issubset(test_skills)
        if in_train:
            train.append(qid)
        elif in_dev:
            dev.append(qid)
        elif in_test:
            test.append(qid)
        else:
            # spans multiple skill-splits -> send to test (the safe sink)
            test.append(qid)
    return Splits(train=sorted(train), dev=sorted(dev), test=sorted(test))


def leave_domain_out_split(
    instances: list[dict],
    *,
    held_out_dataset: str,
    instance_id_field: str = "instance_id",
    dataset_field: str = "dataset",
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> Splits:
    rng = random.Random(seed)
    train_pool, test = [], []
    for inst in instances:
        qid = inst[instance_id_field]
        if inst.get(dataset_field) == held_out_dataset:
            test.append(qid)
        else:
            train_pool.append(qid)
    rng.shuffle(train_pool)
    n_dev = int(len(train_pool) * dev_ratio)
    dev = sorted(train_pool[:n_dev])
    train = sorted(train_pool[n_dev:])
    return Splits(train=train, dev=dev, test=sorted(test))


def build_split(
    protocol: str,
    instances: list[dict],
    **kwargs,
) -> Splits:
    """Dispatch by protocol name. Convenience for CLI plumbing."""
    if protocol == "query_gen":
        return query_generalization_split(instances, **kwargs)
    if protocol == "skill_gen":
        return skill_generalization_split(instances, **kwargs)
    if protocol == "ldo":
        return leave_domain_out_split(instances, **kwargs)
    raise ValueError(f"unknown protocol {protocol!r}")
