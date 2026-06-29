"""Materialize query_gen-test-only bench instances for the end-task eval.

`sragents infer`/`experiment` run over EVERY instance in the instances file
(there is no --split flag). To evaluate only the held-out query_gen-test
(1,079 instances), this writes filtered copies:

    data/bench/instances/{ds}.json   --(keep ids in results/splits/{ds}-query_gen.json["test"])-->
    data/bench/instances_test/{ds}.json

Full instance dicts are preserved (question, skill_annotations, eval_data, ...)
so the evaluators and oracle provider keep working. Point the harness at the
filtered dir:  `sragents experiment ... --instances-dir data/bench/instances_test`.

    python src/kmeans/scripts/make_test_instances.py
"""

from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401
from sragents.config import PROJECT_ROOT, discover_datasets

SRC = PROJECT_ROOT / "data" / "bench" / "instances"
DST = PROJECT_ROOT / "data" / "bench" / "instances_test"
SPLITS = PROJECT_ROOT / "results" / "splits"


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    grand_total = 0
    for ds in discover_datasets():
        inst_path = SRC / f"{ds}.json"
        split_path = SPLITS / f"{ds}-query_gen.json"
        if not inst_path.exists() or not split_path.exists():
            print(f"  [skip] {ds}: missing {inst_path if not inst_path.exists() else split_path}")
            continue
        instances = json.loads(inst_path.read_text())
        test_ids = set(json.loads(split_path.read_text())["test"])
        kept = [i for i in instances if i["instance_id"] in test_ids]
        (DST / f"{ds}.json").write_text(json.dumps(kept, ensure_ascii=False))
        grand_total += len(kept)
        missing = len(test_ids) - len(kept)
        flag = "" if missing == 0 else f"  (WARN: {missing} test ids absent from instances!)"
        print(f"  {ds}: {len(kept)}/{len(test_ids)} test instances -> {DST}/{ds}.json{flag}")
    print(f"TOTAL query_gen-test instances written: {grand_total} (expect 1079)")


if __name__ == "__main__":
    main()
