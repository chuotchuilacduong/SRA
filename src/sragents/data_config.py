"""Load `configs/data.yaml` and expose stable accessors.

The paper-prompt and the actual SRA-Bench schema use different field names.
This module is the single point of mapping so downstream code never has to
care about the difference (or about whether a dataset is multi-label).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from sragents.config import PROJECT_ROOT

_DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"


@dataclass(frozen=True)
class DataConfig:
    corpus_path: Path
    instances_dir: Path
    results_dir: Path
    datasets: list[str]
    multi_label_datasets: set[str]
    # field-mapping (paper-prompt -> actual)
    f_query_id: str
    f_query: str
    f_gold: str
    f_domain: str
    # corpus fields
    cf_id: str
    cf_name: str
    cf_description: str
    cf_content: str
    cf_tools: str

    def is_multi_label(self, dataset: str) -> bool:
        return dataset in self.multi_label_datasets

    def instance_query_id(self, inst: dict) -> str:
        return inst[self.f_query_id]

    def instance_query(self, inst: dict) -> str:
        return inst[self.f_query]

    def instance_gold(self, inst: dict) -> list[str]:
        return list(inst.get(self.f_gold) or [])

    def instance_dataset(self, inst: dict) -> str:
        return inst[self.f_domain]


def load_data_config(path: Path | None = None) -> DataConfig:
    p = Path(path) if path else _DEFAULT_CONFIG
    cfg = yaml.safe_load(p.read_text())
    paths = cfg["paths"]
    fields = cfg["fields"]
    cfields = cfg["corpus_fields"]

    def _abs(rel: str) -> Path:
        rp = Path(rel)
        return rp if rp.is_absolute() else (PROJECT_ROOT / rp)

    return DataConfig(
        corpus_path=_abs(paths["corpus"]),
        instances_dir=_abs(paths["instances_dir"]),
        results_dir=_abs(paths["results_dir"]),
        datasets=list(cfg.get("datasets", [])),
        multi_label_datasets=set(cfg.get("multi_label_datasets", [])),
        f_query_id=fields["query_id"],
        f_query=fields["query"],
        f_gold=fields["gold_skill_ids"],
        f_domain=fields["domain"],
        cf_id=cfields["skill_id"],
        cf_name=cfields["name"],
        cf_description=cfields["description"],
        cf_content=cfields["content"],
        cf_tools=cfields["tools"],
    )


def load_instances(dataset_or_path: str | Path, cfg: DataConfig | None = None) -> list[dict]:
    """Load instances for a dataset name OR a direct path."""
    cfg = cfg or load_data_config()
    p = Path(dataset_or_path)
    if not p.exists() and not p.is_absolute():
        # interpret as dataset name
        p = cfg.instances_dir / f"{dataset_or_path}.json"
    return json.loads(p.read_text())
