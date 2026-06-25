"""Configuration for M4-v2.

Loads ``configs/m4_v2_default.yaml`` into a typed :class:`M4V2Config` and parses
the ablation ``variants`` table into :class:`Variant` objects. All paths resolve
against the repo root (``sragents.config.PROJECT_ROOT``) so the package works
regardless of the current working directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sragents.config import PROJECT_ROOT

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "m4_v2_default.yaml"

# Affinity modes understood by the scorer.
AFFINITY_MODES = {"none", "hard", "hard_calibrated", "soft", "soft_calibrated"}
ALPHA_MODES = {"none", "fixed", "adaptive"}


@dataclass(frozen=True)
class Variant:
    """One ablation variant (a row of spec §7.1)."""

    id: str
    name: str
    pool: int
    prf: bool
    affinity: str          # one of AFFINITY_MODES
    alpha: str             # one of ALPHA_MODES

    def __post_init__(self) -> None:
        if self.affinity not in AFFINITY_MODES:
            raise ValueError(f"{self.id}: bad affinity {self.affinity!r}")
        if self.alpha not in ALPHA_MODES:
            raise ValueError(f"{self.id}: bad alpha {self.alpha!r}")

    @property
    def uses_soft(self) -> bool:
        return self.affinity in ("soft", "soft_calibrated")

    @property
    def uses_calibration(self) -> bool:
        return self.affinity in ("hard_calibrated", "soft_calibrated")

    @property
    def uses_affinity(self) -> bool:
        return self.affinity != "none"

    @property
    def slug(self) -> str:
        """Directory-safe variant slug, e.g. ``A8_full_m4v2``."""
        safe = self.name.lower().replace("+", "").replace("  ", " ").strip()
        safe = "_".join(safe.split())
        return f"{self.id}_{safe}"


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass
class M4V2Config:
    raw: dict[str, Any]
    variants: list[Variant]

    # pool / fusion
    rrf_k: int = 60
    pool_size: int = 500
    pool_size_m1: int = 100
    pool_build_max: int = 1000
    output_top_k: int = 100
    eval_top_k: int = 100

    # clustering
    kmeans_k: int = 300
    soft_cluster_top_l: int = 10
    softmax_tau: float = 20.0

    # PRF
    prf_enabled: bool = True
    prf_rho: float = 0.2
    prf_gamma: float = 5.0
    prf_bm25_rank_cutoff: int = 50
    prf_bge_rank_cutoff: int = 50
    prf_min_safe_candidates: int = 3
    prf_fallback_top_rrf: int = 10

    # calibration
    rel_enabled: bool = True
    rel_min: float = 0.05
    rel_max: float = 1.0

    # adaptive alpha
    adaptive_enabled: bool = True
    alpha0: float = 0.7
    alpha_lambda: float = 0.2
    alpha_min: float = 0.45
    alpha_max: float = 0.90
    conf_rrf_top: int = 1
    conf_rrf_bottom: int = 10

    fixed_alpha: float = 0.7
    eps: float = 1e-8

    # encoder
    bge_model: str = "BAAI/bge-base-en-v1.5"
    bge_prefix: str = "Represent this sentence for searching relevant passages: "
    bge_batch_size: int = 128

    datasets: list[str] = field(default_factory=list)
    a1_use_cached_extended_pool: bool = True

    # resolved paths
    paths: dict[str, Path] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "M4V2Config":
        p = Path(path) if path else _DEFAULT_CONFIG
        raw = yaml.safe_load(p.read_text())
        m = raw["m4_v2"]
        rel = m.get("cluster_reliability", {})
        ad = m.get("adaptive_alpha", {})
        paths = {k: _abs(v) for k, v in m.get("paths", {}).items()}
        variants = [Variant(**v) for v in raw.get("variants", [])]
        return cls(
            raw=raw,
            variants=variants,
            rrf_k=int(m["rrf_k"]),
            pool_size=int(m["pool_size"]),
            pool_size_m1=int(m["pool_size_m1"]),
            pool_build_max=int(m["pool_build_max"]),
            output_top_k=int(m["output_top_k"]),
            eval_top_k=int(m["eval_top_k"]),
            kmeans_k=int(m["kmeans_k"]),
            soft_cluster_top_l=int(m["soft_cluster_top_l"]),
            softmax_tau=float(m["softmax_tau"]),
            prf_enabled=bool(m["prf_enabled"]),
            prf_rho=float(m["prf_rho"]),
            prf_gamma=float(m["prf_gamma"]),
            prf_bm25_rank_cutoff=int(m["prf_bm25_rank_cutoff"]),
            prf_bge_rank_cutoff=int(m["prf_bge_rank_cutoff"]),
            prf_min_safe_candidates=int(m["prf_min_safe_candidates"]),
            prf_fallback_top_rrf=int(m["prf_fallback_top_rrf"]),
            rel_enabled=bool(rel.get("enabled", True)),
            rel_min=float(rel.get("rel_min", 0.05)),
            rel_max=float(rel.get("rel_max", 1.0)),
            adaptive_enabled=bool(ad.get("enabled", True)),
            alpha0=float(ad.get("alpha0", 0.7)),
            alpha_lambda=float(ad.get("lambda", 0.2)),
            alpha_min=float(ad.get("alpha_min", 0.45)),
            alpha_max=float(ad.get("alpha_max", 0.90)),
            conf_rrf_top=int(ad.get("conf_rrf_top", 1)),
            conf_rrf_bottom=int(ad.get("conf_rrf_bottom", 10)),
            fixed_alpha=float(m["fixed_alpha"]),
            eps=float(m["eps"]),
            bge_model=str(m["bge_model"]),
            bge_prefix=str(m["bge_prefix"]),
            bge_batch_size=int(m["bge_batch_size"]),
            datasets=list(m["datasets"]),
            a1_use_cached_extended_pool=bool(m.get("a1_use_cached_extended_pool", True)),
            paths=paths,
        )

    def variant(self, vid: str) -> Variant:
        for v in self.variants:
            if v.id == vid:
                return v
        raise KeyError(f"no variant {vid!r}")

    def query_emb_path(self, ds: str) -> Path:
        return self.paths["query_emb_dir"] / f"{ds}.npy"

    def query_emb_ids_path(self, ds: str) -> Path:
        return self.paths["query_emb_dir"] / f"{ds}_ids.json"

    def fused_pool_path(self, ds: str) -> Path:
        return self.paths["fused_pool_dir"] / f"{ds}.json"

    def extended_pool_path(self, ds: str) -> Path:
        tmpl = str(self.raw["m4_v2"]["paths"]["extended_pool_tmpl"])
        return _abs(tmpl.format(ds=ds))
