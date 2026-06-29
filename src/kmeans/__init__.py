"""Skill-reranking methods, organized into method subpackages.

  - ``kmeans.common`` : shared config, IO, evaluate, embeddings
  - ``kmeans.m4``     : M4-v2 cluster-aware RRF reranker (Steps 1-10)
  - ``kmeans.ltr``    : QSC + LightGBM LambdaRank (incl. the bge_ft feature group)
  - ``kmeans.ce``     : cross-encoder reranker (CE@500)

This package reuses ``sragents`` for metrics / corpus / schema and only adds the
method-specific scoring, ablation, training, and reporting logic.

**Backward compatibility.** The modules used to live flat under ``kmeans/`` (e.g.
``kmeans.io``, ``kmeans.ltr_features``, ``kmeans.config``). They are re-exported
here and re-registered under their former flat paths in ``sys.modules`` so legacy
imports keep working unchanged:
    from kmeans import io, ltr_features         # attribute access
    import kmeans.config                        # submodule path
    from kmeans.qsc_ltr_runner import load_ext  # submodule path
"""

import sys as _sys

from .common import config, io, embeddings, evaluate
from .m4 import (
    ablation_runner, adaptive_alpha, cluster_stats, prf,
    report_writer, rrf_pool, scorer, soft_cluster,
)
from .ltr import (
    base_table, cv, cv_report, ltr_features, qsc,
    qsc_ltr_report, qsc_ltr_runner,
)
from .ce import ce500

# Re-register the former flat module paths (``kmeans.<name>``) pointing at the
# relocated modules, so both ``from kmeans import X`` and ``from kmeans.X import y``
# resolve for every legacy script without touching those scripts.
#
# NOTE: the LambdaRank trainer module ``kmeans.ltr.ltr`` is deliberately NOT
# re-exported here — its former flat name ``kmeans.ltr`` now belongs to the
# subpackage, so a flat alias would shadow the package. Callers that need the
# trainer import it explicitly: ``from kmeans.ltr import ltr``.
_LEGACY = [
    config, io, embeddings, evaluate,
    ablation_runner, adaptive_alpha, cluster_stats, prf,
    report_writer, rrf_pool, scorer, soft_cluster,
    base_table, cv, cv_report, ltr_features, qsc,
    qsc_ltr_report, qsc_ltr_runner, ce500,
]
for _m in _LEGACY:
    _sys.modules[f"{__name__}.{_m.__name__.rsplit('.', 1)[-1]}"] = _m

__all__ = [_m.__name__.rsplit(".", 1)[-1] for _m in _LEGACY]
