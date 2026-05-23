"""SRA-Router: SkillRouter-style retrieve-and-rerank for SRA-Bench.

See sra_skill_router_plan.md for the full design. This package implements a
scaled-down bi-encoder pipeline (Phase 0 — Phase 3 + Phase 6 — Phase 7 of the
plan) that runs on a Mac without GPU. The cross-encoder reranker (Phase 4-5)
is deferred.
"""
