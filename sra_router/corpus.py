"""Skill corpus loader.

Wraps the existing SRA-Bench corpus.json into a typed object the trainer can
use. Source-dataset membership is inferred from the ``skill_id`` prefix:
``theoremqa_*``, ``logicbench_*``, etc. Anything not matching a known prefix is
treated as a distractor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

KNOWN_PREFIXES = (
    "theoremqa", "logicbench", "toolqa", "medcalcbench", "champ", "bigcodebench",
)


@dataclass
class Skill:
    skill_id: str
    name: str
    description: str
    content: str
    source: str         # one of KNOWN_PREFIXES, or "distractor"
    is_gold: bool       # True iff source != "distractor"

    def full_text(self, max_desc: int = 300, max_body: int = 2500) -> str:
        return f"{self.name} | {self.description[:max_desc]} | {self.content[:max_body]}"


def _infer_source(skill_id: str) -> str:
    for p in KNOWN_PREFIXES:
        if skill_id.startswith(p + "_") or skill_id == p:
            return p
    return "distractor"


class SkillCorpus:
    def __init__(self, skills: list[Skill]):
        self.skills: list[Skill] = skills
        self._by_id: dict[str, Skill] = {s.skill_id: s for s in skills}
        self._by_source: dict[str, list[str]] = {}
        for s in skills:
            self._by_source.setdefault(s.source, []).append(s.skill_id)

    @classmethod
    def from_json(cls, path: str | Path) -> "SkillCorpus":
        raw = json.loads(Path(path).read_text())
        out = []
        for d in raw:
            src = _infer_source(d["skill_id"])
            out.append(Skill(
                skill_id=d["skill_id"],
                name=d.get("name", "") or "",
                description=d.get("description", "") or "",
                content=d.get("content", "") or "",
                source=src,
                is_gold=(src != "distractor"),
            ))
        return cls(out)

    def __len__(self) -> int:
        return len(self.skills)

    def __getitem__(self, skill_id: str) -> Skill:
        return self._by_id[skill_id]

    def __contains__(self, skill_id: str) -> bool:
        return skill_id in self._by_id

    def ids(self) -> list[str]:
        return [s.skill_id for s in self.skills]

    def ids_in_source(self, source: str) -> list[str]:
        return list(self._by_source.get(source, []))

    def ids_not_in_sources(self, sources: Iterable[str]) -> list[str]:
        bad = set(sources)
        return [sid for sid, s in self._by_id.items() if s.source not in bad]

    def summary(self) -> dict[str, int]:
        return {k: len(v) for k, v in sorted(self._by_source.items())}
