"""Skill-aware field packing for cross-encoder reranking.

Five packing modes (ablation knobs):

- ``title_only``                  — name only
- ``title_description``           — name + description
- ``title_description_content``   — name + description + content (baseline)
- ``field_tagged``                — explicit tokens `[SKILL_NAME] ... [SKILL_DESCRIPTION] ... [SKILL_CONTENT] ...`
- ``field_tagged_maxp_chunks``    — field-tagged, content sliced for MaxP

The default mode (``field_tagged``) gives the cross-encoder explicit field
boundaries; the chunked variant is paired with :mod:`sragents.retrieve.chunking`
to handle skills whose content exceeds the model's max sequence length.

The original :func:`sragents.corpus.skill_text` is left intact so the
existing baseline retrievers (BM25, BGE) keep their reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

Q_TOKEN = "[QUERY]"
NAME_TOKEN = "[SKILL_NAME]"
DESC_TOKEN = "[SKILL_DESCRIPTION]"
CONTENT_TOKEN = "[SKILL_CONTENT]"
PAYLOAD_TOKEN = "[SKILL_PAYLOAD]"

PACKING_MODES = (
    "title_only",
    "title_description",
    "title_description_content",
    "field_tagged",
    "field_tagged_maxp_chunks",
)


@dataclass
class SkillPacker:
    """Render a (query, skill) pair into a single string for the CE model.

    Attrs:
        mode: One of :data:`PACKING_MODES`.
        include_tools: When True and the skill exposes ``tools``, render
            them as a `[SKILL_PAYLOAD]` block (callable-list string).
        max_content_chars: Hard cap on the content field (after which the
            field is truncated). For ``field_tagged_maxp_chunks`` this is
            ignored — the chunker handles slicing.
    """

    mode: str = "field_tagged"
    include_tools: bool = False
    max_content_chars: int = 1800

    def __post_init__(self) -> None:
        if self.mode not in PACKING_MODES:
            raise ValueError(
                f"unknown packing mode {self.mode!r}; expected one of {PACKING_MODES}"
            )

    # ------------------------------------------------------------------ pack

    def pack_pair(self, query: str, skill: dict) -> str:
        """Render a single (query, skill) text for the cross-encoder."""
        return f"{Q_TOKEN} {query.strip()} {self.pack_skill(skill)}"

    def pack_skill(self, skill: dict) -> str:
        """Render just the skill side (used by chunkers)."""
        name = (skill.get("name") or "").strip()
        desc = (skill.get("description") or "").strip()
        content = (skill.get("content") or "").strip()
        if self.max_content_chars and self.mode != "field_tagged_maxp_chunks":
            content = content[: self.max_content_chars]

        if self.mode == "title_only":
            return name
        if self.mode == "title_description":
            return f"{name}\n{desc}".strip()
        if self.mode == "title_description_content":
            return "\n".join(p for p in (name, desc, content) if p)
        # field_tagged & field_tagged_maxp_chunks share the same template;
        # the chunker decides how to slice ``content`` upstream.
        parts: list[str] = []
        if name:
            parts.append(f"{NAME_TOKEN} {name}")
        if desc:
            parts.append(f"{DESC_TOKEN} {desc}")
        if content:
            parts.append(f"{CONTENT_TOKEN} {content}")
        if self.include_tools:
            payload = _render_tools(skill.get("tools"))
            if payload:
                parts.append(f"{PAYLOAD_TOKEN} {payload}")
        return " ".join(parts)

    def pack_skill_with_chunk(self, skill: dict, chunk: str) -> str:
        """Like :meth:`pack_skill` but using a precomputed content chunk
        instead of the skill's full ``content`` (used for MaxP)."""
        if self.mode != "field_tagged_maxp_chunks":
            # Fall through — caller chose to chunk anyway.
            pass
        name = (skill.get("name") or "").strip()
        desc = (skill.get("description") or "").strip()
        parts = []
        if name:
            parts.append(f"{NAME_TOKEN} {name}")
        if desc:
            parts.append(f"{DESC_TOKEN} {desc}")
        if chunk:
            parts.append(f"{CONTENT_TOKEN} {chunk}")
        if self.include_tools:
            payload = _render_tools(skill.get("tools"))
            if payload:
                parts.append(f"{PAYLOAD_TOKEN} {payload}")
        return " ".join(parts)


def _render_tools(tools) -> str:
    if not tools:
        return ""
    if isinstance(tools, str):
        return tools
    if isinstance(tools, (list, tuple)):
        out: list[str] = []
        for t in tools:
            if isinstance(t, str):
                out.append(t)
            elif isinstance(t, dict):
                # OpenAI-style tool descriptor
                name = t.get("name") or t.get("function", {}).get("name") or ""
                desc = t.get("description") or t.get("function", {}).get("description") or ""
                out.append(f"{name}: {desc}".strip(": "))
            else:
                out.append(repr(t))
        return " | ".join(filter(None, out))
    return str(tools)


def iter_pack_modes() -> Iterable[str]:
    """Convenience for ablation grids."""
    return PACKING_MODES
