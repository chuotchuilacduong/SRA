"""MaxP-style content chunking for long skills.

Each chunk re-attaches the query + skill name + description (cheap and
matters: those are the high-signal fields for the cross-encoder). Only
``content`` is slid with a configurable stride.

Final passage score is ``max`` over chunk scores (a.k.a. MaxP, Dai &
Callan 2019). Returning a stable ordering of chunks is important for
reproducibility — we use the order produced by ``range(0, len, stride)``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MaxPChunker:
    """Slide a window over ``skill['content']`` with optional stride.

    Attrs:
        chunk_size:   Maximum character length of each content chunk.
        stride:       Step in characters between consecutive chunks.
        long_skill_chars: Content shorter than this is **not** chunked
                          (returns a single chunk == full content).
    """

    chunk_size: int = 800
    stride: int = 400
    long_skill_chars: int = 1200

    def __post_init__(self) -> None:
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

    def chunks(self, content: str) -> list[str]:
        """Return the list of content chunks for one skill.

        Always non-empty: returns ``[""]`` for empty input so the caller
        still produces one (query, skill_text) pair.
        """
        if content is None:
            content = ""
        if len(content) <= self.long_skill_chars:
            return [content]
        out: list[str] = []
        n = len(content)
        for start in range(0, n, self.stride):
            piece = content[start : start + self.chunk_size]
            if not piece:
                break
            out.append(piece)
            if start + self.chunk_size >= n:
                break
        return out or [content]


def aggregate_max(scores: list[float]) -> float:
    """MaxP aggregation: highest chunk score wins."""
    if not scores:
        return 0.0
    return float(max(scores))
