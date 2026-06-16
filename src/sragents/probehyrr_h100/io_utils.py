"""JSONL IO, hashing, and cache-key helpers shared across pipeline stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

from sragents.probehyrr_h100 import config as C


def load_json(path: str | Path) -> dict | list:
    return json.loads(Path(path).read_text())


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[dict], *, append: bool = False) -> int:
    """Write rows as JSONL. Returns the number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def sha256_short(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def prompt_hash(system: str, user: str) -> str:
    """Stable hash of the prompt body (pre chat-template), full sha256 hex."""
    payload = json.dumps({"system": system, "user": user}, sort_keys=True, ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_key(*, qid: str, skill_ids: list[str], prompt_hash_val: str, max_new_tokens: int) -> str:
    """Dedup/resume key: a probe is uniquely (model, gen cfg, prompt, skills, qid).

    Mirrors the spec §11 cache key (model + enable_thinking + temperature +
    top_p + max_new_tokens + prompt_hash + skill_id + qid).
    """
    payload = json.dumps(
        {
            "model": C.MODEL_ID,
            "enable_thinking": C.ENABLE_THINKING,
            "temperature": C.TEMPERATURE,
            "top_p": C.TOP_P,
            "max_new_tokens": max_new_tokens,
            "prompt_hash": prompt_hash_val,
            "skill_ids": sorted(skill_ids),
            "qid": qid,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
