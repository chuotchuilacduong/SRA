"""
Reconstruct the full tri-graph (passage–entity–sentence) from saved index files
and export to GraphML.

Usage:
    python parquet_to_graphml.py [--out trigraph.graphml] [--data-dir ...]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import igraph as ig
import pandas as pd

# XML 1.0 forbids most ASCII control characters. Strip them before writing.
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def _sanitize(text: str) -> str:
    if not isinstance(text, str):
        return text
    return _CTRL_RE.sub("", text)

DATA_DIR = Path("../bench_full")


def build_trigraph(data_dir: Path) -> ig.Graph:
    ep = pd.read_parquet(data_dir / "entity_embedding.parquet")[["hash_id", "text"]]
    sp = pd.read_parquet(data_dir / "sentence_embedding.parquet")[["hash_id", "text"]]
    pp = pd.read_parquet(data_dir / "passage_embedding.parquet")[["hash_id", "text"]]

    with open(data_dir / "ner_results.json") as f:
        ner = json.load(f)

    p2e = ner["passage_hash_id_to_entities"]   # passage_hash_id → [entity_text]
    s2e = ner["sentence_to_entities"]           # sentence_text   → [entity_text]

    # lookup: text → hash_id
    entity_text_to_hash   = dict(zip(ep["text"], ep["hash_id"]))
    sentence_text_to_hash = dict(zip(sp["text"], sp["hash_id"]))

    # entity_text → {sentence_hash_id}
    e2s: dict[str, set[str]] = defaultdict(set)
    for sent_text, ents in s2e.items():
        s_hash = sentence_text_to_hash.get(sent_text)
        if s_hash is None:
            continue
        for ent in ents:
            e2s[ent].add(s_hash)

    # --- collect all nodes and edges ---
    # nodes: hash_id → {"type", "text"}
    nodes: dict[str, dict] = {}
    edges: list[tuple[str, str, str]] = []  # (src, dst, edge_type)

    for row in pp.itertuples(index=False):
        nodes[row.hash_id] = {"type": "passage", "text": row.text} # type: ignore

    for row in ep.itertuples(index=False):
        nodes[row.hash_id] = {"type": "entity", "text": row.text} # type: ignore

    for row in sp.itertuples(index=False):
        nodes[row.hash_id] = {"type": "sentence", "text": row.text} # type: ignore

    # passage → entity edges
    for p_hash, ent_list in p2e.items():
        if p_hash not in nodes:
            continue
        for ent_text in ent_list:
            e_hash = entity_text_to_hash.get(ent_text)
            if e_hash and e_hash in nodes:
                edges.append((p_hash, e_hash, "passage_entity"))

    # entity → sentence edges
    for ent_text, sent_hashes in e2s.items():
        e_hash = entity_text_to_hash.get(ent_text)
        if not e_hash or e_hash not in nodes:
            continue
        for s_hash in sent_hashes:
            if s_hash in nodes:
                edges.append((e_hash, s_hash, "entity_sentence"))

    # --- build igraph ---
    node_ids = list(nodes.keys())
    node_idx = {nid: i for i, nid in enumerate(node_ids)}

    g = ig.Graph(n=len(node_ids), directed=False)
    g.vs["name"]  = node_ids
    g.vs["type"]  = [nodes[nid]["type"]  for nid in node_ids]
    # Sanitize text: GraphML/XML 1.0 forbids most ASCII control chars
    # (0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F). Strip them to avoid write errors.
    g.vs["text"]  = [_sanitize(nodes[nid]["text"]) for nid in node_ids]

    edge_list   = [(node_idx[s], node_idx[d]) for s, d, _ in edges]
    edge_types  = [t for _, _, t in edges]
    g.add_edges(edge_list)
    g.es["edge_type"] = edge_types

    print(f"[INFO] nodes : {g.vcount():,}  (passage={sum(1 for v in g.vs if v['type']=='passage')}, "
          f"entity={sum(1 for v in g.vs if v['type']=='entity')}, "
          f"sentence={sum(1 for v in g.vs if v['type']=='sentence')})")
    print(f"[INFO] edges : {g.ecount():,}  (passage_entity={edge_types.count('passage_entity')}, "
          f"entity_sentence={edge_types.count('entity_sentence')})")

    return g


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="../bench_full/trigraph.graphml")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    g = build_trigraph(Path(args.data_dir))
    g.write_graphml(args.out)
    print(f"[INFO] saved → {args.out}")
