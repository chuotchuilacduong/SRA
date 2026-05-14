#!/usr/bin/env python3
"""Standalone embedding script - process entities and sentences with checkpointing.
Loads ner_results.json and creates entity_embedding.parquet + sentence_embedding.parquet.
"""
import json
import os
import sys
import time
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = "import/bench_full"
NER_PATH = f"{OUTPUT_DIR}/ner_results.json"
BATCH_SIZE = 32
CHECKPOINT_EVERY = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_mdhash_id(content, prefix=""):
    return prefix + hashlib.md5(content.encode()).hexdigest()


def embed_with_checkpoint(texts, namespace, parquet_path, model, batch_size=32, checkpoint_every=5000):
    """Embed texts with checkpoint support."""
    # Dedupe texts
    seen = set()
    unique_texts = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique_texts.append(t)

    print(f"\n[{namespace}] Total unique: {len(unique_texts)} (from {len(texts)} input)", flush=True)

    # Build hash IDs
    hash_ids = [compute_mdhash_id(t, prefix=namespace + "-") for t in unique_texts]

    # Resume from existing parquet
    existing_hash_ids = set()
    saved_embeddings = []
    saved_texts = []
    saved_hash_ids = []
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        existing_hash_ids = set(df["hash_id"].values.tolist())
        saved_hash_ids = df["hash_id"].values.tolist()
        saved_texts = df["text"].values.tolist()
        saved_embeddings = df["embedding"].values.tolist()
        print(f"[{namespace}] Resumed: {len(saved_hash_ids)} already done", flush=True)

    # Filter to-encode
    todo_pairs = [(h, t) for h, t in zip(hash_ids, unique_texts) if h not in existing_hash_ids]
    print(f"[{namespace}] To encode: {len(todo_pairs)}", flush=True)

    if not todo_pairs:
        print(f"[{namespace}] All done!", flush=True)
        return

    # Encode in batches with checkpoint
    start_time = time.time()
    total = len(todo_pairs)
    encoded = 0

    for chunk_start in range(0, total, checkpoint_every):
        chunk_end = min(chunk_start + checkpoint_every, total)
        chunk = todo_pairs[chunk_start:chunk_end]
        chunk_hash_ids = [c[0] for c in chunk]
        chunk_texts = [c[1] for c in chunk]

        # Encode this chunk
        embeddings = model.encode(
            chunk_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )

        # Append to saved
        saved_hash_ids.extend(chunk_hash_ids)
        saved_texts.extend(chunk_texts)
        saved_embeddings.extend([emb for emb in embeddings])

        # Save checkpoint
        df = pd.DataFrame({
            "hash_id": saved_hash_ids,
            "text": saved_texts,
            "embedding": saved_embeddings,
        })
        df.to_parquet(parquet_path, index=False)

        encoded += len(chunk)
        elapsed = time.time() - start_time
        rate = encoded / elapsed if elapsed > 0 else 0
        eta = (total - encoded) / rate if rate > 0 else 0
        size_mb = os.path.getsize(parquet_path) / 1024 / 1024
        print(f"[{namespace}] [{encoded}/{total}] {rate:.1f}/s | ETA: {eta:.0f}s | {size_mb:.1f}MB", flush=True)

    print(f"[{namespace}] ✓ Done in {(time.time()-start_time)/60:.1f} min", flush=True)


def main():
    # Load NER results
    print(f"Loading NER results from {NER_PATH}...", flush=True)
    with open(NER_PATH) as f:
        ner = json.load(f)
    passage_entities = ner["passage_hash_id_to_entities"]
    sentence_entities = ner["sentence_to_entities"]

    # Collect unique entities and sentences
    entity_set = set()
    for entities in passage_entities.values():
        entity_set.update(entities)
    entities_list = list(entity_set)

    sentences_list = list(sentence_entities.keys())

    print(f"Unique entities: {len(entities_list)}", flush=True)
    print(f"Unique sentences: {len(sentences_list)}", flush=True)

    # Load model
    print("\nLoading embedding model...", flush=True)
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    print(f"Model loaded: {model.get_sentence_embedding_dimension()} dims", flush=True)

    # Embed entities
    embed_with_checkpoint(
        entities_list,
        namespace="entity",
        parquet_path=f"{OUTPUT_DIR}/entity_embedding.parquet",
        model=model,
        batch_size=BATCH_SIZE,
        checkpoint_every=CHECKPOINT_EVERY,
    )

    # Embed sentences
    embed_with_checkpoint(
        sentences_list,
        namespace="sentence",
        parquet_path=f"{OUTPUT_DIR}/sentence_embedding.parquet",
        model=model,
        batch_size=BATCH_SIZE,
        checkpoint_every=CHECKPOINT_EVERY,
    )

    print("\n✓ All embeddings complete!", flush=True)


if __name__ == "__main__":
    main()
