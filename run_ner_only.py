#!/usr/bin/env python3
"""Standalone NER script - process 26k skills and save ner_results.json.
Runs independently from the main retrieval pipeline.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from sragents.retrieve._linearrag.ner import SpacyNER

CORPUS_PATH = "data/bench/corpus/corpus.json"
OUTPUT_DIR = "import/bench_full"
NER_OUTPUT = f"{OUTPUT_DIR}/ner_results.json"
MAX_CHARS = 3000
CHUNK_SIZE = 500  # Process in chunks to avoid issues
CHECKPOINT_EVERY = 1000  # Save partial results every N skills

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load existing results if any
if os.path.exists(NER_OUTPUT):
    existing = json.load(open(NER_OUTPUT))
    passage_entities = existing.get("passage_hash_id_to_entities", {})
    sentence_entities = existing.get("sentence_to_entities", {})
    print(f"Resumed: {len(passage_entities)} passages already done")
else:
    passage_entities = {}
    sentence_entities = {}

# Load corpus
print(f"Loading corpus from {CORPUS_PATH}...")
corpus = json.load(open(CORPUS_PATH))
print(f"Total skills: {len(corpus)}")

# Build hash_id -> text mapping
import hashlib
def hash_id(text):
    return hashlib.md5(text.encode()).hexdigest()

skill_texts = {}
for skill in corpus:
    text = skill.get("content", "")[:MAX_CHARS]
    skill_texts[hash_id(text)] = text

# Filter out already processed
todo_ids = [k for k in skill_texts if k not in passage_entities]
print(f"To process: {len(todo_ids)} (skipping {len(skill_texts) - len(todo_ids)} already done)")

if not todo_ids:
    print("All done!")
    sys.exit(0)

# Initialize NER
print("Loading spaCy NER (parser/tagger disabled)...")
ner = SpacyNER('en_core_web_sm')
print(f"Pipeline: {ner.spacy_model.pipe_names}")

# Process in chunks
start_time = time.time()
total = len(todo_ids)
processed = 0

for chunk_start in range(0, total, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, total)
    chunk_ids = todo_ids[chunk_start:chunk_end]
    chunk_dict = {hid: skill_texts[hid] for hid in chunk_ids}

    # Run NER on chunk
    chunk_p, chunk_s = ner.batch_ner(chunk_dict, max_workers=1)

    # Merge results
    passage_entities.update(chunk_p)
    for sent, ents in chunk_s.items():
        if sent not in sentence_entities:
            sentence_entities[sent] = []
        for e in ents:
            if e not in sentence_entities[sent]:
                sentence_entities[sent].append(e)

    processed += len(chunk_ids)
    elapsed = time.time() - start_time
    rate = processed / elapsed
    eta = (total - processed) / rate if rate > 0 else 0
    print(f"  [{processed}/{total}] {rate:.1f} skills/sec | ETA: {eta:.0f}s ({eta/60:.1f} min)", flush=True)

    # Checkpoint
    if processed % CHECKPOINT_EVERY < CHUNK_SIZE or processed >= total:
        print(f"  Saving checkpoint at {processed}...", flush=True)
        with open(NER_OUTPUT, 'w') as f:
            json.dump({
                "passage_hash_id_to_entities": passage_entities,
                "sentence_to_entities": sentence_entities
            }, f)
        print(f"  Saved {os.path.getsize(NER_OUTPUT)/1024/1024:.1f} MB", flush=True)

print(f"\n✓ NER complete! Total time: {(time.time()-start_time)/60:.1f} min")
print(f"  Passages: {len(passage_entities)}")
print(f"  Sentences: {len(sentence_entities)}")
