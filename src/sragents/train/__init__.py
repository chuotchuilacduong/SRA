"""Cross-encoder fine-tuning pipeline.

Submodules:

* :mod:`sragents.train.split_builder`  — leakage-resistant train/dev/test splits.
* :mod:`sragents.train.negative_sampler` — HYRR-style hybrid hard-negative mining.
* :mod:`sragents.train.losses` — BCE, pairwise margin, listwise softmax.
* :mod:`sragents.train.train_cross_encoder` — fine-tune loop.
"""
