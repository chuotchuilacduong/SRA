"""Convergence plot: loss + eval metrics vs step."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_convergence(
    train_log: str | Path,
    eval_log: str | Path,
    output_path: str | Path,
) -> None:
    train_rows = [json.loads(l) for l in Path(train_log).read_text().splitlines() if l.strip()]
    eval_rows = [json.loads(l) for l in Path(eval_log).read_text().splitlines() if l.strip()]

    fig, (ax_loss, ax_eval) = plt.subplots(1, 2, figsize=(12, 4.5))

    steps = [r["step"] for r in train_rows]
    losses = [r["loss"] for r in train_rows]
    ax_loss.plot(steps, losses, marker=".", lw=1.0, color="#1f77b4", label="train loss")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("InfoNCE loss")
    ax_loss.set_title("Bi-encoder training loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="upper right")

    if eval_rows:
        e_steps = [r["step"] for r in eval_rows]
        for key, color in [
            ("Recall@1", "#d62728"),
            ("Recall@5", "#ff7f0e"),
            ("Recall@10", "#2ca02c"),
            ("nDCG@10", "#9467bd"),
        ]:
            if key in eval_rows[0]:
                ax_eval.plot(e_steps, [r[key] for r in eval_rows], marker="o", lw=1.2,
                             color=color, label=key)
        ax_eval.set_xlabel("step")
        ax_eval.set_ylabel("score")
        ax_eval.set_title("Held-out retrieval metrics")
        ax_eval.set_ylim(0.0, 1.0)
        ax_eval.grid(True, alpha=0.3)
        ax_eval.legend(loc="lower right")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"  Saved convergence plot → {output_path}")
