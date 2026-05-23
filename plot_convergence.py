"""Generate convergence + dev-metric plots for loss comparison + LR sweep."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PLOT_DIR = Path("results/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Phase 1 — loss comparison
LOSSES = [
    ("BCE",       "results/models/ce-loss-bce/train_summary.json",       "#1f77b4"),
    ("Listwise",  "results/models/ce-loss-listwise/train_summary.json",  "#2ca02c"),
    ("InfoNCE",   "results/models/ce-loss-infonce/train_summary.json",   "#d62728"),
]


def smooth(xs, w=25):
    arr = np.array(xs, dtype=float)
    if len(arr) <= w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


# Plot 1 — train loss vs step
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ax1, ax2 = axes

for name, path, col in LOSSES:
    if not Path(path).exists():
        continue
    d = json.loads(Path(path).read_text())
    steps = [r["global_step"] for r in d["step_loss_history"]]
    losses = [r["loss"] for r in d["step_loss_history"]]
    ax1.plot(steps, smooth(losses, 25), label=f"{name}", color=col, linewidth=1.6)
    # raw thin
    ax1.plot(steps, losses, color=col, alpha=0.15, linewidth=0.5)

ax1.set_xlabel("Training step")
ax1.set_ylabel("Train loss (smoothed window=25)")
ax1.set_title("Train Loss Convergence (3 epochs, lr=2e-5, balanced grouped batches)")
ax1.set_yscale("log")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right")
# Mark epoch boundaries
n_steps_per_epoch = 1261
for e in range(1, 4):
    ax1.axvline(x=e * n_steps_per_epoch, color="gray", linestyle="--", alpha=0.5)
    ax1.text(e * n_steps_per_epoch, ax1.get_ylim()[1] * 0.6,
             f"epoch {e}", rotation=90, va="top", fontsize=8, color="gray")

# Plot 2 — dev nDCG@10 per epoch
for name, path, col in LOSSES:
    if not Path(path).exists():
        continue
    d = json.loads(Path(path).read_text())
    epochs = [h["epoch"] for h in d["history"]]
    ndcgs = [h["dev_metrics"].get("nDCG@10", 0) * 100 for h in d["history"]]
    ax2.plot(epochs, ndcgs, marker="o", label=name, color=col, linewidth=2, markersize=8)
    for i, v in enumerate(ndcgs):
        ax2.annotate(f"{v:.2f}", (epochs[i], v), textcoords="offset points",
                     xytext=(5, 5), fontsize=9, color=col)

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Dev nDCG@10 (%)")
ax2.set_title("Dev nDCG@10 per Epoch (90-query dev pool)")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="lower right")
ax2.set_xticks([0, 1, 2])

plt.tight_layout()
plt.savefig(PLOT_DIR / "loss_convergence.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {PLOT_DIR / 'loss_convergence.png'}")


# Plot 3 — detailed dev metrics per epoch (R@1, R@10, MRR@10, nDCG@10) for best loss
fig, ax = plt.subplots(1, 1, figsize=(9, 6))
best_loss_name = "Listwise"
best_path = next(p for n, p, _ in LOSSES if n == best_loss_name)
d = json.loads(Path(best_path).read_text())
epochs = [h["epoch"] for h in d["history"]]
metrics_to_plot = [("Recall@1", "#1f77b4"), ("Recall@10", "#2ca02c"),
                   ("MRR@10", "#ff7f0e"), ("nDCG@10", "#d62728")]
for m, col in metrics_to_plot:
    vs = [h["dev_metrics"].get(m, 0) * 100 for h in d["history"]]
    ax.plot(epochs, vs, marker="o", label=m, color=col, linewidth=2)
    for i, v in enumerate(vs):
        ax.annotate(f"{v:.1f}", (epochs[i], v), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
ax.set_xlabel("Epoch")
ax.set_ylabel("Dev metric (%)")
ax.set_title(f"Detailed dev metrics — best loss = {best_loss_name}")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")
ax.set_xticks([0, 1, 2])
plt.tight_layout()
plt.savefig(PLOT_DIR / "best_loss_dev_metrics.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {PLOT_DIR / 'best_loss_dev_metrics.png'}")


# Plot 4 — LR sweep (if Phase 2 done)
lr_files = [
    ("5e-6", "results/models/ce-lr-5e-6/train_summary.json"),
    ("2e-5", "results/models/ce-loss-listwise/train_summary.json"),  # reuse Phase 1
    ("5e-5", "results/models/ce-lr-5e-5/train_summary.json"),
]
fig, ax = plt.subplots(1, 1, figsize=(9, 6))
xs, ys_e1, ys_e2 = [], [], []
for lr, path in lr_files:
    if not Path(path).exists():
        continue
    d = json.loads(Path(path).read_text())
    hist = d["history"]
    ndcg_e1 = hist[1]["dev_metrics"].get("nDCG@10", 0) * 100 if len(hist) > 1 else 0
    ndcg_e2 = hist[2]["dev_metrics"].get("nDCG@10", 0) * 100 if len(hist) > 2 else ndcg_e1
    xs.append(lr); ys_e1.append(ndcg_e1); ys_e2.append(ndcg_e2)

if xs:
    x_pos = np.arange(len(xs))
    width = 0.35
    ax.bar(x_pos - width/2, ys_e1, width, label="After epoch 1", color="#aec7e8")
    ax.bar(x_pos + width/2, ys_e2, width, label="After epoch 2 (final)", color="#1f77b4")
    for i, (e1, e2) in enumerate(zip(ys_e1, ys_e2)):
        ax.text(i - width/2, e1 + 0.4, f"{e1:.2f}", ha="center", fontsize=9)
        ax.text(i + width/2, e2 + 0.4, f"{e2:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xs)
    ax.set_ylabel("Dev nDCG@10 (%)")
    ax.set_xlabel("Learning rate")
    ax.set_title("LR sweep on best loss (Listwise)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "lr_sweep.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved {PLOT_DIR / 'lr_sweep.png'}")
else:
    print("No LR sweep data yet")


# Save numeric summary
summary = {}
for name, path, _ in LOSSES:
    if not Path(path).exists():
        continue
    d = json.loads(Path(path).read_text())
    summary[name] = {
        "best_epoch": d["best_epoch"],
        "best_score": d["best_score"],
        "history": [
            {"epoch": h["epoch"],
             "Recall@1": h["dev_metrics"].get("Recall@1", 0) * 100,
             "Recall@10": h["dev_metrics"].get("Recall@10", 0) * 100,
             "nDCG@1": h["dev_metrics"].get("nDCG@1", 0) * 100,
             "nDCG@10": h["dev_metrics"].get("nDCG@10", 0) * 100,
             "MRR@10": h["dev_metrics"].get("MRR@10", 0) * 100,
             "train_loss": h["loss"]}
            for h in d["history"]
        ],
    }
Path("results/comparisons/loss_convergence_summary.json").write_text(
    json.dumps(summary, indent=2))
print(f"\nSaved loss_convergence_summary.json")
for name, data in summary.items():
    print(f"\n[{name}] best_epoch={data['best_epoch']} best_nDCG@10={data['best_score']*100:.2f}")
    for h in data["history"]:
        print(f"  E{h['epoch']}: loss={h['train_loss']:.4f} R@1={h['Recall@1']:.2f} "
              f"R@10={h['Recall@10']:.2f} nDCG@1={h['nDCG@1']:.2f} "
              f"nDCG@10={h['nDCG@10']:.2f} MRR@10={h['MRR@10']:.2f}")
