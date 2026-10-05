"""Plot the complete isolated-reader development results from published JSON."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals/results/isolated-reader-v1"
summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
audit = json.loads((RESULTS / "analysis.json").read_text(encoding="utf-8"))
intervals = json.loads((RESULTS / "paired-intervals.json").read_text(encoding="utf-8"))
assert audit["method_predictions"] == 640 and audit["failed_predictions"] == 0
assert summary["questions"] == 128

methods = [
    ("jev_blocking_rerank", "Original Jev · 12 candidates"),
    ("jev_pool_8192", "Jev · expanded pool"),
    ("jev_fusion_025", "Jev · rank fusion"),
    ("jev_topic_merge", "Jev · parent expansion"),
    ("codex_pool_8192", "Hybrid + Codex reranking"),
]
colors = ["#aab8ac", "#759785", "#345847", "#759785", "#ad8650"]
background, ink, muted = "#f7f3ea", "#263d33", "#53645b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "svg.fonttype": "none", "svg.hashsalt": "isolated-reader-v1"})
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), sharey=True, facecolor=background)
for ax, dataset, title in zip(axes, ("qasper", "quality"),
                              ("QASPER answer F1", "QuALITY-HARD accuracy (%)")):
    values = [100 * summary["methods"][method][dataset]["answer_score"] for method, _ in methods]
    assert all(summary["methods"][method][dataset]["questions"] == 64 for method, _ in methods)
    ax.set_facecolor(background)
    bars = ax.barh(range(len(methods)), values, color=colors, height=.58, zorder=3)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=6, color=ink, fontsize=11)
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks(range(len(methods)), [label for _, label in methods])
    ax.set_title(title, loc="left", color=ink, fontsize=12, weight="bold", pad=15)
    ax.grid(axis="x", color="#ded9cc", linewidth=.7, zorder=0)
    ax.tick_params(axis="both", length=0, labelcolor=muted, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
axes[0].invert_yaxis()
fig.text(.035, .94, "Our best completed development comparison", color=ink, fontsize=20, weight="bold")
fig.text(.035, .884, "64 questions per benchmark · 2,048-token source budget · same isolated reader", color=muted, fontsize=11)
delta = intervals["jev_fusion_025/codex_pool_8192/qasper"]["answer_score"]
low, high = [100 * value for value in delta["ci95"]]
fig.text(.035, .115,
         f"Rank fusion vs Codex: QASPER {100 * delta['difference']:+.2f} points (95% paired interval {low:+.2f} to {high:+.2f}); QuALITY tied.",
         color=ink, fontsize=10)
fig.text(.035, .065,
         "Descriptive development results; superiority is not established. Embeddings enabled; no generative LLM indexing.",
         color=muted, fontsize=9)
fig.subplots_adjust(left=.245, right=.97, top=.765, bottom=.245, wspace=.16)
fig.savefig(ROOT / "assets/development-results.svg", facecolor=background, metadata={"Date": None})
(ROOT / "output").mkdir(exist_ok=True)
fig.savefig(ROOT / "output/development-results.png", dpi=160, facecolor=background)
plt.close(fig)
