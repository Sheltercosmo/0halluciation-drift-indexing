"""Render published pilot measurements; matplotlib is a development dependency."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "evals/results/pilot-v1-live/results.json").read_text())["summary"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), facecolor="#f7f3ea")
panels = [
    ("Boundary F1", ["Lexical", "Fixed 2", "Jev"],
     [data["boundaries"][key]["f1"] for key in ("lexical", "fixed_two_paragraphs", "jev_prior_0.7")]),
    ("Representative agreement", ["First", "Budget 2", "Jev full"],
     [data["representatives"][key]["accuracy"] for key in ("first_sentence", "jev_budget_2", "jev_full")]),
    ("Retrieval hit@1", ["BM25 / L", "BM25 / J", "Jev / J"],
     [data["retrieval"][key]["hit_at_1"] for key in ("lexical_tree_bm25", "jev_tree_bm25", "jev_tree_jev")]),
]
for ax, (title, labels, values) in zip(axes, panels):
    ax.set_facecolor("#f7f3ea")
    bars = ax.bar(labels, values, width=.58, color=["#c4caba", "#bca278", "#345847"], zorder=3)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=6, color="#263d33", fontsize=11)
    ax.set_title(title, loc="left", fontsize=12, color="#263d33", pad=14, weight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, .5, 1], ["0", "50%", "100%"])
    ax.grid(axis="y", color="#ded9cc", linewidth=.7, zorder=0)
    ax.tick_params(axis="both", length=0, labelcolor="#53645b")
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.suptitle("A small, auditable development pilot", x=.045, y=.98, ha="left", fontsize=18, color="#263d33", weight="bold")
fig.text(.045, .04, "6 synthetic documents · 36 paragraphs · 24 queries  |  L = lexical tree · J = Jev tree\nAssistant-authored labels; no claim of general performance. Full results and failures are included.", fontsize=9, color="#53645b", linespacing=1.6)
fig.subplots_adjust(left=.06, right=.98, top=.78, bottom=.24, wspace=.3)
fig.savefig(ROOT / "assets/pilot-results.svg", facecolor=fig.get_facecolor())
(ROOT / "output").mkdir(exist_ok=True)
fig.savefig(ROOT / "output/pilot-results.png", dpi=160, facecolor=fig.get_facecolor())
