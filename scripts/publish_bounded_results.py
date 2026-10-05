"""Export completed scores and provenance without document text or provider secrets."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_clients import save


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def export(source, destination):
    status = read(source / "status.json")
    manifest, summary, rows = [read(source / name) for name in ("manifest.json", "summary.json", "scores.json")]
    if status["status"] != "scored_complete" or len(rows) != manifest["questions"] * len(manifest["methods"]):
        raise ValueError("Only the complete registered run can be exported as results")
    expected = {(x["id"], m) for x in read(source / "selection.json") for m in manifest["methods"]}
    if len({(x["id"], x["method"]) for x in rows}) != len(rows) or {(x["id"], x["method"]) for x in rows} != expected:
        raise ValueError("Score coverage is incomplete or duplicated")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "selection.json", "summary.json", "scores.json", "budget.json", "runtime.json", "index-integrity.json"):
        (destination / name).write_bytes((source / name).read_bytes())
    for name, expected_hash in manifest["source_hashes"].items():
        payload = (source / "source" / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError("Frozen source hash mismatch: " + name)
        target = destination / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    # This data-source manifest is needed by prepare when using the frozen runner.
    target = destination / "source/evals/frontier/sources.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((ROOT / "evals/frontier/sources.json").read_bytes())
    provenance = {"manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
                  "stages": {}, "indexes": {}}
    for provider in ("embedding", "jev", "codex"):
        audit = [json.loads(s) for s in (source / (provider + "-audit.jsonl")).read_text(encoding="utf-8").splitlines()]
        # These clients write whitelisted metadata, hashes and usage, never request bodies.
        save(destination / (provider + "-audit.json"), audit)
        for stage in sorted({x["stage"] for x in audit}):
            part = [x for x in audit if x["stage"] == stage]
            usage = Counter()
            for row in part:
                usage.update({k: v for k, v in row.get("usage", {}).items() if type(v) is int})
                if provider == "jev":
                    usage.update({k: row[k] for k in ("input_tokens", "output_tokens") if type(row.get(k)) is int})
            provenance["stages"][provider + "/" + stage] = {
                "requests": len(part), "questions": sum(x.get("questions", 0) for x in part),
                "inputs": sum(x.get("inputs", 0) for x in part),
                "provider_counted_tokens": sum(x.get("provider_counted_tokens", 0) for x in part),
                "tokens": dict(usage), "summed_request_seconds": sum(x["seconds"] for x in part)}
    indexes = [read(p) for p in (source / "indexes").glob("*.json")]
    if len(indexes) != manifest["documents"]:
        raise ValueError("Index coverage mismatch")
    provenance["indexes"] = {
        "documents": len(indexes), "comparisons_consumed": sum(len(x["trace"]) for x in indexes),
        "topic_cuts": sum(t["cut"] for x in indexes for t in x["trace"]),
        "chunks": {m: sum(len(x[m]) for x in indexes) for m in ("recursive", "semantic", "jev")},
        "mean_document_index_seconds": statistics.mean(x["index_seconds"] for x in indexes)}
    save(destination / "provenance.json", provenance)
    figure(summary, destination)
    print(json.dumps({"exported_predictions": len(rows), "destination": str(destination)}))


def figure(summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"hybrid_recursive": "Hybrid + paragraph packing",
              "hybrid_semantic": "Hybrid + semantic chunking",
              "hybrid_codex_rerank": "Hybrid + Codex reranking"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    fig.patch.set_facecolor("#f7f8fa")
    for ax, dataset, title in zip(axes, ("quality", "qasper"),
                                  ("QuALITY-HARD · accuracy", "QASPER · answer F1")):
        ax.set_facecolor("#f7f8fa")
        for i, (baseline, label) in enumerate(labels.items()):
            result = summary["paired_document_bootstrap"][dataset][baseline]["answer_score"]
            point, low, high = result["difference"]*100, result["ci95"][0]*100, result["ci95"][1]*100
            ax.plot([low, high], [i, i], color="#147d79", linewidth=2.8)
            ax.scatter([point], [i], color="#147d79", s=55, zorder=3)
        ax.axvline(0, color="#697586", linewidth=1, linestyle="--")
        ax.set_yticks(range(len(labels)), list(labels.values()), fontsize=9)
        ax.invert_yaxis()
        ax.set_ylim(2.5, -.5)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=16)
        ax.set_xlabel("Jev pipeline minus baseline (percentage points)", fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=.12)
    fig.suptitle("Decision-based blocking and reranking: paired comparison", fontsize=16, fontweight="bold")
    fig.text(.5, -.025, "384 selected questions · 307 documents · 2,048-token context · 95% document-bootstrap intervals",
             ha="center", fontsize=9, color="#475467")
    fig.savefig(destination / "comparison.svg", bbox_inches="tight")
    fig.savefig(destination / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "output/bounded-v1")
    parser.add_argument("--destination", type=Path, default=ROOT / "evals/results/bounded-v1")
    args = parser.parse_args()
    export(args.source, args.destination)
