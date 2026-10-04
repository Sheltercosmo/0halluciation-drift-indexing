"""Frozen live pilot and exact offline decision replay. Run from the repo root."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

from zero_index import Config, JevScorer, LexicalJaccard, build_index, find
from zero_index.parse import parse_blocks, sentence_spans
from zero_index.segment import central_sentences, segment

ROOT = Path(__file__).resolve().parents[1]


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def manifest():
    paths = sorted([*ROOT.glob("zero_index/*.py"), *ROOT.glob("evals/*.py"),
                    ROOT / "evals/pilot.json", ROOT / "evals/PROTOCOL.md"])
    return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


class Transport:
    """One global budget and ordered, credential-free transcript for all clients."""

    def __init__(self, output, api_key=None, replay=None):
        self.output, self.api_key = output, api_key
        self.entries = []
        self.replay = ([json.loads(line) for line in (replay / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
                       if replay else None)
        self.attempts = self.questions = 0
        self.stage = "initializing"

    def client(self, batch_size=64):
        owner = self

        class AuditedJev(JevScorer):
            def _request(self, state, questions):
                body = {"model": self.model, "state": state, "questions": questions}
                if owner.attempts >= 220 or owner.questions + len(questions) > 2000:
                    raise RuntimeError("Global pilot request/question budget exceeded")
                owner.attempts += 1
                owner.questions += len(questions)
                if owner.replay is not None:
                    if len(owner.entries) >= len(owner.replay):
                        raise ValueError("Replay transcript exhausted")
                    entry = owner.replay[len(owner.entries)]
                    if entry["request"] != body or entry["stage"] != owner.stage:
                        raise ValueError("Replay request or stage mismatch")
                    if "error" in entry:
                        raise RuntimeError("Recorded provider failure")
                    self.calls += 1
                    self.questions_answered += len(questions)
                    self.request_metrics.append(entry["metric"])
                    self.response_models.update(entry["response_models"])
                    owner.entries.append(entry)
                    return entry["values"]
                entry = {"number": owner.attempts, "stage": owner.stage, "request": body}
                try:
                    values = super()._request(state, questions)
                    entry.update(values=values, response_models=sorted(self.response_models))
                    return values
                except (ValueError, RuntimeError) as exc:
                    entry["error"] = str(exc)
                    raise
                finally:
                    entry["metric"] = self.request_metrics[-1] if self.request_metrics else {"status": "failed"}
                    owner.entries.append(entry)
                    with (owner.output / "decisions.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return AuditedJev(api_key="offline-replay" if self.replay is not None else self.api_key,
                          provider="typesafe", batch_size=batch_size, max_calls=220)


def precision_recall_f1(expected, actual):
    tp = len(set(expected) & set(actual))
    fp, fn = len(set(actual) - set(expected)), len(set(expected) - set(actual))
    return {"tp": tp, "fp": fp, "fn": fn}


def boundary_summary(rows):
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def retrieval_metrics(index, matches, gold):
    positions = [(m["start"], m["end"]) for m in matches]
    relevant = [i + 1 for i, span in enumerate(positions) if span in gold]
    top = index.read(matches[0]["node_id"])
    paragraph = index.read(index.parent(top["node_id"])["node_id"])
    citation = paragraph["citation"]
    assert paragraph["text"] == index.source[citation["start"]:citation["end"]]
    assert citation["source_sha256"] == hashlib.sha256(index.source.encode()).hexdigest()
    return {"hit_at_1": int(bool(relevant) and relevant[0] == 1),
            "recall_at_3": len(set(positions[:3]) & gold) / len(gold),
            "reciprocal_rank": 1 / relevant[0] if relevant else 0.0,
            "paragraph_evidence_recall": sum(citation["start"] <= a < b <= citation["end"] for a, b in gold) / len(gold),
            "read_characters": len(top["text"]) + len(paragraph["text"]),
            "citation_exact": True,
            "top_three": [{"text": m["text"], "start": m["start"], "end": m["end"],
                           "relevance": m["relevance"]} for m in matches[:3]],
            "read_paragraph": paragraph}


def summarize(documents, transport, probe):
    boundaries, representatives, retrieval = {}, {}, {}
    for method in documents[0]["boundaries"]:
        boundaries[method] = boundary_summary([doc["boundaries"][method]["counts"] for doc in documents])
    for method in documents[0]["representatives"]:
        rows = [row for doc in documents for row in doc["representatives"][method]]
        representatives[method] = {"correct": sum(row["correct"] for row in rows), "total": len(rows),
                                   "accuracy": statistics.mean(row["correct"] for row in rows)}
    for method in documents[0]["retrieval"]:
        rows = [row for doc in documents for row in doc["retrieval"][method]]
        retrieval[method] = {"queries": len(rows), **{key: statistics.mean(row[key] for row in rows) for key in
            ("hit_at_1", "recall_at_3", "reciprocal_rank", "paragraph_evidence_recall", "read_characters")},
            "exact_citations": sum(row["citation_exact"] for row in rows)}
    deltas = [statistics.mean(a["hit_at_1"] - b["hit_at_1"] for a, b in
                              zip(doc["retrieval"]["jev_tree_jev"], doc["retrieval"]["jev_tree_bm25"]))
              for doc in documents]
    rng = random.Random(1729)
    samples = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(10000))
    metrics = [entry["metric"] for entry in transport.entries]
    token_usage = {}
    for key in ("input_tokens", "output_tokens"):
        values = [metric.get(key) for metric in metrics]
        token_usage[key] = sum(values) if all(type(value) is int for value in values) else None
    return {"boundaries": boundaries, "representatives": representatives, "retrieval": retrieval,
            "paired_document_bootstrap_hit_at_1_delta": {"mean": statistics.mean(deltas),
                "percentile_95": [samples[249], samples[9749]], "clusters": len(deltas), "resamples": 10000, "seed": 1729},
            "usage": {"attempts": len(metrics), "decision_questions": transport.questions,
                "failed_attempts": sum(metric["status"] != "ok" for metric in metrics), **token_usage,
                "request_seconds": sum(metric.get("elapsed_seconds", 0) for metric in metrics),
                "dollar_cost": None}, "batch_probe": probe}


def evaluate(transport):
    corpus = json.loads((ROOT / "evals/pilot.json").read_text(encoding="utf-8"))
    jev = transport.client()
    documents = []
    first_pairs = None
    for doc in corpus["documents"]:
        source = "# " + doc["title"] + "\n\n" + "\n\n".join(" ".join(p["sentences"]) for p in doc["paragraphs"]) + "\n"
        blocks = [block for block in parse_blocks(source) if block.kind == "paragraph"]
        spans = [sentence_spans(source, block) for block in blocks]
        assert len(blocks) == len(doc["paragraphs"]) == 6
        for gold, positions in zip(doc["paragraphs"], spans):
            assert [source[a:b] for a, b in positions] == gold["sentences"]
        expected = [i for i in range(1, len(blocks)) if doc["paragraphs"][i]["topic"] != doc["paragraphs"][i - 1]["topic"]]
        starts = {block.start: i for i, block in enumerate(blocks)}
        row = {"id": doc["id"], "gold_boundaries": expected, "boundaries": {}, "representatives": {}, "retrieval": {}}
        def add_boundaries(method, actual, trace):
            row["boundaries"][method] = {"cuts_before_paragraph": actual, "counts": precision_recall_f1(expected, actual), "trace": trace}
        add_boundaries("headings_only", [], [])
        add_boundaries("fixed_two_paragraphs", [2, 4], [])
        transport.stage = doc["id"] + ":index"
        lexical = build_index(source, source_name=doc["id"] + ".md")
        indexed = build_index(source, source_name=doc["id"] + ".md", scorer=jev)
        for method, index in (("lexical", lexical), ("jev_prior_0.7", indexed)):
            add_boundaries(method, [starts[d["candidate_start"]] for d in index.decisions if d["cut"]], index.decisions)
            assert index.metadata["structure"]["model_calls"] == 0
        save(transport.output / (doc["id"] + ".index.json"), indexed.to_dict())
        transport.stage = doc["id"] + ":prior_ablation"
        _, trace = segment(source, blocks, jev, Config(same_topic_prior=.5))
        add_boundaries("jev_prior_0.5", [starts[d["candidate_start"]] for d in trace if d["cut"]], trace)
        for method, index in (("lexical", lexical), ("jev_full", indexed)):
            paragraphs = sorted((n for n in index.root.walk() if n.kind == "paragraph"), key=lambda n: n.start)
            row["representatives"][method] = [{"paragraph": i, "selected": node.central["text"],
                "gold": gold["sentences"][gold["central"]], "correct": node.central["text"] == gold["sentences"][gold["central"]]}
                for i, (node, gold) in enumerate(zip(paragraphs, doc["paragraphs"]))]
        row["representatives"]["first_sentence"] = [{"paragraph": i, "selected": p["sentences"][0],
            "gold": p["sentences"][p["central"]], "correct": p["central"] == 0} for i, p in enumerate(doc["paragraphs"])]
        transport.stage = doc["id"] + ":budget_replay"
        before = transport.attempts
        budgeted = central_sentences(source, spans, jev, budget=2)
        assert transport.attempts == before, "Budget ablation must only reuse scored decisions"
        row["representatives"]["jev_budget_2"] = [{"paragraph": i, "selected": central["text"],
            "gold": p["sentences"][p["central"]], "correct": central["text"] == p["sentences"][p["central"]]}
            for i, (central, p) in enumerate(zip(budgeted, doc["paragraphs"]))]
        for method, index, reranker in (("lexical_tree_bm25", lexical, None), ("jev_tree_bm25", indexed, None), ("jev_tree_jev", indexed, jev)):
            queries = []
            for number, query in enumerate(doc["queries"]):
                transport.stage = f"{doc['id']}:retrieval:{number}:{method}"
                matches = find(index, query["question"], query["question"], reranker=reranker,
                               candidate_limit=None, top_k=18)["matches"]
                gold = {spans[p][s] for p, s in query["gold"]}
                queries.append({"question": query["question"], "kind": query["kind"],
                                "gold_spans": sorted(gold), **retrieval_metrics(index, matches, gold)})
            row["retrieval"][method] = queries
        documents.append(row)
        save(transport.output / "documents.json", documents)
        print(f"Evaluated {doc['id']}: {transport.attempts} total API attempts", flush=True)
        if first_pairs is None:
            first_pairs = [(source[a:b], "\n".join(source[x:y] for x, y in positions))
                           for positions in spans[:2] for a, b in positions]
    probe = []
    for repeat in range(3):
        measurements = {}
        for name in (["serial", "batch"] if repeat % 2 == 0 else ["batch", "serial"]):
            transport.stage = f"efficiency:{repeat}:{name}"
            client = transport.client(batch_size=1 if name == "serial" else 64)
            values = client.representatives(first_pairs)
            measurements[name] = {"values": values, "metadata": client.metadata(),
                                  "request_seconds": sum(m["elapsed_seconds"] for m in client.request_metrics)}
        measurements["max_probability_delta"] = max(abs(a - b) for a, b in zip(measurements["serial"]["values"], measurements["batch"]["values"]))
        measurements["paragraph_winner_agreement"] = [
            max(range(3), key=lambda i: measurements["serial"]["values"][offset + i]) ==
            max(range(3), key=lambda i: measurements["batch"]["values"][offset + i]) for offset in (0, 3)]
        probe.append(measurements)
    if transport.replay is not None:
        assert len(transport.entries) == len(transport.replay), "Unused replay decisions"
    return {"documents": documents, "summary": summarize(documents, transport, probe)}


def run(output, *, api_key=None, replay=None):
    output = Path(output)
    replay = Path(replay) if replay else None
    frozen = manifest()
    if replay and json.loads((replay / "manifest.json").read_text())["files"] != frozen:
        raise ValueError("Frozen source/dataset/protocol hashes differ; refusing replay")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "manifest.json", {"created_at": datetime.now(timezone.utc).isoformat(), "files": frozen,
         "mode": "replay" if replay else "live", "config": asdict(Config()), "provider": "typesafe", "model": "jev-1.13.0"})
    transport = Transport(output, api_key, replay)
    started = time.perf_counter()
    try:
        result = evaluate(transport)
        result["mode"] = "replay" if replay else "live"
        result["wall_seconds"] = time.perf_counter() - started
        save(output / "results.json", result)
        if replay:
            original = json.loads((replay / "results.json").read_text(encoding="utf-8"))
            assert result["summary"] == original["summary"]
            assert result["documents"] == original["documents"]
        return result
    except Exception as exc:
        save(output / "failure.json", {"type": type(exc).__name__, "message": str(exc), "stage": transport.stage,
                                       "attempts": transport.attempts, "questions": transport.questions})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    run(args.output, replay=args.replay)


if __name__ == "__main__":
    main()
