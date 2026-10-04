"""Prepare, run or replay a small public-data reranking evaluation."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from zero_index import JevScorer
from zero_index.index import DocumentIndex, Node
from zero_index.ranking import candidates, rank_candidates


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(data):
    corpus = {d["doc_id"]: d for d in map(json.loads, (data / "corpus.jsonl").read_text().splitlines())}
    claims = {d["id"]: d for d in map(json.loads, (data / "claims_dev.jsonl").read_text().splitlines())}
    return corpus, claims


def prepare(data):
    corpus, claims = load_data(data)
    eligible = []
    for claim in sorted(claims.values(), key=lambda c: c["id"]):
        labels = {r["label"] for rs in claim["evidence"].values() for r in rs}
        if len(claim["evidence"]) == 1 and len(labels) == 1:
            eligible.append((claim, next(iter(labels))))
    random.Random(1729).shuffle(eligible)
    chosen, counts, used = [], Counter(), set()
    for claim, label in eligible:
        doc_id = int(next(iter(claim["evidence"])))
        if counts[label] < 6 and doc_id not in used:
            chosen.append((claim, label, doc_id)); counts[label] += 1; used.add(doc_id)
        if counts["SUPPORT"] == counts["CONTRADICT"] == 6:
            break
    assert len(chosen) == 12
    terms = lambda s: re.findall(r"\w+", s.casefold())
    counters = {key: Counter(terms(doc["title"] + " " + " ".join(doc["abstract"]))) for key, doc in corpus.items()}
    frequency = Counter(t for c in counters.values() for t in c)
    average = statistics.mean(sum(c.values()) for c in counters.values())
    rows = []
    for claim, label, gold_doc in chosen:
        query = sorted(set(terms(claim["claim"])))
        scores = {}
        for key, counter in counters.items():
            length = sum(counter.values())
            scores[key] = sum(math.log(1 + (len(counters) - frequency[t] + .5) / (frequency[t] + .5)) *
                counter[t] * 2.2 / (counter[t] + 1.2 * (.25 + .75 * length / average))
                for t in query if counter[t])
        others = [key for key in sorted(scores, key=lambda k: (-scores[k], k)) if key != gold_doc][:4]
        rows.append({"claim_id": claim["id"], "label": label, "gold_document": gold_doc,
                     "candidate_documents": sorted([gold_doc, *others])})
    selection = {"seed": 1729, "data_hashes": {p.name: digest(p) for p in
                 (data / "corpus.jsonl", data / "claims_dev.jsonl")}, "claims": rows}
    save(ROOT / "evals/scifact-selection.json", selection)
    print({"selected_claims": len(rows), "labels": dict(counts)})


def make_index(papers):
    source, headings, lookup = "", [], {}
    for paper in papers:
        doc_id = paper["doc_id"]
        hstart = len(source)
        source += "# " + paper["title"] + "\n\n"
        start, sentences = len(source), []
        for number, sentence in enumerate(paper["abstract"]):
            a = len(source); source += sentence
            node = Node(f"d{doc_id}s{number}", "sentence", sentence, a, len(source))
            sentences.append(node); lookup[node.node_id] = (doc_id, number)
            source += " "
        end = len(source) - 1
        first = sentences[0]
        central = {"text": source[first.start:first.end], "start": first.start, "end": first.end}
        paragraph = Node(f"p{doc_id}", "paragraph", "Abstract", start, end, central, sentences)
        headings.append(Node(f"h{doc_id}", "heading", paper["title"], hstart, end, children=[paragraph]))
        source += "\n\n"
    root = Node("root", "document", "SciFact candidate abstracts", 0, len(source), children=headings)
    return DocumentIndex(source, "scifact", root, {"source_sha256": hashlib.sha256(source.encode()).hexdigest()}, []), lookup


def metrics(ranking, gold):
    ranks = [i + 1 for i, span in enumerate(ranking) if span in gold]
    return {"hit_at_1": int(bool(ranks) and ranks[0] == 1), "hit_at_3": int(bool(ranks) and ranks[0] <= 3),
            "recall_at_3": len(set(ranking[:3]) & gold) / len(gold), "mrr": 1 / ranks[0] if ranks else 0}


def run(data, output, api_key=None, replay=None):
    corpus, claims = load_data(data)
    selection_path = ROOT / "evals/scifact-selection.json"
    selection = json.loads(selection_path.read_text())
    for name, expected in selection["data_hashes"].items():
        assert digest(data / name) == expected, "Dataset hash changed"
    paths = [*ROOT.glob("zero_index/*.py"), Path(__file__), selection_path, ROOT / "evals/SCIFACT_PROTOCOL.md"]
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in paths}
    if replay:
        assert json.loads((replay / "manifest.json").read_text())["files"] == hashes
    output.mkdir(parents=True, exist_ok=False)
    save(output / "manifest.json", {"created_at": datetime.now(timezone.utc).isoformat(), "files": hashes,
                                   "data_hashes": selection["data_hashes"], "mode": "replay" if replay else "live"})
    if not replay:
        for name in hashes:
            target = output / "source" / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
    audit = []
    recorded = [json.loads(line) for line in (replay / "decisions.jsonl").read_text().splitlines()] if replay else None

    class RecordedJev(JevScorer):
        def _request(self, state, questions):
            body_hash = hashlib.sha256(json.dumps({"state": state, "questions": questions, "model": self.model},
                                                  ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if sum(row["question_count"] for row in audit) + len(questions) > 1500:
                raise RuntimeError("Question budget exceeded")
            if recorded is not None:
                entry = recorded[len(audit)]
                assert entry["request_sha256"] == body_hash
                self.calls += 1; self.questions_answered += len(questions)
                self.request_metrics.append(entry["metric"])
                self.response_models.update(entry["response_models"])
                audit.append(entry)
                return entry["values"]
            entry = {"request_sha256": body_hash, "question_count": len(questions)}
            try:
                values = super()._request(state, questions)
                entry.update(values=values, response_models=sorted(self.response_models))
                return values
            except (ValueError, RuntimeError) as exc:
                entry["error"] = str(exc)
                raise
            finally:
                entry["metric"] = self.request_metrics[-1]
                audit.append(entry)
                with (output / "decisions.jsonl").open("a") as f:
                    f.write(json.dumps(entry) + "\n")

    scorer = RecordedJev(api_key="replay-only" if replay else api_key, provider="typesafe", max_calls=60)
    rows = []
    try:
        for item in selection["claims"]:
            claim = claims[item["claim_id"]]
            index, lookup = make_index([corpus[k] for k in item["candidate_documents"]])
            pool = candidates(index, claim["claim"], claim["claim"], limit=None)
            gold = {(int(k), s) for k, rs in claim["evidence"].items() for r in rs for s in r["sentences"]}
            row = {**item, "candidates": pool["candidate_count"], "gold_sentences": sorted(gold), "methods": {}}
            for name, reranker in (("bm25", None), ("jev", scorer)):
                ranked = rank_candidates(claim["claim"], claim["claim"], pool, reranker=reranker, top_k=pool["candidate_count"])["matches"]
                positions = [lookup[m["node_id"]] for m in ranked]
                row["methods"][name] = {**metrics(positions, gold), "ranking": positions,
                                         "scores": [m["relevance"] for m in ranked]}
            rows.append(row); save(output / "claims.json", rows)
            print(f"SciFact claim {item['claim_id']}: {len(audit)} recorded requests", flush=True)
        if recorded is not None:
            assert len(audit) == len(recorded)
        summary = {name: {key: statistics.mean(r["methods"][name][key] for r in rows)
                         for key in ("hit_at_1", "hit_at_3", "recall_at_3", "mrr")} for name in ("bm25", "jev")}
        for name in summary:
            summary[name]["hit_at_1_by_label"] = {label: statistics.mean(r["methods"][name]["hit_at_1"]
                for r in rows if r["label"] == label) for label in ("SUPPORT", "CONTRADICT")}
        summary["usage"] = scorer.metadata()
        result = {"claims": rows, "summary": summary}
        save(output / "results.json", result)
        if replay:
            original = json.loads((replay / "results.json").read_text())
            assert json.loads(json.dumps(result)) == original
        print(json.dumps({k:v for k,v in summary.items() if k != "usage"}, indent=2))
        return result
    except Exception as exc:
        save(output / "failure.json", {"error": type(exc).__name__, "message": str(exc), "completed_claims": len(rows)})
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "run"))
    p.add_argument("--data", type=Path, default=Path("output/scifact"))
    p.add_argument("--output", type=Path)
    p.add_argument("--replay", type=Path)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.data)
    else:
        if args.output is None:
            p.error("run requires --output")
        run(args.data, args.output, replay=args.replay)
