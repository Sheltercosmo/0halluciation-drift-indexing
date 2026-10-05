"""Verify public predictions against official labels, without model calls."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Metric mismatch: " + name)


def replay(results, inputs):
    frozen = module("frozen_bounded", results / "source/scripts/bounded_eval.py")
    manifest = read(results / "manifest.json")
    for name, expected in manifest["source_hashes"].items():
        if frozen.sha(results / "source" / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    for name, expected in manifest["data_hashes"].items():
        if frozen.sha(inputs / name) != expected:
            raise ValueError("Reconstructed input mismatch: " + name)
    cases, docs, gold = [read(inputs / name) for name in ("cases.json", "documents.json", "gold.json")]
    rows, summary = read(results / "scores.json"), read(results / "summary.json")
    strata = {r["id"]: r["stratum"] for r in read(inputs / "selection.json")}
    if summary["questions"] != len(cases) or summary["documents"] != len(docs):
        raise ValueError("Global denominator mismatch")
    expected = {(c["id"], m) for c in cases for m in manifest["methods"]}
    if len(rows) != len(expected) or {(r["id"], r["method"]) for r in rows} != expected:
        raise ValueError("Incomplete or duplicate prediction coverage")
    official = module("official_qasper", inputs / "qasper_evaluator.py")
    refs = official.get_answers_and_evidence({"paper": {"qas": [
        {"question_id": c["id"], "answers": [{"answer": a} for a in gold[c["id"]]["answers"]]}
        for c in cases if c["dataset"] == "qasper"]}}, False)
    by_id, enc = {c["id"]: c for c in cases}, frozen.tokenizer()
    for row in rows:
        case = by_id[row["id"]]
        if row["doc_id"] != case["doc_id"] or row["dataset"] != case["dataset"] or row["stratum"] != strata[row["id"]] or row["status"] not in ("ok", "failed"):
            raise ValueError("Invalid prediction identity or status")
        doc = docs[case["doc_id"]]
        chunks = []
        for a, b in row["spans"]:
            if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(doc["text"]):
                raise ValueError("Invalid source span")
            heading = next(u["heading"] for u in doc["units"] if u["start"] <= a < u["end"])
            chunks.append({"start": a, "end": b, "heading": heading, "text": doc["text"][a:b]})
        context = "Title: " + doc["title"] + "\n" + "\n\n".join(
            f"[{i+1}] {c['heading']}\n{c['text']}" for i, c in enumerate(sorted(chunks, key=lambda x: x["start"])))
        tokens = len(enc.encode(context, disallowed_special=()))
        if tokens != row["context_tokens"] or tokens > 2048:
            raise ValueError("Context token count mismatch")
        if case["dataset"] == "quality":
            answer = int(row["status"] == "ok" and row["answer"] == gold[case["id"]]["label"])
        else:
            references = refs[case["id"]]
            answer = max(official.token_f1_score(row["answer"], r["answer"]) for r in references) if row["status"] == "ok" else 0
            evidence = frozen.covered_paragraphs(doc, row["spans"])
            close(row["retrieved_evidence_f1"], max(official.paragraph_f1_score(evidence, r["evidence"]) for r in references), "retrieved evidence F1")
            eligible = [r for r in references if r["evidence"]]
            if eligible:
                recall = max(len(set(evidence) & set(r["evidence"]))/len(set(r["evidence"])) for r in eligible)
                complete = int(any(set(r["evidence"]) <= set(evidence) for r in eligible))
                close(row["evidence_recall"], recall, "evidence recall")
                close(row["complete_evidence"], complete, "complete evidence")
            elif row["evidence_recall"] is not None or row["complete_evidence"] is not None:
                raise ValueError("Undefined evidence metric received a score")
        close(row["answer_score"], answer, "answer score")
    for dataset, methods in summary["methods"].items():
        subset = [r for r in rows if r["dataset"] == dataset]
        for method, metrics in methods.items():
            part = [r for r in subset if r["method"] == method]
            if sum(r["status"] != "ok" for r in part) != metrics["failed"]:
                raise ValueError("Failure denominator mismatch")
            for metric, value in metrics.items():
                if metric == "failed":
                    continue
                eligible = [r[metric] for r in part if r[metric] is not None]
                if not eligible:
                    if value is not None:
                        raise ValueError("Undefined aggregate metric")
                else:
                    if len(eligible) != value["n"]:
                        raise ValueError("Metric denominator mismatch")
                    close(value["mean"], statistics.mean(eligible), metric)
        for baseline, metrics in summary["paired_document_bootstrap"][dataset].items():
            for metric, recorded in metrics.items():
                calculated = frozen.paired_interval(subset, metric, baseline)
                for key in ("difference", "documents", "questions", "replicates"):
                    close(calculated[key], recorded[key], "bootstrap " + key)
                for actual, expected in zip(calculated["ci95"], recorded["ci95"]):
                    close(actual, expected, "bootstrap interval")
    for stratum, methods in summary["qasper_strata"].items():
        for method, metrics in methods.items():
            part = [r for r in rows if r["stratum"] == stratum and r["method"] == method]
            if len(part) != metrics["n"]:
                raise ValueError("Stratum denominator mismatch")
            close(statistics.mean(r["answer_score"] for r in part), metrics["answer_f1"], "stratum F1")
    result = {"validated_predictions": len(rows), "questions": len(cases),
              "source_and_data_hashes": "passed", "source_contexts": "passed",
              "official_answer_and_evidence_scores": "passed", "paired_bootstrap": "passed", "model_calls": 0}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "evals/results/bounded-v1")
    parser.add_argument("--inputs", type=Path, default=ROOT / "output/bounded-replay")
    args = parser.parse_args()
    replay(args.results, args.inputs)
