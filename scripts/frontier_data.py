"""Fetch and audit complete benchmark releases; never calls an inference API."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "evals/frontier/sources.json"
DOMAINS = ("biology", "earth_science", "economics", "psychology", "robotics",
           "stackoverflow", "sustainable_living")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def fetch(data, sources, download=False):
    for row in sources:
        dest = data / row["name"]
        if not dest.is_file():
            if not download:
                raise ValueError(f"Missing input: {row['name']}; run with --download")
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_suffix(dest.suffix + ".partial")
            with urlopen(row["url"], timeout=120) as response, temporary.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            if sha256(temporary) != row["sha256"]:
                raise ValueError(f"Download checksum mismatch: {row['name']}")
            temporary.replace(dest)
        if dest.stat().st_size != row["bytes"] or sha256(dest) != row["sha256"]:
            raise ValueError(f"Input checksum mismatch: {row['name']}")


def unique_ids(rows):
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate evaluation IDs")
    return ids


def validate_coverage(expected, predictions):
    """A run with omitted or duplicated difficult cases cannot pass as complete."""
    expected_ids, predicted_ids = set(unique_ids(expected)), set(unique_ids(predictions))
    if expected_ids != predicted_ids:
        raise ValueError(f"Coverage mismatch: missing={len(expected_ids-predicted_ids)}, "
                         f"unexpected={len(predicted_ids-expected_ids)}")
    for row in predictions:
        if row.get("status") not in {"ok", "failed"}:
            raise ValueError("Every prediction must record status: ok or failed")
    return {"expected": len(expected_ids), "recorded": len(predicted_ids),
            "failed": sum(row["status"] == "failed" for row in predictions)}


def qasper_inventory(data):
    # Read the exact member; do not extract executable evaluator code or arbitrary paths.
    with tarfile.open(data / "qasper-test.tgz") as archive:
        docs = json.load(archive.extractfile("qasper-test-v0.3.json"))
    rows, types = [], Counter()
    paragraph_count = chars = 0
    for doc_id, doc in sorted(docs.items()):
        paragraphs = [p for section in doc["full_text"] for p in section["paragraphs"]]
        paragraph_count += len(paragraphs)
        chars += sum(map(len, paragraphs))
        for qa in doc["qas"]:
            answers = [x["answer"] for x in qa["answers"]]
            all_unanswerable = bool(answers) and all(x["unanswerable"] for x in answers)
            if all_unanswerable:
                types["all_annotations_unanswerable"] += 1
            elif any(x["unanswerable"] for x in answers):
                types["mixed_answerability_annotations"] += 1
            else:
                types["no_unanswerable_annotation"] += 1
            rows.append({"id": "qasper/" + qa["question_id"], "cluster": doc_id,
                         "all_annotations_unanswerable": all_unanswerable,
                         "max_evidence_items": max((len(x["evidence"]) for x in answers), default=0)})
    return rows, {"questions": len(rows), "documents": len(docs),
                  "paragraphs": paragraph_count, "body_characters": chars,
                  "answerability": dict(types), "split": "test-v0.3"}


def quality_inventory(data):
    docs = [json.loads(line) for line in (data / "quality-dev.jsonl").read_text(encoding="utf-8").splitlines()]
    rows, articles = [], {}
    for doc in docs:
        doc_id = doc["article_id"]
        if doc_id in articles and articles[doc_id] != doc["article"]:
            raise ValueError("QuALITY article text differs between question sets")
        articles[doc_id] = doc["article"]
        for qa in doc["questions"]:
            rows.append({"id": "quality/" + qa["question_unique_id"], "cluster": doc_id,
                         "hard": bool(qa["difficult"])})
    return rows, {"questions": len(rows), "hard_questions": sum(x["hard"] for x in rows),
                  "articles": len(articles), "question_sets": len(docs),
                  "body_characters": sum(map(len, articles.values())), "split": "v1.0.1-dev"}


def longbench_inventory(data):
    docs = json.loads((data / "longbench-v2.json").read_text(encoding="utf-8"))
    rows = [{"id": "longbench-v2/" + x["_id"],
             "cluster": hashlib.sha256(x["context"].encode()).hexdigest(),
             "domain": x["domain"], "difficulty": x["difficulty"], "length": x["length"]}
            for x in docs]
    contexts = {r["cluster"]: d["context"] for r, d in zip(rows, docs)}
    return rows, {"questions": len(rows), "exact_distinct_contexts": len(contexts),
                  "body_characters": sum(map(len, contexts.values())),
                  "split": "full release (upstream calls it train; no tuning on it)",
                  **{key: dict(Counter(x[key] for x in rows)) for key in ("domain", "difficulty", "length")}}


def bright_inventory(data):
    import pyarrow.parquet as pq  # Optional evaluation dependency only.
    rows, domains = [], {}
    for domain in DOMAINS:
        def read(config):
            return pq.read_table(data / f"bright-pro/{config}/{domain}.parquet").to_pylist()
        examples, aspects, documents = read("examples"), read("aspects"), read("documents")
        document_ids = set(unique_ids(documents))
        example_ids = {str(x["id"]) for x in examples}
        unique_ids(examples)
        unique_ids(aspects)
        for example in examples:
            if not set(example["gold_ids"]) <= document_ids:
                raise ValueError(f"Gold document missing from {domain} corpus")
            rows.append({"id": f"bright-pro/{domain}/{example['id']}", "cluster": f"{domain}/{example['id']}",
                         "domain": domain})
        for aspect in aspects:
            owner = aspect["id"].removeprefix(domain + "-").rsplit("-a", 1)[0]
            if owner not in example_ids or not set(aspect["supporting_docs"]) <= document_ids:
                raise ValueError(f"Invalid reasoning-aspect link in {domain}")
        domains[domain] = {"queries": len(examples), "documents": len(documents),
                           "aspects": len(aspects), "body_characters": sum(len(x["content"]) for x in documents)}
    return rows, {"questions": len(rows), "domains": domains,
                  "documents": sum(x["documents"] for x in domains.values()),
                  "aspects": sum(x["aspects"] for x in domains.values()),
                  "body_characters": sum(x["body_characters"] for x in domains.values()),
                  "split": "all seven domains; full corresponding corpus per query"}


def inventory(data):
    selection, summary = {}, {}
    for name, fn, count in (("qasper", qasper_inventory, 1451),
                            ("quality", quality_inventory, 2086),
                            ("longbench-v2", longbench_inventory, 503),
                            ("bright-pro", bright_inventory, 739)):
        rows, stats = fn(data)
        unique_ids(rows)
        if len(rows) != count:
            raise ValueError(f"Unexpected release size for {name}: {len(rows)} != {count}")
        selection[name] = sorted(rows, key=lambda x: x["id"])
        summary[name] = stats
    return selection, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "output/frontier-data")
    parser.add_argument("--output", type=Path, default=ROOT / "output/frontier-manifest")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--check-predictions", type=Path)
    parser.add_argument("--dataset", choices=("qasper", "quality", "longbench-v2", "bright-pro"))
    args = parser.parse_args()
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    fetch(args.data, sources["files"], args.download)
    selection, summary = inventory(args.data)
    if args.check_predictions:
        if not args.dataset:
            parser.error("--check-predictions requires --dataset")
        predictions = [json.loads(x) for x in args.check_predictions.read_text(encoding="utf-8").splitlines()]
        print(json.dumps(validate_coverage(selection[args.dataset], predictions)))
        return
    save(args.output / "selection.json", selection)
    save(args.output / "inventory.json", {"status": "data_verified_no_model_results",
         "source_manifest_sha256": sha256(SOURCES),
         "selection_sha256": sha256(args.output / "selection.json"),
         "total_questions_across_separate_tasks": sum(x["questions"] for x in summary.values()),
         "datasets": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
