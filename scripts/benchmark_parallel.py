"""Offline scheduling benchmark. Synthetic latency and scores, never live Jev."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import platform
import re
import statistics
import sys
from threading import Lock
import time
from unittest.mock import patch

from zero_index import JevScorer, build_index

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "evals/results/pilot-v1-live"


def legacy_package():
    """Use the archived implementation, verifying its original file hashes."""
    manifest = json.loads((ARCHIVE / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (ARCHIVE / "source" / name).resolve()
        if not path.is_relative_to(ARCHIVE / "source"):
            raise ValueError("Invalid archived source path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Archived source hash mismatch")
    package = ARCHIVE / "source/zero_index"
    spec = importlib.util.spec_from_file_location(
        "parallel_benchmark_legacy", package / "__init__.py", submodule_search_locations=[str(package)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SyntheticTransport:
    """Thread-safe fake network with fixed per-request latency and stable scores."""

    def __init__(self, delay):
        self.delay = delay
        self.lock = Lock()
        self.active = self.peak = self.requests = self.questions = 0

    @staticmethod
    def topic(anchor, candidate):
        # Paragraphs 0/1 share a topic; 2/3 share another. This exercises resets.
        left = int(re.search(r"paragraph (\d+)", anchor)[1])
        right = int(re.search(r"paragraph (\d+)", candidate)[1])
        return .95 if left // 2 == right // 2 else .05

    def __call__(self, request, timeout):
        body = json.loads(request.data)
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.requests += 1
            self.questions += len(body["questions"])
        try:
            time.sleep(self.delay)
            state, answers = body["state"], {}
            for key, question in body["questions"].items():
                if "anchor" in state:
                    value = self.topic(state["anchor"], state["candidate"])
                elif "pairs" in state:
                    pair = state["pairs"][f"p{key[1:]}"]
                    value = self.topic(pair["anchor"], pair["candidate"])
                else:
                    sentence_key = re.search(r"`sentences\.(s\d+)`", question["instructions"])[1]
                    context_key = re.search(r"`contexts\.(c\d+)`", question["instructions"])[1]
                    identity = [state["sentences"][sentence_key], state["contexts"][context_key]]
                    digest = hashlib.sha256(json.dumps(identity).encode()).digest()
                    value = int.from_bytes(digest[:2], "big") / 65535
                answers[key] = {"type": "noul", "noul": value}
            return io.BytesIO(json.dumps({"model": "synthetic-fixed-scores", "answers": answers}).encode())
        finally:
            with self.lock:
                self.active -= 1


def corpus(headings, paragraphs, sentences):
    return "\n\n".join(
        f"# Heading {h}\n\n" + "\n\n".join(
            " ".join(f"Heading {h} paragraph {p} sentence {s} describes its topic."
                     for s in range(sentences)) for p in range(paragraphs)
        ) for h in range(headings)
    )


def run(delay=.04, repeats=3, headings=8, paragraphs=4, sentences=8, batch_size=64, workers=4):
    legacy = legacy_package()
    source = corpus(headings, paragraphs, sentences)
    variants = [("legacy_waves_serial", legacy.build_index, legacy.JevScorer, 1),
                ("coalesced_serial", build_index, JevScorer, 1),
                ("coalesced_concurrent", build_index, JevScorer, workers)]
    reference = None
    results = {}
    for label, builder, client, concurrency in variants:
        rows = []
        for _ in range(repeats):
            transport = SyntheticTransport(delay)
            options = {} if client is legacy.JevScorer else {"max_concurrency": concurrency}
            scorer = client(api_key="offline-placeholder", batch_size=batch_size, max_calls=100000, **options)
            started = time.perf_counter()
            with patch(f"{client.__module__}.urlopen", transport):
                index = builder(source, scorer=scorer)
            elapsed = time.perf_counter() - started
            evidence = {"tree": asdict(index.root), "decisions": index.decisions}
            if reference is None:
                reference = evidence
            if evidence != reference:
                raise AssertionError("Scheduling changed fixed-score tree, spans, representatives, or decisions")
            rows.append({"elapsed_seconds": elapsed, "requests": transport.requests,
                         "questions": transport.questions, "peak_in_flight": transport.peak})
        results[label] = {"median_seconds": statistics.median(r["elapsed_seconds"] for r in rows),
                          "runs": rows}
    baseline = results["legacy_waves_serial"]["median_seconds"]
    for result in results.values():
        result["speedup_vs_legacy"] = baseline / result["median_seconds"]
    paths = [Path(__file__).resolve(), *sorted((ROOT / "zero_index").glob("*.py"))]
    return {"mode": "offline-synthetic-transport", "network_calls": 0,
            "fixed_score_tree_and_decisions_equal": True,
            "python": platform.python_version(), "platform": platform.system(),
            "parameters": {"delay_seconds": delay, "repeats": repeats, "headings": headings,
                           "paragraphs_per_heading": paragraphs, "sentences_per_paragraph": sentences,
                           "batch_size": batch_size, "max_concurrency": workers},
            "runtime_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in paths},
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay-ms", type=float, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.delay_ms) or args.delay_ms < 0 or args.repeats < 1 or args.max_concurrency < 1:
        parser.error("delay must be finite and nonnegative; repeats and concurrency must be positive")
    result = run(delay=args.delay_ms / 1000, repeats=args.repeats, workers=args.max_concurrency)
    content = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    print(content)


if __name__ == "__main__":
    main()
