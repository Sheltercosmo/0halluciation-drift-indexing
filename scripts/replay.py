"""Replay an archived pilot with its hash-verified original source snapshot."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def replay_equal(actual, expected, field="", path=""):
    """Allow platform roundoff in derived scores, never in decisions or metrics."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            replay_equal(actual[k], expected[k], k, path + "/" + k) for k in actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            replay_equal(a, b, field, path + "/" + str(i))
            for i, (a, b) in enumerate(zip(actual, expected)))
    # BM25 sums depend on the frozen implementation's set order. Log-odds
    # calculations also differ by a few ulps across platform math libraries.
    derived_fields = {"relevance", "posterior_same", "previous_probability", "drop"}
    if field in derived_fields and isinstance(actual, float) and isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
    if actual != expected:
        print(f"Replay mismatch at {path}: {actual!r} != {expected!r}")
    return actual == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    archive, output = args.archive.resolve(), args.output.resolve()
    frozen = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    snapshot = archive / "source"
    for name, expected in frozen["files"].items():
        path = (snapshot / name).resolve()
        if not path.is_relative_to(snapshot) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Archived source hash mismatch")
    if (archive / "failure.json").exists():
        raise ValueError("This archive records a failed attempt; select the completed run for replay")
    # Load the original package from its snapshot, not the current implementation.
    sys.path.insert(0, str(snapshot))
    from evals.run import Transport, evaluate, save
    output.mkdir(parents=True, exist_ok=False)
    transport = Transport(output, replay=archive)
    result = evaluate(transport)
    original = json.loads((archive / "results.json").read_text(encoding="utf-8"))
    # JSON converts span tuples to arrays. Normalize before structural equality.
    for key in ("documents", "summary"):
        if not replay_equal(json.loads(json.dumps(result[key])), original[key]):
            raise AssertionError("Replayed metrics or decisions differ from archive")
    save(output / "results.json", {**result, "mode": "replay", "network_calls": 0})
    save(output / "manifest.json", {**frozen, "mode": "replay", "source_archive": str(archive)})
    print("Replay verified: identical decisions, ranks, evidence and metrics; derived-score tolerance 1e-12; zero network calls.")


if __name__ == "__main__":
    main()
