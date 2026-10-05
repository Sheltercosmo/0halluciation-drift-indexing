"""Download only the two checksum-pinned inputs needed by the bounded study."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.frontier_data import fetch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "output/frontier-data")
    args = parser.parse_args()
    names = {"qasper-test.tgz", "quality-dev.jsonl"}
    sources = json.loads((ROOT / "evals/frontier/sources.json").read_text(encoding="utf-8"))["files"]
    selected = [x for x in sources if x["name"] in names]
    if {x["name"] for x in selected} != names:
        raise ValueError("Incomplete bounded data-source manifest")
    fetch(args.data, selected, download=True)
    print(json.dumps({"verified_files": sorted(names), "bytes": sum(x["bytes"] for x in selected)}))
