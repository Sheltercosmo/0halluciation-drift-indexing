"""Rebuild the public split registry and hashes with no inference or test opening."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_eval import read_json
from scripts.frontier_data import fetch
from scripts.iteration_splits import prepare


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--data', type=Path, default=ROOT / 'output/frontier-data')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/splits-replay')
    args = parser.parse_args()
    registered = ROOT / 'evals/iterations/v1'
    expected = read_json(registered / 'manifest.json')
    fetch(args.data, expected['sources'], download=args.download)
    public = args.output / 'public'
    prepare(args.data, args.output, public)
    actual = read_json(public / 'manifest.json')
    # Metadata about implementation/protocol revisions is retained but is not a
    # data split. Every assignment, input, label file hash and sample must match.
    for key in ('seed', 'counts', 'exposed_source', 'duplicate_moves', 'sources',
                'data_hashes', 'registry_sha256', 'development_screening_ids'):
        if actual[key] != expected[key]:
            raise ValueError('Split replay mismatch: ' + key)
    if read_json(public / 'registry.json') != read_json(registered / 'registry.json'):
        raise ValueError('Assignment registry mismatch')
    if (args.output / 'test-opened.json').exists():
        raise ValueError('Replay must never open the test')
    print('PASS: all assignments and prepared input/gold hashes match; test remains locked; zero model calls.')
