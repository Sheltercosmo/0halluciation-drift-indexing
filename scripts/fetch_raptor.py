"""Fetch exact MIT-licensed RAPTOR sources; no installation or inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'evals/frontier/raptor-source-lock.json'


def verify(root):
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    for row in lock['files']:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Unsafe source path')
        data = path.read_bytes()
        blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if len(data) != row['bytes'] or blob != row['git_blob_sha1']:
            raise ValueError('RAPTOR upstream source changed: ' + row['path'])
    return lock


def fetch(root):
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    def one(row):
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Unsafe source path')
        if path.exists():
            return
        url = f"https://raw.githubusercontent.com/parthsarthi03/raptor/{lock['commit']}/{row['path']}"
        with urlopen(url, timeout=60) as response:
            data = response.read()
        blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if len(data) != row['bytes'] or blob != row['git_blob_sha1']:
            raise ValueError('Downloaded source checksum mismatch: ' + row['path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, lock['files']))
    verify(root)
    print(json.dumps({'commit': lock['commit'], 'verified_files': len(lock['files']), 'model_calls': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/vendor/raptor')
    args = parser.parse_args()
    fetch(args.output)
