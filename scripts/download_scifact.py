"""Download the authors' public dataset into ignored output/, without extracting paths."""

import hashlib
import io
from pathlib import Path
import tarfile
from urllib.request import urlopen

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
EXPECTED = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"


if __name__ == "__main__":
    root = Path("output/scifact")
    root.mkdir(parents=True, exist_ok=True)
    data = urlopen(URL, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != EXPECTED:
        raise ValueError("Upstream archive changed; refusing to replace the pinned data")
    (root / "data.tar.gz").write_bytes(data)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for name in ("data/corpus.jsonl", "data/claims_dev.jsonl"):
            (root / Path(name).name).write_bytes(archive.extractfile(name).read())
    print("Downloaded and verified the pinned SciFact archive.")
