"""Run frozen v4 inference with bounded Windows atomic-file replacement retries.

Only retry the same filesystem operation. Never resend an API request or change
inference, predictions, reservations, prompts, or statistical settings here.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


@contextmanager
def retry_atomic_replaces():
    original=Path.replace
    def replace(path,target):
        for attempt in range(7):
            try:return original(path,target)
            except PermissionError as error:
                if getattr(error,'winerror',None) not in (5,32) or attempt==6:raise
                # The original temp file and destination are unchanged. A budget
                # reservation remains locked and no provider call can begin yet.
                time.sleep(.025*2**attempt)
    Path.replace=replace
    try:yield
    finally:Path.replace=original


if __name__=='__main__':
    from scripts.retrieval_v4 import run,V4
    from scripts.bounded_clients import save
    from scripts.bounded_eval import sha
    p=argparse.ArgumentParser();p.add_argument('partition',choices=['development','test']);p.add_argument('--workers',type=int,default=16)
    a=p.parse_args()
    with retry_atomic_replaces():
        save(V4/a.partition/'execution-runtime.json',{'wrapper_sha256':sha(Path(__file__)),
            'inference_sources_and_settings':'unchanged; validated against frozen manifest',
            'operation':'bounded retries of identical atomic file replacement on Windows sharing/access errors',
            'provider_requests_retried_by_wrapper':False})
        run(V4/a.partition,a.workers)
