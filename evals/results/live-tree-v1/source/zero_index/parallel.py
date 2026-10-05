"""Bounded I/O scheduling with completion-driven replenishment."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def completed_map(function, items, workers):
    """Yield (item, result), keeping at most workers futures submitted.

    Consumers restore source order. A slow first request must not prevent a
    completed worker from taking another batch. On failure, stop submitting,
    cancel queued work and join requests already in flight before raising.
    """
    if workers == 1:
        for item in items:
            yield item, function(item)
        return
    iterator = iter(items)
    pending = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            def fill():
                while len(pending) < workers:
                    item = next(iterator, None)
                    if item is None:
                        break
                    pending[pool.submit(function, item)] = item

            fill()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Check all completed requests before scheduling replacements.
                results = [(pending.pop(future), future.result()) for future in done]
                for item, result in results:
                    yield item, result
                fill()
        finally:
            for future in pending:
                future.cancel()
