"""Source-preserving candidate selection under a common token allowance."""
from collections.abc import Callable, Iterable


def candidate_pool(
    ranked: Iterable[dict], token_count: Callable[[str], int], *,
    max_source_tokens: int = 8192, max_candidates: int = 128,
) -> dict:
    """Select complete ranked chunks without penalizing small-chunk indexes.

    Counts source text only; callers separately record prompt wrapper tokens.
    Chunks that do not fit are skipped, preserving the relative order of all
    admitted candidates. Duplicate IDs are counted once. Nothing is truncated.
    """
    for name, value in [('max_source_tokens', max_source_tokens), ('max_candidates', max_candidates)]:
        if type(value) is not int or value < 1:
            raise ValueError(name + ' must be a positive integer')
    selected, seen, tokens, skipped = [], set(), 0, 0
    for chunk in ranked:
        if chunk['id'] in seen:
            continue
        seen.add(chunk['id'])
        size = token_count(chunk['text'])
        if type(size) is not int or size < 0:
            raise ValueError('token_count must return a non-negative integer')
        if tokens + size > max_source_tokens:
            skipped += 1
            continue
        selected.append(chunk)
        tokens += size
        if len(selected) == max_candidates:
            break
    return {'candidates': selected, 'source_tokens': tokens,
            'candidate_count': len(selected), 'skipped_for_token_budget': skipped,
            'source_token_limit': max_source_tokens, 'candidate_limit': max_candidates}
