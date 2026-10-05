"""Experimental bottom-up context assembly from exact topic-block source spans.

The adaptive coverage idea follows HiChunk's published auto-merge retrieval:
https://github.com/TencentCloudADP/hichunk/blob/main/retrieval_algo.py
This implementation uses source offsets and a strict rendered token budget.
It is not a reproduction of HiChunk's trained document chunker.
"""


def topic_parents(doc, trace):
    """Recover topic-block parents from the already-computed boundary decisions."""
    cuts = {r['candidate'] for r in trace if r['cut']}
    groups = []
    for unit in doc['units']:
        if not groups or unit['section'] != groups[-1][-1]['section'] or unit['start'] in cuts:
            groups.append([])
        groups[-1].append(unit)
    return [{'id': f"topic:{g[0]['start']}:{g[-1]['end']}", 'start': g[0]['start'], 'end': g[-1]['end'],
             'heading': g[0]['heading'], 'text': doc['text'][g[0]['start']:g[-1]['end']]} for g in groups]


def merge_context(doc, ranking, parents, token_count, *, budget=2048, minimum_coverage=None):
    """Read ranked children, then include their complete parent when it fits.

    At least two selected children are required. The adaptive threshold rises
    from one third to two thirds as the reader budget fills. A fixed threshold
    can be used for explicitly registered development comparisons.
    """
    if type(budget) is not int or budget < 1:
        raise ValueError('budget must be a positive integer')
    if minimum_coverage is not None and not 0 < minimum_coverage <= 1:
        raise ValueError('minimum_coverage must be in (0, 1]')
    ordered_parents = sorted(parents, key=lambda p: p['start'])
    for i, parent in enumerate(ordered_parents):
        if not 0 <= parent['start'] < parent['end'] <= len(doc['text']):
            raise ValueError('Invalid parent span')
        if parent['text'] != doc['text'][parent['start']:parent['end']]:
            raise ValueError('Parent text must match source')
        if i and ordered_parents[i-1]['end'] > parent['start']:
            raise ValueError('Topic parents must be disjoint')

    def render(items):
        return 'Title: ' + doc['title'] + '\n' + '\n\n'.join(
            f"[{i+1}] {c['heading']}\n{c['text']}" for i, c in enumerate(sorted(items, key=lambda c: c['start'])))
    if token_count(render([])) > budget:
        raise ValueError('Document title alone exceeds the context budget')
    chosen, merges = [], []
    for chunk in ranking:
        if chunk['text'] != doc['text'][chunk['start']:chunk['end']]:
            raise ValueError('Candidate text must match source')
        if any(c['start'] <= chunk['start'] and chunk['end'] <= c['end'] for c in chosen):
            continue
        if any(max(c['start'], chunk['start']) < min(c['end'], chunk['end']) for c in chosen):
            raise ValueError('Partially overlapping candidates must be deduplicated before packing')
        if token_count(render(chosen + [chunk])) > budget:
            continue
        chosen.append(chunk)
        parent = next((p for p in ordered_parents if p['start'] <= chunk['start'] and chunk['end'] <= p['end']), None)
        if parent is None:
            continue
        children = [c for c in chosen if parent['start'] <= c['start'] and c['end'] <= parent['end']]
        if len(children) < 2:
            continue
        coverage = sum(c['end'] - c['start'] for c in children) / (parent['end'] - parent['start'])
        threshold = minimum_coverage if minimum_coverage is not None else (1 + token_count(render(chosen))/budget)/3
        replacement = [c for c in chosen if c not in children] + [parent]
        if coverage >= threshold and token_count(render(replacement)) <= budget:
            chosen = replacement
            merges.append({'parent': parent['id'], 'children': [c['id'] for c in children],
                           'coverage': coverage, 'threshold': threshold})
    context = render(chosen)
    return {'context': context, 'spans': [(c['start'], c['end']) for c in chosen],
            'context_tokens': token_count(context), 'merges': merges}
