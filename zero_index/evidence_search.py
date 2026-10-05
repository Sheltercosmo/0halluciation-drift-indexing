"""Evidence search v3: global embedding ranking or promising-branch exploration.

No gold labels enter this module. All output refers to original source spans.
"""
from dataclasses import dataclass, asdict
import math

from .segment import outside_in


@dataclass(frozen=True)
class EvidenceSearchConfig:
    beam: int = 3
    acceptance: float = 0.2
    global_hits: int = 30
    ancestor_paragraphs: int = 3
    child_cues: int = 6
    extra_sentences: int = 8
    refine_below: float = 0.85
    max_decisions: int = 4096

    def __post_init__(self):
        for key in ('beam', 'global_hits', 'ancestor_paragraphs', 'child_cues', 'extra_sentences', 'max_decisions'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError('Positive integer required: ' + key)
        if not 0 <= self.acceptance <= self.refine_below <= 1:
            raise ValueError('Invalid acceptance/refinement thresholds')


def source_card(index, node, config, *, detail=False):
    """Full paragraphs at evidence level; bounded source cues above that level.

    Detail opens additional non-central sentences, never an entire unseen tree.
    Every excerpt is traceable; omitted middle candidates remain a budget limit.
    """
    card = {'node_id': node.node_id, 'kind': node.kind,
            'heading_path': [p['title'] for p in index.path(node.node_id) if p['kind'] == 'heading'],
            'central_sentence': node.central['text'] if node.central else '', 'excerpts': []}
    spans = set()
    def add(a, b, role):
        if (a, b) not in spans:
            spans.add((a, b))
            card['excerpts'].append({'start': a, 'end': b, 'text': index.source[a:b], 'role': role})
    if node.kind in ('paragraph', 'sentence'):
        add(node.start, node.end, 'full_source')
    else:
        card['title'] = node.title if node.kind in ('heading', 'document') else ''
        if node.central:
            add(node.central['start'], node.central['end'], 'central')
        children = node.children
        for i in list(outside_in(len(children)))[:config.child_cues]:
            child = children[i]
            if child.central:
                add(child.central['start'], child.central['end'], 'child_central')
        if detail:
            leaves = [n for n in node.walk() if n.kind == 'sentence']
            # Spread inspection over paragraphs by taking outside-in waves.
            groups = [[s for s in p.children if s.kind == 'sentence']
                      for p in node.walk() if p.kind == 'paragraph']
            orders = [list(outside_in(len(g))) for g in groups]
            chosen = []
            for depth in range(max(map(len, orders), default=0)):
                for i in outside_in(len(groups)):
                    if depth < len(orders[i]):
                        leaf = groups[i][orders[i][depth]]
                        if (leaf.start, leaf.end) not in spans:
                            chosen.append(leaf)
                            add(leaf.start, leaf.end, 'additional_source')
                            if len(chosen) == config.extra_sentences:
                                break
                if len(chosen) == config.extra_sentences:
                    break
            card['source_sentences_total'] = len(leaves)
    return card


def card_text(card):
    return '\n'.join([*card['heading_path'], card.get('title', ''),
                      card['central_sentence'], *[x['text'] for x in card['excerpts']]])


def validate_scores(values, count):
    values = list(values)
    if len(values) != count or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Invalid decision scores')
    return values


def fuse_ids(rankings, constant=60):
    scores = {}
    for ranking in rankings:
        for position, node_id in enumerate(dict.fromkeys(ranking), 1):
            scores[node_id] = scores.get(node_id, 0.) + 1 / (constant + position)
    return sorted(scores, key=lambda n: (-scores[n], n))


def global_embedding_search(index, needs, score, config=None):
    """Score every depth; a rejected ancestor cannot hide a matching leaf.

    score(need, cards) returns cosine scores. Internal hits open the best-scoring
    descendant paragraphs, while all paragraph and sentence hits remain eligible.
    """
    config = config or EvidenceSearchConfig()
    nodes = [n for n in index.root.walk() if n.kind != 'document']
    cards = [source_card(index, n, config) for n in nodes]
    by_id = {n.node_id: n for n in nodes}
    rankings, trace = [], []
    for need in needs:
        values = list(score(need, cards))
        if len(values) != len(nodes) or not all(math.isfinite(v) for v in values):
            raise ValueError('Invalid embedding scores')
        values = dict(zip(by_id, values))
        ordered = sorted(nodes, key=lambda n: (-values[n.node_id], n.start, n.node_id))[:config.global_hits]
        paragraphs, seen = [], set()
        for node in ordered:
            if node.kind == 'sentence':
                candidates = [index._node(index._parents[node.node_id])]
            elif node.kind == 'paragraph':
                candidates = [node]
            else:
                candidates = sorted([n for n in node.walk() if n.kind == 'paragraph'],
                                    key=lambda n: (-values[n.node_id], n.start))[:config.ancestor_paragraphs]
            for candidate in candidates:
                if candidate.node_id not in seen:
                    seen.add(candidate.node_id); paragraphs.append(candidate.node_id)
        rankings.append(paragraphs)
        trace.append({'need': need, 'global_hits': [n.node_id for n in ordered],
                      'paragraphs': paragraphs, 'node_scores': len(nodes)})
    return {'ranking': fuse_ids(rankings), 'need_rankings': rankings, 'trace': trace,
            'status': 'complete', 'config': asdict(config), 'decisions': 0,
            'node_scores': len(nodes)*len(needs), 'search': 'global_embedding'}


def promising_search(index, question, needs, decide, config=None):
    """Jev explores a layer of children of retained parents for each need.

    decide(question, need, cards) evaluates full paragraph evidence or possible
    descendant evidence. Uncertain internal nodes get additional source cues
    before pruning, including those initially below the acceptance threshold.
    """
    config = config or EvidenceSearchConfig()
    frontiers = [[index.root] for _ in needs]
    ranked = [[] for _ in needs]
    trace, decisions, payload_chars = [], 0, 0
    status, depth = 'complete', 0
    while any(frontiers):
        depth += 1
        upcoming = [[c for p in frontier for c in p.children if c.kind != 'sentence'] for frontier in frontiers]
        if decisions + sum(map(len, upcoming)) > config.max_decisions:
            status = 'truncated'; break
        next_frontiers = [[] for _ in needs]
        for ni, (need, candidates) in enumerate(zip(needs, upcoming)):
            if not candidates:
                continue
            cards = [source_card(index, n, config) for n in candidates]
            values = validate_scores(decide(question, need, cards), len(cards))
            decisions += len(cards); payload_chars += sum(len(card_text(c)) for c in cards)
            refine = [i for i,n in enumerate(candidates) if n.kind != 'paragraph' and values[i] < config.refine_below]
            extra = [source_card(index, candidates[i], config, detail=True) for i in refine]
            if decisions + len(extra) <= config.max_decisions:
                refined = validate_scores(decide(question, need, extra), len(extra)) if extra else []
                for i, value in zip(refine, refined):
                    values[i] = value
                decisions += len(extra); payload_chars += sum(len(card_text(c)) for c in extra)
            else:
                status = 'truncated'; refine, extra = [], []
            accepted = sorted([i for i,v in enumerate(values) if v >= config.acceptance],
                              key=lambda i: (-values[i], candidates[i].start, candidates[i].node_id))[:config.beam]
            trace.append({'depth': depth, 'need_index': ni, 'parents': [n.node_id for n in frontiers[ni]],
                          'candidates': [c['node_id'] for c in cards], 'scores': values,
                          'refined_ids': [c['node_id'] for c in extra],
                          'selected': [candidates[i].node_id for i in accepted]})
            for i in accepted:
                node = candidates[i]
                if node.kind == 'paragraph':
                    ranked[ni].append((values[i], node.node_id))
                else:
                    next_frontiers[ni].append(node)
        frontiers = next_frontiers
        if status == 'truncated':
            break
    rankings = [[node_id for _,node_id in sorted(rows, key=lambda x:(-x[0], x[1]))] for rows in ranked]
    return {'ranking': fuse_ids(rankings), 'need_rankings': rankings, 'trace': trace,
            'status': status, 'config': asdict(config), 'decisions': decisions,
            'payload_chars': payload_chars, 'search': 'jev_promising_frontier'}


def pack_paragraphs(index, ranking, token_count, budget):
    chosen, skipped, tokens = [], [], 0
    for node_id in dict.fromkeys(ranking):
        node = index._node(node_id)
        if node.kind != 'paragraph':
            raise ValueError('Evidence ranking must resolve to original paragraphs')
        size = token_count(index.source[node.start:node.end])
        if tokens + size <= budget:
            tokens += size; chosen.append(node_id)
        else:
            skipped.append(node_id)
    return {'paragraphs': chosen, 'skipped': skipped, 'source_tokens': tokens}
