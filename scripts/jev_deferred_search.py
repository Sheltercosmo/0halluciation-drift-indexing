"""Bounded backtracking over promising nodes deferred by the first tree pass.

The first pass is supplied intact, making discovery a controlled additional
stage. Only already-scored, accepted nodes can be expanded. Original branch
decisions, source cards, full paragraph leaves and need fusion are unchanged.
"""
from copy import deepcopy
from zero_index.evidence_search import EvidenceSearchConfig, validate_scores, fuse_ids
from scripts.retrieval_v4_search import navigation_card


def resume_frontier(index, question, needs, decide, initial, extra_decisions):
    if type(extra_decisions) is not int or extra_decisions < 0:
        raise ValueError('Nonnegative decision allowance required')
    config = EvidenceSearchConfig(**initial['config'])
    remaining = min(extra_decisions, max(0, config.max_decisions-initial['decisions']))
    expanded = [set() for _ in needs]
    observed = [{} for _ in needs]
    leaves = [{} for _ in needs]
    for step in initial['trace']:
        ni = step['need_index']
        expanded[ni].update(step['parents'])
        for nid,p in zip(step['candidates'], step['scores']):
            observed[ni][nid] = p
            if index._node(nid).kind == 'paragraph':
                leaves[ni][nid] = p
    queues = [{nid:p for nid,p in row.items() if nid not in expanded[i]
               and index._node(nid).kind != 'paragraph' and p >= config.acceptance}
              for i,row in enumerate(observed)]
    trace, spent = [], 0
    while remaining:
        progress = False
        # Round robin prevents the first generated need consuming every retry.
        for ni,queue in enumerate(queues):
            ordered = sorted(queue, key=lambda n: (-queue[n], index._node(n).start, n))
            parent = next((n for n in ordered if len([c for c in index._node(n).children
                          if c.kind != 'sentence']) <= remaining), None)
            if parent is None:
                continue
            prior = queue.pop(parent)
            expanded[ni].add(parent)
            children = [c for c in index._node(parent).children if c.kind != 'sentence']
            if not children:
                progress = True
                continue
            cards = [navigation_card(index,c,config) for c in children]
            scores = validate_scores(decide(question, needs[ni], cards), len(cards))
            spent += len(cards)
            remaining -= len(cards)
            refine = [i for i,c in enumerate(children)
                      if c.kind != 'paragraph' and scores[i] < config.refine_below]
            # Partial refinement cannot exceed the global allowance.
            refine = refine[:remaining]
            if refine:
                detailed = [navigation_card(index,children[i],config,detail=True) for i in refine]
                rescored = validate_scores(decide(question,needs[ni],detailed), len(detailed))
                for i,p in zip(refine,rescored):
                    scores[i] = p
                spent += len(refine)
                remaining -= len(refine)
            accepted, terminals = [], []
            for c,p in zip(children,scores):
                observed[ni][c.node_id] = p
                if c.kind == 'paragraph':
                    leaves[ni][c.node_id] = p
                    terminals.append(c.node_id)
                elif p >= config.acceptance and c.node_id not in expanded[ni]:
                    queue[c.node_id] = p
                    accepted.append(c.node_id)
            trace.append({'need_index': ni, 'parent': parent, 'parent_score': prior,
                          'candidates': [c.node_id for c in children], 'scores': scores,
                          'refined_ids': [children[i].node_id for i in refine],
                          'queued_internal': accepted, 'scored_paragraphs': terminals})
            progress = True
        if not progress:
            break
    per_need = [[nid for nid,p in sorted(row.items(), key=lambda item: (-item[1],item[0]))]
                for row in leaves]
    result = deepcopy(initial)
    result.update(ranking=fuse_ids(per_need), need_rankings=per_need,
                  deferred_trace=trace, deferred_frontier=queues,
                  extra_decisions=spent, decisions=initial['decisions']+spent,
                  status='budget_exhausted' if any(queues) or initial['status'] != 'complete' else 'complete',
                  search='jev_deferred_frontier')
    return result
