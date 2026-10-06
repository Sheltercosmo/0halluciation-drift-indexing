"""Shared evidence runtime, extracted without changing the measured algorithms."""
from copy import deepcopy
from dataclasses import asdict
from itertools import combinations
import hashlib
import json
import math
import re
from .index import DocumentIndex, Node
from .evidence_search import EvidenceSearchConfig, source_card, card_text, validate_scores, fuse_ids

def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def restore_heading_hierarchy(index):
    """Rebuild contiguous native ``A ::: B`` paths without touching source leaves.

    An absent ancestor becomes a structural node with no generated summary.
    Repeated paths after a different section are separate occurrences. Existing
    heading IDs, topic groups, paragraph IDs, central sentences and source spans
    stay intact except that ancestor heading spans expand to contain children.
    """
    result=DocumentIndex.from_dict(index.to_dict())
    if result.metadata.get('native_hierarchy_repair')=='v4':return result
    if any(child.kind=='heading' for heading in result.root.children for child in heading.children):
        raise ValueError('Expected flat native-heading adapter input')
    old_children=result.root.children;result.root.children=[]
    used={node.node_id for node in index.root.walk()};sequence=0;active=[]
    for heading in old_children:
        if heading.kind!='heading':
            result.root.children.append(heading);active=[];continue
        native_title=heading.title
        parts=[part.strip() for part in native_title.split(' ::: ')]
        if any(not part for part in parts):parts=[native_title]
        common=0
        # The final component is a new native section, even if its title repeats.
        while common<min(len(active),len(parts)-1) and active[common][0]==parts[common]:common+=1
        active=active[:common]
        for depth in range(common,len(parts)):
            parent=active[-1][1] if active else result.root
            if depth==len(parts)-1:
                node=heading;node.title=parts[depth]
                node.metadata.update(native_heading_title=native_title,native_heading_path=parts)
            else:
                while True:
                    sequence+=1;node_id=f'v4h{sequence}'
                    if node_id not in used:break
                used.add(node_id)
                node=Node(node_id,'heading',parts[depth],heading.start,heading.end,
                          metadata={'origin':'native_heading_path','implicit_ancestor':True,
                                    'native_heading_path':parts[:depth+1]})
            parent.children.append(node);active.append((parts[depth],node))
    def expand(node):
        for child in node.children:expand(child)
        if node.kind=='heading' and node.children:
            node.start=min(node.start,*(c.start for c in node.children))
            node.end=max(node.end,*(c.end for c in node.children))
    expand(result.root)
    result.metadata.update(native_hierarchy_repair='v4',
        structure='dataset-native nested heading paths and unchanged source paragraph offsets')
    return DocumentIndex.from_dict(result.to_dict())


def navigation_card(index,node,config,*,detail=False):
    card=source_card(index,node,config,detail=detail)
    headings=[child for child in node.children if child.kind=='heading']
    if headings:
        card['child_headings']=[{'node_id':n.node_id,'title':n.title} for n in headings]
    return card


def navigation_text(card):
    titles=[n['title'] for n in card.get('child_headings',[])]
    return card_text(card)+ ('\n'+'\n'.join(titles) if titles else '')


def search_tree(index,question,needs,decide,config=None):
    """Layer-wise internal beam; final paragraph selection is a separate stage.

    Every scored paragraph remains eligible for the common final pool. Internal
    nodes still require acceptance and a beam slot before descendants are read.
    """
    config=config or EvidenceSearchConfig(beam=5)
    frontiers=[[index.root] for _ in needs];ranked=[[] for _ in needs]
    trace=[];decisions=0;status='complete';depth=0
    while any(frontiers):
        depth+=1
        upcoming=[[child for parent in parents for child in parent.children if child.kind!='sentence']
                  for parents in frontiers]
        if decisions+sum(map(len,upcoming))>config.max_decisions:
            status='truncated';break
        remaining_base=sum(map(len,upcoming))
        next_frontiers=[[] for _ in needs]
        for ni,(need,candidates) in enumerate(zip(needs,upcoming)):
            if not candidates:continue
            cards=[navigation_card(index,n,config) for n in candidates]
            values=validate_scores(decide(question,need,cards),len(cards));decisions+=len(cards)
            remaining_base-=len(cards)
            refine=[i for i,n in enumerate(candidates) if n.kind!='paragraph' and values[i]<config.refine_below]
            extra=[navigation_card(index,candidates[i],config,detail=True) for i in refine]
            if decisions+remaining_base+len(extra)<=config.max_decisions:
                refreshed=validate_scores(decide(question,need,extra),len(extra)) if extra else []
                for i,value in zip(refine,refreshed):values[i]=value
                decisions+=len(extra)
            else:
                refine=[];extra=[];status='truncated'
            internal=sorted([i for i,n in enumerate(candidates) if n.kind!='paragraph' and values[i]>=config.acceptance],
                key=lambda i:(-values[i],candidates[i].start,candidates[i].node_id))[:config.beam]
            terminals=[i for i,n in enumerate(candidates) if n.kind=='paragraph']
            trace.append({'depth':depth,'need_index':ni,'parents':[n.node_id for n in frontiers[ni]],
                'candidates':[n.node_id for n in candidates],'scores':values,
                'refined_ids':[candidates[i].node_id for i in refine],
                'selected_internal':[candidates[i].node_id for i in internal],
                'scored_paragraphs':[candidates[i].node_id for i in terminals]})
            next_frontiers[ni]=[candidates[i] for i in internal]
            ranked[ni].extend((values[i],candidates[i].node_id) for i in terminals)
        frontiers=next_frontiers
        if status=='truncated':break
    per_need=[[node for score,node in sorted(rows,key=lambda row:(-row[0],row[1]))] for rows in ranked]
    return {'ranking':fuse_ids(per_need),'need_rankings':per_need,'trace':trace,
            'decisions':decisions,'status':status,'config':asdict(config),'search':'jev_hierarchical_v4'}


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


def whole_paragraph_cards(doc, ranking, limit=30):
    if type(limit) is not int or limit < 1:
        raise ValueError('Positive paragraph limit required')
    cards = []
    for pid in list(dict.fromkeys(ranking))[:limit]:
        if not pid.startswith('p') or not pid[1:].isdigit():
            raise ValueError('Expected original paragraph ID')
        i = int(pid[1:])
        unit = doc['units'][i]
        if unit['paragraph'] != i:
            raise ValueError('Paragraph identity mismatch')
        cards.append({'node_id': pid, 'kind': 'paragraph', 'heading_path': [unit['heading']],
                      'central_sentence': '', 'excerpts': [{'start': unit['start'], 'end': unit['end'],
                      'text': doc['text'][unit['start']:unit['end']], 'role': 'full_source'}]})
    return cards


def pairwise_rerank(doc, question, ranking, compare, limit=30, *, threshold=0.5):
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0.5 <= threshold <= 1:
        raise ValueError('Pairwise threshold must be finite in [0.5, 1]')
    cards = whole_paragraph_cards(doc, ranking, limit)
    edges = list(combinations(range(len(cards)), 2))
    oriented = [pair for i,j in edges for pair in ((cards[i], cards[j]), (cards[j], cards[i]))]
    values = validate_scores(compare(question, oriented), len(oriented)) if oriented else []
    wins, trace = [0.] * len(cards), []
    for e,(i,j) in enumerate(edges):
        ab, ba = values[2*e:2*e+2]
        # Both orientations must agree; conflicts, equality and uncertainty tie.
        w = 1. if ab > threshold and ba < 1-threshold else 0. if ab < 1-threshold and ba > threshold else .5
        wins[i] += w
        wins[j] += 1-w
        trace.append({'a': cards[i]['node_id'], 'b': cards[j]['node_id'],
                      'p_ab': ab, 'p_ba': ba, 'a_win': w})
    pool = [c['node_id'] for c in cards]
    ordered = sorted(range(len(cards)), key=lambda i: (-wins[i], i))
    return {'ranking': [pool[i] for i in ordered], 'candidate_pool': pool,
            'scores': wins, 'comparisons': trace, 'decisions': len(values),
            'input_policy': 'whole_original_paragraph', 'status': 'complete',
            'method': 'bidirectional_pairwise_evidence_utility', 'question': question}


INTRO=re.compile(r'\b(?:following|below)\b.{0,100}\b(?:methods|models|tasks|approaches|settings|variants|steps)\b|:\s*$',re.I)


def contextual_paragraph_cards(doc,ranking,limit=30,*,context=True,max_context_chars=6000):
    if type(max_context_chars) is not int or max_context_chars<0:
        raise ValueError('Nonnegative context allowance required')
    cards=whole_paragraph_cards(doc,ranking,limit)
    for card in cards:
        if not context:continue
        i=int(card['node_id'][1:]);unit=doc['units'][i]
        card['heading_path']=[s.strip() for s in unit['heading'].split(':::') if s.strip()]
        card['context']=[];card['omitted_context']=[]
        text=doc['text'][unit['start']:unit['end']]
        # A new introduction starts its own scope, not that of the previous list.
        if INTRO.search(text):continue
        previous=[]
        for j in range(i-1,-1,-1):
            u=doc['units'][j]
            if (u['section'],u['heading'])!=(unit['section'],unit['heading']):break
            previous.append(j)
        intro=next((j for j in previous if INTRO.search(doc['text'][doc['units'][j]['start']:doc['units'][j]['end']])),None)
        chosen=[]
        if intro is not None:chosen.append((intro,'active_source_introduction'))
        if previous and previous[0]!=intro:chosen.append((previous[0],'preceding_source_paragraph'))
        used=0
        for j,role in chosen:
            u=doc['units'][j];source=doc['text'][u['start']:u['end']]
            if used+len(source)>max_context_chars:
                card['omitted_context'].append({'node_id':'p'+str(j),'reason':'context_allowance'})
                continue
            card['context'].append({'node_id':'p'+str(j),'role':role,'start':u['start'],'end':u['end'],'text':source})
            used+=len(source)
    return cards


CONTINUATION=re.compile(r'^(?:where\b|such that\b|which\b|and\b|or\b|with\b|respectively\b)',re.I)


def evidence_packet(doc,ranking,*,base_limit=12,context=True,expand=False,max_targets=20):
    if type(base_limit) is not int or base_limit<1:raise ValueError('Positive base limit required')
    if type(max_targets) is not int or max_targets<base_limit:raise ValueError('Invalid target cap')
    if len(set(ranking))!=len(ranking):raise ValueError('Duplicate source candidates')
    if expand and not context:raise ValueError('Expansion requires source context')
    base=[c['node_id'] for c in whole_paragraph_cards(doc,ranking,base_limit)]
    targets=list(base);links=[];sources={};omitted=[]
    def put(pid):
        u=doc['units'][int(pid[1:])]
        sources[pid]={'node_id':pid,'section':u['section'],'heading_path':[s.strip() for s in u['heading'].split(':::') if s.strip()],
                      'start':u['start'],'end':u['end'],'text':doc['text'][u['start']:u['end']]}
    def connect(pid):
        put(pid)
        if not context:return
        card=contextual_paragraph_cards(doc,[pid],limit=1)[0]
        for item in card['context']:
            other=item['node_id'];put(other)
            links.append({'target':pid,'context':other,'role':item['role']})
        omitted.extend({'target':pid,**o} for o in card['omitted_context'])
        i=int(pid[1:]);u=doc['units'][i]
        if i+1<len(doc['units']):
            neighbor=doc['units'][i+1]
            if (neighbor['section'],neighbor['heading'])==(u['section'],u['heading']):
                other='p'+str(i+1);put(other)
                role='dependent_source_continuation' if (sources[pid]['text'].rstrip().endswith((',',':')) or
                    CONTINUATION.match(sources[other]['text'].lstrip())) else 'following_source_paragraph'
                links.append({'target':pid,'context':other,'role':role})
    for pid in base:connect(pid)
    expanded=[]
    if expand:
        # Two bounded source-link hops can recover a formula followed by its
        # explanation, without scanning unrelated sections or the full paper.
        for depth in (1,2):
            proposals=[]
            for link in links:
                if link['role'] not in ('active_source_introduction','dependent_source_continuation'):continue
                if link['context'] in targets or link['context'] in [p['node_id'] for p in proposals]:continue
                proposals.append({'node_id':link['context'],'from':link['target'],'reason':link['role'],'depth':depth})
            added=proposals[:max_targets-len(targets)]
            if not added:break
            for proposal in added:
                targets.append(proposal['node_id']);expanded.append(proposal);connect(proposal['node_id'])
    # Source order preserves document relationships and exposes no retrieval rank.
    order=lambda pid:doc['units'][int(pid[1:])]['start']
    return {'document_title':doc.get('title',''),'selection_size':5,
            'target_ids':sorted(targets,key=order),'passages':[sources[p] for p in sorted(sources,key=order)],
            'links':list({(x['target'],x['context'],x['role']):x for x in links}.values()),
            'omitted_context':omitted,'expanded_targets':expanded}


def joint_select(question,packet,ranking,decide,*,k=5):
    if type(k) is not int or k<1:raise ValueError('Positive selection size required')
    ids=packet['target_ids']
    if len(set(ids))!=len(ids) or len(set(ranking))!=len(ranking):raise ValueError('Duplicate IDs')
    if not set(ids)<={p['node_id'] for p in packet['passages']}:raise ValueError('Missing target source')
    state=deepcopy(packet);state['selection_size']=k
    values=[]
    for reverse in (False,True):
        presented=deepcopy(state)
        if reverse:
            presented['passages'].reverse();presented['target_ids'].reverse();presented['links'].reverse()
        scores=validate_scores(decide(question,presented,ids),len(ids)) if ids else []
        values.append(scores)
    mean=[sum(x)/2 for x in zip(*values)]
    prior=list(dict.fromkeys(list(ranking)+ids))
    ordered=sorted(ids,key=lambda p:(-mean[ids.index(p)],prior.index(p)))
    return {'ranking':ordered+[p for p in prior if p not in ids],
            'selected':ordered[:k],'candidate_pool':ids,'scores':dict(zip(ids,mean)),
            'scores_source_order':dict(zip(ids,values[0])),
            'scores_reverse_order':dict(zip(ids,values[1])),
            'decisions':2*len(ids),'status':'complete','question':question,'selection_k':k,
            'input_policy':'whole_original_paragraph','ranking_prior':False,'repetition_penalty':False,
            'context_key':signature(state)}
