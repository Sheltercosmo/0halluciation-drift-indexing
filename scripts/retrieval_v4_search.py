"""Repaired hierarchy traversal and common full-paragraph Jev final ranking."""
from dataclasses import asdict
from zero_index.evidence_search import (EvidenceSearchConfig,source_card,card_text,
    validate_scores,fuse_ids)


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


def final_paragraph_rerank(doc,question,ranking,decide,limit=30,score_weight=1.):
    """Use exact original paragraphs and the complete question; no summaries."""
    if type(limit) is not int or limit<1:raise ValueError('Positive candidate limit required')
    if not 0<=score_weight<=1:raise ValueError('Score weight must be in [0,1]')
    pool=list(dict.fromkeys(ranking))[:limit];cards=[]
    for paragraph in pool:
        if not paragraph.startswith('p') or not paragraph[1:].isdigit():raise ValueError('Expected paragraph ID')
        unit=doc['units'][int(paragraph[1:])]
        if unit['paragraph']!=int(paragraph[1:]):raise ValueError('Paragraph identity mismatch')
        cards.append({'node_id':paragraph,'kind':'paragraph','heading_path':[unit['heading']],
            'central_sentence':'','excerpts':[{'start':unit['start'],'end':unit['end'],
                'text':doc['text'][unit['start']:unit['end']],'role':'full_source'}]})
    decisions=validate_scores(decide(question,question,cards),len(cards)) if cards else []
    scores=[score_weight*p+(1-score_weight)*(1-i/max(1,len(pool)-1)) for i,p in enumerate(decisions)]
    ordered=sorted(range(len(pool)),key=lambda i:(-scores[i],i))
    return {'ranking':[pool[i] for i in ordered],'candidate_pool':pool,'scores':scores,
            'decision_scores':decisions,'final_score_weight':score_weight,
            'question':question,'input_policy':'whole_original_paragraph','status':'complete'}
