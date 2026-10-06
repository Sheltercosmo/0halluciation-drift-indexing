"""Standard measured policy: Jev traversal, pairwise ranking, shared context.

This deliberately calls the frozen implementations that produced the retained
91.45% / 92.44% historical scores. Later experimental selectors are not defaults.
"""
from math import ceil
from scripts.jev_scoped_client import ScopedEvidenceJev
from scripts.jev_evidence_comparison import EvidenceComparisonJev,pairwise_rerank
from scripts.jev_joint_evidence import SharedSetJev,evidence_packet,joint_select
from scripts.jev_deferred_search import resume_frontier
from scripts.retrieval_v4_search import search_tree
from scripts.retrieval_v4_structure import restore_heading_hierarchy
from zero_index.evidence_search import EvidenceSearchConfig,fuse_ids

STANDARD_POLICY={
    'name':'jev-shared-context-v1','model':'jev-1.13.0',
    'beam':5,'acceptance':.2,'global_hits':30,'ancestor_paragraphs':3,
    'child_cues':6,'extra_sentences':8,'refine_below':.85,'max_decisions':4096,
    'deferred_search_fraction':.25,'pairwise_candidates':30,'shared_targets':12,
    'shared_context':True,'expand_targets':False,'selection_k':5,
    'presentations':['source order','reverse source order'],
    'aggregation':'mean','tie_break':'earlier pairwise rank',
    'bayesian_prior_scope':'topic splitting only',
}


def select_standard_evidence(doc,question,pairwise_ranking,decide):
    """Select from the shared packet using the exact retained measured policy."""
    packet=evidence_packet(doc,pairwise_ranking,base_limit=12,context=True,expand=False,max_targets=20)
    return {'packet':packet,'result':joint_select(question,packet,pairwise_ranking,decide,k=5)}


def retrieve_standard(doc,index,question,needs,route,compare,select,*,dense_ranking=None):
    """Full standard retrieval; callbacks supply Jev decisions, never gold labels.

    dense_ranking is an independent embedding hit list. Omit it for full Jev.
    A query-time LLM supplies needs; the original question is included in routing.
    Final selection receives only the original question and source packet.
    """
    if index.source!=doc['text']:raise ValueError('Document and index source differ')
    if not isinstance(question,str) or not question.strip():raise ValueError('Original question required')
    if not isinstance(needs,(list,tuple)) or any(not isinstance(n,str) or not n.strip() for n in needs):
        raise ValueError('Search needs must be nonempty strings')
    index=restore_heading_hierarchy(index)
    requests=list(dict.fromkeys([*needs,question]))
    fields=EvidenceSearchConfig.__dataclass_fields__
    config=EvidenceSearchConfig(**{k:STANDARD_POLICY[k] for k in fields})
    initial=search_tree(index,question,requests,route,config)
    retrieval=resume_frontier(index,question,requests,route,initial,
                             ceil(.25*initial['decisions']))
    raw=retrieval['ranking']
    if dense_ranking is not None:
        if len(set(dense_ranking))!=len(dense_ranking):raise ValueError('Duplicate dense paragraph IDs')
        raw=fuse_ids([raw,list(dense_ranking)])
    pairs=pairwise_rerank(doc,question,raw,compare,limit=30)
    chosen=select_standard_evidence(doc,question,pairs['ranking'],select)
    return {'policy':STANDARD_POLICY['name'],'mode':'hybrid' if dense_ranking is not None else 'full_jev',
        'needs':requests,'initial_retrieval':initial,'retrieval':retrieval,'pairwise':pairs,
        'selection':chosen,'ranking':chosen['result']['ranking'],
        'selected':chosen['result']['selected'],'input_policy':'whole_original_paragraph'}


def make_standard_clients(cache,budget):
    """Separate adapters preserve the evaluated request scoping for each stage."""
    clients=(ScopedEvidenceJev(cache,budget),EvidenceComparisonJev(cache,budget),SharedSetJev(cache,budget))
    for client in clients:client.max_state_chars=250000
    return clients
