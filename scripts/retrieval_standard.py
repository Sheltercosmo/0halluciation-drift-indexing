"""Configurable EEJ default with retained Jev search and shared-context selection."""
from dataclasses import dataclass, field
import hashlib
import json
from math import ceil
from scripts.jev_scoped_client import ScopedEvidenceJev
from scripts.jev_evidence_comparison import EvidenceComparisonJev,pairwise_rerank
from scripts.jev_joint_evidence import SharedSetJev,evidence_packet,joint_select
from scripts.jev_deferred_search import resume_frontier
from scripts.retrieval_v4_search import search_tree
from scripts.retrieval_v4_structure import restore_heading_hierarchy
from zero_index.evidence_search import fuse_ids
from zero_index.configuration import RetrievalConfig, MEASURED_JJJ_CONFIG
from scripts.standard_index import build_standard_index, check_index_factors, validate_document

# Frozen archive policy, retained for reproducibility. Deployment defaults live
# in RetrievalConfig and configs/retrieval-standard.json, not this manifest.
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


def select_standard_evidence(doc,question,pairwise_ranking,decide,*,config=None):
    """Keep shared context and both presentation orders at any configured effort."""
    config=config or RetrievalConfig()
    packet=evidence_packet(doc,pairwise_ranking,base_limit=config.shared_targets,
                           context=True,expand=False,max_targets=max(20,config.shared_targets))
    packet['selection_size']=config.output_paragraphs
    return {'packet':packet,'result':joint_select(question,packet,pairwise_ranking,decide,k=config.output_paragraphs)}


def retrieve_standard(doc,index,question,needs,route,compare,select,*,dense_ranking=None,config=None):
    """Full standard retrieval; callbacks supply Jev decisions, never gold labels.

    dense_ranking is an independent embedding hit list. Omit it for tree only.
    A query-time LLM supplies needs; the original question is included in routing.
    Final selection receives only the original question and source packet.
    """
    config=config or RetrievalConfig()
    validate_document(doc)
    if index.source!=doc['text']:raise ValueError('Document and index source differ')
    check_index_factors(index,config)
    paragraphs={n.node_id:(n.start,n.end) for n in index.root.walk() if n.kind=='paragraph'}
    if paragraphs!={f'p{i}':(u['start'],u['end']) for i,u in enumerate(doc['units'])}:
        raise ValueError('Index paragraph identities do not match the document')
    if not isinstance(question,str) or not question.strip():raise ValueError('Original question required')
    if not isinstance(needs,(list,tuple)) or any(not isinstance(n,str) or not n.strip() for n in needs):
        raise ValueError('Search needs must be nonempty strings')
    if dense_ranking is not None:
        if not isinstance(dense_ranking,(list,tuple)) or any(not isinstance(p,str) or p not in paragraphs for p in dense_ranking):
            raise ValueError('Dense ranking must contain original paragraph IDs from this document')
        if len(set(dense_ranking))!=len(dense_ranking):raise ValueError('Duplicate dense paragraph IDs')
    index=restore_heading_hierarchy(index)
    requests=list(dict.fromkeys([*needs,question]))
    initial=search_tree(index,question,requests,route,config.search.as_evidence_config())
    retrieval=resume_frontier(index,question,requests,route,initial,
                             ceil(config.deferred_search_fraction*initial['decisions']))
    raw=retrieval['ranking']
    if dense_ranking is not None:
        raw=fuse_ids([raw,list(dense_ranking)[:config.dense_candidates]])
    pairs=pairwise_rerank(doc,question,raw,compare,limit=config.pairwise_candidates,threshold=config.pairwise_threshold)
    chosen=select_standard_evidence(doc,question,pairs['ranking'],select,config=config)
    identity={'pipeline':STANDARD_POLICY['name'],'configuration':config.to_dict()}
    return {'policy':STANDARD_POLICY['name'],'variant':config.variant,
        'configuration':config.to_dict(),
        'configuration_sha256':hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'mode':'hybrid' if dense_ranking is not None else 'full_jev' if config.variant=='JJJ' else 'tree',
        'needs':requests,'initial_retrieval':initial,'retrieval':retrieval,'pairwise':pairs,
        'selection':chosen,'ranking':chosen['result']['ranking'],
        'selected':chosen['result']['selected'],'input_policy':'whole_original_paragraph'}


@dataclass(frozen=True)
class RetrievalAdapter:
    """Bind explicit provider callbacks and one validated index/search policy.

    No provider is called during construction. embed(texts, purpose) returns
    vectors in input order; jev is needed only for J split/central stages.
    """

    route: object
    compare: object
    select: object
    config: RetrievalConfig = field(default_factory=RetrievalConfig)
    embed: object = None
    embedding_model: str | None = None
    jev: object = None

    def __post_init__(self):
        if not isinstance(self.config,RetrievalConfig):raise ValueError('Expected RetrievalConfig')
        if not all(callable(fn) for fn in (self.route,self.compare,self.select)):
            raise ValueError('Routing, pairwise comparison and shared selection callbacks are required')

    def build_index(self,doc):
        return build_standard_index(doc,config=self.config,embed=self.embed,
                                    embedding_model=self.embedding_model,jev=self.jev)

    def retrieve(self,doc,index,question,needs=(),*,dense_ranking=None):
        return retrieve_standard(doc,index,question,needs,self.route,self.compare,self.select,
                                 dense_ranking=dense_ranking,config=self.config)


def make_standard_clients(cache,budget):
    """Separate adapters preserve the evaluated request scoping for each stage."""
    clients=(ScopedEvidenceJev(cache,budget),EvidenceComparisonJev(cache,budget),SharedSetJev(cache,budget))
    for client in clients:client.max_state_chars=250000
    return clients
