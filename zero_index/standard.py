"""Configurable EEJ default with retained Jev search and shared-context selection."""
from dataclasses import dataclass, field, replace
import hashlib
import json
from math import ceil
from ._evidence_runtime import pairwise_rerank, evidence_packet, joint_select, resume_frontier, search_tree, restore_heading_hierarchy
from .evidence_search import fuse_ids
from .configuration import RetrievalConfig, MEASURED_JJJ_CONFIG
from .standard_index import build_standard_index, check_index_factors, validate_document

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

    @classmethod
    def from_models(cls, embeddings, decisions, *, config=None):
        """Connect embedding/decision objects without writing stage callbacks.

        A generic model setup uses no provider-specific embedding text prefix.
        An explicitly supplied configuration is honored exactly as written.
        """
        from .providers import DecisionModel
        if not isinstance(decisions,DecisionModel):decisions=DecisionModel(decisions)
        if config is None:
            config=RetrievalConfig()
            config=replace(config,indexing=replace(config.indexing,paragraph_embedding_prefix=''))
        return cls(route=decisions.route_content,compare=decisions.compare,select=decisions.score_pool,
                   config=config,embed=embeddings.embed if embeddings is not None else None,
                   embedding_model=embeddings.model if embeddings is not None else None,jev=decisions)

    def build_index(self,doc):
        return build_standard_index(doc,config=self.config,embed=self.embed,
                                    embedding_model=self.embedding_model,jev=self.jev)

    def retrieve(self,doc,index,question,needs=(),*,dense_ranking=None):
        result=retrieve_standard(doc,index,question,needs,self.route,self.compare,self.select,
                                 dense_ranking=dense_ranking,config=self.config)
        result['paragraphs']=[{'node_id':pid,'text':doc['text'][u['start']:u['end']],
                              'start':u['start'],'end':u['end'],'heading':u['heading']}
                             for pid in result['selected'] for u in [doc['units'][int(pid[1:])]]]
        return result

    def direct_embedding_ranking(self,doc,question,needs=()):
        """Independent direct paragraph retrieval for optional hybrid fusion."""
        from .standard_index import _vectors
        if not callable(self.embed):raise ValueError('Hybrid retrieval requires an embedding provider')
        if not isinstance(question,str) or not question.strip():raise ValueError('Original question required')
        if not isinstance(needs,(list,tuple)) or any(not isinstance(n,str) or not n.strip() for n in needs):
            raise ValueError('Search needs must be nonempty strings')
        validate_document(doc)
        texts=[doc['text'][u['start']:u['end']] for u in doc['units']]
        vectors=_vectors(self.embed,texts,'retrieval-document')
        requests=list(dict.fromkeys([question,*needs]))
        queries=_vectors(self.embed,requests,'retrieval-query')
        if vectors and any(len(q)!=len(vectors[0]) for q in queries):
            raise ValueError('Query and document embedding dimensions differ')
        rankings=[]
        for q in queries:
            scores=[sum(a*b for a,b in zip(q,v)) for v in vectors]
            rankings.append([f'p{i}' for i in sorted(range(len(vectors)),key=lambda i:(-scores[i],i))])
        return fuse_ids(rankings)
