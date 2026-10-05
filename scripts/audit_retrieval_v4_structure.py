"""Check the isolated heading repair on validation source trees, without models."""
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bounded_clients import save
from scripts.bounded_eval import read_json,sha
from scripts.retrieval_v3 import load_run,index_path
from scripts.retrieval_v4_structure import restore_heading_hierarchy
from zero_index.index import DocumentIndex


def audit():
    _,_,docs=load_run(ROOT/'output/retrieval-v3-confirmatory/validation')
    before=[];after=[];paragraphs=0;changed=0;affected_before=[];affected_after=[]
    for doc_id in docs:
        original=DocumentIndex.from_dict(read_json(index_path(doc_id,'J','J')))
        repaired=restore_heading_hierarchy(original)
        original_leaves={n.node_id:n for n in original.root.walk() if n.kind in ('paragraph','sentence','section')}
        repaired_leaves={n.node_id:n for n in repaired.root.walk() if n.kind in ('paragraph','sentence','section')}
        assert original_leaves==repaired_leaves
        assert original.source==repaired.source and original.decisions==repaired.decisions
        assert original.metadata['source_sha256']==repaired.metadata['source_sha256']
        assert len(list(repaired.root.walk()))==len(repaired._nodes)
        assert repaired.to_dict()==restore_heading_hierarchy(repaired).to_dict()
        DocumentIndex.from_dict(repaired.to_dict())
        paragraphs+=sum(n.kind=='paragraph' for n in original_leaves.values())
        before.append(len(original.root.children));after.append(len(repaired.root.children))
        is_changed=any(' ::: ' in n.title for n in original.root.children)
        changed+=is_changed
        if is_changed:affected_before.append(before[-1]);affected_after.append(after[-1])
    summary={'partition':'validation','audit':'source and topology invariants only; retrieval accuracy not measured',
        'repair_source_sha256':sha(ROOT/'scripts/retrieval_v4_structure.py'),
        'audit_source_sha256':sha(Path(__file__)),'papers':len(docs),'papers_with_repaired_native_paths':changed,
        'original_paragraphs_preserved':paragraphs,
        'root_children_before':{'median':statistics.median(before),'maximum':max(before)},
        'root_children_after':{'median':statistics.median(after),'maximum':max(after)},
        'affected_papers_root_children_before_median':statistics.median(affected_before),
        'affected_papers_root_children_after_median':statistics.median(affected_after),
        'source_spans_ids_topic_groups_centrals_preserved':True,'model_calls':0}
    save(ROOT/'output/retrieval-v3-validation-diagnostics/structure-repair.json',summary)
    print(summary)


if __name__=='__main__':audit()
