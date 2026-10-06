"""Publish the retained shared-context policy from verified saved predictions."""
import ast,gzip,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bounded_eval import read_json,sha
from scripts.bounded_clients import save,signature
from scripts.retrieval_standard import STANDARD_POLICY
from scripts.retrieval_v3_statistics import cluster_summary,contrast,holm
FULL=ROOT/'output/joint-evidence/full_historical'
V5=ROOT/'output/retrieval-v5/test';V4=ROOT/'output/retrieval-v4/test'
DEST=ROOT/'evals/results/shared-context-standard/test'
METHODS=['JJJ_pairwise','hybrid_pairwise','JJJ_shared','hybrid_shared']
LABELS={'JJJ_shared':'Full Jev — shared context (standard)',
 'hybrid_shared':'Jev + independent direct embeddings — shared context (standard)',
 'JJJ_pairwise':'Full Jev — traversal + pairwise ranking',
 'hybrid_pairwise':'Hybrid — traversal + pairwise ranking',
 'JJJ':'Full Jev v4','hybrid':'Hybrid v4','gemini_dense':'Gemini Embedding 2, direct paragraphs',
 'gemini_jev_rerank':'Direct Gemini + Jev v4 reranking','bm25_dense_rrf':'BM25 + direct Gemini, RRF',
 'qwen_dense':'Qwen3 Embedding 0.6B','qwen_rerank':'Qwen3 Embedding + Qwen3 Reranker 0.6B',
 'qwen_jev_rerank':'Qwen3 Embedding + Jev v4 reranking','bge_dense':'BGE-M3 dense',
 'bge_rerank':'BGE-M3 + BGE Reranker v2-M3','bge_jev_rerank':'BGE-M3 + Jev v4 reranking'}
def pct(x):return f'{100*x:.2f}'
def interval(x):return f"{pct(x['mean'])} [{pct(x['ci95'][0])}, {pct(x['ci95'][1])}]"


def sources():
    seen=set();pending=['scripts/retrieval_standard.py','scripts/retrieval_v3_statistics.py','scripts/retrieval_v3_metrics.py']
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path)
        for node in ast.walk(ast.parse((ROOT/path).read_text(encoding='utf-8'))):
            modules=([node.module] if isinstance(node,ast.ImportFrom) and node.level==0 and node.module
              else [n.name for n in node.names] if isinstance(node,ast.Import) else [])
            for module in modules:
                if not module.startswith(('scripts.','zero_index')):continue
                p=module.replace('.','/')+'.py'
                if (ROOT/p).exists():pending.append(p)
                elif (ROOT/module.replace('.','/')/'__init__.py').exists():pending.append(module.replace('.','/')+'/__init__.py')
    seen.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'zero_index').glob('*.py'))
    return sorted(seen)


def publish():
    replay=read_json(ROOT/'output/standard-pipeline-replay.json')
    assert replay['passed'] and replay['matched']==1456 and replay['new_model_calls']==0
    assert replay['source_sha256']==sha(ROOT/'scripts/retrieval_standard.py')
    cases=read_json(FULL/'cases.json');docs=read_json(FULL/'documents.json')
    assert cases==read_json(V4/'cases.json') and docs==read_json(V4/'documents.json')
    assert read_json(FULL/'conformance.json')['all_passed']
    original_manifest=read_json(ROOT/'output/joint-evidence/manifest.json')
    for name,h in original_manifest['sources'].items():assert sha(ROOT/name)==h,name
    rows=read_json(V4/'aligned-scores.json');old_rows=rows[:];expected={r['id']:r['recall@5'] is not None for r in rows}
    for row in read_json(FULL/'scores.json'):
        arm='JJJ' if row['arm']=='Jev' else 'hybrid'
        method=arm+('_pairwise' if row['method']=='baseline' else '_shared')
        assert (row['recall'] is not None)==expected[row['id']]
        rows.append({'id':row['id'],'doc_id':row['doc_id'],'method':method,'status':'complete',
            **{k+'@5':row[k] for k in ('recall','complete','precision','f1')},
            'prediction':row['prediction']})
    methods=list(dict.fromkeys(r['method'] for r in rows));assert len(methods)==22
    import numpy as np
    stats={'version':'shared-context-standard','partition':'previously inspected historical evaluation',
      'questions':728,'eligible_questions':640,'eligible_documents':217,
      'draws':10000,'seed':20261005,'cluster':'document','mean':'question-weighted','metrics':{}}
    for metric in ('recall@5','complete@5','precision@5','f1@5'):
        means,boot,sums,counts=cluster_summary(rows,methods,metric)
        assert int(counts.sum())==640 and len(counts)==217
        stats['metrics'][metric]={'questions':640,'documents':217,'methods':{
            name:{'mean':float(means[i]),'ci95':np.quantile(boot[:,i],[.025,.975]).tolist()} for i,name in enumerate(methods)}}
        if metric=='recall@5':
            contrasts=[]
            for a,b in [('JJJ','JJJ_shared'),('hybrid','hybrid_shared'),
                        ('JJJ_pairwise','JJJ_shared'),('hybrid_pairwise','hybrid_shared'),
                        ('gemini_jev_rerank','JJJ_shared'),('gemini_jev_rerank','hybrid_shared')]:
                contrasts.append({'negative':a,'positive':b,**contrast(sums,counts,boot,[int(n==b)-int(n==a) for n in methods])})
            for c,p in zip(contrasts,holm([c['p_two_sided'] for c in contrasts])):c['p_holm']=p
            stats['retrospective_contrasts']=contrasts
    src={p:sha(ROOT/p) for p in sources()}
    manifest={'version':'shared-context-standard','policy':STANDARD_POLICY,'questions':728,'documents':224,
       'eligible_questions':640,'eligible_documents':217,'output_unit':'whole_original_paragraph',
       'source_experiment_manifest_sha256':sha(ROOT/'output/joint-evidence/manifest.json'),
       'full_run_manifest_sha256':sha(FULL/'manifest.json'),
       'original_selection':'joint_context selected on 64 development questions before historical scoring',
       'standard_selection':'User retained the better measured shared-context policy after later experiments',
       'partition':'previously inspected historical evaluation; not untouched confirmation',
       'sources':src,'input_hashes':{n:sha(FULL/n) for n in ('cases.json','documents.json')},
       'methods':METHODS,'unchanged_baseline_methods':sorted({r['method'] for r in old_rows}),
       'new_inference_for_publication':False,'statistics':'10,000 paired document-cluster bootstrap draws; Holm correction across six retrospective contrasts',
       'primary_metric':'aligned paragraph recall@5',
       'limits':['Within-paper retrieval, not full-corpus search',
          'Shared-context gains over pairwise controls are not statistically established',
          'No cost or latency superiority claim; all systems return whole paragraphs']}
    DEST.mkdir(parents=True,exist_ok=True);save(DEST/'manifest.json',manifest)
    stats['manifest_sha256']=sha(DEST/'manifest.json')
    save(DEST/'statistics.json',stats);save(DEST/'conformance.json',read_json(FULL/'conformance.json'))
    save(DEST/'standard-replay.json',replay)
    def gz(name,records):
        with (DEST/name).open('wb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as stream:
                for row in records:stream.write((json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode())
    gz('scores.jsonl.gz',rows)
    gz('predictions.jsonl.gz',(read_json(FULL/'predictions'/(signature(c['id'])+'.json')) for c in cases))
    def upstream():
        for c in cases:
            a=read_json(V5/'retrieval'/(signature(c['id'])+'.json'))
            yield {k:v for k,v in a.items() if k!='methods'}|{'pairwise':{arm:a['methods'][m]['pairwise'] for arm,m in [('Jev','JJJ_v5'),('hybrid','hybrid_v5')]}}
    gz('upstream-predictions.jsonl.gz',upstream())
    gz('runtime-sources.jsonl.gz',({'path':p,'sha256':h,'source':(ROOT/p).read_text(encoding='utf-8')} for p,h in src.items()))
    save(DEST/'catalog.json',{'questions':728,'methods':22,'unchanged_baseline_rows':13104,
       'files':{p.name:sha(p) for p in sorted(DEST.iterdir()) if p.name!='catalog.json'}})
    rec=stats['metrics']['recall@5']['methods'];complete=stats['metrics']['complete@5']['methods'];f1=stats['metrics']['f1@5']['methods']
    lines=['# Standard shared-context retrieval comparison','',
      '**The retained standard is Jev traversal, bidirectional pairwise ranking and shared-context evidence selection.** It reaches **91.45% full-Jev Recall@5 and 92.44% hybrid Recall@5** on the same 640 historical questions with aligned evidence references.','',
      'All 728 questions from 224 QASPER papers have predictions for both systems. The primary evidence population contains 640 questions from 217 papers. Every result returns up to five whole original paragraphs. No generated-answer reader is used.','',
      '## Comparable progression','',
      '| Version | Full Jev Recall@5 | Hybrid Recall@5 |','| --- | ---: | ---: |']
    for label,a,b in [('V4','JJJ','hybrid'),('Improved traversal + pairwise ranking','JJJ_pairwise','hybrid_pairwise'),('Shared-context selection — standard','JJJ_shared','hybrid_shared')]:
        lines.append(f'| {label} | {pct(rec[a]["mean"])}% | {pct(rec[b]["mean"])}% |')
    lines+=['','These rows use the same question IDs, source paragraphs, reference alignment and five-paragraph output budget. The standard preserves the shared-context method; later experimental selectors do not replace it.','',
      '## Complete systems and cached baselines','',
      '| System | Evidence Recall@5, % [95% CI] | Complete evidence@5, % | Evidence F1@5, % |',
      '| --- | ---: | ---: | ---: |']
    for name in LABELS:lines.append(f'| {LABELS[name]} | {interval(rec[name])} | {pct(complete[name]["mean"])} | {pct(f1[name]["mean"])} |')
    lines+=['','All 18 historical baseline rows are reused without model calls and preserved in the score archive. Qwen uses 0.6B checkpoints; BGE-M3 uses dense retrieval. The [historical crossed experiment](RETRIEVAL_V4_REPORT.md) still isolates splitting / central sentences / search under its own common final selector.','',
      '## Paired comparisons','',
      '| Comparison | Recall difference, percentage points [95% CI] | Holm-adjusted p |','| --- | ---: | ---: |']
    for c in stats['retrospective_contrasts']:
        lines.append(f'| {LABELS[c["positive"]]} minus {LABELS[c["negative"]]} | {pct(c["delta"])} [{pct(c["ci95"][0])}, {pct(c["ci95"][1])}] | {c["p_holm"]:.4f} |')
    lines+=['','Intervals use 10,000 paired document-cluster bootstrap draws with question-weighted means. Two-sided document sign-randomization tests use Holm correction across these six retrospective comparisons. The small shared-context gains over the pairwise controls have intervals including zero.','',
      'Compared with the pairwise controls, shared selection improves recall on 34 full-Jev questions and worsens 19; the hybrid improves 38 and worsens 18. The aggregate gains do not imply every question improves.','',
      '## Standard method','',
      'Headings and the Jev topic/central-sentence index provide the tree. Jev evaluates children of promising nodes from root to paragraphs, using the original question and LLM-proposed evidence needs. The evaluated beam is five, the acceptance threshold is 0.2, and accepted deferred branches receive up to 25% additional routing decisions within the 4,096-decision global ceiling.','',
      'The hybrid independently retrieves direct-embedding hits and fuses them with Jev candidates. Both systems then compare the top 30 whole paragraphs in both orientations with Jev. Consistent preferences contribute a win; conflicting preferences tie.','',
      'The top 12 pairwise-ranked paragraphs become selectable targets in a shared evidence packet. Complete source paragraphs, heading paths, nearby introductions, preceding/following context and source links are visible together. Jev scores each target in source order and reverse source order; the mean score determines the final five. Exact ties retain earlier pairwise order. Context-only paragraphs clarify the source but are not returned or credited as targets.','',
      'Selection evaluates evidence useful for the original question; it does not classify answerability or penalize useful repetition. No Bayesian or weighted candidate-position prior enters ranking. The statistical prior is confined to topic-boundary detection. Central sentences guide navigation and never replace evidence paragraphs.','',
      '## Validation scope and reproduction','',
      '**These are previously inspected historical questions, not an untouched confirmation set.** Shared-context selection was originally chosen on separate development papers before its historical run. The retained standard was chosen after observing subsequent experiments. This is within-paper evidence retrieval and does not establish full-corpus or frontier superiority, or a compute advantage.','',
      'Evidence alignment is unchanged: exact or unambiguous whitespace-normalized paragraph matching, best acceptable reference per metric, and no artificial credit for empty references. Complete evidence@5 means an accepted reference set is fully retrieved. Evidence F1 measures paragraph precision and recall together.','',
      'The new standard entrypoint was replayed using exact routing caches and input-keyed saved comparison scores: **all 1,456 system predictions match the retained traversal, pairwise ranking, source packet and final selection**. This verification required no new model calls.','',
      '[Standard configuration](../configs/retrieval-standard.json) · [Implementation and usage](../docs/retrieval-standard.md) · [Manifest](results/shared-context-standard/test/manifest.json) · [Statistics](results/shared-context-standard/test/statistics.json) · [Artifact catalog](results/shared-context-standard/test/catalog.json)','',
      'Run `python scripts/replay_standard_results.py` to verify archive hashes, per-question scores, exact baseline reuse and the shared selector offline. The archive contains source packets, probabilities, upstream traversal/pairwise traces and a snapshot of the runtime sources.','']
    (ROOT/'evals/RETRIEVAL_STANDARD_REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    measured=['## Measured performance','',
      '**The standard method is Jev traversal + pairwise ranking + shared-context selection.** On the same **640 historical QASPER questions with aligned evidence**, full Jev reaches **91.45% evidence Recall@5** and the independent Jev + direct-embedding hybrid reaches **92.44%**. All 728 questions from 224 papers have predictions; every output contains whole source paragraphs.','',
      '| System | Evidence Recall@5 | 95% document-cluster interval |','| --- | ---: | ---: |']
    for n in ('JJJ_shared','hybrid_shared','JJJ_pairwise','hybrid_pairwise','gemini_dense','gemini_jev_rerank'):
        v=rec[n];measured.append(f'| {LABELS[n]} | {pct(v["mean"])}% | {pct(v["ci95"][0])}–{pct(v["ci95"][1])}% |')
    measured+=['',
      'Full Jev progresses from **87.56% → 90.80% → 91.45%**, and hybrid from **89.51% → 91.51% → 92.44%**: v4, improved traversal with pairwise ranking, then shared-context selection. Unchanged baselines reuse their saved predictions. [Full comparison, complete evidence recovery and paired tests →](evals/RETRIEVAL_STANDARD_REPORT.md)','',
      'Jev selects five paragraphs while considering a shared packet of promising targets and local source context. All five positions can change. The original question governs evidence selection; useful corroboration is retained. **There is no Bayesian ranking prior or weighted candidate-position score.** [Standard configuration](configs/retrieval-standard.json) · [Standard retrieval entrypoint](docs/retrieval-standard.md)','',
      'These historical questions have been inspected; the results are not untouched confirmation or proof of frontier superiority. The small shared-context gains over pairwise ranking are not statistically established. The report includes paired intervals and regressions.','',
      'The [historical eight-way v4 experiment](evals/RETRIEVAL_V4_REPORT.md) found higher recall for Jev tree search in all four matched splitting/central-sentence configurations. Its average matched search effect was **+4.09 percentage points [95% CI: +1.26, +6.98]**; one conditional comparison passed the prespecified 12-test Holm correction. Its final selector differs from the current standard.','',
      'Earlier [v3 validation](evals/RETRIEVAL_V3_VALIDATION.md), [blocking comparisons](evals/BOUNDED_REPORT.md) and [tree evaluations](evals/LIVE_TREE_REPORT.md) remain available.','']
    p=ROOT/'README.md';text=p.read_text(encoding='utf-8');a=text.index('## Measured performance');b=text.index('## Quick start',a)
    p.write_text(text[:a]+'\n'.join(measured)+'\n'+text[b:],encoding='utf-8',newline='\n')
    print({'standard':'shared context','full_jev':rec['JJJ_shared'],'hybrid':rec['hybrid_shared'],'methods':22})

if __name__=='__main__':publish()
