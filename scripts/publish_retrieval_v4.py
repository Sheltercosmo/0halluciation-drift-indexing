"""Publish completed v4 measurements, without private logs or failure narratives."""
import gzip
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bounded_clients import save,signature
from scripts.bounded_eval import read_json,sha
from scripts.retrieval_v4 import V4,ARMS,METHODS,validate,load_run
from scripts.audit_retrieval_v4 import audit

LABELS={'JJJ':'Jev splitting, central sentences and tree search',
    'hybrid':'Independent Jev tree + direct Gemini retrieval',
    'gemini_dense':'Gemini Embedding 2, direct paragraphs',
    'gemini_jev_rerank':'Direct Gemini + Jev reranking',
    'bm25_dense_rrf':'BM25 + direct Gemini, RRF',
    'qwen_dense':'Qwen3 Embedding 0.6B',
    'qwen_rerank':'Qwen3 Embedding + Qwen3 Reranker 0.6B',
    'qwen_jev_rerank':'Qwen3 Embedding + Jev reranking',
    'bge_dense':'BGE-M3 dense',
    'bge_rerank':'BGE-M3 + BGE Reranker v2-M3',
    'bge_jev_rerank':'BGE-M3 + Jev reranking'}


def pct(v):return f'{100*v:.2f}'
def interval(row):return f"{pct(row['mean'])} [{pct(row['ci95'][0])}, {pct(row['ci95'][1])}]"
def effect(row):return f"{100*row['delta']:+.2f} [{100*row['ci95'][0]:+.2f}, {100*row['ci95'][1]:+.2f}]"


def publish():
    out=V4/'test';validate(out);audit();manifest,cases,docs=load_run(out)
    stats=read_json(out/'statistics.json');summary=read_json(out/'aligned-summary.json')
    if stats['manifest_sha256']!=sha(out/'manifest.json') or stats['scores_sha256']!=sha(out/'aligned-scores.json'):
        raise ValueError('Statistics provenance mismatch')
    destination=ROOT/'evals/results/tree-retrieval-v4/test';destination.mkdir(parents=True,exist_ok=True)
    for name in ('manifest.json','aligned-summary.json','statistics.json','conformance.json','independent-audit.json','execution-runtime.json'):
        shutil.copyfile(out/name,destination/name)
    shutil.copyfile(V4/'development-final-ranking-grid.json',destination/'development-ranking-grid.json')
    def gz(name,rows):
        with (destination/name).open('wb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as stream:
                for row in rows:stream.write((json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode())
    gz('predictions.jsonl.gz',(read_json(out/'retrieval'/(signature(c['id'])+'.json')) for c in cases))
    gz('scores.jsonl.gz',read_json(out/'aligned-scores.json'))
    save(destination/'catalog.json',{'version':'tree-retrieval-v4','partition':'test','questions':len(cases),
        'documents':len(docs),'methods':METHODS,'predictions':len(cases)*len(METHODS),
        'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(destination.iterdir()) if p.name!='catalog.json'}})
    primary=stats['metrics']['recall@5'];r=primary['methods']
    metric=lambda key,m:stats['metrics'][key]['methods'][m]['mean']
    search_pairs=[row for row in stats['components'] if row['negative'][:2]==row['positive'][:2]]
    significant=sum(row['p_holm']<.05 for row in search_pairs)
    direct_comparison=next(row for row in stats['systems'] if row['negative']=='gemini_jev_rerank')
    interpretation=(f"Jev search has higher recall in all four matched configurations; {significant} of four comparisons passes the prespecified 12-test Holm correction. "
        f"JJJ's {100*direct_comparison['delta']:.2f}-point difference from direct Gemini + Jev ranking is not statistically significant "
        f"(adjusted p = {direct_comparison['p_holm']:.3f}). The hybrid has the highest observed recall, {pct(r['hybrid']['mean'])}%; JJJ remains the prespecified primary system.")
    lines=['# Repaired paragraph retrieval comparison','',
        f"Version 4 evaluates **{len(cases):,} questions from {len(docs):,} QASPER papers**, with all **18 methods** completed. The primary evidence-retrieval metric uses **{primary['questions']:,} questions from {primary['documents']:,} papers** with at least one fully aligned, nonempty paragraph reference. No answer-generation reader is used.",'',
        interpretation,'',
        'Every result returns whole original paragraphs. The eight crossed configurations, hybrid and Jev-reranked dense controls share the same final rule: the complete original question, complete candidate paragraphs, at most 30 candidates, and five returned paragraphs for the primary score. Final scores combine 25% Jev direct-evidence probability with 75% normalized candidate-rank prior. Central sentences guide navigation; they do not replace the source paragraphs.','',
        'For candidate position i starting at zero in a pool of n paragraphs, the final score is `0.25 * Jev probability + 0.75 * (1 - i / max(1, n - 1))`. Ties preserve candidate order. This is a retrieval pipeline comparison, not a claim that pure Jev scores alone outperform every native reranker.','',
        '![Evidence paragraph retrieval with document-cluster confidence intervals](../assets/figures/tree-retrieval-v4-test.svg)','',
        '## All eight crossed configurations','',
        'Letters mean **splitting / central-sentence selection / search**. E means embeddings and J means Jev. E search scores all depths globally. J evaluates children of retained parents layer by layer. **The common final Jev reranker is outside these three factors**, including for EEE.','',
        '| Arm | Evidence recall@5, % [95% CI] | Complete evidence@5, % | Evidence F1@5, % |',
        '| --- | ---: | ---: | ---: |']
    for m in ARMS:lines.append(f"| {m} | {interval(r[m])} | {pct(metric('complete@5',m))} | {pct(metric('f1@5',m))} |")
    lines+=['','## Complete systems and external baselines','',
        'Native Qwen and BGE predictions were reused unchanged from the completed frozen baseline runs. Their Jev variants hold the dense top-30 candidate pool and original question fixed. Direct Gemini uses the same shared search requests as the factorial. The hybrid fuses independent Jev-tree and direct-embedding candidates before final Jev reranking.','',
        '| System | Evidence recall@5, % [95% CI] | Complete evidence@5, % | Evidence F1@5, % |',
        '| --- | ---: | ---: | ---: |']
    order=['JJJ','hybrid',*[m for m in METHODS if m not in ARMS and m!='hybrid']]
    for m in order:lines.append(f"| {LABELS[m]} | {interval(r[m])} | {pct(metric('complete@5',m))} | {pct(metric('f1@5',m))} |")
    lines+=['','## Matched component comparisons','',
        'Each comparison changes one factor and holds the other two fixed. Differences are the second arm minus the first, in percentage points. Twelve component tests form one Holm family.','',
        '| Change | Recall@5 difference [95% CI], points | Holm-adjusted p |','| --- | ---: | ---: |']
    for row in stats['components']:lines.append(f"| {row['comparison']} | {effect(row)} | {row['p_holm']:.4f} |")
    for key,title in [('systems','Prespecified JJJ system comparisons'),('rerankers','Same-pool native versus Jev rerankers')]:
        lines+=['',f'## {title}','','| Change | Recall@5 difference [95% CI], points | Holm-adjusted p |','| --- | ---: | ---: |']
        for row in stats[key]:lines.append(f"| {row['comparison']} | {effect(row)} | {row['p_holm']:.4f} |")
    lines+=['','## Equal source-token budgets','',
        'Whole paragraphs are packed in ranked order under the same token cap. Oversized paragraphs are skipped and logged; no paragraph is truncated. These are secondary metrics.','',
        '| System | Recall, 512 tokens | Recall, 1,024 tokens | Recall, 2,048 tokens |','| --- | ---: | ---: | ---: |']
    for m in order:lines.append('| '+LABELS[m]+' | '+' | '.join(pct(metric('recall@tokens'+str(b),m))+'%' for b in (512,1024,2048))+' |')
    lines+=['','## Design and scoring','',
        'Repairs were checked on a fixed 16-paper development pilot before the v4 test manifest was frozen. Three final-ranking instructions/aggregations were examined on the same candidates. A cached sweep of direct-evidence weights 0, 0.25, 0.5, 0.75 and 1 selected 0.25 by JJJ development recall@5. Those development scores are tuning results, not held-out evidence. Earlier v3 validation informed the repairs. JJJ was fixed as the primary system before opening test outcomes; the best test row is not used to select a new primary system. Test questions and documents had previously been processed under v3, but test evidence labels and scores were unopened until the complete v4 prediction audit passed.','',
        'The shared, previously frozen gpt-6.1-sol planner sees questions, titles and headings. Its evidence needs are augmented with the literal original question for every factorial arm and direct Gemini search. Indexing uses no generative LLM. Native heading nesting is restored for every arm. Jev uses beam 5, acceptance 0.20, additional source inspection below 0.85, and at most 4,096 decisions per search. Scored terminal paragraphs remain eligible for final ranking. Embedding search uses 30 global node hits per need. Central selection uses eight outside-in candidates without early stopping. Splitting settings and model versions are recorded in the manifest.','',
        'Evidence must align to an original paragraph exactly or by unambiguous whitespace normalization. An acceptable reference must map all non-figure text evidence and contain at least one paragraph. Each metric uses its best acceptable reference; references are not unioned. Questions with no eligible reference receive predictions but are excluded from the paragraph-evidence denominator.','',
        f"There are {summary['population']['all']-summary['population']['aligned_paragraph_evidence']} questions outside that denominator, including {summary['population'].get('unmapped_only',0)} with text evidence but no fully aligned reference. Unanswerable, empty and figure-only evidence are not awarded artificial retrieval credit.",'',
        'Confidence intervals use 10,000 paired document-cluster bootstrap draws and question-weighted means. Two-sided document sign randomization supplies p-values. Holm adjustment is separate for 12 component, 10 system and two native-reranker comparisons. Intervals are marginal, not simultaneous.','',
        'This is retrieval within a supplied paper, not retrieval over a full corpus. Qwen uses compact 0.6B checkpoints and BGE-M3 uses dense mode. This comparison does not establish frontier or universal superiority. Model training exposure to public benchmark data cannot be excluded. Shared caches make recorded operation times unsuitable for standalone latency comparisons. V3 validation and v4 test use different pipelines and must not be compared as an isolated repair-effect estimate.','',
        '## Reproduction','',
        '[Frozen manifest](results/tree-retrieval-v4/test/manifest.json) · [Artifact catalog](results/tree-retrieval-v4/test/catalog.json) · [Statistics](results/tree-retrieval-v4/test/statistics.json) · [Conformance audit](results/tree-retrieval-v4/test/conformance.json)','',
        'The catalog contains compressed per-question paragraph rankings, traversal traces, candidate pools and scores, with SHA-256 hashes. Provider credentials, spending ledgers and private failure investigations are excluded. The source dataset is [QASPER](https://arxiv.org/abs/2105.03011).','',
        'With the prepared, hashed QASPER allocation and model caches described by the manifest, run `python scripts/run_retrieval_v4.py test`, `python scripts/retrieval_v4.py validate test`, then `python scripts/retrieval_v4_evaluate.py test`. The runtime wrapper only retries temporary Windows file locks; its hash is recorded separately from the frozen inference sources. Native baseline files are reused; their models are not executed again.','']
    (ROOT/'evals/RETRIEVAL_V4_REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    measured=['## Measured performance','',
        f"The [completed v4 comparison](evals/RETRIEVAL_V4_REPORT.md) tests **18 methods on {len(cases):,} questions from {len(docs):,} held-out QASPER papers**. Evidence paragraph recall@5 is scored on **{primary['questions']:,} questions with fully aligned evidence**. Every method returns whole original paragraphs; no generated-answer reader is used.",'',
        '| System | Evidence recall@5 | 95% document-cluster interval |','| --- | ---: | ---: |']
    for m in ('JJJ','hybrid','gemini_dense','gemini_jev_rerank'):
        row=r[m];measured.append(f"| {LABELS[m]} | {pct(row['mean'])}% | {pct(row['ci95'][0])}–{pct(row['ci95'][1])}% |")
    measured+=['','The crossed experiment varies **topic splitting / central-sentence selection / search** between embeddings (E) and Jev (J). Central sentences are navigation cues, with full paragraphs available throughout. Embeddings search globally across depths; Jev explores promising branches from root to paragraphs. All eight configurations use the same final ranking rule: 25% Jev direct-evidence score and 75% normalized candidate-rank prior, selected on development samples.','',
        '| Fixed splitting / central sentences | Global embedding search + Jev final ranking | Jev tree search + Jev final ranking | Recall difference |',
        '| --- | ---: | ---: | ---: |']
    for prefix,label in [('EE','Embedding / Embedding'),('EJ','Embedding / Jev'),('JE','Jev / Embedding'),('JJ','Jev / Jev')]:
        a,b=prefix+'E',prefix+'J';measured.append(f"| {label} | {a}: {pct(r[a]['mean'])}% | {b}: {pct(r[b]['mean'])}% | {100*(r[b]['mean']-r[a]['mean']):+.2f} points |")
    average=stats['factor_effects']['search']
    measured+=['',f"The average matched search effect is **{effect(average)} percentage points** (95% document-cluster interval). The full report provides each conditional comparison, adjusted significance tests, complete evidence recovery, equal-token-budget metrics, and native Qwen/BGE baselines with Jev replacements on identical candidate pools.",'',
        interpretation,'',
        'This is evidence retrieval within the same supplied paper. It does not establish full-corpus or frontier superiority. Development and earlier validation informed engineering repairs; the repaired implementation was frozen before test outcomes were opened. Unchanged native baselines reuse their saved predictions.','',
        'The [historical v3 validation](evals/RETRIEVAL_V3_VALIDATION.md) found a 7.19–8.64-point benefit from Jev search in its original pipeline. V4 adds a common final reranker and structural repairs, so the two versions are separate comparisons. Earlier [blocking](evals/BOUNDED_REPORT.md) and [tree](evals/LIVE_TREE_REPORT.md) studies remain available.','']
    p=ROOT/'README.md';text=p.read_text(encoding='utf-8');a=text.index('## Measured performance');b=text.index('## Quick start',a)
    p.write_text(text[:a]+'\n'.join(measured)+'\n'+text[b:],encoding='utf-8',newline='\n')
    print({'report':'evals/RETRIEVAL_V4_REPORT.md','questions':len(cases),'predictions':len(cases)*len(METHODS)})


if __name__=='__main__':publish()
