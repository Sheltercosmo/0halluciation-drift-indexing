"""Publish the complete validation partition without implying a completed test."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bounded_eval import read_json
from scripts.report_retrieval_v3 import NAMES,pct,interval,effect


def report():
    base=ROOT/'evals/results/tree-retrieval-v3/validation'
    stats=read_json(base/'statistics.json');catalog=read_json(base/'catalog.json')
    assert stats['partition']==catalog['partition']=='validation'
    assert catalog['questions']==1005 and len(catalog['methods'])==18
    primary=stats['metrics']['recall@5'];methods=primary['methods']
    lines=['# Whole-paragraph retrieval: completed validation comparison','',
        '**Validation results only.** All 18 methods completed 1,005 questions across 281 QASPER papers. The separate 728-question test is not complete; no partial-test accuracy is reported here.','',
        f"The primary metric uses **{primary['questions']} questions across {primary['documents']} papers** with a fully aligned, nonempty paragraph-evidence reference. Every method returns whole original paragraphs.",'',
        '## What the comparison shows','',
        f"JJJ achieves **{pct(methods['JJJ']['mean'])}% evidence recall@5**, versus **{pct(methods['gemini_dense']['mean'])}%** for direct Gemini dense retrieval. However, **direct Gemini retrieval + Jev reranking reaches {pct(methods['gemini_jev_rerank']['mean'])}%** using the same shared search requests. Validation therefore does not establish an advantage for tree traversal over that direct-retrieval control.",'',
        'The four matched search replacements favor Jev by 7.19–8.64 percentage points, with Holm-adjusted p = 0.0012 in validation. Splitting and central-sentence replacements show no significant validation improvement in their matched comparisons. Same-pool Jev reranking improves over the native compact Qwen and BGE rerankers; all effects and intervals appear below.','',
        'The registered system-selection rule compared JJJ with the independent JJJ + dense hybrid and selected **JJJ** before opening the test. JEJ has the highest numerical validation score among the crossed arms, but it was not one of the selectable final-system candidates. All eight arms remain fixed for test evaluation.','',
        'These findings concern evidence retrieval inside a supplied paper. They do not establish full-corpus performance, generated-answer quality or superiority over frontier RAG systems.','',
        '![Validation evidence recall and document-cluster intervals](../assets/figures/tree-retrieval-v3-validation.svg)','',
        '## Eight crossed configurations','',
        'Letters mean **split / central-sentence selection / search**. E uses embeddings; J uses Jev. Embedding search ranks nodes globally across depths. Jev explores the children of retained promising parents from the root downward. Central sentences are cues, while the source remains accessible; paragraph-level decisions use full paragraphs.','',
        '| Arm | Recall@5, % [95% CI] | Complete evidence@5, % | F1@5, % |',
        '| --- | ---: | ---: | ---: |']
    def row(method):
        return f"| {NAMES.get(method,method)} | {interval(methods[method])} | {pct(stats['metrics']['complete@5']['methods'][method]['mean'])} | {pct(stats['metrics']['f1@5']['methods'][method]['mean'])} |"
    for arm in ['EEE','EEJ','EJE','EJJ','JEE','JEJ','JJE','JJJ']:lines.append(row(arm))
    lines+=['','## Complete systems and controls','',
        'Gemini and the crossed methods share the same planner requests. Qwen/BGE retain their native original-question interface. Each Jev reranker swap uses the exact dense top-30 candidate pool and full paragraph text used by its corresponding native pipeline; the pool is checked before scoring.','',
        '| Method | Recall@5, % [95% CI] | Complete evidence@5, % | F1@5, % |',
        '| --- | ---: | ---: | ---: |']
    order=['JJJ','hybrid','gemini_dense','gemini_jev_rerank','qwen_dense','qwen_rerank',
           'qwen_jev_rerank','bge_dense','bge_rerank','bge_jev_rerank','bm25_dense_rrf']
    for method in order:lines.append(row(method))
    lines+=['','## Recall under equal source-token budgets','',
        'All methods apply the same whole-paragraph packing rule to their own rankings. Oversized paragraphs are skipped and recorded, never truncated. These budgets count source text, excluding headings and wrappers.','',
        '| Method | 512 tokens, % | 1,024 tokens, % | 2,048 tokens, % |',
        '| --- | ---: | ---: | ---: |']
    for method in catalog['methods']:
        values=[pct(stats['metrics']['recall@tokens'+str(b)]['methods'][method]['mean']) for b in (512,1024,2048)]
        lines.append('| '+NAMES.get(method,method)+' | '+' | '.join(values)+' |')
    token_methods=stats['metrics']['recall@tokens2048']['methods']
    lines+=['',f"At 2,048 source tokens, the independent hybrid reaches **{pct(token_methods['hybrid']['mean'])}% recall**, compared with **{pct(token_methods['JJJ']['mean'])}%** for JJJ and **{pct(token_methods['gemini_jev_rerank']['mean'])}%** for direct Gemini + Jev. This descriptive secondary result shows a benefit from retaining direct retrieval for broader evidence coverage under a larger reading budget. It does not change the registered recall@5 selection rule or the frozen test candidate."]
    for key,title in [('components','Matched component effects'),('reranker_contrasts','Same-pool reranker effects')]:
        lines+=['','## '+title,'',
            '| Comparison | Recall difference, percentage points [95% CI] | Holm-adjusted p |',
            '| --- | ---: | ---: |']
        for result in stats[key]:
            lines.append(f"| {result['comparison']} | {effect(result)} | {result['p_holm']:.4f} |")
    lines+=['','## Population, controls and limitations','',
        'There are 864 primary-eligible questions, 28 with text evidence but no fully aligned reference, and 113 without usable text evidence. The latter includes unanswerable, empty and figure-only annotations. All 1,005 questions have predictions for all 18 methods.','',
        'A reference must map every non-figure evidence item to an original paragraph exactly or through unambiguous whitespace normalization, and contain at least one paragraph. Each metric uses its best acceptable reference; alternative annotations are never unioned. Raw official string F1 is a separate diagnostic in the summaries.','',
        'Development, validation and test are document-disjoint. The 16-paper development pilot fixed settings before validation. The native-heading parser, outside-in central candidates, Bayesian splitting rule, beam 5, acceptance 0.20, refinement below 0.85 and global 30-hit search were frozen. No test-result tuning has occurred.','',
        'Intervals use 10,000 paired document-cluster bootstrap draws with question-weighted means. Document-level paired sign randomization supplies two-sided p-values, with Holm adjustment separately across 12 component contrasts and two reranker contrasts. Intervals are marginal rather than simultaneous. These validation statistics do not replace the registered final test comparisons.','',
        'Models are Gemini Embedding 2 (768 dimensions), Jev 1.13.0, Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B, BGE-M3 dense and BGE-reranker-v2-m3. The shared query planner is gpt-6.1-sol; it sees questions, titles and headings. Indexing uses no generative LLM. Exact revisions and source hashes are registered.','',
        'Qwen uses compact checkpoints and BGE-M3 uses dense mode. ColBERTv2, RTriever and RAPTOR were not evaluated. Shared caches prevent interpreting stage timings as standalone deployment latency. Public-benchmark exposure during model training cannot be excluded.','',
        '## Reproduction and artifacts','',
        '[Engineering failure-case inspection](RETRIEVAL_V3_FAILURE_ANALYSIS.md) separates beam, cutoff, cue and native-hierarchy issues. Its isolated hierarchy repair has passed source-preservation checks but has no measured retrieval result.','',
        '[Protocol](TREE_SYSTEM_PROTOCOL.md) · [Registration](registrations/tree-retrieval-v3.json) · [Pre-outcome amendment](registrations/tree-retrieval-v3-jev-rerank-amendment.json) · [Reproduction](RETRIEVAL_V3_REPRODUCTION.md)','',
        '[Validation catalog and file hashes](results/tree-retrieval-v3/validation/catalog.json) · [Statistics](results/tree-retrieval-v3/validation/statistics.json) · [Metric summary](results/tree-retrieval-v3/validation/aligned-summary.json)','',
        'The export contains all 18,090 method predictions, compressed per-question scores, traversal traces, paragraph packs and reranker pools. The benchmark is [QASPER](https://arxiv.org/abs/2105.03011); these retrieval results are not comparable to the historical answer-based studies.','']
    path=ROOT/'evals/RETRIEVAL_V3_VALIDATION.md'
    path.write_text('\n'.join(lines),encoding='utf-8',newline='\n');print(path)


if __name__=='__main__':report()
