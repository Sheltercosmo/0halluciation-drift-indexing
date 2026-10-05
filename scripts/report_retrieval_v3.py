"""Render complete registered results; inference and metric code remain frozen."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.bounded_eval import read_json

NAMES={'gemini_dense':'Gemini dense (shared requests)',
       'gemini_jev_rerank':'Gemini dense + Jev reranker (shared requests)',
       'bm25_dense_rrf':'BM25 + Gemini dense RRF','hybrid':'JJJ + independent Gemini dense RRF',
       'qwen_dense':'Qwen3-Embedding-0.6B','qwen_rerank':'Qwen dense + Qwen3-Reranker-0.6B',
       'qwen_jev_rerank':'Qwen dense + Jev reranker','bge_dense':'BGE-M3 dense',
       'bge_rerank':'BGE dense + BGE-reranker-v2-m3','bge_jev_rerank':'BGE dense + Jev reranker'}


def pct(x):return f'{100*x:.2f}'
def interval(row):return f"{pct(row['mean'])} [{pct(row['ci95'][0])}, {pct(row['ci95'][1])}]"
def effect(row):return f"{100*row['delta']:+.2f} [{100*row['ci95'][0]:+.2f}, {100*row['ci95'][1]:+.2f}]"


def report():
    base=ROOT/'evals/results/tree-retrieval-v3'
    stats={p:read_json(base/p/'statistics.json') for p in ('validation','test')}
    summaries={p:read_json(base/p/'aligned-summary.json') for p in stats}
    catalogs={p:read_json(base/p/'catalog.json') for p in stats}
    test=stats['test'];winner=test['selected_system'];primary=test['metrics']['recall@5']
    all_win=all(r['delta']>0 and r['p_holm']<.05 for r in test['system_contrasts'])
    verdict=('The selected system has positive recall differences with Holm-adjusted p < 0.05 against all ten registered controls in this supplied-paper test.'
             if all_win else 'The selected system does not establish superiority over all registered controls. Individual gains, ties and regressions are shown below.')
    lines=['# Whole-paragraph evidence retrieval on QASPER','',
        f"**Completed:** {catalogs['validation']['questions']:,} validation questions and {catalogs['test']['questions']:,} test questions, with 18 methods per question. All methods return whole original source paragraphs.",'',
        f"Validation selected **{NAMES.get(winner,winner)}** before test access. Its test evidence recall@5 is **{pct(primary['methods'][winner]['mean'])}%**, with a 95% document-cluster bootstrap interval of **{pct(primary['methods'][winner]['ci95'][0])}–{pct(primary['methods'][winner]['ci95'][1])}%**. {verdict}",'',
        'This measures retrieval of annotated evidence inside the same supplied paper. It is not an answer-generation evaluation, full-corpus search benchmark or frontier-superiority claim.','',
        '## Complete systems and direct-retrieval controls','',
        'The direct Gemini + Jev control uses the same LLM search requests and decision model as JJJ. Qwen and BGE supply external embedding pipelines; swapping their rerankers for Jev keeps each exact top-30 candidate pool, original question and complete paragraph text fixed.','',
        '| Method | Test recall@5, % [95% CI] | Complete evidence@5, % | F1@5, % |',
        '| --- | ---: | ---: | ---: |']
    order=['JJJ','hybrid','gemini_dense','gemini_jev_rerank','qwen_dense','qwen_rerank','qwen_jev_rerank','bge_dense','bge_rerank','bge_jev_rerank','bm25_dense_rrf','EEE']
    for m in order:
        lines.append(f"| {NAMES.get(m,m)} | {interval(primary['methods'][m])} | {pct(test['metrics']['complete@5']['methods'][m]['mean'])} | {pct(test['metrics']['f1@5']['methods'][m]['mean'])} |")
    lines+=['','## All eight crossed configurations','',
        'Letters mean **split / central-sentence selection / search**. E uses embeddings; J uses Jev. E search ranks all depths globally. J explores the children of promising parents layer by layer. Central sentences are cues; full source paragraphs remain accessible and are the final output.','',
        '| Arm | Validation recall@5, % | Test recall@5, % [95% CI] | Test complete@5, % |','| --- | ---: | ---: | ---: |']
    for arm in ['EEE','EEJ','EJE','EJJ','JEE','JEJ','JJE','JJJ']:
        lines.append(f"| {arm} | {pct(stats['validation']['metrics']['recall@5']['methods'][arm]['mean'])} | {interval(primary['methods'][arm])} | {pct(test['metrics']['complete@5']['methods'][arm]['mean'])} |")
    lines+=['','## Recall under equal source-token budgets','',
        'These secondary scores use complete paragraphs under each source-token limit. Each method applies the same packing rule to its own ranking. They help distinguish ranking gains from gains that require longer passages.','',
        '| Method | 512 tokens, % | 1,024 tokens, % | 2,048 tokens, % |','| --- | ---: | ---: | ---: |']
    for m in order:
        values=[pct(test['metrics']['recall@tokens'+str(b)]['methods'][m]['mean']) for b in (512,1024,2048)]
        lines.append('| '+NAMES.get(m,m)+' | '+' | '.join(values)+' |')
    for key,title in [('components','Matched component effects'),('reranker_contrasts','Same-pool reranker effects'),('system_contrasts','Selected-system comparisons')]:
        lines+=['',f'## {title}','',
            '| Comparison | Recall difference, percentage points [95% CI] | Holm-adjusted p |','| --- | ---: | ---: |']
        for r in test[key]:lines.append(f"| {r['comparison']} | {effect(r)} | {r['p_holm']:.4f} |")
    lines+=['','## Average component effects','',
        'These descriptive averages give equal weight to the four matched settings of the other two components. The conditional comparisons above show whether a component behaves differently across those settings.','',
        '| Component changed from embeddings to Jev | Average recall difference, percentage points [95% CI] |',
        '| --- | ---: |']
    labels={'split':'Topic splitting','central':'Central-sentence selection','representative':'Central-sentence selection',
            'central_sentence':'Central-sentence selection','central_sentences':'Central-sentence selection','search':'Search procedure'}
    for factor,row in test['descriptive_factor_effects'].items():
        lines.append(f"| {labels.get(factor,factor)} | {effect(row)} |")
    failure_path=base/'test/failure-analysis.json'
    if failure_path.is_file():
        failures=read_json(failure_path)
        lines+=['','## Where reference evidence was lost','',
            'This supplemental descriptive analysis was added after validation and is separate from the registered hypothesis tests. It locates lost evidence along the retrieval procedure; it does not establish the cause of a relevance decision.','',
            '| System | Eligible questions | Questions with incomplete top-five evidence | Missed reference paragraph instances |',
            '| --- | ---: | ---: | ---: |']
        for method,row in failures['methods'].items():
            lines.append(f"| {NAMES.get(method,method)} | {row['eligible_questions']} | {row['incomplete_top5_questions']} | {row['missed_reference_paragraph_instances']} |")
        lines+=['','| System | Loss location | Paragraph instances |','| --- | --- | ---: |']
        location_labels={'reached_but_ranked_below_5':'Retrieved but ranked below the top five',
            'decision_budget':'Decision budget stopped traversal','outside_dense_top30':'Absent from the dense top 30',
            'candidate_reranked_below_5':'Candidate reranked below the top five'}
        for kind,title in [('heading','Heading'),('topic','Topic block'),('paragraph','Paragraph')]:
            location_labels[kind+':below_threshold']=title+' score below cutoff'
            location_labels[kind+':outside_beam']=title+' excluded by beam limit'
        for method,row in failures['methods'].items():
            for location,count in row['loss_locations'].items():
                lines.append(f"| {NAMES.get(method,method)} | {location_labels.get(location,location)} | {count} |")
        lines+=['',
            'For each method, use the fully aligned reference with highest recall@5, with ties resolved by annotation order. Count missed paragraph instances in that reference. For tree misses, attribute the loss to the deepest evidence-path node reached by any shared request. Methods can select different acceptable references for this diagnostic.','',
            'A five-paragraph output cannot contain a reference with more than five paragraphs. The minimum missed instances imposed by that limit for the chosen references are '+
            '; '.join(f"{NAMES.get(m,m)}: {r['minimum_misses_for_chosen_reference_at_k5']}" for m,r in failures['methods'].items())+'. These unavoidable capacity limits are included in the counts above.','']
    lines+=['','## Population and scoring','',
        '| Partition | All questions | Papers | Eligible paragraph-evidence questions | Papers with eligible evidence |','| --- | ---: | ---: | ---: | ---: |']
    for p in ('validation','test'):
        metric=stats[p]['metrics']['recall@5'];catalog=catalogs[p]
        lines.append(f"| {p.title()} | {catalog['questions']} | {catalog['documents']} | {metric['questions']} | {metric['documents']} |")
    lines+=['','| Evidence eligibility | Validation questions | Test questions |','| --- | ---: | ---: |']
    for label,key in [('Has a fully aligned paragraph reference','aligned_paragraph_evidence'),
                      ('Has text evidence but no fully aligned reference','unmapped_only')]:
        lines.append('| '+label+' | '+' | '.join(str(summaries[p]['population'].get(key,0)) for p in ('validation','test'))+' |')
    lines.append('| No usable text evidence annotation | '+' | '.join(str(summaries[p]['population']['all']-summaries[p]['population']['raw_text_evidence']) for p in ('validation','test'))+' |')
    lines+=['','No usable text evidence includes unanswerable, empty and figure-only annotations; it is not a claim that every such question is unanswerable. Annotation-alignment issue counts in the raw summaries count evidence items across annotations, not distinct questions.']
    lines+=['',
        'The primary score is paragraph-identity recall@5, adapted from QASPER evidence annotations. Evidence must match an original paragraph exactly or by unambiguous whitespace normalization. An annotation is eligible only when every non-figure text item maps and it contains at least one paragraph. Questions need one eligible alternative annotation. Each metric takes its best acceptable reference; references are never unioned. Empty/unanswerable, heading-only and unmapped evidence are reported separately, rather than receiving artificial retrieval credit.','',
        'All questions receive predictions. The primary evidence denominator excludes questions without a fully mapped nonempty paragraph reference. The published summaries include those counts, annotation issues and raw official string F1 as a separate diagnostic.','',
        'Secondary results include precision/F1, complete-set recovery, recall@1/3/10, and complete-paragraph packing under 512/1,024/2,048 source-token limits. Oversized paragraphs are skipped and logged, never truncated. Source-token budgets exclude headings and wrapper syntax.','',
        'Intervals use 10,000 paired document-cluster bootstrap draws with question-weighted means. Paired document-level sign randomization supplies two-sided p-values. Holm adjustment is applied separately to 12 component contrasts, ten selected-system comparisons and two native reranker comparisons. Intervals are marginal, not simultaneous. The prespecified practical target was a two-percentage-point recall gain.','',
        '## Frozen settings and limitations','',
        'A fixed 16-paper development pilot calibrated six shared-search settings. The final configuration uses Jev beam 5, acceptance 0.20, extra source inspection below 0.85, and 30 global embedding hits per need. Central selectors have eight matched outside-in candidates and no early stop. Native titles/headings are parsed without a model. The split prior/drop rule is one method, not a separate ablation.','',
        'The shared query planner is gpt-6.1-sol with low reasoning effort. Gemini Embedding 2 uses 768 dimensions; Jev uses 1.13.0. Query planning is separate from indexing, which uses no generative LLM. Qwen and BGE retain their native original-question interfaces. Exact checkpoint revisions and source hashes are registered.','',
        'Qwen uses compact 0.6B checkpoints; BGE-M3 uses its dense mode. ColBERTv2, RTriever and RAPTOR were not run here. Shared experiment caching means stage times are not standalone deployed-system latency measurements. No speed-superiority conclusion is supported by them. Public benchmark exposure during model training cannot be excluded.','',
        'The original registration preceded validation access. At the user’s request, a documented amendment added the three same-pool Jev reranker controls during indexing/planning, before any held-out scores were inspected. No retrieval threshold or selectable system was changed by that amendment.','',
        '## Reproduction and artifacts','',
        '[Protocol](TREE_SYSTEM_PROTOCOL.md) · [Original registration](registrations/tree-retrieval-v3.json) · [Pre-outcome amendment](registrations/tree-retrieval-v3-jev-rerank-amendment.json) · [Reproduction instructions](RETRIEVAL_V3_REPRODUCTION.md)','',
        '[Validation artifacts](results/tree-retrieval-v3/validation/catalog.json) · [Test artifacts](results/tree-retrieval-v3/test/catalog.json) · [Test statistics](results/tree-retrieval-v3/test/statistics.json)','',
        'Each partition includes compressed per-question rankings, whole-paragraph packs, traversal traces, reranker pools and scores, with file hashes. The task uses [QASPER](https://arxiv.org/abs/2105.03011); these results are not comparable to the historical answer-based studies.','']
    if (ROOT/'assets/figures/tree-retrieval-v3-test.svg').is_file():
        position=lines.index('## Complete systems and direct-retrieval controls')
        lines[position:position]=['![Test evidence recall with document-cluster confidence intervals](../assets/figures/tree-retrieval-v3-test.svg)','']
    path=ROOT/'evals/RETRIEVAL_V3_REPORT.md';path.write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(path)


if __name__=='__main__':report()
