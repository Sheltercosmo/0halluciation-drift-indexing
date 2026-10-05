"""Render the complete live comparison, including every registered loss."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bounded_eval import read_json

RESULT = ROOT / 'evals/results/live-tree-complete-v2'
LABELS = {
    'dense_recursive': 'Direct dense, recursive chunks', 'rrf_recursive': 'BM25 + dense, recursive chunks',
    'rerank_recursive': 'Generative reranking, recursive chunks', 'dense_semantic': 'Direct dense, semantic chunks',
    'rrf_semantic': 'BM25 + dense, semantic chunks', 'rerank_semantic': 'Generative reranking, semantic chunks',
    'hybrid_full': 'Jev tree + direct dense, full budgets', 'hybrid_matched': 'Jev tree + direct dense, divided budget',
    'EEE_bottom_up': 'Embedding tree, retry + bottom-up reading', 'JJJ_bottom_up': 'Jev tree, retry + bottom-up reading',
    'EEE_ancestor': 'Embedding tree, successive ancestor expansion', 'JJJ_ancestor': 'Jev tree, successive ancestor expansion'}


def effect(value):
    return f"{100*value['difference']:+.2f} [{100*value['ci95'][0]:+.2f}, {100*value['ci95'][1]:+.2f}]"


def render():
    summary, manifest = read_json(RESULT / 'summary.json'), read_json(RESULT / 'manifest.json')
    intervals, analysis = read_json(RESULT / 'paired-intervals.json'), read_json(RESULT / 'retrieval-component-audit.json')
    methods = manifest['methods']
    eligible = sum(r['dataset'] == 'qasper' and r['method'] == 'JJJ' and r['evidence_recall'] is not None
                   for r in read_json(RESULT / 'scores.json'))
    lines = ['# Live central-sentence, tree-search and complete-system comparison', '',
        f"**Complete:** {summary['predictions']:,} scored method/question records across 384 exposed development questions and 307 documents. "
        'QASPER and QuALITY-HARD each contribute 192 questions. Validation and test remain unopened.', '',
        'All arms use the same Gemini 2.5 Flash planner and reader (temperature 0, thinking disabled), '
        'Gemini Embedding 2 at 768 dimensions where applicable, and Jev 1.13.0 for decision-based stages. '
        'One isolated answer is reused for identical reader inputs. Final source contexts are capped at 2,048 cl100k tokens. '
        'The original sixteen-arm study contributes 6,144 records; two frozen follow-ups add 768 records each, '
        'comparing both embedding and Jev systems at each iteration.', '',
        'This is development evidence, not a frontier or held-out superiority claim. The generative reranker also uses '
        'Gemini 2.5 Flash. RAPTOR, PageIndex and stronger independent rerankers are not measured in this run.', '',
        '## Complete systems and all factorial arms', '',
        '`E` means embedding and `J` means Jev. The three positions are **split / representative / router**. '
        'For example, `EJE` uses embedding splits, Jev central sentences and embedding tree search. '
        'Every tree arm uses the same native headings and source paragraph offsets.', '',
        '| Method | QASPER answer F1 | QASPER evidence recall | QuALITY-HARD accuracy |',
        '| --- | ---: | ---: | ---: |']
    for method in methods:
        q, h = summary['methods']['qasper'][method], summary['methods']['quality'][method]
        lines.append(f"| {LABELS.get(method, method)} | {q['answer_score']*100:.2f} | {q['evidence_recall']*100:.2f}% | {h['answer_score']*100:.2f}% |")
    lines += ['', f'Evidence recall is averaged over {eligible} questions with annotated evidence and uses source paragraphs fully present in the final context. '
              'Representative sentences and unselected previews are not counted as retrieved evidence. '
              'No subjective central-sentence gold labels were invented.', '',
              f"There are {sum(r['status'] != 'ok' for r in read_json(RESULT / 'scores.json'))} explicit reader failure records across {len(summary['questions_with_any_reader_failure'])} questions. "
              'They receive zero answer score in the table above. Provider-blocked inputs are terminal and are not retried or sent to another model. '
              'The common-unblocked fields in the [summary](results/live-tree-complete-v2/summary.json) '
              'restrict all methods to the same questions without any reader block; matching paired intervals are included in the interval file. '
              'The forty records include skipped calls for the two questions already blocked in the original study.', '',
              '## Controlled component effects', '',
              'Each row changes exactly one component from embedding to Jev. Values are percentage-point differences '
              'with descriptive 95% paired document-bootstrap intervals (10,000 replicates). '
              'They are not multiplicity-adjusted confirmatory tests.', '',
              '| Changed component | Jev arm − embedding arm | QASPER F1 difference [95% interval] | QuALITY accuracy difference [95% interval] |',
              '| --- | --- | ---: | ---: |']
    for factor in ['split', 'representative', 'router']:
        for row in [r for r in intervals if r['factor'] == factor and r['dataset'] == 'qasper']:
            other = next(r for r in intervals if r['factor'] == factor and r['candidate'] == row['candidate'] and r['baseline'] == row['baseline'] and r['dataset'] == 'quality')
            lines.append(f"| {factor} | {row['candidate']} − {row['baseline']} | {effect(row['metrics']['answer_score'])} | {effect(other['metrics']['answer_score'])} |")
    lines += ['', '![Matched component effects](results/live-tree-complete-v2/component-effects.png)', '',
              '## Failure-driven iteration: retry and bottom-up reading', '',
              'The primary Jev tree supplied only 297 context tokens on average for QASPER, and exhausted its search '
              'budget with an empty context on 109 of 192 QuALITY questions. Searching several proposed needs consumed '
              'the budget before a sentence leaf was reached. Direct methods used about 2,000 context tokens.', '',
              'The follow-up was frozen from those retrieval diagnostics. For both embedding and Jev trees, exhausted '
              'searches retry using the first shared need. Reached paragraphs are then read first, followed by topic '
              'parents and nearby paragraphs inside the same topics, all under the original final context limit. '
              'This allows at most two search attempts and therefore uses extra routing compute. Both variants are '
              'reported regardless of outcome.', '',
              '| Follow-up contrast | QASPER F1 difference [95% interval] | QuALITY accuracy difference [95% interval] |',
              '| --- | ---: | ---: |']
    for row in [r for r in intervals if r['factor'] == 'development-repair' and r['dataset'] == 'qasper']:
        other = next(r for r in intervals if r['factor'] == 'development-repair' and r['candidate'] == row['candidate'] and r['baseline'] == row['baseline'] and r['dataset'] == 'quality')
        lines.append(f"| {row['candidate']} − {row['baseline']} | {effect(row['metrics']['answer_score'])} | {effect(other['metrics']['answer_score'])} |")
    lines += ['', 'The first repair still averaged only 678 Jev context tokens for QASPER. A second policy was '
              'frozen from those context-length diagnostics before inspecting first-follow-up QA aggregates. It reuses '
              'the exact same routes, preserves previously selected evidence, then interleaves nearby paragraphs '
              'through topic, heading and document ancestors. It makes no additional routing or embedding calls. '
              'The reader sees one final source context; this is deterministic ancestor expansion, not interactive '
              'LLM traversal.', '',
              '| Second-iteration contrast | QASPER F1 difference [95% interval] | QuALITY accuracy difference [95% interval] |',
              '| --- | ---: | ---: |']
    for row in [r for r in intervals if r['factor'] == 'ancestor-repair' and r['dataset'] == 'qasper']:
        other = next(r for r in intervals if r['factor'] == 'ancestor-repair' and r['candidate'] == row['candidate'] and r['baseline'] == row['baseline'] and r['dataset'] == 'quality')
        lines.append(f"| {row['candidate']} − {row['baseline']} | {effect(row['metrics']['answer_score'])} | {effect(other['metrics']['answer_score'])} |")
    lines += ['', '| Tree | QASPER mean context tokens | QuALITY empty contexts |', '| --- | ---: | ---: |']
    for m in ['EEE', 'JJJ', 'EEE_bottom_up', 'JJJ_bottom_up', 'EEE_ancestor', 'JJJ_ancestor']:
        lines.append(f"| {m} | {summary['methods']['qasper'][m]['context_tokens']:.1f} | {summary['methods']['quality'][m]['empty_contexts']} / 192 |")
    lines += ['',
              '### Inspected failures and recoveries', '',
              'These are illustrative exposed-development cases, not a representative subsample or a separate evaluation:', '',
              '- `qasper/f6346828c2f44529dc307abf04dd246bfeb4a9b2`: asked whether compression methods were compared. Topic expansion increased context from 470 to 901 tokens and changed an incorrect “Unanswerable” to the correct “Yes.” Both contexts already contained all annotated evidence, showing that evidence recall alone does not determine reader success.',
              '- `qasper/d5bce5da746a075421c80abe10c97ad11a96c6cd`: asked which baseline was used. The 182-token topic context yielded the correct “memorization baseline”; expansion to 2,045 tokens yielded “Unanswerable,” despite retaining full annotated evidence. This is a concrete regression from adding surrounding text.',
              '- `qasper/d9354c0bb32ec037ff2aacfed58d57887a713163`: asked which input language was used. Topic expansion still missed the annotated evidence and the reader abstained. The direct reranker recovered it and answered “English.” Expanding the chosen branch cannot reliably repair an incorrect branch choice.', '',
              '## Retrieval and compute controls', '',
              '- Representatives search the same outside-in sequence, with at most eight candidates per node. The optional 0.9 early stop is disabled for this comparison.',
              '- Tree search starts at the root and visits headings, topic blocks, paragraphs and sentence leaves. Each shared need has beam width 2. Each query allows at most 256 node scores and 8,192 preview-payload tokens.',
              '- Reached sentence leaves expand to source paragraphs. All methods use the same source-union renderer and final token limit.',
              '- Flat methods search the whole document independently. Multiple shared needs combine through RRF with constant 60; the reranker receives at most 8,192 source tokens.',
              '- The full-budget hybrid combines 8,192 tree-preview tokens with 8,192 direct-passage tokens. The divided-budget hybrid gives each path 4,096 tokens. Both fuse whole-path outputs with equal-weight RRF; neither mixes routing scores.',
              '- Query-time previews and flat source pools contain different information even at equal token ceilings. Shared caches reduce experimental spending; summed API time is not end-to-end single-system latency.', '',
              '## Integrity, spending and limitations', '',
              f"The cumulative Gemini reservation is **${summary['budget']['gemini_reserved_upper_bound_usd']:.4f} / $30**, "
              'including earlier work, conservative output allowances and failed/retried requests. '
              'This reservation is an upper bound, not a billing statement. Codex was not used for this run’s reader. '
              'Jev usage is recorded separately. The legacy Jev audit stage name “reranking” includes representative '
              'selection and tree routing; native input/output tokens and decision counts are preserved in usage totals.', '',
              'All 614 native-to-topic topologies were checked against the original recursive packing of the cached '
              'Jev and embedding split groups. Metadata and transport repairs are recorded in the manifest amendments; '
              'no answer outcomes were inspected to choose those repairs. The provider input quota required paced embeddings.', '',
              f"The generative reranker needed deterministic permutation normalization on {analysis['normalized_ranker_responses']} responses. "
              'The adapter takes the earliest usable reply, keeps its first-occurrence priorities and appends omitted '
              'candidates in original RRF order. This policy was applied uniformly before reader inference. '
              '[Raw and normalized orders](results/live-tree-complete-v2/ranker-normalizations.json) remain available.', '',
              f"The reader adapter repaired {analysis['reader_identity_repairs']} copied-ID typos by attaching the canonical identity of the single isolated request. "
              'Answer text was not edited, and the earliest usable reply is selected uniformly. '
              '[Identity-repair audit](results/live-tree-complete-v2/reader-identity-normalizations.json).', '',
              '**Reader contract correction:** the first mixed-task reader frequently answered multiple-choice questions '
              'with free text or “Unanswerable,” making strict A–D accuracy misleading. All sixteen QuALITY arms were '
              'rerun uniformly with explicitly labelled options and an A–D enum response schema; no labels were inferred '
              'from gold. Original QASPER answers and every retrieval context were preserved. Known provider-blocked '
              'questions were not sent again. The follow-up uses the same corrected reader. '
              '[Original outputs](results/live-tree-v1/scores.json) and the '
              '[contract audit](results/live-tree-complete-v2/reader-contract-audit.json) are retained.', '',
              'The sample is deliberately difficult and evidence-dense, but already exposed. It does not represent '
              'a complete benchmark, multi-reader replication, open-corpus retrieval, image reasoning, or a published '
              'RAPTOR/PageIndex comparison. All registered arms, including regressions and exhausted routes, remain in the denominators.', '',
              '[All predictions and source spans](results/live-tree-complete-v2/scores.json) · '
              '[Paired intervals](results/live-tree-complete-v2/paired-intervals.json) · '
              '[Routing diagnostics](results/live-tree-complete-v2/routing-analysis.json) · '
              '[Failure comparisons](results/live-tree-complete-v2/failure-comparisons.json) · '
              '[Usage](results/live-tree-complete-v2/usage.json) · [Registration](results/live-tree-complete-v2/manifest.json)', '',
              '## Replay', '',
              'Reconstruct the original bounded inputs as described in [the data protocol](BOUNDED_PROTOCOL.md), '
              'then verify this report without provider credentials:', '',
              '```sh', 'python scripts/export_live_tree_complete.py replay', '```', '',
              'The replay checks frozen source and official-data hashes, all 7,680 predictions, source context hashes, '
              'token limits, official answer/evidence metrics, aggregates and paired intervals.', '']
    (ROOT / 'evals/LIVE_TREE_REPORT.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    intervals = read_json(RESULT / 'paired-intervals.json')
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), layout='constrained')
    fig.patch.set_facecolor('#faf9f5')
    for row_index, dataset in enumerate(['qasper', 'quality']):
        for col, factor in enumerate(['split', 'representative', 'router']):
            ax = axes[row_index, col]; ax.set_facecolor('#faf9f5')
            rows = [r for r in intervals if r['factor'] == factor and r['dataset'] == dataset]
            for i, row in enumerate(rows):
                effect = row['metrics']['answer_score']; point = effect['difference'] * 100
                lo, hi = [v * 100 for v in effect['ci95']]
                color = '#35725b' if point >= 0 else '#a75d48'
                ax.plot([lo, hi], [i, i], color=color, linewidth=2)
                ax.scatter([point], [i], color=color, s=38, zorder=3)
            ax.set_yticks(range(len(rows)), [r['candidate'] + ' − ' + r['baseline'] for r in rows], fontsize=9)
            ax.invert_yaxis(); ax.axvline(0, color='#7c8079', linewidth=1, linestyle='--')
            ax.set_title(('QASPER F1' if dataset == 'qasper' else 'QuALITY-HARD accuracy') + '\n' + factor.capitalize(), loc='left', fontsize=12)
            ax.set_xlabel('Jev − embedding (percentage points)', fontsize=9)
            ax.spines[['top', 'right', 'left']].set_visible(False); ax.grid(axis='x', alpha=.12)
    fig.suptitle('Live component comparison on 384 development questions', fontsize=17, fontweight='bold')
    fig.savefig(RESULT / 'component-effects.png', dpi=170, bbox_inches='tight')
    fig.savefig(RESULT / 'component-effects.svg', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    render()
    figure()
