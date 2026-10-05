"""Source-backed figure for the completed repaired paragraph retrieval study."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    source=ROOT/'evals/results/tree-retrieval-v4/test'
    stats=json.loads((source/'statistics.json').read_text(encoding='utf-8'))
    catalog=json.loads((source/'catalog.json').read_text(encoding='utf-8'))
    assert catalog['questions']==728 and len(catalog['methods'])==18 and stats['partition']=='test'
    primary=stats['metrics']['recall@5'];methods=primary['methods']
    assert set(methods)==set(catalog['methods'])
    names={'JJJ':'Jev splitting / central / tree search','hybrid':'Jev tree + independent Gemini dense',
        'gemini_dense':'Gemini dense','gemini_jev_rerank':'Gemini dense + Jev ranking',
        'qwen_dense':'Qwen dense (0.6B)','qwen_rerank':'Qwen dense + Qwen reranker (0.6B)',
        'qwen_jev_rerank':'Qwen dense + Jev ranking','bge_dense':'BGE-M3 dense',
        'bge_rerank':'BGE dense + BGE reranker','bge_jev_rerank':'BGE dense + Jev ranking',
        'bm25_dense_rrf':'BM25 + Gemini dense'}
    panels=[['EEE','EEJ','EJE','EJJ','JEE','JEJ','JJE','JJJ'],list(names)]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(15,8),gridspec_kw={'width_ratios':[.9,1.25]})
    for panel,(ax,order) in enumerate(zip(axes,panels)):
        for i,m in enumerate(order):
            row=methods[m];mean=row['mean'];lo,hi=row['ci95']
            green=(m.endswith('J') if panel==0 else m=='JJJ' or m.endswith('_jev_rerank'))
            color='#ad8650' if m=='hybrid' else '#345847' if green else '#68786f'
            ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',markersize=6,
                        color=color,ecolor=color,capsize=3,linewidth=1.4)
            ax.annotate(f'{mean:.1%}',(mean,i),xytext=(6,7),textcoords='offset points',fontsize=9,color=color)
        ax.set_yticks(range(len(order)),order if panel==0 else [names[m] for m in order])
        ax.set_ylim(len(order)-.5,-.7);ax.set_xlim(0,1.03)
        ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_xticks([0,.2,.4,.6,.8,1])
        ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0,pad=8)
        for spine in ax.spines.values():spine.set_visible(False)
        ax.set_xlabel('Evidence paragraph recall@5')
        ax.set_title('Crossed components' if panel==0 else 'Complete systems and controls',loc='left',pad=22,fontweight='bold')
    fig.suptitle('Evidence paragraph retrieval',x=.055,y=.975,ha='left',fontsize=23,fontweight='bold',color='#263d31')
    fig.text(.055,.922,f"Repaired v4 test · {primary['questions']} eligible questions across {primary['documents']} papers · whole paragraphs",fontsize=12,color='#53645b')
    fig.text(.055,.042,'E/J letters: splitting / central selection / search. Whiskers: 95% document-cluster intervals.\nJev-ranking rule: 25% evidence score + 75% candidate-rank prior. Native Qwen/BGE controls are unchanged.',fontsize=10,color='#53645b')
    fig.subplots_adjust(left=.055,right=.97,bottom=.14,top=.83,wspace=.95)
    dest=ROOT/'assets/figures';dest.mkdir(exist_ok=True,parents=True)
    for ext in ('svg','png'):
        path=dest/('tree-retrieval-v4-test.'+ext)
        fig.savefig(path,dpi=220,facecolor='white',metadata={'Creator':'0halluciation drift indexing'})
        if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
        print(path)
    plt.close(fig)


if __name__=='__main__':plot()
