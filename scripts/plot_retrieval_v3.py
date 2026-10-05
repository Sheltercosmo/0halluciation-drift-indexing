"""Plot a complete partition using its published document-cluster intervals."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def plot(partition='test'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    source=ROOT/'evals/results/tree-retrieval-v3'/partition
    stats=json.loads((source/'statistics.json').read_text(encoding='utf-8'))
    catalog=json.loads((source/'catalog.json').read_text(encoding='utf-8'))
    assert catalog['questions']=={'validation':1005,'test':728}[partition] and len(catalog['methods'])==18
    assert stats['partition']==catalog['partition']==partition
    primary=stats['metrics']['recall@5'];methods=primary['methods']
    assert set(methods)==set(catalog['methods'])
    names={'JJJ':'Jev split / central / tree search','hybrid':'Jev tree + independent dense',
        'gemini_dense':'Gemini dense','gemini_jev_rerank':'Gemini dense + Jev',
        'qwen_dense':'Qwen dense (0.6B)','qwen_rerank':'Qwen dense + Qwen reranker (0.6B)',
        'qwen_jev_rerank':'Qwen dense + Jev','bge_dense':'BGE-M3 dense',
        'bge_rerank':'BGE dense + BGE reranker','bge_jev_rerank':'BGE dense + Jev',
        'bm25_dense_rrf':'BM25 + Gemini dense','EEE':'Embedding split / central / global search'}
    panels=[['EEE','EEJ','EJE','EJJ','JEE','JEJ','JJE','JJJ'],
            ['JJJ','hybrid','gemini_dense','gemini_jev_rerank','qwen_dense','qwen_rerank',
             'qwen_jev_rerank','bge_dense','bge_rerank','bge_jev_rerank','bm25_dense_rrf','EEE']]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(15,8),gridspec_kw={'width_ratios':[.9,1.25]})
    for panel,(ax,order) in enumerate(zip(axes,panels)):
        for i,m in enumerate(order):
            row=methods[m];mean=row['mean'];lo,hi=row['ci95']
            color='#ad8650' if m=='hybrid' else '#345847' if m=='JJJ' or m.endswith('_jev_rerank') else '#68786f'
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
    label='Locked QASPER test' if partition=='test' else 'QASPER validation — test results pending'
    fig.text(.055,.922,f"{label} · {primary['questions']} eligible questions across {primary['documents']} papers · whole-paragraph outputs",fontsize=12,color='#53645b')
    fig.text(.055,.045,'Letters: split / central-sentence selection / search.  E = embedding; J = Jev.\nWhiskers: 95% document-cluster bootstrap intervals. Retrieval inside the supplied paper.',fontsize=10,color='#53645b')
    fig.subplots_adjust(left=.055,right=.97,bottom=.14,top=.83,wspace=.95)
    destination=ROOT/'assets/figures';destination.mkdir(parents=True,exist_ok=True)
    for extension in ('svg','png'):
        path=destination/('tree-retrieval-v3-'+partition+'.'+extension)
        fig.savefig(path,dpi=220,facecolor='white',metadata={'Creator':'0halluciation drift indexing'})
        if extension=='svg':
            text='\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n'
            path.write_text(text,encoding='utf-8',newline='\n')
        print(path)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--partition',choices=('validation','test'),default='test')
    plot(parser.parse_args().partition)
