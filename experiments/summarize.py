"""Aggregate independent training seeds and render the Vietnamese report."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import platform
import re
import statistics as stats

ROOT = Path(__file__).resolve().parents[1]


def aggregate(rows):
    groups = defaultdict(list)
    reflex_seed = min((int(r['seed']) for r in rows if r['agent']=='reflex' and r['phase']=='eval'), default=None)
    for row in rows:
        if row['phase'] == 'eval':
            # Reflex has no training seed. Do not count repeated identical runs.
            if row['agent'] == 'reflex' and int(row['seed']) != reflex_seed:
                continue
            groups[int(row['size']), row['agent'], int(row['seed'])].append(row)
    seeds = []
    for (size, agent, seed), games in sorted(groups.items()):
        mean = lambda field: stats.mean(float(g[field]) for g in games)
        seeds.append(dict(size=size, agent=agent, seed=seed, games=len(games),
                          score=mean('score'), food_fraction=mean('food_fraction'), steps=mean('steps'),
                          win_rate=stats.mean(g['outcome']=='win' for g in games),
                          timeout_rate=stats.mean(g['outcome']=='timeout' for g in games),
                          decision_ms=1000*sum(float(g['decision_seconds']) for g in games)/sum(int(g['steps']) for g in games),
                          model_entries=mean('model_entries'), train_seconds=mean('train_seconds'),
                          winning_steps=stats.mean(int(g['steps']) for g in games if g['outcome']=='win') if any(g['outcome']=='win' for g in games) else None))
    methods = defaultdict(list)
    for row in seeds:
        methods[row['size'],row['agent']].append(row)
    summary = []
    for (size, agent), data in sorted(methods.items()):
        record = dict(size=size,agent=agent,seeds=len(data),games=sum(r['games'] for r in data))
        for key in ['score','food_fraction','steps','win_rate','timeout_rate','decision_ms','model_entries','train_seconds']:
            values = [r[key] for r in data]
            record[key] = stats.mean(values)
            record[key+'_sd'] = stats.stdev(values) if len(values)>1 else 0
        summary.append(record)
    return seeds, summary


def findings(summary):
    paragraphs=[]
    for size in sorted({r['size'] for r in summary}):
        subset={r['agent']:r for r in summary if r['size']==size}
        best=max(subset.values(),key=lambda r:r['food_fraction'])
        text=f"- Bản {size}×{size}: {best['agent']} có tỷ lệ thức ăn cao nhất ({best['food_fraction']:.1%})."
        if all(m in subset for m in ('improved','simple','no-danger')):
            improved,simple,ablation=(subset[m] for m in ('improved','simple','no-danger'))
            text+=f" improved so với simple chênh {(improved['food_fraction']-simple['food_fraction'])*100:+.1f} điểm phần trăm; so với no-danger chênh {(improved['food_fraction']-ablation['food_fraction'])*100:+.1f} điểm phần trăm. Đây là chênh lệch mô tả, chưa phải kiểm định ý nghĩa thống kê."
        paragraphs.append(text)
    return '\n'.join(paragraphs)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='results/main')
    args=parser.parse_args()
    folder=ROOT/args.input
    rows=list(csv.DictReader((folder/'episodes.csv').open(encoding='utf8')))
    seeds, summary=aggregate(rows)
    for name,data in [('per_seed',seeds),('summary',summary)]:
        with (folder/f'{name}.csv').open('w',newline='',encoding='utf8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]))
            writer.writeheader();writer.writerows(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=ROOT/'report/figures';figdir.mkdir(parents=True,exist_ok=True)
    sizes=sorted({r['size'] for r in summary})
    methods=['reflex','tabular','simple','improved','no-danger']
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,key,label in zip(axes,['win_rate','food_fraction','score'],['Win rate','Food collected','Score']):
        for method in methods:
            data=sorted([r for r in summary if r['agent']==method],key=lambda r:r['size'])
            ax.errorbar([r['size'] for r in data],[r[key] for r in data],
                        yerr=[r[key+'_sd'] for r in data],marker='o',capsize=3,label=method)
        ax.set(xlabel='Maze side length',ylabel=label,xticks=sizes)
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'comparison.png',dpi=170);plt.close(fig)
    # Episode returns binned by consumed interaction steps, then averaged by seed.
    fig,axes=plt.subplots(1,len(sizes),figsize=(15,4),squeeze=False)
    for ax,size in zip(axes[0],sizes):
        bins=defaultdict(list)
        for row in rows:
            if row['phase']=='train' and int(row['size'])==size:
                step=int(row['training_steps']); bucket=max(1,(step+1999)//2000)*2000
                bins[row['agent'],int(row['seed']),bucket].append(float(row['score']))
        by_method=defaultdict(list)
        for (method,seed,bucket),scores in bins.items():
            by_method[method,bucket].append(stats.mean(scores))
        for method in methods[1:]:
            x=sorted(b for m,b in by_method if m==method)
            ax.plot(x,[stats.mean(by_method[method,b]) for b in x],marker='.',label=method)
        ax.set(title=f'{size} x {size}',xlabel='Training interactions',ylabel='Training episode score')
        ax.grid(alpha=.2)
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'learning.png',dpi=170);plt.close(fig)
    # Show actual saved layouts, not an illustrative approximation.
    fig,axes=plt.subplots(1,len(sizes),figsize=(12,4),squeeze=False)
    for ax,size in zip(axes[0],sizes):
        layout=(ROOT/f'experiments/layouts/extended{size}.lay').read_text().splitlines()
        colors={'%':0,' ':1,'.':2,'o':3,'G':4,'P':5}
        ax.imshow([[colors[c] for c in row] for row in layout],cmap='viridis',vmin=0,vmax=5,interpolation='nearest')
        ax.set_title(f'{size} x {size}');ax.axis('off')
    from matplotlib.patches import Patch
    cmap=plt.get_cmap('viridis')
    fig.legend(handles=[Patch(color=cmap(i/5),label=label) for i,label in enumerate(
        ['Wall','Corridor','Food','Capsule','Ghost','Pac-Man'])],loc='lower center',ncol=6)
    fig.tight_layout(rect=(0,.07,1,1));fig.savefig(figdir/'layouts.png',dpi=170);plt.close(fig)
    report=ROOT/'report';report.mkdir(exist_ok=True)
    table='| Cỡ | Tác tử | Thắng | Thức ăn | Điểm TB ± SD seed | Hết bước | ms/quyết định |\n|---|---|---:|---:|---:|---:|---:|\n'
    for r in summary:
        table+=f"| {r['size']} | {r['agent']} | {r['win_rate']:.1%} | {r['food_fraction']:.1%} | {r['score']:.1f} ± {r['score_sd']:.1f} | {r['timeout_rate']:.1%} | {r['decision_ms']:.3f} |\n"
    details='| Cỡ | Tác tử | Giây học TB | Số mục mô hình TB |\n|---|---|---:|---:|\n'
    for r in summary:
        details+=f"| {r['size']} | {r['agent']} | {r['train_seconds']:.2f} | {r['model_entries']:.0f} |\n"
    text=(report/'template.md').read_text(encoding='utf8')
    text=text.replace('{{RESULTS}}',table).replace('{{COSTS}}',details)
    config=json.loads((folder/'config.json').read_text())
    text=text.replace('{{CONFIG}}',json.dumps(config,ensure_ascii=False,indent=2))
    text=text.replace('{{ENV}}',f'Python {platform.python_version()}; {platform.platform()}; CPU, không dùng GPU.')
    layoutstats='| Kích thước | Ô đi được | Thức ăn | Capsule | Ma |\n|---|---:|---:|---:|---:|\n'
    for size in sizes:
        layout=(ROOT/f'experiments/layouts/extended{size}.lay').read_text()
        layoutstats+=f"| {size}×{size} | {sum(c in ' .oGP' for c in layout)} | {layout.count('.')} | {layout.count('o')} | {layout.count('G')} |\n"
    text=text.replace('{{LAYOUTSTATS}}',layoutstats)
    grades=[]
    for n in [1,2,3,5]:
        log=(ROOT/f'logs/project{n}.txt').read_text(encoding='utf8')
        total=re.findall(r'Total:\s*(\d+/\d+)',log)
        grades.append(f'| Project {n} | {total[-1] if total else "Chưa có kết quả"} | logs/project{n}.txt |')
    text=text.replace('{{GRADES}}','\n'.join(grades))
    ml_log=(ROOT/'logs/project5.txt').read_text(encoding='utf8')
    regression=re.search(r'Your final loss is:\s*([0-9.]+)',ml_log)
    accuracies=re.findall(r'Your final test set accuracy is:\s*([0-9.]+)%',ml_log)
    metrics=f"MSE hồi quy: {regression.group(1) if regression else 'chưa có'}. "
    if len(accuracies)==3:
        metrics+=f"Độ chính xác chữ số: {accuracies[0]}%; ngôn ngữ: {accuracies[1]}%; CNN trên tập kiểm tra gốc có chồng lặp: {accuracies[2]}%."
    text=text.replace('{{MLMETRICS}}',metrics)
    text=text.replace('{{FINDINGS}}',findings(summary))
    (report/'bao_cao.md').write_text(text,encoding='utf8')
    build_docx(text,report/'Bao_cao_PacMan.docx')
    print('Wrote summary CSVs, figures, bao_cao.md and Bao_cao_PacMan.docx')


def build_docx(text,target):
    from docx import Document
    from docx.shared import Inches, Pt
    doc=Document()
    section=doc.sections[0]
    section.left_margin=section.right_margin=Inches(1)
    section.top_margin=section.bottom_margin=Inches(.8)
    normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(12)
    doc.sections[0].header.paragraphs[0].text='BÀI TẬP LỚN CƠ SỞ TRÍ TUỆ NHÂN TẠO — PAC-MAN'
    lines=text.splitlines();i=0;code=False
    while i<len(lines):
        line=lines[i]
        if line.startswith('```'):
            code=not code;i+=1;continue
        if line.startswith('|'):
            tablelines=[]
            while i<len(lines) and lines[i].startswith('|'):
                if not re.match(r'^\|[\s:|\-]+\|$',lines[i]):
                    tablelines.append([c.strip() for c in lines[i].strip('|').split('|')])
                i+=1
            table=doc.add_table(rows=0,cols=len(tablelines[0]));table.style='Light Shading Accent 1'
            for row in tablelines:
                cells=table.add_row().cells
                for cell,value in zip(cells,row):
                    cell.text=value
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:run.font.size=Pt(9)
            continue
        image=re.match(r'!\[.*?\]\((.*?)\)',line)
        if image:
            doc.add_picture(str(ROOT/'report'/image.group(1)),width=Inches(6.2))
        elif line.startswith('#'):
            level=len(line)-len(line.lstrip('#'))
            doc.add_heading(line.lstrip('# ').replace('**',''),level=min(level,3))
        elif line.strip():
            paragraph=doc.add_paragraph(line.replace('**','').replace('`',''))
            if code:
                for run in paragraph.runs: run.font.name='Consolas';run.font.size=Pt(9)
        i+=1
    doc.save(target)


if __name__=='__main__':main()
