"""Report figures (PNG, light surface) from the saved val results; numbers read from the JSON files.

  reports/figures/01_pipeline_progression.png : AP50 after each step of the final pipeline
  reports/figures/02_ap50_by_size.png         : AP50 by bird size, YOLO vs GFL vs ensemble
With --en: English labels, no in-figure titles (the paper captions carry them), saved to github_release/paper/figures/.
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EN = '--en' in sys.argv
OUT = ROOT/('github_release/paper/figures' if EN else 'reports/figures'); OUT.mkdir(parents=True, exist_ok=True)
T = (lambda vi, en: en) if EN else (lambda vi, en: vi)
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE})
J = lambda p: json.loads((ROOT/p).read_text(encoding='utf-8'))


def progression():
    rows = [('Baseline M\n(pretrain Drone → MVA)', J('reports/sahi_tta/baseline_m_e30_full.json')['AP50']),
            ('+ Slicing (train)', J('reports/sahi_tta/slicing_m_e30_full.json')['AP50']),
            (T('+ TTA lật ngang, ghép WBF', '+ flip TTA, WBF'), J('reports/error_analysis/slicing_m_e30_tta_wbf.json')['by_size']['all']['AP50']),
            (T('+ Ensemble GFL (WBF 3 nguồn)', '+ GFL ensemble (3-source WBF)'), J('reports/error_analysis/final_ensemble.json')['by_size']['all']['AP50'])]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = range(len(rows))[::-1]
    ax.barh(list(y), [v*100 for _, v in rows], height=.55, color=BLUE, edgecolor=SURFACE, linewidth=2)
    prev = None
    for yi, (label, v) in zip(y, rows):
        txt = f'{v*100:.1f}%' + (f'  (+{(v-prev)*100:.1f})' if prev is not None else '')
        ax.text(v*100+.4, yi, txt, va='center', color=INK, fontsize=11); prev = v
    ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.set_xlim(0, 100); ax.set_xlabel(T('AP50 trên val audit (%)', 'AP50 on MVA validation (%)'))
    ax.xaxis.grid(True, color=GRID, linewidth=.8); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    if not EN: ax.set_title('Pipeline cuối: đóng góp của từng bước', loc='left', color=INK, fontsize=13, pad=10)
    fig.tight_layout(); fig.savefig(OUT/'01_pipeline_progression.png', dpi=200); plt.close(fig)


def by_size():
    series = [('YOLOv8-M Slicing + TTA', 'reports/error_analysis/slicing_m_e30_tta_wbf.json', BLUE),
              ('GFL ResNet-50 FPN', 'reports/error_analysis/gfl_ep38.json', ORANGE),
              ('Ensemble (WBF)', 'reports/error_analysis/final_ensemble.json', AQUA)]
    buckets = ['8-16', '16-32', '32-64']
    counts = J(series[0][1])['by_size']
    fig, ax = plt.subplots(figsize=(8, 4.6))
    w = .26
    for k, (name, path, color) in enumerate(series):
        s = J(path)['by_size']; xs = [b + (k-1)*w for b in range(len(buckets))]
        vals = [s[b]['AP50']*100 for b in buckets]
        ax.bar(xs, vals, width=w, color=color, edgecolor=SURFACE, linewidth=2, label=name)
        for x, v in zip(xs, vals): ax.text(x, v+.8, f'{v:.1f}', ha='center', color=INK, fontsize=9.5)
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels([f'{b} px\n({counts[b]["gt_count"]} ' + T('chim', 'birds') + ')' for b in buckets], color=INK)
    ax.set_ylim(0, 100); ax.set_ylabel('AP50 (%)')
    ax.yaxis.grid(True, color=GRID, linewidth=.8); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=3, fontsize=10, labelcolor=INK)
    if not EN:
        ax.set_title('AP50 theo kích thước chim (√diện tích): hai mô hình bù cho nhau', loc='left', color=INK, fontsize=13, pad=10)
    fig.text(.01, .01, T('Bỏ nhóm 0–8 px (7 chim) và 64+ px (15 chim) vì quá ít mẫu.',
                         'Buckets 0–8 px (7 birds) and 64+ px (15 birds) are omitted: too few samples.'), color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(OUT/'02_ap50_by_size.png', dpi=200); plt.close(fig)


if __name__ == '__main__':
    progression(); by_size(); print('saved', sorted(p.name for p in OUT.glob('*.png')))
