"""README figures in the style of the DSCMR repository: framework.png (pipeline), workflow.png (tensor shapes from the
4K input to the output boxes) and Result.png (results table).
Numbers are read from results/*.json.

python scripts/make_repo_figures.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, LINE = '#0b0b0b', '#52514e', '#7f7f7f'
FILL = {'data': '#eeeeec', 'model': '#d6e6fa', 'op': '#fde2d6', 'out': '#d4efe3'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})


def framework():
    fig, ax = plt.subplots(figsize=(11, 4.2)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.2); ax.axis('off')
    def box(x, y, w, h, text, kind):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=FILL[kind], ec=LINE, lw=1))
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', color=INK, fontsize=9.5)
    def arrow(x1, y1, x2, y2, label=None):
        ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='-|>', color=LINE, lw=1.2))
        if label: ax.text((x1+x2)/2, (y1+y2)/2 + .12, label, ha='center', fontsize=8, color=MUTED)
    ax.text(0.1, 3.95, 'Training', fontsize=11, fontweight='bold', color=INK)
    box(0.1, 2.9, 1.9, .8, 'Drone2021\n51k frames', 'data')
    box(2.5, 2.9, 2.0, .8, 'YOLOv8-M\npretrain (ep. 7)', 'model')
    box(5.0, 2.9, 2.3, .8, 'MVA fine-tune, slicing\n(all bird tiles, 30 ep.)', 'op')
    arrow(2.0, 3.3, 2.5, 3.3); arrow(4.5, 3.3, 5.0, 3.3)
    box(7.8, 2.9, 3.0, .8, 'GFL R50-FPN\nDrone2021 → MVA (ep. 38)', 'model')
    ax.text(0.1, 2.35, 'Inference', fontsize=11, fontweight='bold', color=INK)
    box(0.1, 1.1, 1.9, .9, '4K frame\n3840×2160\n(no downscale)', 'data')
    box(2.5, 1.55, 2.0, .6, 'YOLO, full frame', 'model')
    box(2.5, 0.85, 2.0, .6, 'YOLO, mirrored', 'model')
    box(2.5, 0.15, 2.0, .6, 'GFL (2560×1440)', 'model')
    arrow(2.0, 1.55, 2.5, 1.85); arrow(2.0, 1.55, 2.5, 1.15); arrow(2.0, 1.4, 2.5, .45)
    box(5.4, 0.55, 2.3, 1.2, 'Weighted boxes\nfusion\nIoU 0.55, 1:1:1', 'op')
    arrow(4.5, 1.85, 5.4, 1.4); arrow(4.5, 1.15, 5.4, 1.15); arrow(4.5, .45, 5.4, .9)
    box(8.4, 0.75, 2.4, .8, 'Bird boxes\nAP50 84.5% (val)', 'out')
    arrow(7.7, 1.15, 8.4, 1.15)
    ax.text(5.5, 2.45, 'trained weights are used below', ha='center', fontsize=8.5, color=MUTED, style='italic')
    fig.tight_layout(); fig.savefig(ROOT/'framework.png', dpi=200, facecolor='white'); plt.close(fig)

def workflow():
    """Data flow of the final detector, top to bottom, with the shape of every intermediate result (one 4K frame)."""
    from matplotlib import font_manager
    font = 'Segoe UI' if any(f.name == 'Segoe UI' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
    BG, TXT, SUB, ARR = '#1e1e2e', '#dcdcf5', '#c9c3e6', '#8a8aa0'
    LIGHT, DARK = '#424252', '#30283a'
    C = {'in': '#4a90d9', 'yolo': '#3fae6a', 'flip': '#e0952a', 'gfl': '#9b6fd6', 'fuse': '#d85c8a', 'out': '#5a5470'}
    W, Hgt = 12, 15.6
    fig = plt.figure(figsize=(W*.78, Hgt*.78), facecolor=BG); ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG)
    ax.set_xlim(0, W); ax.set_ylim(0, Hgt); ax.axis('off')
    bh = .82
    def box(cx, cy, w, title, sub, edge, fill):
        ax.add_patch(FancyBboxPatch((cx - w/2, cy - bh/2), w, bh, boxstyle='round,pad=0,rounding_size=0.16',
                                    fc=fill, ec=edge, lw=1.1))
        ax.text(cx, cy + (.13 if sub else 0), title, ha='center', va='center', color=TXT, fontsize=9.6, fontweight='bold', family=font)
        if sub: ax.text(cx, cy - .17, sub, ha='center', va='center', color=SUB, fontsize=8.8, fontweight='bold', family=font)
    def line(pts, head=True):
        xs, ys = zip(*pts)
        ax.plot(xs[:-1] + (xs[-1],), ys, color=ARR, lw=.9, solid_joinstyle='round')
        if head: ax.annotate('', pts[-1], (pts[-1][0], pts[-1][1] + .14), arrowprops=dict(arrowstyle='->', color=ARR, lw=.9))
    xs = {'yolo': 2.2, 'flip': 6.0, 'gfl': 9.8}; bw = 3.3
    y0 = 14.95
    box(6.0, y0, 3.6, '4K FRAME', '3840 × 2160 × 3 · uint8 · BGR', C['in'], LIGHT)
    head = {'yolo': ('YOLO · FULL FRAME', C['yolo']), 'flip': ('YOLO · MIRRORED (FLIP TTA)', C['flip']), 'gfl': ('GFL R50-FPN', C['gfl'])}
    split = y0 - .95
    for k, x in xs.items():
        line([(6.0 if k != 'yolo' and k != 'gfl' else 6.0, y0 - bh/2), (6.0, split), (x, split), (x, 12.75 + bh/2)] if k != 'flip'
             else [(6.0, y0 - bh/2), (6.0, 12.75 + bh/2)])
        ax.text(x, 12.75 + bh/2 + .32, head[k][0], ha='center', va='center', color=head[k][1], fontsize=8.6, fontweight='bold', family=font, zorder=3, bbox=dict(fc=BG, ec='none', pad=2.5))
    rows = [  # y, fill, {branch: (title, subtitle)}
        (12.75, LIGHT, {'yolo': ('PAD (NO RESIZE)', '→ 3840 × 2176 · ÷255'), 'flip': ('MIRROR + PAD', '→ 3840 × 2176 · ÷255'),
                        'gfl': ('RESIZE ×2/3', '→ 2560 × 1440 · mean/std')}),
        (11.35, LIGHT, {'yolo': ('INPUT TENSOR', '1 × 3 × 2176 × 3840'), 'flip': ('INPUT TENSOR', '1 × 3 × 2176 × 3840'),
                        'gfl': ('INPUT TENSOR (FP16)', '1 × 3 × 1440 × 2560')}),
        (9.95, LIGHT, {'yolo': ('YOLOV8-M (EP. 30)', 'heads at stride 8 / 16 / 32'), 'flip': ('YOLOV8-M', 'same weights · 2nd pass'),
                       'gfl': ('RESNET-50 + FPN (EP. 38)', 'P3–P7 · stride 8 … 128')}),
        (8.55, DARK, {'yolo': ('RAW OUTPUT', '1 × 5 × 171,360 (cx, cy, w, h, s)'), 'flip': ('RAW OUTPUT', '1 × 5 × 171,360 · mirrored'),
                      'gfl': ('RAW OUTPUT', '76,760 pts · score + 4 × 17 bins')}),
        (7.15, DARK, {'yolo': ('SCORE ≥ 0.001 + NMS 0.6', 'xyxy · keep ≤ 300'), 'flip': ('NMS 0.6 + UN-MIRROR', 'x → 3840 − x · keep ≤ 300'),
                      'gfl': ('NMS 0.6 + RESCALE ×1.5', 'back to 4K · keep ≤ 100')}),
        (5.75, DARK, {'yolo': ('BOXES', '≤ 300 × 5'), 'flip': ('BOXES', '≤ 300 × 5'), 'gfl': ('BOXES', '≤ 100 × 5')}),
    ]
    for i, (y, fill, cells) in enumerate(rows):
        for k, (t, s) in cells.items():
            box(xs[k], y, bw, t, s, head[k][1], fill)
            if i: line([(xs[k], rows[i-1][0] - bh/2), (xs[k], y + bh/2)])
    yf = 4.0
    merge = yf + bh/2 + .45
    for k, x in xs.items():
        line([(x, 5.75 - bh/2), (x, merge), (6.0, merge), (6.0, yf + bh/2)] if k != 'flip' else [(x, 5.75 - bh/2), (6.0, yf + bh/2)],
             head=(k == 'flip'))
    box(6.0, yf, 4.6, 'WEIGHTED BOXES FUSION', 'IoU 0.55 · weights 1 : 1 : 1 · ≤ 700 boxes in', C['fuse'], DARK)
    box(6.0, 2.6, 4.6, 'CLIP TO FRAME', '0 ≤ x ≤ 3840 · 0 ≤ y ≤ 2160', C['out'], DARK)
    box(6.0, 1.2, 4.6, 'FINAL BOXES + SCORES', 'n × 5 → COCO JSON · val AP50 84.5%', C['out'], DARK)
    line([(6.0, yf - bh/2), (6.0, 2.6 + bh/2)]); line([(6.0, 2.6 - bh/2), (6.0, 1.2 + bh/2)])
    fig.savefig(ROOT/'workflow.png', dpi=220, facecolor=BG); plt.close(fig)


def result():
    J = lambda p: json.loads((ROOT/'results'/p).read_text(encoding='utf-8'))
    t = {k: J(f'training/{k}/epoch_30_metrics.json') for k in ('baseline_m_e30', 'augmentation_m_e30', 'copy_paste_m_e30', 'slicing_m_e30')}
    tw = J('yolo_tables.json')['slicing_m_e30']['stages']['tta_wbf']   # values already in %
    inf = {k: tw[k]/100 for k in ('AP50', 'AP75', 'AR100')}
    ens = J('ensemble/gfl+plain+flip.json')['rules']['wbf_i0.55_w1']
    gfl = J('gfl/gfl_rescored_audited.json')['per_epoch']['38']
    sahi = J('inference/slicing_m_e30_sahi_z1.5.json')
    rows = [('GFL R50-FPN', gfl), ('YOLOv8-M baseline', t['baseline_m_e30']), ('  + colour augmentation', t['augmentation_m_e30']),
            ('  + copy-paste', t['copy_paste_m_e30']), ('  + slicing', t['slicing_m_e30']), ('  slicing + SAHI ×1.5', sahi),
            ('  slicing + flip TTA (WBF)', inf),
            ('  slicing + flip TTA + GFL (WBF, final)', ens)]
    cell = lambda v: '–' if v is None else f'{100*v:.1f}'
    data = [[n, cell(m.get('AP50')), cell(m.get('AP75')), cell(m.get('AR100'))] for n, m in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.6)); ax.axis('off')
    tb = ax.table(cellText=data, colLabels=['Method', 'AP50', 'AP75', 'AR100'], loc='center', cellLoc='center', colLoc='center',
                  colWidths=[.58, .14, .14, .14])
    tb.auto_set_font_size(False); tb.set_fontsize(9.5); tb.scale(1, 1.45)
    for (r, c), cl in tb.get_celld().items():
        cl.set_edgecolor('#d0d0d0'); cl.set_linewidth(.6)
        if c == 0 and r > 0: cl.set_text_props(ha='left'); cl.PAD = .03
        if r == 0: cl.set_facecolor('#e8e8e6'); cl.set_text_props(fontweight='bold')
        if r == len(data): cl.set_text_props(fontweight='bold'); cl.set_facecolor('#eef6f1')
    ax.set_title('MVA 2023 SOD4SB, audited validation split (879 frames), COCO maxDets 100, %', fontsize=9.5, color=MUTED, pad=4)
    fig.tight_layout(); fig.savefig(ROOT/'Result.png', dpi=200, facecolor='white'); plt.close(fig)


if __name__ == '__main__':
    framework(); workflow(); result(); print('framework.png, workflow.png, Result.png')
