"""Paper figures for the BRMA study (reads Revision/study-results/analysis/*.csv)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A = Path(__file__).resolve().parents[1] / 'results/tables'; FIG = Path(__file__).resolve().parents[1] / 'results/figures'; FIG.mkdir(exist_ok=True)
S = pd.read_csv(A / 'summary-mean-ci.csv'); Q = pd.read_csv(A / 'audit-quality.csv'); D = pd.read_csv(A / 'per-class-delta.csv')
ATT = ['K1', 'K3', 'K3H', 'K5']
ARMLAB = {'C': 'clean', 'K1': '1 attacker\n50%', 'K3': '3 attackers\n50%', 'K3H': '3 attackers\n90%', 'K5': '5 attackers\n50%'}
COL = {'BRMA': '#2a78d6', 'RAND': '#eb6834', 'LOSS': '#1baf7a', 'FT': '#eda100', 'EI': '#e87ba4', 'DC': '#008300',
       'MEDIAN': '#6250d6', 'DG': '#e34948', 'NONE': '#8a8985', 'ORACLE': '#0b0b0b'}
MK = {'BRMA': 'o', 'RAND': 's', 'LOSS': 'D', 'FT': 'v', 'EI': '^', 'DC': 'P', 'MEDIAN': 'X', 'DG': '<', 'NONE': 'o', 'ORACLE': None}
LAB = {'BRMA': 'BRMA (ours)', 'RAND': 'Random deletion', 'LOSS': 'High-loss deletion', 'ORACLE': 'Oracle purge', 'NONE': 'No defence',
       'FT': 'FT', 'EI': 'EI (EWC)', 'DC': 'DC filter', 'DG': 'DG filter', 'MEDIAN': 'Median aggregation'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': .6, 'axes.axisbelow': True, 'legend.frameon': False, 'savefig.dpi': 300, 'figure.facecolor': 'white'})


def get(m, col, arms):
    d = S[S.method == m].set_index('arm').loc[arms]; return d[col].to_numpy(), d[col + '_lo'].to_numpy(), d[col + '_hi'].to_numpy()


def save(fig, name):
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight'); fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight'); plt.close(fig)


# ---- Figure 1: attack effect H by attack strength, two panels
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
x = np.arange(len(ATT))
for ax, group, title in ((axes[0], ['NONE', 'RAND', 'LOSS', 'ORACLE', 'BRMA'], '(a) Memory actions (same budget as BRMA)'),
                         (axes[1], ['FT', 'EI', 'DC', 'DG', 'MEDIAN', 'BRMA'], '(b) Published-style defences')):
    for m in group:
        y, lo, hi = get(m, 'H', ATT)
        ls = '--' if m == 'ORACLE' else '-'
        ax.fill_between(x, lo, hi, color=COL[m], alpha=.12, lw=0)
        ax.plot(x, y, ls, color=COL[m], lw=2 if m == 'BRMA' else 1.6, marker=MK[m], ms=6 if MK[m] else 0, mec='white', mew=1, label=LAB[m], zorder=3 if m == 'BRMA' else 2)
    ax.set_xticks(x, [ARMLAB[a] for a in ATT], fontsize=7.5); ax.set_title(title, fontsize=9, color=INK, loc='left')
    ax.set_ylim(-.02, 1.0); ax.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2, handlelength=2.2)
axes[0].set_ylabel('H: rise in Hulk-as-benign rate\nover clean training (lower is better)')
save(fig, 'fig1_attack_effect_H')

# ---- Figure 2: robustness vs utility under the strongest attack (K3H)
fig, ax = plt.subplots(figsize=(4.6, 3.3))
for m in ['NONE', 'RAND', 'LOSS', 'ORACLE', 'BRMA', 'DC', 'DG', 'MEDIAN', 'FT', 'EI']:
    h, hl, hh = get(m, 'H', ['K3H']); f, fl, fh = get(m, 'macro_f1', ['K3H'])
    ax.errorbar(h, f, xerr=[[h[0] - hl[0]], [hh[0] - h[0]]], yerr=[[f[0] - fl[0]], [fh[0] - f[0]]], fmt=MK[m] or '*', ls='none', ms=7 if m == 'BRMA' else 6,
                color=COL[m], mec='white', mew=.8, ecolor=COL[m], elinewidth=1, capsize=0, zorder=3)
    pos = {'BRMA': (.07, .70), 'ORACLE': (.07, .665), 'LOSS': (.07, .535), 'RAND': (.15, .615), 'NONE': (.54, .655), 'MEDIAN': (.99, .535),
           'DG': (.70, .465), 'DC': (.44, .40), 'FT': (.40, .135), 'EI': (.40, .06)}[m]
    arrow = dict(arrowstyle='-', color=INK2, lw=.6) if m in ('BRMA', 'ORACLE', 'LOSS') else None
    ax.annotate(LAB[m], (h[0], f[0]), xytext=pos, textcoords='data', fontsize=7.5, color=INK, ha='right' if m == 'MEDIAN' else 'left', va='center', arrowprops=arrow)
ax.set_xlabel('H under 3 attackers flipping 90% (lower is better)'); ax.set_ylabel('Macro-F1 after stage 4 (higher is better)')
ax.set_xlim(-.05, 1.0); ax.set_ylim(0, .74)
save(fig, 'fig2_robustness_vs_utility')

# ---- Figure 3: audit quality after stage 2 (detection recall and false removals vs alpha)
q = Q[Q.after_stage == 2].set_index('arm')
fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.9), gridspec_kw=dict(width_ratios=[4, 5]))
arms = ['C'] + ATT; xx = np.arange(len(arms))
axes[0].bar(np.arange(len(ATT)), 100 * q.loc[ATT, 'detection_recall'], width=.55, color=COL['BRMA'])
for i, a in enumerate(ATT):
    axes[0].text(i, 100 * q.loc[a, 'detection_recall'] - 6, f"{100 * q.loc[a, 'detection_recall']:.2f}%", ha='center', va='top', color='white', fontsize=7.5)
axes[0].set_xticks(np.arange(len(ATT)), [ARMLAB[a] for a in ATT], fontsize=7); axes[0].set_ylim(0, 100)
axes[0].set_ylabel('Poisoned records flagged (%)'); axes[0].set_title('(a) Detection of poisoned memory', fontsize=9, loc='left')
axes[1].bar(xx, 100 * q.loc[arms, 'false_removal_rate'], width=.55, color=COL['NONE'])
axes[1].axhline(5, color=INK, ls='--', lw=1); axes[1].text(len(arms) - .55, 5.25, 'conformal level α = 5%', ha='right', va='bottom', fontsize=7.5, color=INK)
for i, a in enumerate(arms):
    axes[1].text(i, 100 * q.loc[a, 'false_removal_rate'] / 2, f"{100 * q.loc[a, 'false_removal_rate']:.1f}%", ha='center', color='white', fontsize=7.5)
axes[1].set_xticks(xx, [ARMLAB[a] for a in arms], fontsize=7); axes[1].set_ylim(0, 7)
axes[1].set_ylabel('Genuine benign records removed (%)'); axes[1].set_title('(b) False removals', fontsize=9, loc='left')
fig.tight_layout(); save(fig, 'fig3_audit_quality')

# ---- Figure 4: per-class F1 after stage 4, BRMA vs no defence, under the strongest attack
d = D[D.arm == 'K3H'].sort_values('none')
fig, ax = plt.subplots(figsize=(4.8, 3.4)); y = np.arange(len(d))
for i, r in enumerate(d.itertuples()):
    ax.plot([r.none, r.brma], [i, i], color=GRID, lw=2.5, zorder=1)
ax.scatter(d.none, y, s=34, color=COL['NONE'], edgecolor='white', lw=.8, zorder=2, label='No defence')
ax.scatter(d.brma, y, s=34, color=COL['BRMA'], edgecolor='white', lw=.8, zorder=3, label='BRMA (ours)')
ax.set_yticks(y, d.cls); ax.set_xlabel('F1 after stage 4 (3 attackers, 90% flipped)'); ax.set_xlim(0, 1.02)
ax.legend(loc='lower right', fontsize=7.5)
save(fig, 'fig4_per_class_f1')
print('figures:', sorted(p.name for p in FIG.glob('*.png')))
