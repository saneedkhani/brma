"""Exploratory paired tests for BRMA-M / BRMA-C on TON_IoT (fresh seeds 201-212, test split) and CIC-IDS2017 (seeds 1-11, 42)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, str(Path(__file__).parent))
from cic_main_analysis import boot_ci, holm
R = Path(__file__).resolve().parents[1] / 'results'
ton = pd.read_csv(R / 'raw/ton_variants/phase2/test-metrics.csv')
cic = pd.read_csv(R / 'raw/cic_brma_m_c/metrics.csv')

def pair(df, arm, a, b, col):
    x = df[(df.arm == arm) & (df.method == a)].set_index('seed')[col]; y = df[(df.arm == arm) & (df.method == b)].set_index('seed')[col].loc[x.index]
    return x.to_numpy(), y.to_numpy()

rows = []
for name, df, arms in (('TON', ton, ['K3', 'K3H', 'K5']), ('CIC', cic, ['K3', 'K3H', 'K5', 'S3H', 'S5'])):
    for arm in arms:
        for a, b in (('BRMA-C', 'NONE'), ('BRMA-C', 'BRMA'), ('BRMA-C', 'BRMA-M')):
            x, y = pair(df, arm, a, b, 'H'); d = x - y
            rows.append(dict(data=name, arm=arm, a=a, b=b, a_mean=x.mean(), b_mean=y.mean(), wins=int((d < 0).sum()),
                             p_less=float(wilcoxon(d, alternative='less').pvalue) if (d != 0).any() else 1.,
                             p_two=float(wilcoxon(d).pvalue) if (d != 0).any() else 1.))
t = pd.DataFrame(rows); t['p_holm_less'] = holm(t.p_less)
pd.set_option('display.width', 220); print(t.round(5).to_string(index=False))
for name, df in (('TON', ton), ('CIC', cic)):
    for m in ('BRMA', 'BRMA-M', 'BRMA-C'):
        x, y = pair(df, 'C', m, 'NONE', 'macro_f1'); d = x - y; lo, hi = boot_ci(d)
        print(f'{name} clean macro-F1 {m} - NONE: {100 * d.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] points, p two-sided {wilcoxon(d).pvalue:.3f}')
t.to_csv(R / 'tables/variants-exploratory.csv', index=False)
