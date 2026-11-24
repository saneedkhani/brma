"""Analysis of the adaptive-attack study (brma-adaptive, 12 seeds, 672 runs)."""
import json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, str(Path(__file__).parent))
from cic_main_analysis import confusion_metrics, ELIG, ARR, NAME, boot_ci, holm, HULK

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'results/raw/cic_adaptive/runs'; OUT = ROOT / 'results/tables'
ARMS = ['C', 'K3H', 'K5', 'S3H', 'S5', 'M3H', 'M5']
METHODS = ['NONE', 'BRMA', 'RAND', 'LOSS', 'LOSS5', 'ORACLE', 'DC', 'MEDIAN']

rows, aud = [], []
for p in sorted(RUNS.glob('*/result.json')):
    r = json.loads(p.read_text())
    st = [confusion_metrics(s['confusion'], [c for c in ELIG if ARR[c] <= s['stage']]) for s in r['stages']]
    row = dict(seed=r['seed'], arm=r['arm'], method=r['method'], macro_f1=st[-1]['macro_f1'], accuracy=st[-1]['accuracy'],
               far=st[-1]['benign_far'], e3_asr=st[2]['hulk_asr'], e4_asr=st[3]['hulk_asr'], n_poison=r['attack']['n_poison'],
               poison_end=sum(m['poison'] for m in r['memory'] if m) if r.get('memory') else np.nan,
               poison_e2=sum(m['poison'] for m in r['memory_after_stage2'] if m) if r.get('memory_after_stage2') else np.nan)
    rows.append(row)
    for a in r.get('audits', []):
        det = a.get('brma_detection', [])
        aud.append(dict(seed=r['seed'], arm=r['arm'], method=r['method'], stage=a['after_stage'],
                        det_poison=sum(d['poison'] for d in det), det_flag_poison=sum(d['flagged_poison'] for d in det),
                        det_flag=sum(d['flagged'] for d in det), det_benign=sum(d['benign_labelled'] for d in det),
                        removed=sum(c.get('removed', 0) for c in a['clients']), removed_poison=sum(c.get('removed_poison', 0) for c in a['clients']),
                        poison_before=sum(c.get('poison_before', 0) for c in a['clients']), benign_labelled=sum(c.get('benign_labelled', 0) for c in a['clients'])))
df = pd.DataFrame(rows); au = pd.DataFrame(aud)
assert len(df) == 672, len(df)
cl = df[df.arm == 'C'].set_index(['seed', 'method'])
df['H'] = [.5 * ((r.e3_asr - cl.loc[(r.seed, r.method), 'e3_asr']) + (r.e4_asr - cl.loc[(r.seed, r.method), 'e4_asr'])) for r in df.itertuples()]
df.to_csv(OUT / 'adaptive-per-run.csv', index=False); au.to_csv(OUT / 'adaptive-audits.csv', index=False)

pd.set_option('display.width', 220)
summ = []
for arm in ARMS:
    for m in METHODS:
        d = df[(df.arm == arm) & (df.method == m)]
        row = dict(arm=arm, method=m)
        for col in ('H', 'macro_f1', 'accuracy', 'far'):
            lo, hi = boot_ci(d[col]); row.update({col: d[col].mean(), col + '_lo': lo, col + '_hi': hi})
        summ.append(row)
summ = pd.DataFrame(summ); summ.to_csv(OUT / 'adaptive-summary.csv', index=False)
for col, sc in (('H', 1), ('macro_f1', 100), ('accuracy', 100), ('far', 100)):
    print(f'\n=== {col}'); print((sc * summ.pivot(index='method', columns='arm', values=col)).loc[METHODS, ARMS].round(4).to_string())

# BRMA detection (recorded on the NONE branch = memory before any action) and removal by memory actions
det = au[(au.method == 'NONE') & (au.stage == 2)].groupby('arm')[['det_poison', 'det_flag_poison', 'det_flag', 'det_benign']].sum()
det['detection'] = det.det_flag_poison / det.det_poison; det['false_removal'] = (det.det_flag - det.det_flag_poison) / (det.det_benign - det.det_poison)
print('\n=== BRMA detection on memory after the attack stage\n', det.loc[ARMS].round(4).to_string())
rem = au[au.stage == 2].groupby(['arm', 'method'])[['removed', 'removed_poison', 'poison_before']].sum()
rem['share_of_poison_removed'] = rem.removed_poison / rem.poison_before
print('\n=== removal after the attack stage\n', rem.loc[(ARMS[1:], ['BRMA', 'LOSS', 'LOSS5', 'RAND']), :].round(3).to_string())
print('\n=== poisoned records: injected / in memory after E2 / at end\n', df[df.method.isin(['NONE', 'BRMA', 'LOSS5'])].groupby(['arm', 'method'])[['n_poison', 'poison_e2', 'poison_end']].mean().round(1).to_string())

def pair(arm, a, b, col='H'):
    x = df[(df.arm == arm) & (df.method == a)].set_index('seed')[col]; y = df[(df.arm == arm) & (df.method == b)].set_index('seed')[col].loc[x.index]
    return x.to_numpy(), y.to_numpy()
T = [('T7', 'S3H', 'BRMA', 'NONE'), ('T8', 'M3H', 'BRMA', 'NONE'), ('T9', 'K3H', 'BRMA', 'LOSS5'), ('T10', 'S3H', 'BRMA', 'LOSS5')]
res = []
for t, arm, a, b in T:
    x, y = pair(arm, a, b); d = x - y; lo, hi = boot_ci(d)
    res.append(dict(test=t, arm=arm, other=b, brma=x.mean(), other_mean=y.mean(), diff=d.mean(), ci_lo=lo, ci_hi=hi, wins=int((d < 0).sum()),
                    p=float(wilcoxon(d, alternative='less').pvalue) if (d != 0).any() else 1.))
rt = pd.DataFrame(res); rt['p_holm'] = holm(rt.p); rt['confirmed'] = rt.p_holm < .05
rt.to_csv(OUT / 'adaptive-tests.csv', index=False); print('\n=== pre-declared tests\n', rt.round(5).to_string(index=False))
ex = []
for arm in ARMS[1:]:
    for b in ('NONE', 'RAND', 'LOSS', 'LOSS5', 'DC', 'MEDIAN'):
        x, y = pair(arm, 'BRMA', b); d = x - y
        ex.append(dict(arm=arm, other=b, brma=x.mean(), other_mean=y.mean(), wins=int((d < 0).sum()), p=float(wilcoxon(d, alternative='less').pvalue) if (d != 0).any() else 1.))
ex = pd.DataFrame(ex); ex['p_holm'] = holm(ex.p); ex.to_csv(OUT / 'adaptive-exploratory.csv', index=False)
print('\n=== exploratory (BRMA < other on H)\n', ex.round(5).to_string(index=False))
