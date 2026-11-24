"""Full analysis of the BRMA study (CIC-IDS2017, 12 seeds x 5 arms x 10 methods = 600 runs).

Reads results/raw/cic_main/runs/*/result.json and writes tables to results/tables/.
Metric definitions follow scripts/phase3_evidence.confusion_metrics of the audited repository.
"""
import json, glob, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'results/raw/cic_main/runs'
OUT = ROOT / 'results/tables'; OUT.mkdir(parents=True, exist_ok=True)
TAX = json.load(open(ROOT / 'data/cic_taxonomy-v1.json'))
ELIG = [c['source_code'] for c in TAX['classes'] if c['scoring_eligible']]
ARR = {c['source_code']: c['arrival_stage'] for c in TAX['classes']}
NAME = {c['source_code']: c['label'] for c in TAX['classes']}
HULK = 4
ARMS = ['C', 'K1', 'K3', 'K3H', 'K5']
METHODS = ['NONE', 'BRMA', 'RAND', 'LOSS', 'ORACLE', 'FT', 'EI', 'DC', 'DG', 'MEDIAN']
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 42]


def confusion_metrics(confusion, eligible):          # copy of the audited definition
    full = np.asarray(confusion, dtype=np.int64); eligible = sorted(int(c) for c in eligible)
    bank = full.copy(); bank[[c for c in range(16) if c not in eligible], :] = 0
    support = bank.sum(1); predicted = bank.sum(0); tp = bank.diagonal()
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(16), where=(support + predicted) > 0)
    active = [c for c in eligible if support[c] > 0]
    r = dict(macro_f1=float(f1[active].mean()), accuracy=float(tp.sum() / bank.sum()), benign_far=float(1 - bank[0, 0] / support[0]),
             f1={c: float(f1[c]) for c in active})
    if support[HULK]:
        r.update(hulk_recall=float(tp[HULK] / support[HULK]), hulk_asr=float(bank[HULK, 0] / support[HULK]))
    fp, fn, btp, tn = int(bank[0, 1:].sum()), int(bank[1:, 0].sum()), int(bank[1:, 1:].sum()), int(bank[0, 0])
    r['binary_f1'] = 2 * btp / (2 * btp + fp + fn)
    return r


def load():
    rows, audits = [], []
    for p in sorted(RUNS.glob('*/result.json')):
        r = json.loads(p.read_text())
        st = [confusion_metrics(s['confusion'], [c for c in ELIG if ARR[c] <= s['stage']]) for s in r['stages']]
        last = st[-1]
        A = np.mean([np.mean([st[t - 1]['f1'][c] for c in ELIG if ARR[c] == t and c != HULK]) for t in (2, 3, 4)])
        # forgetting: F1 at arrival stage minus F1 after stage 4, classes that arrive in stages 1-3 (Hulk excluded)
        forget = np.mean([st[ARR[c] - 1]['f1'][c] - last['f1'][c] for c in ELIG if ARR[c] <= 3 and c != HULK])
        row = dict(seed=r['seed'], arm=r['arm'], method=r['method'], macro_f1=last['macro_f1'], accuracy=last['accuracy'],
                   binary_f1=last['binary_f1'], benign_far=last['benign_far'], A=A, forgetting=forget,
                   e2_recall=st[1]['hulk_recall'], e3_asr=st[2]['hulk_asr'], e4_asr=st[3]['hulk_asr'],
                   f1_e1=st[0]['macro_f1'], f1_e2=st[1]['macro_f1'], f1_e3=st[2]['macro_f1'], seconds=r['elapsed_seconds'],
                   n_poison=r['attack']['n_poison'], n_attackers=len(r['attack']['attackers']))
        row.update({f'f1_{NAME[c]}': last['f1'][c] for c in ELIG})
        if r.get('memory'):
            row['poison_in_memory_end'] = sum(m['poison'] for m in r['memory'] if m)
        if r.get('memory_after_stage2'):
            row['poison_in_memory_e2'] = sum(m['poison'] for m in r['memory_after_stage2'] if m)
        rows.append(row)
        for a in r.get('audits', []):
            for c in a['clients']:
                if c.get('size'):
                    audits.append(dict(seed=r['seed'], arm=r['arm'], method=r['method'], after_stage=a['after_stage'], **{k: v for k, v in c.items() if k != 'tau'}))
    df = pd.DataFrame(rows)
    clean = df[df.arm == 'C'].set_index(['seed', 'method'])
    df['H'] = [.5 * ((r.e3_asr - clean.loc[(r.seed, r.method), 'e3_asr']) + (r.e4_asr - clean.loc[(r.seed, r.method), 'e4_asr'])) for r in df.itertuples()]
    df['delta_recall'] = [r.e2_recall - clean.loc[(r.seed, r.method), 'e2_recall'] for r in df.itertuples()]
    return df, pd.DataFrame(audits)


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float); g = np.random.default_rng(seed)
    m = g.choice(x, (n, len(x))).mean(1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(p):
    p = np.asarray(p); order = np.argsort(p); adj = np.empty(len(p)); run = 0
    for i, k in enumerate(order):
        run = max(run, (len(p) - i) * p[k]); adj[k] = min(1., run)
    return adj


def paired(df, arm, a, b, col):
    x = df[(df.arm == arm) & (df.method == a)].set_index('seed')[col]; y = df[(df.arm == arm) & (df.method == b)].set_index('seed')[col]
    s = sorted(set(x.index) & set(y.index)); return x.loc[s].to_numpy(), y.loc[s].to_numpy()


def main():
    df, au = load()
    assert len(df) == 600 and sorted(df.seed.unique()) == SEEDS, (len(df), sorted(df.seed.unique()))
    df.to_csv(OUT / 'per-run-metrics.csv', index=False); au.to_csv(OUT / 'per-client-audits.csv', index=False)
    rep = {}

    # ---- main tables: mean and 95% bootstrap CI
    summ = []
    for arm, m in itertools.product(ARMS, METHODS):
        d = df[(df.arm == arm) & (df.method == m)]
        row = dict(arm=arm, method=m, n=len(d))
        for col in ('macro_f1', 'accuracy', 'binary_f1', 'H', 'A', 'benign_far', 'forgetting', 'delta_recall', 'e3_asr', 'e4_asr'):
            lo, hi = boot_ci(d[col]); row[col] = d[col].mean(); row[col + '_lo'] = lo; row[col + '_hi'] = hi; row[col + '_sd'] = d[col].std()
        summ.append(row)
    summ = pd.DataFrame(summ); summ.to_csv(OUT / 'summary-mean-ci.csv', index=False)

    # ---- pre-declared confirmatory tests (one-sided paired Wilcoxon, Holm over 6)
    tests = []
    for arm in ('K3', 'K3H', 'K5'):
        x, y = paired(df, arm, 'BRMA', 'NONE', 'H'); tests.append(('T%d' % (len(tests) + 1), f'BRMA < NONE on H at {arm}', x, y, 0.))
    for b in ('RAND', 'LOSS'):
        x, y = paired(df, 'K3H', 'BRMA', b, 'H'); tests.append(('T%d' % (len(tests) + 1), f'BRMA < {b} on H at K3H', x, y, 0.))
    x, y = paired(df, 'C', 'BRMA', 'NONE', 'macro_f1')
    tests.append(('T6', 'BRMA non-inferior to NONE on clean macro-F1 (margin 0.02)', x, y, -0.02))
    rows = []
    for tid, name, x, y, margin in tests:
        d = x - y
        if tid == 'T6':
            stat = wilcoxon(d - margin, alternative='greater', zero_method='wilcox'); est = d.mean()
        else:
            stat = wilcoxon(d, alternative='less', zero_method='wilcox'); est = d.mean()
        lo, hi = boot_ci(d)
        rows.append(dict(test=tid, hypothesis=name, n=len(d), mean_a=x.mean(), mean_b=y.mean(), mean_diff=est, ci_lo=lo, ci_hi=hi,
                         wins=int((d < 0).sum()) if tid != 'T6' else int((d > margin).sum()), p=float(stat.pvalue)))
    t = pd.DataFrame(rows); t['p_holm'] = holm(t.p); t['reject_H0_at_0.05'] = t.p_holm < .05
    t.to_csv(OUT / 'confirmatory-tests.csv', index=False); rep['confirmatory'] = t.round(5).to_dict('records')

    # ---- exploratory: BRMA vs every other defence on H at every attacked arm (Holm within family)
    ex = []
    for arm in ('K1', 'K3', 'K3H', 'K5'):
        for b in ('RAND', 'LOSS', 'FT', 'EI', 'DC', 'DG', 'MEDIAN'):
            x, y = paired(df, arm, 'BRMA', b, 'H'); d = x - y
            p = float(wilcoxon(d, alternative='less').pvalue) if np.any(d != 0) else 1.
            ex.append(dict(arm=arm, other=b, brma=x.mean(), other_mean=y.mean(), diff=d.mean(), wins=int((d < 0).sum()), p=p))
    ex = pd.DataFrame(ex); ex['p_holm'] = holm(ex.p); ex.to_csv(OUT / 'exploratory-H-tests.csv', index=False)
    ex2 = []
    for arm in ARMS:
        x, y = paired(df, arm, 'BRMA', 'NONE', 'macro_f1'); d = x - y; lo, hi = boot_ci(d)
        ex2.append(dict(arm=arm, brma=x.mean(), none=y.mean(), diff=d.mean(), ci_lo=lo, ci_hi=hi,
                        p_two_sided=float(wilcoxon(d).pvalue)))
    pd.DataFrame(ex2).to_csv(OUT / 'macro-f1-brma-vs-none.csv', index=False); rep['macro_f1_brma_vs_none'] = pd.DataFrame(ex2).round(4).to_dict('records')

    # ---- audit quality (BRMA run; detection is computed on the same memory for every branch)
    b = au[au.method == 'BRMA']
    q = b.groupby(['arm', 'after_stage'])[['size', 'benign_labelled', 'poison_before', 'brma_flagged', 'brma_flagged_poison', 'removed_genuine']].sum()
    q['detection_recall'] = q.brma_flagged_poison / q.poison_before.replace(0, np.nan)
    q['precision'] = q.brma_flagged_poison / q.brma_flagged.replace(0, np.nan)
    q['false_removal_rate'] = q.removed_genuine / (q.benign_labelled - q.poison_before)
    q['genuine_removed_per_client_mean'] = q.removed_genuine / b.groupby(['arm', 'after_stage']).size()
    q.to_csv(OUT / 'audit-quality.csv'); rep['audit'] = q.round(4).reset_index().to_dict('records')
    for m in ('RAND', 'LOSS'):
        r = au[au.method == m].groupby(['arm', 'after_stage'])[['removed', 'removed_poison', 'removed_genuine']].sum()
        r.to_csv(OUT / f'audit-{m.lower()}.csv')
    mem = df[df.method.isin(['NONE', 'BRMA', 'RAND', 'LOSS', 'ORACLE'])].groupby(['arm', 'method'])[['poison_in_memory_e2', 'poison_in_memory_end']].mean()
    mem.to_csv(OUT / 'poison-in-memory.csv')

    # ---- overhead: tree suffix time, BRMA vs NONE
    tr = df[df.method.isin(['NONE', 'BRMA'])].pivot_table(index=['seed', 'arm'], columns='method', values='seconds')
    rep['overhead'] = dict(brma_minus_none_seconds_mean=float((tr.BRMA - tr.NONE).mean()), none_seconds_mean=float(tr.NONE.mean()),
                           relative=float(((tr.BRMA - tr.NONE) / tr.NONE).mean()))

    # ---- per-class F1 after stage 4 (failure cases, RQ3)
    cls = [f'f1_{NAME[c]}' for c in ELIG]
    pc = df[df.method.isin(['NONE', 'BRMA', 'ORACLE'])].groupby(['arm', 'method'])[cls].mean().T
    pc.to_csv(OUT / 'per-class-f1.csv')
    dl = []
    for arm in ARMS:
        for c in ELIG:
            x, y = paired(df, arm, 'BRMA', 'NONE', f'f1_{NAME[c]}'); d = x - y
            dl.append(dict(arm=arm, cls=NAME[c], arrival=ARR[c], brma=x.mean(), none=y.mean(), diff=d.mean(), worse_seeds=int((d < -1e-9).sum()),
                           p_two_sided=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.))
    dl = pd.DataFrame(dl); dl.to_csv(OUT / 'per-class-delta.csv', index=False)

    json.dump(rep, open(OUT / 'report.json', 'w'), indent=1, default=float)
    pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
    for col in ('H', 'macro_f1', 'accuracy', 'A', 'benign_far', 'forgetting'):
        print(f'\n=== {col} (mean over 12 seeds)\n', summ.pivot(index='method', columns='arm', values=col).loc[METHODS, ARMS].round(4).to_string())
    print('\n=== confirmatory tests\n', t.drop(columns=['hypothesis']).round(5).to_string(), '\n', t[['test', 'hypothesis']].to_string())
    print('\n=== BRMA vs NONE macro-F1 by arm\n', pd.DataFrame(ex2).round(4).to_string())
    print('\n=== exploratory H tests (BRMA < other)\n', ex.round(5).to_string())
    print('\n=== audit quality (BRMA)\n', q.round(4).to_string())
    print('\n=== poison in memory\n', mem.round(1).unstack().to_string())
    print('\n=== overhead\n', rep['overhead'])
    print('\n=== per-class delta BRMA-NONE (clean and K3H)\n', dl[dl.arm.isin(['C', 'K3H'])].round(4).to_string())


if __name__ == '__main__':
    main()
