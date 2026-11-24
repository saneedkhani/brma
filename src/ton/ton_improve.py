"""BRMA improvement study on TON_IoT (DDoS weakness). Imports ton_study.py unchanged.

Memory actions after stages 2 and 3 (same points as BRMA):
  NONE, BRMA (Gaussian likelihood ratio, original), LOSS (BRMA's budget + loss ranking), ORACLE,
  BRMA-M  model-margin score  m(x) = max_{c != benign} z_c(x) - z_benign(x)  of the current global model,
          conformal threshold at level alpha on the stage-1 benign calibration half;
  BRMA-G  Gaussian-mixture benign model (5 components) against tied Ledoit-Wolf attack Gaussians,
          log-likelihood ratio, conformal threshold at level alpha;
  BRMA-C  union of the original BRMA test and the model-margin test, each at level alpha/2
          (Bonferroni: false-removal rate still <= alpha).
Phase 1 (selection): development seeds 101-103, arms C/K3H/K5, DEV split. Rule (fixed in advance): among BRMA-M/G/C, keep variants whose
  clean false-removal rate <= alpha + 0.01 and clean dev macro-F1 >= NONE - 1 point; choose the lowest mean dev H over K3H and K5.
Phase 2 (evaluation): fresh seeds 201-212, arms C/K1/K3/K3H/K5, TEST split, all methods.
Pre-specified tests (one-sided paired Wilcoxon, Holm over four), S = selected variant:
  T11 S < BRMA on H at K3H;  T12 S < BRMA on H at K5;  T13 S < NONE on H at K3H;  T14 clean macro-F1 of S non-inferior to NONE (margin 2 points).
Usage:  python ton_improve.py run --data DATA_PORTS --out OUT --seeds 101 102 103 --arms C K3H K5
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse, copy, json, sys, time, traceback
import concurrent.futures as futures
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ton_study as T                                    # noqa: E402

TON_CFG = dict(rounds=24, capacity=2000, features='ports')   # selected earlier on development seeds (ton_study tune)
BRANCH = ('NONE', 'BRMA', 'LOSS', 'ORACLE', 'BRMA-M', 'BRMA-G', 'BRMA-C')
VARIANTS = ('BRMA-M', 'BRMA-G', 'BRMA-C')
DEV_SEEDS, EVAL_SEEDS = (101, 102, 103), tuple(range(201, 213))
GMM_K, GMM_FIT = 5, 20000


class Auditor2(T.Auditor):
    """Original BRMA statistics plus the model-margin and mixture scores."""

    def __init__(self, data, seed):
        super().__init__(data, seed)
        x1 = data[0].x
        ben = T.rng(41, seed).permutation(np.flatnonzero(data[0].y == T.BENIGN))
        self.Xcal = np.asarray(x1[ben[len(ben) // 2:]], np.float32)     # raw calibration half (same split as BRMA)
        self.gmm = {}

    # ---- model margin
    @staticmethod
    def margin(model, X, slots):
        b = slots.index(T.BENIGN); model.eval()
        with torch.no_grad():
            z = model(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))).numpy()
        other = np.delete(z, b, axis=1)
        return other.max(1) - z[:, b] if other.shape[1] else np.full(len(X), -np.inf)

    # ---- Gaussian mixture for benign
    def gmm_model(self, t):
        if t in self.gmm:
            return self.gmm[t]
        from sklearn.mixture import GaussianMixture
        m = self.model(t); d = self.Zref.shape[1]
        Zr = self.Zref if len(self.Zref) <= GMM_FIT else self.Zref[T.rng(46, self.seed, t).choice(len(self.Zref), GMM_FIT, replace=False)]
        g = GaussianMixture(GMM_K, covariance_type='full', reg_covar=1e-2, random_state=int(T.rng(47, self.seed, t).integers(1 << 30)), max_iter=200).fit(Zr)
        sign, logdetP = np.linalg.slogdet(m['P'])
        const = 0.5 * logdetP - 0.5 * d * np.log(2 * np.pi)
        g_m = dict(g=g, const=const)
        g_m['tau'] = T.conformal(self.gmm_score_z(self.Zcal, m, g_m), T.ALPHA); self.gmm[t] = g_m
        return g_m

    @staticmethod
    def gmm_score_z(Z, m, g_m):
        if not len(Z):
            return np.zeros(0)
        att = np.max([g_m['const'] - 0.5 * T.maha_sq(Z, mu, m['P']) for mu in m['means'].values()], axis=0)
        return att - g_m['g'].score_samples(Z)

    def flags_variant(self, method, model, X, slots, t):
        if method == 'BRMA-G':
            m = self.model(t); g_m = self.gmm_model(t); return self.gmm_score_z(self.std(X), m, g_m) > g_m['tau']
        tau_m = lambda a: T.conformal(self.margin(model, self.Xcal, slots), a)
        if method == 'BRMA-M':
            return self.margin(model, X, slots) > tau_m(T.ALPHA)
        m = self.model(t)                                                   # BRMA-C: union at alpha/2 each
        tau_g = T.conformal(self.score_z(self.Zcal, m), T.ALPHA / 2)
        return (self.score_z(self.std(X), m) > tau_g) | (self.margin(model, X, slots) > tau_m(T.ALPHA / 2))


def audit(method, model, clients, slots, t, auditor, poison_ids, seed):
    if method not in VARIANTS:
        return T.audit(method, model, clients, slots, t, auditor, poison_ids, seed)
    rows = []
    for k, c in enumerate(clients):
        m = c['replay']
        if m is None or m['size'] == 0:
            rows.append(dict(client=k, size=0)); continue
        n = m['size']; ids = m['ids'][:n]; is_p = np.isin(ids, poison_ids); ben = np.flatnonzero(m['y'][:n] == T.BENIGN)
        flag = auditor.flags_variant(method, model, m['x'][:n][ben], slots, t) if len(ben) else np.zeros(0, bool)
        drop = ben[flag].astype(np.int64)
        rows.append(dict(client=k, size=int(n), benign_labelled=int(len(ben)), poison_before=int(is_p.sum()), brma_flagged=int(flag.sum()),
                         brma_flagged_poison=int(is_p[ben][flag].sum()) if len(ben) else 0, removed=int(len(drop)),
                         removed_poison=int(is_p[drop].sum()), removed_genuine=int(len(drop) - is_p[drop].sum())))
        T.remove_rows(m, drop)
    return rows


def worker(job):
    T.setup()
    folder, out, seed, arm = job['data'], job['out'], job['seed'], job['arm']
    t0 = time.perf_counter()
    data, attack = T.inputs(folder, seed, arm); d = data[0].x.shape[1]
    pid = np.asarray(attack['poison_ids'], np.uint64)
    base = dict(seed=seed, arm=arm, arm_spec=T.ARMS[arm], attack=dict(attackers=attack['attackers'], k=attack['k'], dose=attack['dose'], n_poison=len(pid)),
                alpha=T.ALPHA, dataset='TON_IoT train_test_network (ports)', target=T.CLASSES[T.TARGET], cfg=TON_CFG)
    s = T.Settings(**T.cfg(dict(cfg=TON_CFG)), seed=seed, dose=T.ARMS[arm][1:])
    clients = [T.client_state(s, i, d) for i in range(5)]; slots = []
    model, prefix = T.run_stages(folder, None, clients, slots, data, s, 1, 2)
    c2 = T.census(clients, pid); pre = time.perf_counter() - t0
    auditor = Auditor2(data, seed); keys = []
    for mth in BRANCH:
        if T.done(out, seed, arm, mth):
            continue
        t1 = time.perf_counter()
        bm, bc, bs = copy.deepcopy(model), copy.deepcopy(clients), list(slots)
        audits = [dict(after_stage=2, clients=audit(mth, bm, bc, bs, 2, auditor, pid, seed))]
        bm, st3 = T.run_stages(folder, bm, bc, bs, data, s, 3, 3)
        audits.append(dict(after_stage=3, clients=audit(mth, bm, bc, bs, 3, auditor, pid, seed)))
        bm, st4 = T.run_stages(folder, bm, bc, bs, data, s, 4, 4)
        T.save(out, T.key(seed, arm, mth), dict(base, method=mth, settings=asdict(s), stages=prefix + st3 + st4, memory_after_stage2=c2,
               memory=T.census(bc, pid), audits=audits, elapsed_seconds=pre + time.perf_counter() - t1, shared_prefix_seconds=pre))
        keys.append(T.key(seed, arm, mth))
    return keys


def run(a):
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    (out / 'protocol.json').write_text(json.dumps(dict(cfg=TON_CFG, branch=BRANCH, variants=VARIANTS, seeds=a.seeds, arms=a.arms, alpha=T.ALPHA,
                                                         gmm_components=GMM_K, doc=__doc__), indent=1))
    jobs = [dict(seed=sd, arm=ar, data=str(a.data), out=str(out)) for sd in a.seeds for ar in a.arms
            if not all(T.done(out, sd, ar, m) for m in BRANCH)]
    print(f'{len(jobs)} jobs pending', flush=True); t0 = time.time(); fin = 0
    with futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        pend = {pool.submit(worker, j): j for j in jobs}
        for f in futures.as_completed(pend):
            j = pend[f]
            try:
                ks = f.result()
            except Exception:
                (out / f"failure-s{j['seed']}-{j['arm']}.txt").write_text(traceback.format_exc()); print('FAILED', j['seed'], j['arm'], flush=True); continue
            fin += 1; el = time.time() - t0
            print(f"[{fin}/{len(jobs)}] s{j['seed']} {j['arm']} -> {len(ks)} results | {el / 60:.1f} min elapsed, ~{el / fin * (len(jobs) - fin) / 60:.0f} min left", flush=True)


def table(out, split):
    """Per-run metrics (macro-F1, H, false removal, detection) for one split."""
    import pandas as pd
    rows = []
    for p in sorted(Path(out).glob('runs/*/result.json')):
        r = json.loads(p.read_text()); row = T.result_row(r)
        a2 = [c for c in r['audits'][0]['clients'] if c.get('size')] if r['audits'] else []
        ben = sum(c.get('benign_labelled', 0) for c in a2); pb = sum(c.get('poison_before', 0) for c in a2)
        rp = sum(c.get('removed_poison', 0) for c in a2); rg = sum(c.get('removed_genuine', 0) for c in a2)
        rows.append(dict(seed=r['seed'], arm=r['arm'], method=r['method'], macro_f1=row[f'{split}_macro_f1'], accuracy=row[f'{split}_accuracy'],
                         e3=row[f'{split}_e3_asr'], e4=row[f'{split}_e4_asr'], poison=pb, removed_poison=rp,
                         false_removal=rg / max(1, ben - pb), poison_end=sum(m['poison'] for m in r['memory'] if m)))
    df = pd.DataFrame(rows); cl = df[df.arm == 'C'].set_index(['seed', 'method'])
    df['H'] = [.5 * ((x.e3 - cl.loc[(x.seed, x.method), 'e3']) + (x.e4 - cl.loc[(x.seed, x.method), 'e4'])) for x in df.itertuples()]
    df['detection'] = df.removed_poison / df.poison.replace(0, np.nan)
    return df


def select(a):
    df = table(a.out, 'dev'); df.to_csv(Path(a.out) / 'dev-metrics.csv', index=False)
    import pandas as pd
    pd.set_option('display.width', 200)
    g = df.groupby(['method', 'arm'])[['H', 'macro_f1', 'false_removal', 'detection']].mean().unstack('arm')
    print('Phase 1 (development seeds, dev split): mean over seeds'); print(g.round(4).to_string())
    clean = df[df.arm == 'C'].groupby('method')[['macro_f1', 'false_removal']].mean()
    hatt = df[df.arm.isin(['K3H', 'K5'])].groupby('method').H.mean()
    ok = [v for v in VARIANTS if clean.loc[v, 'false_removal'] <= T.ALPHA + 0.01 and clean.loc[v, 'macro_f1'] >= clean.loc['NONE', 'macro_f1'] - 0.01]
    best = min(ok, key=lambda v: hatt[v]) if ok else 'BRMA-C'
    print('\neligible variants:', ok, '| mean dev H (K3H, K5):', {v: round(float(hatt[v]), 4) for v in VARIANTS}, '| BRMA:', round(float(hatt['BRMA']), 4))
    print('SELECTED:', best)
    (Path(a.out) / 'selected.json').write_text(json.dumps(dict(selected=best, eligible=ok, mean_dev_H={k: float(v) for k, v in hatt.items()},
                                                              rule='eligible: clean false removal <= alpha+0.01 and clean dev macro-F1 >= NONE-1pt; choose lowest mean dev H over K3H,K5')))


def evaluate_tests(a):
    from scipy.stats import wilcoxon
    import pandas as pd
    S = json.loads(Path(a.sel).read_text())['selected']
    df = table(a.out, 'test'); df.to_csv(Path(a.out) / 'test-metrics.csv', index=False)
    pd.set_option('display.width', 200)
    order = ['NONE', 'BRMA', 'LOSS', 'ORACLE', 'BRMA-M', 'BRMA-G', 'BRMA-C']; arms = [x for x in ['C', 'K1', 'K3', 'K3H', 'K5'] if x in set(df.arm)]
    for col, sc in (('H', 1), ('macro_f1', 100), ('accuracy', 100), ('detection', 100), ('false_removal', 100)):
        print(f'\n{col} (test split, mean over seeds)'); print((sc * df.pivot_table(index='method', columns='arm', values=col)).reindex(index=order, columns=arms).round(4).to_string())
    def pair(arm, x, y, col):
        u = df[(df.arm == arm) & (df.method == x)].set_index('seed')[col]; v = df[(df.arm == arm) & (df.method == y)].set_index('seed')[col].loc[u.index]
        return u.to_numpy(), v.to_numpy()
    res = []
    for tid, arm, other in (('T11', 'K3H', 'BRMA'), ('T12', 'K5', 'BRMA'), ('T13', 'K3H', 'NONE')):
        u, v = pair(arm, S, other, 'H'); d = u - v
        res.append(dict(test=tid, hypothesis=f'{S} < {other} on H at {arm}', selected=u.mean(), other=v.mean(), wins=int((d < 0).sum()),
                        p=float(wilcoxon(d, alternative='less').pvalue) if (d != 0).any() else 1.))
    u, v = pair('C', S, 'NONE', 'macro_f1'); d = u - v + 0.02
    res.append(dict(test='T14', hypothesis=f'{S} clean macro-F1 non-inferior to NONE (2 points)', selected=u.mean(), other=v.mean(), wins=int((d > 0).sum()),
                    p=float(wilcoxon(d, alternative='greater').pvalue) if (d != 0).any() else 1.))
    r = pd.DataFrame(res).sort_values('p'); k = len(r); mx = 0; adj = []
    for i, p in enumerate(r.p):
        mx = max(mx, (k - i) * p); adj.append(min(1., mx))
    r['p_holm'] = adj; r = r.sort_index(); r['confirmed'] = r.p_holm < .05
    print(f'\nPre-specified tests (selected variant: {S})'); print(r.round(5).to_string(index=False)); r.to_csv(Path(a.out) / 'tests.csv', index=False)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--data', required=True); r.add_argument('--out', required=True)
    r.add_argument('--seeds', type=int, nargs='+', required=True); r.add_argument('--arms', nargs='+', required=True); r.add_argument('--workers', type=int, default=4)
    s = sub.add_parser('select'); s.add_argument('--out', required=True)
    e = sub.add_parser('evaluate'); e.add_argument('--out', required=True); e.add_argument('--sel', required=True)
    a = ap.parse_args()
    {'run': run, 'select': select, 'evaluate': evaluate_tests}[a.cmd](a)


if __name__ == '__main__':
    main()
