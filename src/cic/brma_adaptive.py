"""BRMA adaptive-attack study (extension of brma_study.py; same audited engine, same 12 seeds, CIC-IDS2017).

Adds, without changing brma_study.py:
  * two adaptive attacks by attackers who know BRMA:
      sel - each attacker scores its own DoS Hulk records with its own copy of the BRMA likelihood ratio
            (tied Gaussian, Ledoit-Wolf, fitted on its own genuine benign and Hulk records) and flips the
            floor(rho * n) records that look MOST benign;
      mix - the usual random nested flip, plus feature camouflage: each flipped record is moved a fraction
            GAMMA of the way towards the attacker's own benign mean (x' = x + GAMMA (m_benign - x));
  * a self-sufficient baseline LOSS5 that removes a FIXED 5% of benign-labelled memory with the highest
    loss under the current model (no BRMA information).
Pre-declared tests (Holm over four, one-sided paired Wilcoxon over 12 seeds):
  T7 BRMA < NONE on H at S3H;  T8 BRMA < NONE on H at M3H;  T9 BRMA < LOSS5 on H at K3H;  T10 BRMA < LOSS5 on H at S3H.

Usage (from the repository root, next to brma_study.py):  python brma_adaptive.py --out OUT --workers 4
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
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import brma_study as B                                   # noqa: E402  (frozen main-study code, imported unchanged)
from src.phase3.common import rng                       # noqa: E402

ARMS = {                    # name: (attackers k, dose numerator, denominator, attack mode)
    'C':   (0, 0, 1, 'std'),
    'K3H': (3, 9, 10, 'std'),
    'K5':  (5, 1, 2, 'std'),
    'S3H': (3, 9, 10, 'sel'),
    'S5':  (5, 1, 2, 'sel'),
    'M3H': (3, 9, 10, 'mix'),
    'M5':  (5, 1, 2, 'mix'),
}
BRANCH_METHODS = ('NONE', 'BRMA', 'RAND', 'LOSS', 'ORACLE', 'LOSS5')
FULL_METHODS = ('DC', 'MEDIAN')
GAMMA, FIXED_FRACTION, HULK, BENIGN = 0.5, 0.05, B.HULK, B.BENIGN


# ---------------------------------------------------------------- attacker-side BRMA copy
def attacker_score(Xb, Xh, X, seed, a):
    """The attacker's own likelihood ratio log p(x|Hulk) - log p(x|benign) with a tied Ledoit-Wolf covariance."""
    from sklearn.covariance import LedoitWolf
    cap = 50000
    if len(Xb) > cap:
        Xb = Xb[rng(45, seed, a).choice(len(Xb), cap, replace=False)]
    m, s = Xb.mean(0), Xb.std(0) + 1e-6
    Zb, Zh, Z = (np.asarray(Xb, np.float64) - m) / s, (np.asarray(Xh, np.float64) - m) / s, (np.asarray(X, np.float64) - m) / s
    mb, mh = Zb.mean(0), Zh.mean(0)
    P = LedoitWolf(assume_centered=True).fit(np.concatenate([Zb - mb, Zh - mh])).precision_
    d = lambda Q, mu: np.einsum('ij,ij->i', (Q - mu) @ P, Q - mu)
    return 0.5 * (d(Z, mb) - d(Z, mh))                  # high = looks like Hulk, low = looks benign


def corrupt_adaptive(clean, x, ids, owner, seed, stage, k, num, den, mode, benign_pool):
    """Returns supplied labels, possibly modified features (stage 2 copy) and attack info.
    benign_pool[a] = the attacker's genuine benign records (stages 1-2) in raw feature space."""
    if mode == 'std' or stage != 2 or k == 0 or num == 0:
        y, info = B.corrupt_multi(clean, ids, owner, seed, stage, k, num, den)
        info['mode'] = mode
        return y, x, info
    supplied = clean.copy(); xs = np.array(x, dtype=np.float32, copy=True)
    counts = np.bincount(owner[clean == HULK], minlength=5)
    eligible = np.flatnonzero(counts >= 10)
    first = int(rng(2, seed).choice(eligible))
    rest = [int(c) for c in rng(12, seed).permutation(eligible) if c != first]
    attackers = [first] + rest[:k - 1]
    poison, moved = [], 0
    for a in attackers:
        source = np.flatnonzero((owner == a) & (clean == HULK))
        source = source[np.argsort(ids[source], kind='stable')]
        q = num * len(source) // den
        if mode == 'sel':
            sc = attacker_score(benign_pool[a], xs[source], xs[source], seed, a)
            chosen = source[np.lexsort((ids[source], sc))[:q]]          # most benign-looking first
        else:                                                           # 'mix': same nested random order as std
            chosen = rng(3, seed, a).permutation(source)[:q]
            mb = np.asarray(benign_pool[a], np.float64).mean(0)
            xs[chosen] = (xs[chosen] + GAMMA * (mb - xs[chosen])).astype(np.float32); moved += len(chosen)
        supplied[chosen] = BENIGN; poison.extend(str(int(v)) for v in ids[chosen])
    return supplied, xs, dict(stage=stage, k=k, dose=[num, den], attackers=attackers, poison_ids=poison, mode=mode, moved=moved)


def inputs(seed, arm, out):
    """As brma_study.inputs, with adaptive attacks."""
    with np.load(Path(out) / 'ownership-source.npz', allow_pickle=False) as f:
        source_ids, components, labels = f['ids'], f['components'], f['labels']
    owners, _ = B.allocate(components, labels, seed)
    order = np.argsort(source_ids, kind='stable')
    k, num, den, mode = ARMS[arm]
    raw = []
    for stage in range(1, 5):
        folder = B.BASE / f'train/E{stage}'
        x = np.load(folder / 'x.npy', mmap_mode='r', allow_pickle=False)
        ids = np.load(folder / 'ids.npy', allow_pickle=False)
        clean = np.load(folder / 'y.npy', allow_pickle=False)
        index = order[np.searchsorted(source_ids[order], ids)]
        assert np.array_equal(source_ids[index], ids) and np.array_equal(labels[index], clean)
        raw.append((stage, x, ids, clean, owners[index].astype('i1')))
    benign_pool = {}
    if mode != 'std':
        for a in range(5):
            parts = [np.asarray(x)[(o == a) & (y == BENIGN)] for (s, x, ids, y, o) in raw[:2]]
            pool = np.concatenate(parts)
            if len(pool) < 20:                                          # attacker without benign traffic: use all its stage-2 non-Hulk rows
                s, x, ids, y, o = raw[1]; pool = np.asarray(x)[(o == a) & (y != HULK)]
            benign_pool[a] = pool
    data, attack = [], None
    for stage, x, ids, clean, owner in raw:
        y, xs, info = corrupt_adaptive(clean, x, ids, owner, seed, stage, k, num, den, mode, benign_pool)
        if stage == 2:
            attack = info
        data.append(B.StageInput(stage, xs, y, ids, owner))
    return data, attack


# ---------------------------------------------------------------- memory actions (adds LOSS5)
def audit(method, model, clients, slots, t, auditor, poison_ids, seed):
    if method != 'LOSS5':
        return B.audit(method, model, clients, slots, t, auditor, poison_ids, seed)
    rows = []; bslot = slots.index(BENIGN)
    for k, c in enumerate(clients):
        m = c.replay
        if m is None or m['size'] == 0:
            rows.append(dict(client=k, size=0)); continue
        n = m['size']; y, ids = m['y'][:n], m['ids'][:n]
        is_p = np.isin(ids, poison_ids); ben = np.flatnonzero(y == BENIGN)
        budget = int(np.floor(FIXED_FRACTION * len(ben) + 0.5))
        drop = np.zeros(0, np.int64)
        if budget:
            model.eval()
            with torch.no_grad():
                loss = F.cross_entropy(model(torch.from_numpy(m['x'][:n][ben].copy())), torch.full((len(ben),), bslot, dtype=torch.int64),
                                       reduction='none').numpy()
            drop = ben[np.lexsort((ids[ben], -loss))[:budget]].astype(np.int64)
        rows.append(dict(client=k, size=int(n), benign_labelled=int(len(ben)), poison_before=int(is_p.sum()), budget=budget,
                         removed=int(len(drop)), removed_poison=int(is_p[drop].sum()), removed_genuine=int(len(drop) - is_p[drop].sum())))
        B.remove_rows(m, drop)
    return rows


def brma_detection(clients, t, auditor, poison_ids):
    """BRMA detection on the memory as it stands (recorded for every branch, no removal)."""
    out = []
    for c in clients:
        m = c.replay
        if m is None or m['size'] == 0:
            continue
        n = m['size']; ben = np.flatnonzero(m['y'][:n] == BENIGN)
        if not len(ben):
            continue
        flag, _, _ = auditor.flags(m['x'][:n][ben], t); is_p = np.isin(m['ids'][:n][ben], poison_ids)
        out.append(dict(benign_labelled=int(len(ben)), poison=int(is_p.sum()), flagged=int(flag.sum()), flagged_poison=int((flag & is_p).sum())))
    return out


# ---------------------------------------------------------------- jobs
def worker(job):
    B.setup()
    kind, seed, arm, out = job['kind'], job['seed'], job['arm'], Path(job['out'])
    t0 = time.perf_counter()
    data, attack = inputs(seed, arm, out)
    poison_ids = np.asarray([int(v) for v in attack['poison_ids']], dtype=np.uint64)
    base = dict(seed=seed, arm=arm, arm_spec=ARMS[arm], attack=dict(attackers=attack['attackers'], k=attack['k'], dose=attack['dose'], mode=attack.get('mode'),
                n_poison=len(poison_ids), moved=attack.get('moved', 0)), alpha=B.ALPHA, gamma=GAMMA, fixed_fraction=FIXED_FRACTION, development_bank_only=True)
    if kind == 'full':
        method = job['method']
        s = B.Settings(seed=seed, dose=ARMS[arm][1:3], admission=method if method in ('DC', 'DG') else 'NONE',
                       aggregation='median' if method == 'MEDIAN' else 'fedavg')
        clients = [B.L.client_state(s, i) for i in range(5)]
        model, stages = B.run_stages(None, clients, [], data, s, 1, 4)
        B.save(out, B.job_key(seed, arm, method), dict(base, method=method, settings=asdict(s), stages=stages,
               memory=B.memory_census(clients, poison_ids), audits=[], elapsed_seconds=time.perf_counter() - t0), model, stages[-1]['slots'])
        return [B.job_key(seed, arm, method)]
    s = B.Settings(seed=seed, dose=ARMS[arm][1:3])
    clients = [B.L.client_state(s, i) for i in range(5)]; slots = []
    model, prefix = B.run_stages(None, clients, slots, data, s, 1, 2)
    census2 = B.memory_census(clients, poison_ids); prefix_seconds = time.perf_counter() - t0
    auditor = B.Auditor(data, seed); keys = []
    for method in BRANCH_METHODS:
        if B.done(out, seed, arm, method):
            continue
        t1 = time.perf_counter()
        bm, bc, bs = copy.deepcopy(model), copy.deepcopy(clients), list(slots)
        det2 = brma_detection(bc, 2, auditor, poison_ids)
        audits = [dict(after_stage=2, clients=audit(method, bm, bc, bs, 2, auditor, poison_ids, seed), brma_detection=det2)]
        bm, st3 = B.run_stages(bm, bc, bs, data, s, 3, 3)
        det3 = brma_detection(bc, 3, auditor, poison_ids)
        audits.append(dict(after_stage=3, clients=audit(method, bm, bc, bs, 3, auditor, poison_ids, seed), brma_detection=det3))
        bm, st4 = B.run_stages(bm, bc, bs, data, s, 4, 4)
        B.save(out, B.job_key(seed, arm, method), dict(base, method=method, settings=asdict(s), stages=prefix + st3 + st4,
               memory_after_stage2=census2, memory=B.memory_census(bc, poison_ids), audits=audits,
               elapsed_seconds=prefix_seconds + time.perf_counter() - t1, shared_prefix_seconds=prefix_seconds), bm, bs)
        keys.append(B.job_key(seed, arm, method))
    return keys


def build_jobs(out, seeds):
    jobs = []
    for seed in seeds:
        for arm in ARMS:
            if not all(B.done(out, seed, arm, m) for m in BRANCH_METHODS):
                jobs.append(dict(kind='tree', seed=seed, arm=arm, out=str(out)))
            for m in FULL_METHODS:
                if not B.done(out, seed, arm, m):
                    jobs.append(dict(kind='full', seed=seed, arm=arm, method=m, out=str(out)))
    jobs.sort(key=lambda j: (seeds.index(j['seed']), j['kind'] != 'tree'))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--seeds', type=int, nargs='+', default=list(B.SEEDS))
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args(); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if not (out / 'ownership-source.npz').exists():
        with np.load(B.BASE / 'builder/ownership.npz', allow_pickle=False) as f:
            positions, components = f['source_positions'], f['canonical_components']
        with np.load(B.ROOT / 'data/raw/phase3-feasibility-corrected/no-headers-74/conditional-components-120.npz', allow_pickle=False) as f:
            ids, labels = f['row_ids'][positions], f['labels'][positions].astype(np.int64)
        np.savez_compressed(out / 'ownership-source.npz', ids=ids, components=components, labels=labels)
    (out / 'protocol.json').write_text(json.dumps(dict(arms=ARMS, branch_methods=BRANCH_METHODS, full_methods=FULL_METHODS, alpha=B.ALPHA,
        gamma=GAMMA, fixed_fraction=FIXED_FRACTION, seeds=a.seeds, tests=['T7 BRMA<NONE H S3H', 'T8 BRMA<NONE H M3H', 'T9 BRMA<LOSS5 H K3H',
        'T10 BRMA<LOSS5 H S3H'], engine='brma_study.py engine unchanged'), indent=1))
    jobs = build_jobs(out, a.seeds); t0 = time.time(); fin = 0
    print(f'{len(jobs)} jobs pending for seeds {a.seeds}', flush=True)
    with futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        pend = {pool.submit(worker, j): j for j in jobs}
        for f in futures.as_completed(pend):
            j = pend[f]
            try:
                keys = f.result()
            except Exception:
                (out / f"failure-{j['kind']}-s{j['seed']}-{j['arm']}-{j.get('method', 'tree')}.txt").write_text(traceback.format_exc())
                print('FAILED', j, flush=True); continue
            fin += 1; el = time.time() - t0
            print(f"[{fin}/{len(jobs)}] {j['kind']} s{j['seed']} {j['arm']} {j.get('method', '')} -> {len(keys)} results | "
                  f"{el / 60:.1f} min elapsed, ~{el / fin * (len(jobs) - fin) / 60:.0f} min left", flush=True)
    print('complete' if not build_jobs(out, a.seeds) else 'incomplete: re-run to resume', flush=True)


if __name__ == '__main__':
    main()
