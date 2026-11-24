"""BRMA full study: Benign-Reference Memory Audit for federated class-incremental intrusion detection.

Runs on the audited Phase 3 CPU engine of the project repository (src/phase3, src/phase3_v2; the
'core' study of results/phase3-rigorous-audit-v1, 12 ownership seeds 1-11 and 42).  Engine code is
imported unchanged; this file adds (i) multi-client label-flip attacks, (ii) end-of-stage memory
interventions (BRMA and matched baselines), (iii) a coordinate-wise median aggregation baseline.
Everything is CPU, single-thread, deterministic; evaluation uses the development bank only.

Usage (from the repository root):  python brma_study.py --out OUT --seeds 1 2 3 4 --workers 4
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse, copy, json, sys, time, traceback
import concurrent.futures as futures
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(os.environ.get('BRMA_ROOT', Path.cwd())).resolve()
sys.path.insert(0, str(ROOT))
from src.phase3.common import rng                      # noqa: E402
from src.phase3.prepare import allocate               # noqa: E402
from src.phase3.model import ReferenceMLP             # noqa: E402
from src.phase3.reference import snapshot, model_identity  # noqa: E402
from src.phase3_v2 import learning as L               # noqa: E402
from src.phase3_v2.data import StageInput             # noqa: E402
from src.phase3_v2.prepare import corrupt             # noqa: E402

BASE = ROOT / 'data/processed/phase3-reference74-v1'
SEEDS = tuple(range(1, 12)) + (42,)
CAPACITY, BENIGN, HULK = 500, 0, 4

# ---------------------------------------------------------------- pre-declared protocol
ARMS = {                     # name: (number of attacking clients k, dose numerator, denominator)
    'C':   (0, 0, 1),        # clean
    'K1':  (1, 1, 2),        # one attacker, 50% of its Hulk rows (identical to the audited study's dose 0.5)
    'K3':  (3, 1, 2),        # three coordinated attackers, 50%
    'K3H': (3, 9, 10),       # three coordinated attackers, 90%
    'K5':  (5, 1, 2),        # all five clients, 50%
}
BRANCH_METHODS = ('NONE', 'BRMA', 'RAND', 'LOSS', 'ORACLE')   # share stages 1-2, differ by end-of-stage memory action
FULL_METHODS = ('FT', 'EI', 'DC', 'DG', 'MEDIAN')               # separate full trajectories
ALPHA = 0.05                 # conformal level, fixed by pilots 1-3 on development seeds 101-103
FIT_CAP, POOL_CAP, MIN_ROWS = 50000, 200000, 20


@dataclass(frozen=True)
class Settings:              # field names follow src/phase3_v2 so the engine reads them unchanged
    learner: str = 'DR'
    admission: str = 'NONE'
    seed: int = 1
    rounds: int = 12
    epochs: int = 1
    learning_rate: float = .001
    lambda_E: float = 100.
    lambda_y: float = 1.
    lambda_z: float = .1
    q: float = .75
    eviction_policy: str = 'reservoir'
    aggregation: str = 'fedavg'
    dose: tuple = (0, 1)


def setup():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')


# ---------------------------------------------------------------- engine patches (behaviour-preserving)
def insert_refill(client, x, labels, ids, teachers, width):
    """Reservoir insertion that first fills slots freed by an audit.  Identical to
    src.phase3.reference.insert_batch whenever no record was removed (size == min(500, seen))."""
    for i in range(len(labels)):
        client['seen'] += 1
        seen = client['seen']
        if client['size'] < min(CAPACITY, seen - 1):          # a freed slot exists
            slot = client['size']; client['size'] += 1
        else:
            slot = seen - 1 if seen <= CAPACITY else int(client['insert'].integers(0, seen))
            client['size'] = min(CAPACITY, seen)
        if slot < CAPACITY:
            client['x'][slot] = x[i]; client['y'][slot] = labels[i]; client['ids'][slot] = ids[i]
            client['z'][slot].fill(0); client['z'][slot, :width] = teachers[i]; client['width'][slot] = width


L.insert_batch = insert_refill
_FEDAVG = L.aggregate


def median_aggregate(broadcast, updates, counts):
    """Coordinate-wise median of the active client models (robust-aggregation baseline)."""
    active = [u for u, n in zip(updates, counts) if n]
    if len(active) < 3:
        return _FEDAVG(broadcast, updates, counts)
    combined = {k: torch.quantile(torch.stack([u[k] for u in active]), .5, dim=0) for k in broadcast}
    return combined, [1. / len(active) if n else 0. for n in counts]


# ---------------------------------------------------------------- data and attacks
def corrupt_multi(clean, ids, owner, seed, stage, k, num, den):
    """k-client nested label flip of DoS Hulk to benign in stage 2.  The first attacker and its row order
    use the audited single-attacker streams (rng 2 and rng 3), so k = 1 reproduces src.phase3_v2.prepare.corrupt."""
    supplied = clean.copy()
    info = dict(stage=stage, k=k, dose=[num, den], attackers=[], poison_ids=[])
    if stage != 2 or k == 0 or num == 0:
        return supplied, info
    if k == 1 and (num, den) in {(1, 10), (3, 10), (1, 2)}:
        supplied, a = corrupt(clean, ids, owner, seed, stage, (num, den))
        info.update(attackers=[a['attacked_client']], poison_ids=a['poison_ids'])
        return supplied, info
    counts = np.bincount(owner[clean == HULK], minlength=5)
    eligible = np.flatnonzero(counts >= 10)
    first = int(rng(2, seed).choice(eligible))
    rest = [int(c) for c in rng(12, seed).permutation(eligible) if c != first]
    attackers = [first] + rest[:k - 1]
    poison = []
    for a in attackers:
        source = np.flatnonzero((owner == a) & (clean == HULK))
        source = source[np.argsort(ids[source], kind='stable')]
        prefix = rng(3, seed, a).permutation(source)
        chosen = prefix[:num * len(source) // den]
        supplied[chosen] = BENIGN; poison.extend(str(int(v)) for v in ids[chosen])
    info.update(attackers=attackers, poison_ids=poison)
    return supplied, info


def inputs(seed, arm, out):
    with np.load(out / 'ownership-source.npz', allow_pickle=False) as f:
        source_ids, components, labels = f['ids'], f['components'], f['labels']
    owners, _ = allocate(components, labels, seed)
    order = np.argsort(source_ids, kind='stable')
    k, num, den = ARMS[arm]
    data, attack = [], None
    for stage in range(1, 5):
        folder = BASE / f'train/E{stage}'
        x = np.load(folder / 'x.npy', mmap_mode='r', allow_pickle=False)
        ids = np.load(folder / 'ids.npy', allow_pickle=False)
        clean = np.load(folder / 'y.npy', allow_pickle=False)
        index = order[np.searchsorted(source_ids[order], ids)]
        assert np.array_equal(source_ids[index], ids) and np.array_equal(labels[index], clean)
        owner = owners[index].astype('i1')
        y, info = corrupt_multi(clean, ids, owner, seed, stage, k, num, den)
        if stage == 2:
            attack = info
        data.append(StageInput(stage, x, y, ids, owner))
    return data, attack


def evaluate(model, slots, stage):
    """Frozen development bank only (as in the audited study)."""
    folder = BASE / 'development'
    x = np.load(folder / 'x.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(folder / 'y.npy', mmap_mode='r', allow_pickle=False)
    keep = np.flatnonzero(np.isin(y, slots))
    matrix = np.zeros((16, 16), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(keep), 4096):
            ix = keep[start:start + 4096]
            pred = np.asarray(slots)[model(torch.from_numpy(x[ix].copy())).argmax(1).numpy()]
            np.add.at(matrix, (y[ix], pred), 1)
    return dict(stage=stage, slots=list(slots), confusion=matrix.tolist(), model_sha256=model_identity(model, slots))


# ---------------------------------------------------------------- BRMA: server statistics and audit
def ledoit_wolf_precision(X):
    from sklearn.covariance import LedoitWolf
    return LedoitWolf(assume_centered=True).fit(X).precision_


def maha_sq(Z, mu, P, chunk=65536):
    out = np.empty(len(Z))
    for i in range(0, len(Z), chunk):
        D = Z[i:i + chunk] - mu; out[i:i + chunk] = np.einsum('ij,ij->i', D @ P, D)
    return out


class Auditor:
    """Server-side Gaussian statistics.  Benign: stage-1 rows only (before any attack), split into a
    reference half (statistics) and a calibration half (conformal threshold).  Attack classes: rows the
    clients label as that class in stages <= t (supplied labels; flipped rows are labelled benign and
    therefore never enter attack statistics).  Score = 0.5 (d_benign^2 - min_c d_c^2) under one shared
    Ledoit-Wolf covariance (tied-covariance likelihood ratio)."""

    def __init__(self, data, seed):
        self.data, self.seed = data, seed
        x1 = np.asarray(data[0].x)
        self.mu0, self.sd0 = x1.mean(0), x1.std(0) + 1e-6
        ben = rng(41, seed).permutation(np.flatnonzero(data[0].y == BENIGN))
        self.Zref = self.std(x1[ben[: len(ben) // 2]]); self.Zcal = self.std(x1[ben[len(ben) // 2:]])
        self.models = {}

    def std(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mu0) / self.sd0

    def model(self, t):
        if t in self.models:
            return self.models[t]
        means, centred = {}, [self.Zref - self.Zref.mean(0)]
        for c in range(1, 16):
            parts = [np.asarray(self.data[s].x)[self.data[s].y == c] for s in range(t)]
            X = np.concatenate(parts) if parts else np.zeros((0, 74))
            if len(X) < MIN_ROWS:
                continue
            if len(X) > FIT_CAP:
                X = X[rng(42, self.seed, t, c).choice(len(X), FIT_CAP, replace=False)]
            Z = self.std(X); means[c] = Z.mean(0); centred.append(Z - means[c])
        pooled = np.concatenate(centred)
        if len(pooled) > POOL_CAP:
            pooled = pooled[rng(43, self.seed, t).choice(len(pooled), POOL_CAP, replace=False)]
        P = ledoit_wolf_precision(pooled)
        m = dict(P=P, mub=self.Zref.mean(0), means=means)
        m['tau'] = conformal(self.score_z(self.Zcal, m), ALPHA)
        self.models[t] = m
        return m

    def score_z(self, Z, m):
        if not len(Z):
            return np.zeros(0)
        da = np.min([maha_sq(Z, mu, m['P']) for mu in m['means'].values()], axis=0)
        return 0.5 * (maha_sq(Z, m['mub'], m['P']) - da)

    def flags(self, X, t):
        m = self.model(t); s = self.score_z(self.std(X), m)
        return s > m['tau'], s, m['tau']


def conformal(cal, alpha):
    n = len(cal); return float(np.quantile(cal, min(1., np.ceil((n + 1) * (1 - alpha)) / n), method='higher'))


def remove_rows(memory, drop):
    keep = np.setdiff1d(np.arange(memory['size']), drop)
    for key in ('x', 'y', 'ids', 'z', 'width'):
        memory[key][:len(keep)] = memory[key][keep]
    memory['size'] = len(keep)


def audit(method, model, clients, slots, t, auditor, poison_ids, seed):
    """End-of-stage memory action.  RAND and LOSS remove exactly as many benign-labelled records per
    client as BRMA flags on the same memory (matched removal budget)."""
    rows = []
    benign_slot = slots.index(BENIGN)
    for k, c in enumerate(clients):
        m = c.replay
        if m is None or m['size'] == 0:
            rows.append(dict(client=k, size=0)); continue
        n = m['size']; y, ids = m['y'][:n], m['ids'][:n]
        is_poison = np.isin(ids, poison_ids); ben = np.flatnonzero(y == BENIGN)
        flag, _, tau = auditor.flags(m['x'][:n][ben], t) if len(ben) else (np.zeros(0, bool), None, None)
        budget = int(flag.sum())
        if method == 'BRMA':
            drop = ben[flag]
        elif method == 'RAND':
            drop = rng(44, seed, t, k).choice(ben, budget, replace=False) if budget else np.zeros(0, int)
        elif method == 'LOSS':
            model.eval()
            with torch.no_grad():
                loss = F.cross_entropy(model(torch.from_numpy(m['x'][:n][ben].copy())),
                                       torch.full((len(ben),), benign_slot, dtype=torch.int64), reduction='none').numpy()
            drop = ben[np.lexsort((ids[ben], -loss))[:budget]]
        elif method == 'ORACLE':
            drop = np.flatnonzero(is_poison)
        else:
            drop = np.zeros(0, int)
        drop = np.asarray(drop, dtype=np.int64)
        rows.append(dict(client=k, size=int(n), benign_labelled=int(len(ben)), poison_before=int(is_poison.sum()),
                         brma_flagged=budget, brma_flagged_poison=int(is_poison[ben][flag].sum()) if len(ben) else 0,
                         removed=int(len(drop)), removed_poison=int(is_poison[drop].sum()),
                         removed_genuine=int(len(drop) - is_poison[drop].sum()), tau=tau))
        remove_rows(m, drop)
    return rows


# ---------------------------------------------------------------- training
def run_stages(model, clients, slots, data, s, start, end, folder_tag=None):
    stages = []
    L.aggregate = median_aggregate if s.aggregation == 'median' else _FEDAVG
    for stage in range(start, end + 1):
        d = data[stage - 1]
        new_codes = [int(c) for c in np.unique(d.y) if int(c) not in slots]
        slots.extend(new_codes)
        if model is None:
            model = ReferenceMLP(len(slots), s.seed)
        elif new_codes:
            model.expand(len(slots), stage, s.seed)
        if s.learner == 'EI':
            for c in clients:
                L.expand_importance(c, model)
        L.admit(model, d, slots, set(new_codes), clients, s, [])
        for r in range(1, s.rounds + 1):
            broadcast = snapshot(model); updates = []
            for i, c in enumerate(clients):
                update, _ = L.train_client(broadcast, model, c, d, slots, s, r, i); updates.append(update)
            combined, _ = L.aggregate(broadcast, updates, [len(c.accepted) for c in clients])
            model.load_state_dict(combined)
        if s.learner == 'EI':
            L.commit_importance(model, clients, d, slots, s, [])
        stages.append(evaluate(model, slots, stage))
    return model, stages


def memory_census(clients, poison_ids):
    out = []
    for c in clients:
        m = c.replay
        if m is None:
            out.append(None); continue
        n = m['size']; p = np.isin(m['ids'][:n], poison_ids)
        out.append(dict(size=int(n), poison=int(p.sum()), benign_labelled=int((m['y'][:n] == BENIGN).sum())))
    return out


def save(out, key, payload, model, slots):
    folder = out / 'runs' / key; folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'E4-weights.npz', **{k: v.numpy() for k, v in snapshot(model).items()}, source_codes=np.asarray(slots))
    tmp = folder / 'result.json.tmp'; tmp.write_text(json.dumps(payload)); tmp.replace(folder / 'result.json')


def job_key(seed, arm, method):
    return f's{seed}-{arm}-{method}'


def done(out, seed, arm, method):
    return (out / 'runs' / job_key(seed, arm, method) / 'result.json').exists()


def worker(job):
    setup()
    kind, seed, arm, out = job['kind'], job['seed'], job['arm'], Path(job['out'])
    t0 = time.perf_counter()
    data, attack = inputs(seed, arm, out)
    poison_ids = np.asarray([int(v) for v in attack['poison_ids']], dtype=np.uint64)
    base = dict(seed=seed, arm=arm, arm_spec=ARMS[arm], attack=dict(attackers=attack['attackers'], k=attack['k'],
                dose=attack['dose'], n_poison=len(poison_ids)), alpha=ALPHA, development_bank_only=True)
    if kind == 'full':
        method = job['method']
        s = Settings(seed=seed, dose=ARMS[arm][1:],
                     learner=method if method in ('FT', 'EI') else 'DR',
                     admission=method if method in ('DC', 'DG') else 'NONE',
                     aggregation='median' if method == 'MEDIAN' else 'fedavg')
        clients = [L.client_state(s, i) for i in range(5)]
        model, stages = run_stages(None, clients, [], data, s, 1, 4)
        slots = stages[-1]['slots']
        save(out, job_key(seed, arm, method), dict(base, method=method, settings=asdict(s), stages=stages,
             memory=memory_census(clients, poison_ids), audits=[], elapsed_seconds=time.perf_counter() - t0), model, slots)
        return [job_key(seed, arm, method)]
    # branch tree: shared stages 1-2, then one suffix per memory action
    s = Settings(seed=seed, dose=ARMS[arm][1:])
    clients = [L.client_state(s, i) for i in range(5)]
    slots = []
    model, prefix = run_stages(None, clients, slots, data, s, 1, 2)
    census2 = memory_census(clients, poison_ids)
    prefix_seconds = time.perf_counter() - t0
    auditor = Auditor(data, seed)
    keys = []
    for method in BRANCH_METHODS:
        if done(out, seed, arm, method):
            continue
        t1 = time.perf_counter()
        bm, bc, bs = copy.deepcopy(model), copy.deepcopy(clients), list(slots)
        audits = []
        audits.append(dict(after_stage=2, clients=audit(method, bm, bc, bs, 2, auditor, poison_ids, seed)))
        bm, st3 = run_stages(bm, bc, bs, data, s, 3, 3)
        audits.append(dict(after_stage=3, clients=audit(method, bm, bc, bs, 3, auditor, poison_ids, seed)))
        bm, st4 = run_stages(bm, bc, bs, data, s, 4, 4)
        save(out, job_key(seed, arm, method), dict(base, method=method, settings=asdict(s), stages=prefix + st3 + st4,
             memory_after_stage2=census2, memory=memory_census(bc, poison_ids), audits=audits,
             elapsed_seconds=prefix_seconds + time.perf_counter() - t1, shared_prefix_seconds=prefix_seconds), bm, bs)
        keys.append(job_key(seed, arm, method))
    return keys


def build_jobs(out, seeds):
    jobs = []
    for seed in seeds:
        for arm in ARMS:
            if not all(done(out, seed, arm, m) for m in BRANCH_METHODS):
                jobs.append(dict(kind='tree', seed=seed, arm=arm, out=str(out)))
            for m in FULL_METHODS:
                if not done(out, seed, arm, m):
                    jobs.append(dict(kind='full', seed=seed, arm=arm, method=m, out=str(out)))
    # clean replay trees first (validation gate), then by seed
    jobs.sort(key=lambda j: (j['arm'] != 'C' or j['kind'] != 'tree', j['seed'], j['kind'] != 'tree'))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--hours', type=float, default=11.0)
    a = ap.parse_args(); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if not (out / 'ownership-source.npz').exists():
        with np.load(BASE / 'builder/ownership.npz', allow_pickle=False) as f:
            positions, components = f['source_positions'], f['canonical_components']
        with np.load(ROOT / 'data/raw/phase3-feasibility-corrected/no-headers-74/conditional-components-120.npz', allow_pickle=False) as f:
            ids, labels = f['row_ids'][positions], f['labels'][positions].astype(np.int64)
        np.savez_compressed(out / 'ownership-source.npz', ids=ids, components=components, labels=labels)
    (out / 'protocol.json').write_text(json.dumps(dict(arms=ARMS, branch_methods=BRANCH_METHODS, full_methods=FULL_METHODS,
        alpha=ALPHA, seeds=a.seeds, engine='src/phase3 + src/phase3_v2 (audited core), 12 rounds x 1 epoch, lr .001, lambda_E 100, lambda_y 1, lambda_z .1'), indent=1))
    jobs = build_jobs(out, a.seeds); t0 = time.time(); finished = 0
    print(f'{len(jobs)} jobs pending for seeds {a.seeds}', flush=True)
    with futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        pending = {pool.submit(worker, j): j for j in jobs}
        for f in futures.as_completed(pending):
            j = pending[f]
            try:
                keys = f.result()
            except Exception:
                (out / f"failure-{j['kind']}-s{j['seed']}-{j['arm']}-{j.get('method', 'tree')}.txt").write_text(traceback.format_exc())
                print('FAILED', j, flush=True); continue
            finished += 1; el = time.time() - t0
            print(f"[{finished}/{len(jobs)}] {j['kind']} s{j['seed']} {j['arm']} {j.get('method', '')} -> {len(keys)} results | "
                  f"{el / 60:.1f} min elapsed, ~{el / finished * (len(jobs) - finished) / 60:.0f} min left", flush=True)
    print('complete' if all(done(out, s, ar, m) for s in a.seeds for ar in ARMS for m in BRANCH_METHODS + FULL_METHODS) else 'incomplete: re-run to resume', flush=True)


if __name__ == '__main__':
    main()
