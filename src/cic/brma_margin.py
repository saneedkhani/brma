"""CIC-IDS2017 check of BRMA-M (model-margin conformal audit, selected on TON_IoT development seeds) and BRMA-C.
Imports brma_study.py and brma_adaptive.py unchanged; same engine and seeds (1-11, 42) as the main study.

Arms: C, K3, K3H, K5 (standard) and S3H, S5 (selective adaptive attack).  Methods: NONE, BRMA, BRMA-M, BRMA-C, ORACLE.
  BRMA-M  m(x) = max_{c != benign} z_c(x) - z_benign(x) of the current global model; conformal threshold at alpha on the
          stage-1 benign calibration half (same split as BRMA); removes benign-labelled memory records with m(x) > tau.
  BRMA-C  union of the BRMA Gaussian test and the margin test, each at alpha/2.
Pre-specified tests (one-sided paired Wilcoxon over 12 seeds, Holm over four):
  T15 BRMA-M non-inferior to BRMA on H at K3H (margin 0.01);  T16 BRMA-M < NONE on H at K3H;
  T17 BRMA-M < NONE on H at S3H;  T18 clean macro-F1 of BRMA-M non-inferior to NONE (margin 0.02).
Usage (repository root):  python brma_margin.py --out OUT --workers 4
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
import brma_study as B                                   # noqa: E402
import brma_adaptive as A                                # noqa: E402
from src.phase3.common import rng                       # noqa: E402

A.ARMS['K3'] = (3, 1, 2, 'std')
ARMS = ('C', 'K3', 'K3H', 'K5', 'S3H', 'S5')
BRANCH = ('NONE', 'BRMA', 'BRMA-M', 'BRMA-C', 'ORACLE')


def margin(model, X, slots):
    b = slots.index(B.BENIGN); model.eval()
    with torch.no_grad():
        z = model(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))).numpy()
    other = np.delete(z, b, axis=1)
    return other.max(1) - z[:, b]


class Auditor2(B.Auditor):
    def __init__(self, data, seed):
        super().__init__(data, seed)
        ben = rng(41, seed).permutation(np.flatnonzero(data[0].y == B.BENIGN))
        self.Xcal = np.asarray(np.asarray(data[0].x)[ben[len(ben) // 2:]], np.float32)

    def flags_variant(self, method, model, X, slots, t):
        tau_m = lambda a: B.conformal(margin(model, self.Xcal, slots), a)
        if method == 'BRMA-M':
            return margin(model, X, slots) > tau_m(B.ALPHA)
        m = self.model(t); tau_g = B.conformal(self.score_z(self.Zcal, m), B.ALPHA / 2)
        return (self.score_z(self.std(X), m) > tau_g) | (margin(model, X, slots) > tau_m(B.ALPHA / 2))


def audit(method, model, clients, slots, t, auditor, poison_ids, seed):
    if method not in ('BRMA-M', 'BRMA-C'):
        return B.audit(method, model, clients, slots, t, auditor, poison_ids, seed)
    rows = []
    for k, c in enumerate(clients):
        m = c.replay
        if m is None or m['size'] == 0:
            rows.append(dict(client=k, size=0)); continue
        n = m['size']; ids = m['ids'][:n]; is_p = np.isin(ids, poison_ids); ben = np.flatnonzero(m['y'][:n] == B.BENIGN)
        flag = auditor.flags_variant(method, model, m['x'][:n][ben], slots, t) if len(ben) else np.zeros(0, bool)
        drop = ben[flag].astype(np.int64)
        rows.append(dict(client=k, size=int(n), benign_labelled=int(len(ben)), poison_before=int(is_p.sum()), brma_flagged=int(flag.sum()),
                         brma_flagged_poison=int(is_p[ben][flag].sum()) if len(ben) else 0, removed=int(len(drop)),
                         removed_poison=int(is_p[drop].sum()), removed_genuine=int(len(drop) - is_p[drop].sum())))
        B.remove_rows(m, drop)
    return rows


def worker(job):
    B.setup()
    seed, arm, out = job['seed'], job['arm'], Path(job['out'])
    t0 = time.perf_counter()
    data, attack = A.inputs(seed, arm, out)
    pid = np.asarray([int(v) for v in attack['poison_ids']], dtype=np.uint64)
    base = dict(seed=seed, arm=arm, arm_spec=A.ARMS[arm], attack=dict(attackers=attack['attackers'], k=attack['k'], dose=attack['dose'],
                mode=attack.get('mode'), n_poison=len(pid)), alpha=B.ALPHA, development_bank_only=True)
    s = B.Settings(seed=seed, dose=A.ARMS[arm][1:3])
    clients = [B.L.client_state(s, i) for i in range(5)]; slots = []
    model, prefix = B.run_stages(None, clients, slots, data, s, 1, 2)
    c2 = B.memory_census(clients, pid); pre = time.perf_counter() - t0
    auditor = Auditor2(data, seed); keys = []
    for mth in BRANCH:
        if B.done(out, seed, arm, mth):
            continue
        t1 = time.perf_counter()
        bm, bc, bs = copy.deepcopy(model), copy.deepcopy(clients), list(slots)
        audits = [dict(after_stage=2, clients=audit(mth, bm, bc, bs, 2, auditor, pid, seed))]
        bm, st3 = B.run_stages(bm, bc, bs, data, s, 3, 3)
        audits.append(dict(after_stage=3, clients=audit(mth, bm, bc, bs, 3, auditor, pid, seed)))
        bm, st4 = B.run_stages(bm, bc, bs, data, s, 4, 4)
        B.save(out, B.job_key(seed, arm, mth), dict(base, method=mth, settings=asdict(s), stages=prefix + st3 + st4, memory_after_stage2=c2,
               memory=B.memory_census(bc, pid), audits=audits, elapsed_seconds=pre + time.perf_counter() - t1, shared_prefix_seconds=pre), bm, bs)
        keys.append(B.job_key(seed, arm, mth))
    return keys


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
    (out / 'protocol.json').write_text(json.dumps(dict(arms=ARMS, branch=BRANCH, seeds=a.seeds, alpha=B.ALPHA,
        tests=['T15 BRMA-M non-inferior to BRMA on H at K3H (0.01)', 'T16 BRMA-M<NONE H K3H', 'T17 BRMA-M<NONE H S3H', 'T18 clean macro-F1 BRMA-M non-inferior NONE (0.02)']), indent=1))
    jobs = [dict(seed=sd, arm=ar, out=str(out)) for sd in a.seeds for ar in ARMS if not all(B.done(out, sd, ar, m) for m in BRANCH)]
    print(f'{len(jobs)} jobs pending', flush=True); t0 = time.time(); fin = 0
    with futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        pend = {pool.submit(worker, j): j for j in jobs}
        for f in futures.as_completed(pend):
            j = pend[f]
            try:
                ks = f.result()
            except Exception:
                (out / f"failure-s{j['seed']}-{j['arm']}.txt").write_text(traceback.format_exc()); print('FAILED', j, flush=True); continue
            fin += 1; el = time.time() - t0
            print(f"[{fin}/{len(jobs)}] s{j['seed']} {j['arm']} -> {len(ks)} results | {el / 60:.1f} min elapsed, ~{el / fin * (len(jobs) - fin) / 60:.0f} min left", flush=True)
    print('complete' if all(B.done(out, sd, ar, m) for sd in a.seeds for ar in ARMS for m in BRANCH) else 'incomplete: re-run to resume', flush=True)


if __name__ == '__main__':
    main()
