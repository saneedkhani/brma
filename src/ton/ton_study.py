"""BRMA on TON_IoT (extension study): Benign-Reference Memory Audit for federated class-incremental IDS.

Self-contained re-implementation of the project's audited Phase 3 learner (src/phase3 + src/phase3_v2)
for a second dataset, generalised to any feature width and class count:
  * MLP d-128-64-width (dropout .1), head expanded when new classes arrive (same seeded initialisation);
  * 5 clients, FedAvg weighted by admitted rows, 12 rounds x 1 epoch per stage, Adam lr 1e-3 reset each broadcast;
  * DR learner = current CE + 1.0 replay CE + 0.1 dark-logit distillation, 500-record reservoir per client;
  * FT (no replay), EI (empirical-Fisher EWC, lambda 100), DC/DG low-loss admission filters, MEDIAN aggregation;
  * k-client label flip of the stage-2 target class (DDoS) to benign; BRMA, RAND, LOSS, ORACLE memory actions.
BRMA is identical to the CIC-IDS2017 study (tied-covariance likelihood ratio, Ledoit-Wolf, conformal alpha .05).

Usage:
  python ton_study.py prepare --csv train_test_network.csv --data DATA_DIR
  python ton_study.py run --data DATA_DIR --out OUT --seeds 1 2 3 4 --workers 4 [--arms C K3H] [--full MEDIAN DC]
  python ton_study.py summarize --data DATA_DIR --out OUT
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse, copy, json, sys, time, traceback
import concurrent.futures as futures
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# ---------------------------------------------------------------- dataset protocol (fixed before any run)
CLASSES = ['normal', 'scanning', 'dos', 'ddos', 'password', 'backdoor', 'injection', 'xss', 'ransomware', 'mitm']
ARRIVAL = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4]          # stage in which each class first appears (normal: every stage)
K, BENIGN = len(CLASSES), 0
TARGET = CLASSES.index('ddos')                    # flipped class (plays the role of DoS Hulk in CIC-IDS2017)
CAPACITY, BATCH = 500, 128
SEEDS = tuple(range(1, 12)) + (42,)
ARMS = {'C': (0, 0, 1), 'K1': (1, 1, 2), 'K3': (3, 1, 2), 'K3H': (3, 9, 10), 'K5': (5, 1, 2)}
BRANCH_METHODS = ('NONE', 'BRMA', 'RAND', 'LOSS', 'ORACLE')
FULL_METHODS = ('FT', 'EI', 'DC', 'DG', 'MEDIAN')
ALPHA = 0.05
FIT_CAP, POOL_CAP, MIN_ROWS = 50000, 200000, 20
DIRICHLET = 1.0                                   # non-IID client allocation per class
CARILLO_F1_ALL = 74.49                            # Carillo et al., IEEE TNSM 2026, Table IV (TON_IoT, FedProx+MEMENTO+)


@dataclass(frozen=True)
class Settings:
    learner: str = 'DR'
    admission: str = 'NONE'
    seed: int = 1
    rounds: int = int(os.environ.get('TON_ROUNDS', '12'))
    epochs: int = 1
    learning_rate: float = .001
    lambda_E: float = 100.
    lambda_y: float = 1.
    lambda_z: float = .1
    aggregation: str = 'fedavg'
    dose: tuple = (0, 1)
    capacity: int = 500          # replay records per client
    hidden: tuple = (128, 64)    # MLP widths
    class_weight: bool = False   # client-local inverse-sqrt-frequency weights on the current-data CE


def rng(namespace, seed, *scope):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([7, namespace, seed, *scope])))


def torch_seed(namespace, seed, *scope):
    return int(rng(namespace, seed, *scope).integers(0, 2**63, dtype=np.int64))


def setup():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')


# ---------------------------------------------------------------- preparation (train/dev/test, features)
def prepare(csv, folder, ports=False):
    import pandas as pd
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if 'type' not in df.columns:
        raise ValueError(f'No "type" column; columns are {list(df.columns)}')
    t = df['type'].astype(str).str.strip().str.lower()
    unknown = sorted(set(t) - set(CLASSES))
    if unknown:
        raise ValueError(f'Unexpected classes {unknown}')
    y = t.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(np.int64)
    drop = {'ts', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label', 'type'}
    feats, names, dropped = [], [], []
    for c in df.columns:
        if c in drop:
            continue
        s = df[c]
        if s.dtype == bool or pd.api.types.is_numeric_dtype(s):
            v = pd.to_numeric(s, errors='coerce').fillna(0).to_numpy(np.float64)
            feats.append(np.sign(v) * np.log1p(np.abs(v))); names.append(c); continue
        ss = s.astype(str).str.strip()
        placeholder = ss.isin(['-', '', 'nan'])
        num = pd.to_numeric(ss.where(~placeholder), errors='coerce')
        nunique = ss.nunique()
        if (~placeholder).sum() and num[~placeholder].notna().mean() >= 0.99 and nunique > 50:
            v = num.fillna(0).to_numpy(np.float64); feats.append(np.sign(v) * np.log1p(np.abs(v))); names.append(c)
        elif nunique <= 50:
            for val in sorted(ss.unique()):
                feats.append((ss == val).to_numpy(np.float64)); names.append(f'{c}={val}')
        else:
            dropped.append(c)
    X = np.column_stack(feats).astype(np.float32)
    n_raw = len(y)
    frame = pd.DataFrame(X); frame['__y'] = y
    keep = ~frame.duplicated().to_numpy()            # de-duplication always on the base features: same rows and splits in every variant
    if ports:                                        # destination-port features: signed log1p value + one-hot of the 30 most frequent ports
        p = pd.to_numeric(df['dst_port'], errors='coerce').fillna(-1).astype(np.int64).to_numpy()
        top = pd.Series(p[keep]).value_counts().index[:30]
        extra = [np.log1p(np.maximum(p, 0)).astype(np.float64)] + [(p == v).astype(np.float64) for v in top]
        X = np.column_stack([X] + extra).astype(np.float32); names += ['dst_port'] + [f'dst_port={v}' for v in top]
    X, y = X[keep], y[keep]
    conflict = int(pd.DataFrame(X).duplicated(keep=False).sum())
    split = np.zeros(len(y), np.int8); stage = np.zeros(len(y), np.int8)   # split 0 train, 1 dev, 2 test
    for c in range(K):
        idx = rng(1, 0, c).permutation(np.flatnonzero(y == c))
        ntr, ndv = int(.7 * len(idx)), int(.15 * len(idx))
        split[idx[ntr:ntr + ndv]] = 1; split[idx[ntr + ndv:]] = 2
        tr = idx[:ntr]
        if c == BENIGN:
            for s, part in enumerate(np.array_split(rng(2, 0).permutation(tr), 4)):
                stage[part] = s + 1
        else:
            stage[tr] = ARRIVAL[c]
    tr = np.flatnonzero(split == 0)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd < 1e-6] = 1.
    X = np.clip((X - mu) / sd, -10, 10).astype(np.float32)
    order = tr[rng(3, 0).permutation(len(tr))]                # global record order -> ids 1..n
    dv, te = np.flatnonzero(split == 1), np.flatnonzero(split == 2)
    np.savez(folder / 'ton.npz', xtr=X[order], ytr=y[order], stage=stage[order], ids=np.arange(1, len(order) + 1, dtype=np.uint64),
             xdev=X[dv], ydev=y[dv], xte=X[te], yte=y[te], names=np.asarray(names))
    counts = {CLASSES[c]: {sp: int(((y == c) & (split == i)).sum()) for i, sp in enumerate(('train', 'dev', 'test'))} for c in range(K)}
    report = dict(csv=str(csv), raw_rows=n_raw, duplicate_rows_removed=int(n_raw - len(y)), rows=int(len(y)),
                  feature_vectors_shared_by_two_classes=conflict, features=len(names), dropped_columns=dropped,
                  arrival={c: ARRIVAL[i] for i, c in enumerate(CLASSES)}, target=CLASSES[TARGET], counts=counts,
                  train_rows_per_stage={s: int((stage[order] == s).sum()) for s in range(1, 5)},
                  destination_port_features=bool(ports), scaling='signed log1p for numeric columns, one-hot for <=50 categories, z-score with train statistics, clip +-10')
    (folder / 'prepare-report.json').write_text(json.dumps(report, indent=1))
    return report


_DATA = {}


def load(folder):
    folder = str(folder)
    if folder not in _DATA:
        with np.load(Path(folder) / 'ton.npz', allow_pickle=False) as f:
            _DATA[folder] = {k: f[k] for k in f.files}
    return _DATA[folder]


class Stage:
    def __init__(self, stage, x, y, ids, owner):
        self.stage, self.x, self.y, self.ids, self.owner = stage, x, y, ids, owner

    def targets(self, slots):
        lut = np.full(K, -1, np.int64); lut[slots] = np.arange(len(slots)); return lut[self.y]


def ownership(seed, y):
    owner = np.zeros(len(y), np.int8)
    for c in range(K):
        idx = np.flatnonzero(y == c)
        p = rng(20, seed, c).dirichlet(np.full(5, DIRICHLET))
        owner[idx] = rng(21, seed, c).choice(5, len(idx), p=p)
    return owner


def corrupt_multi(clean, ids, owner, seed, stage, k, num, den):
    """k-client nested label flip of the target class to benign in stage 2 (same streams as the CIC study)."""
    supplied = clean.copy()
    info = dict(stage=stage, k=k, dose=[num, den], attackers=[], poison_ids=[])
    if stage != 2 or k == 0 or num == 0:
        return supplied, info
    counts = np.bincount(owner[clean == TARGET], minlength=5)
    eligible = np.flatnonzero(counts >= 10)
    first = int(rng(22, seed).choice(eligible))
    rest = [int(c) for c in rng(23, seed).permutation(eligible) if c != first]
    attackers = [first] + rest[:k - 1]
    poison = []
    for a in attackers:
        source = np.flatnonzero((owner == a) & (clean == TARGET))
        source = source[np.argsort(ids[source], kind='stable')]
        chosen = rng(24, seed, a).permutation(source)[:num * len(source) // den]
        supplied[chosen] = BENIGN; poison.extend(int(v) for v in ids[chosen])
    info.update(attackers=attackers, poison_ids=poison)
    return supplied, info


def inputs(folder, seed, arm):
    D = load(folder)
    owner = ownership(seed, D['ytr'])
    k, num, den = ARMS[arm]
    data, attack = [], None
    for s in range(1, 5):
        idx = np.flatnonzero(D['stage'] == s)
        y, info = corrupt_multi(D['ytr'][idx], D['ids'][idx], owner[idx], seed, s, k, num, den)
        if s == 2:
            attack = info
        data.append(Stage(s, D['xtr'][idx], y, D['ids'][idx], owner[idx]))
    return data, attack


# ---------------------------------------------------------------- model and learner (mirrors src/phase3 + src/phase3_v2)
class MLP(nn.Module):
    def __init__(self, d, width, seed, hidden=(128, 64)):
        super().__init__()
        h1, h2 = hidden; self.h2 = h2
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch_seed(4, seed))
            self.backbone = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Dropout(.1), nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(.1))
            self.classifier = nn.Linear(h2, width)
        self.initialize_rows(0, 1, seed)

    def initialize_rows(self, old, stage, seed):
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(torch_seed(5, seed, stage))
            nn.init.xavier_uniform_(self.classifier.weight[old:], gain=1); self.classifier.bias[old:].zero_()

    def expand(self, width, stage, seed):
        old = self.classifier
        with torch.random.fork_rng(devices=[]):
            new = nn.Linear(self.h2, width)
        with torch.no_grad():
            new.weight[:old.out_features].copy_(old.weight); new.bias[:old.out_features].copy_(old.bias)
        self.classifier = new
        self.initialize_rows(old.out_features, stage, seed)

    def forward(self, x):
        return self.classifier(self.backbone(x))


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def client_state(s, i, d):
    g = torch.Generator().manual_seed(torch_seed(10, s.seed, i))
    replay = None
    if s.learner == 'DR':
        cap = s.capacity
        replay = dict(x=np.zeros((cap, d), np.float32), y=np.zeros(cap, np.int64), ids=np.zeros(cap, np.uint64),
                      z=np.zeros((cap, K), np.float32), width=np.zeros(cap, np.uint8), size=0, seen=0, cap=cap,
                      draw=rng(7, s.seed, i), insert=rng(8, s.seed, i))
    return dict(accepted=np.empty(0, np.int64), dropout=g.get_state().numpy().copy(), replay=replay, importance={}, anchor={})


def insert_refill(m, x, labels, ids, teachers, width):
    """Reservoir (Algorithm R) that first fills slots freed by an audit; plain reservoir when nothing was removed."""
    cap = m.get('cap', CAPACITY)
    for i in range(len(labels)):
        m['seen'] += 1; seen = m['seen']
        if m['size'] < min(cap, seen - 1):
            slot = m['size']; m['size'] += 1
        else:
            slot = seen - 1 if seen <= cap else int(m['insert'].integers(0, seen))
            m['size'] = min(cap, seen)
        if slot < cap:
            m['x'][slot] = x[i]; m['y'][slot] = labels[i]; m['ids'][slot] = ids[i]
            m['z'][slot].fill(0); m['z'][slot, :width] = teachers[i]; m['width'][slot] = width


def ce_scores(model, x, targets):
    model.eval(); out = np.empty(len(targets), np.float32)
    with torch.no_grad():
        for a in range(0, len(targets), 4096):
            out[a:a + 4096] = F.cross_entropy(model(torch.from_numpy(x[a:a + 4096])), torch.from_numpy(targets[a:a + 4096]), reduction='none').numpy()
    return out


def admit(model, d, slots, new_codes, clients, s):
    targets = d.targets(slots)
    scores = ce_scores(model, d.x, targets) if (s.admission in ('DC', 'DG') and d.stage > 1) else None
    for i, c in enumerate(clients):
        idx = np.flatnonzero(d.owner == i)
        if scores is None:
            chosen = idx
        elif s.admission == 'DG':
            chosen = idx[np.lexsort((d.ids[idx], scores[idx]))][:(3 * len(idx) + 2) // 4]
        else:
            chosen = []
            for code in np.unique(d.y[idx]):
                g = idx[d.y[idx] == code]
                n = len(g) if (int(code) in new_codes or len(g) < 20) else (3 * len(g) + 3) // 4
                chosen.extend(g[np.lexsort((d.ids[g], scores[g]))][:n].tolist())
            chosen = np.asarray(chosen, np.int64)
        c['accepted'] = np.sort(chosen).astype(np.int64)


def expand_importance(c, model):
    for name, value in model.named_parameters():
        prior, anchor = c['importance'].get(name), c['anchor'].get(name)
        upd, anc = torch.zeros_like(value), value.detach().clone()
        if prior is not None:
            region = tuple(slice(0, n) for n in prior.shape); upd[region] = prior; anc[region] = anchor
        c['importance'][name], c['anchor'][name] = upd, anc


def empirical_importance(model, x, targets, chunk=64):
    params = {k: v.detach() for k, v in model.named_parameters()}; buffers = dict(model.named_buffers())
    totals = {k: torch.zeros_like(p) for k, p in params.items()}
    model.eval()

    def single(p, r, l):
        return F.cross_entropy(torch.func.functional_call(model, (p, buffers), (r.unsqueeze(0),)), l.unsqueeze(0))
    grad = torch.func.vmap(torch.func.grad(single), in_dims=(None, 0, 0), randomness='error')
    for a in range(0, len(targets), chunk):
        for k, g in grad(params, x[a:a + chunk], targets[a:a + chunk]).items():
            totals[k].add_(g.square().sum(0))
    return {k: v / len(targets) for k, v in totals.items()}


def commit_importance(model, clients, d, slots, s):
    targets = d.targets(slots); shared = snapshot(model)
    for i, c in enumerate(clients):
        if len(c['accepted']):
            sel = rng(9, s.seed, i, d.stage).choice(c['accepted'], min(1024, len(c['accepted'])), replace=False)
            for k, v in empirical_importance(model, torch.from_numpy(d.x[sel]), torch.from_numpy(targets[sel])).items():
                c['importance'][k].add_(v)
        c['anchor'] = {k: v.clone() for k, v in shared.items()}


def dark_loss(pred, teachers, widths):
    mask = torch.arange(pred.shape[1])[None, :] < widths[:, None]
    return (torch.where(mask, pred - teachers[:, :pred.shape[1]], 0.).square().sum(1) / widths).mean()


def train_client(broadcast, template, c, d, slots, s, round_id, cid):
    if not len(c['accepted']):
        return None
    local = copy.deepcopy(template); local.load_state_dict(broadcast)
    opt = torch.optim.Adam(local.parameters(), lr=s.learning_rate, betas=(.9, .999), eps=1e-8, foreach=False)
    targets = d.targets(slots); lut = np.full(K, -1, np.int64); lut[slots] = np.arange(len(slots))
    m = c['replay']
    cw = None
    if s.class_weight:                                   # client-local, from its own admitted supplied labels
        n = np.bincount(targets[c['accepted']], minlength=len(slots)).astype(np.float64)
        w = np.where(n > 0, 1. / np.sqrt(np.maximum(n, 1.)), 0.)
        w = w / (w[targets[c['accepted']]].mean())       # mean weight over the client's rows = 1
        cw = torch.tensor(w, dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.from_numpy(c['dropout'].copy()))
        mask = np.zeros(len(d.y), bool); mask[c['accepted']] = True
        raw = np.flatnonzero(d.owner == cid)
        for epoch in range(1, s.epochs + 1):
            perm = rng(6, s.seed, cid, d.stage, round_id, epoch).permutation(raw)
            order = perm[mask[perm]]
            for a in range(0, len(order), BATCH):
                cur = order[a:a + BATCH]
                x, y = torch.from_numpy(d.x[cur]), torch.from_numpy(targets[cur])
                first = m is not None and round_id == 1 and epoch == 1
                if first:
                    local.eval()
                    with torch.no_grad():
                        teachers = local(x).numpy().copy()
                replay = None
                if m is not None and m['size']:
                    draw = m['draw'].choice(m['size'], min(BATCH, m['size']), replace=False)
                    replay = {k: m[k][draw].copy() for k in ('x', 'y', 'z', 'width')}
                opt.zero_grad(set_to_none=True); local.train()
                total = F.cross_entropy(local(x), y, weight=cw)
                if replay is not None:
                    rx = torch.from_numpy(replay['x'])
                    total = total + s.lambda_y * F.cross_entropy(local(rx), torch.from_numpy(lut[replay['y']]))
                    local.eval()
                    total = total + s.lambda_z * dark_loss(local(rx), torch.from_numpy(replay['z']), torch.from_numpy(replay['width'].astype(np.int64)))
                    local.train()
                if s.learner == 'EI':
                    pen = torch.zeros(())
                    for name, p in local.named_parameters():
                        pen = pen + (c['importance'][name] * (p - c['anchor'][name]).square()).sum()
                    total = total + pen * (s.lambda_E / 2)
                if not torch.isfinite(total):
                    raise FloatingPointError('non-finite objective')
                total.backward(); opt.step()
                if first:
                    insert_refill(m, d.x[cur], d.y[cur], d.ids[cur], teachers, len(slots))
        c['dropout'] = torch.get_rng_state().numpy().copy()
    return snapshot(local)


def aggregate(broadcast, updates, counts, how):
    active = [u for u, n in zip(updates, counts) if n]
    if how == 'median' and len(active) >= 3:
        return {k: torch.quantile(torch.stack([u[k] for u in active]), .5, dim=0) for k in broadcast}
    total = sum(counts); out = {k: torch.zeros_like(v) for k, v in broadcast.items()}
    for n, u in zip(counts, updates):
        if n:
            w = float(np.float32(n / total))
            for k in out:
                out[k].add_(u[k] * w)
    return out


def evaluate(folder, model, slots, stage):
    D = load(folder); res = dict(stage=stage, slots=list(map(int, slots)))
    model.eval()
    for split, xs, ys in (('dev', 'xdev', 'ydev'), ('test', 'xte', 'yte')):
        x, y = D[xs], D[ys]; keep = np.flatnonzero(np.isin(y, slots)); M = np.zeros((K, K), np.int64)
        with torch.no_grad():
            for a in range(0, len(keep), 8192):
                ix = keep[a:a + 8192]
                np.add.at(M, (y[ix], np.asarray(slots)[model(torch.from_numpy(x[ix])).argmax(1).numpy()]), 1)
        res[split] = M.tolist()
    return res


def run_stages(folder, model, clients, slots, data, s, start, end):
    stages = []; d0 = data[0].x.shape[1]
    for st in range(start, end + 1):
        d = data[st - 1]
        new = [int(c) for c in np.unique(d.y) if int(c) not in slots]; slots.extend(new)
        if model is None:
            model = MLP(d0, len(slots), s.seed, tuple(s.hidden))
        elif new:
            model.expand(len(slots), st, s.seed)
        if s.learner == 'EI':
            for c in clients:
                expand_importance(c, model)
        admit(model, d, slots, set(new), clients, s)
        for r in range(1, s.rounds + 1):
            b = snapshot(model)
            ups = [train_client(b, model, c, d, slots, s, r, i) for i, c in enumerate(clients)]
            model.load_state_dict(aggregate(b, ups, [len(c['accepted']) for c in clients], s.aggregation))
        if s.learner == 'EI':
            commit_importance(model, clients, d, slots, s)
        stages.append(evaluate(folder, model, slots, st))
    return model, stages


# ---------------------------------------------------------------- BRMA (identical to the CIC-IDS2017 study)
def maha_sq(Z, mu, P, chunk=65536):
    out = np.empty(len(Z))
    for i in range(0, len(Z), chunk):
        D = Z[i:i + chunk] - mu; out[i:i + chunk] = np.einsum('ij,ij->i', D @ P, D)
    return out


def conformal(cal, alpha):
    n = len(cal); return float(np.quantile(cal, min(1., np.ceil((n + 1) * (1 - alpha)) / n), method='higher'))


class Auditor:
    def __init__(self, data, seed):
        self.data, self.seed = data, seed
        x1 = data[0].x; self.mu0, self.sd0 = x1.mean(0), x1.std(0) + 1e-6
        ben = rng(41, seed).permutation(np.flatnonzero(data[0].y == BENIGN))
        self.Zref = self.std(x1[ben[:len(ben) // 2]]); self.Zcal = self.std(x1[ben[len(ben) // 2:]]); self.models = {}

    def std(self, X):
        return (np.asarray(X, np.float64) - self.mu0) / self.sd0

    def model(self, t):
        if t in self.models:
            return self.models[t]
        from sklearn.covariance import LedoitWolf
        means, centred = {}, [self.Zref - self.Zref.mean(0)]
        for c in range(1, K):
            X = np.concatenate([self.data[s].x[self.data[s].y == c] for s in range(t)])
            if len(X) < MIN_ROWS:
                continue
            if len(X) > FIT_CAP:
                X = X[rng(42, self.seed, t, c).choice(len(X), FIT_CAP, replace=False)]
            Z = self.std(X); means[c] = Z.mean(0); centred.append(Z - means[c])
        pooled = np.concatenate(centred)
        if len(pooled) > POOL_CAP:
            pooled = pooled[rng(43, self.seed, t).choice(len(pooled), POOL_CAP, replace=False)]
        m = dict(P=LedoitWolf(assume_centered=True).fit(pooled).precision_, mub=self.Zref.mean(0), means=means)
        m['tau'] = conformal(self.score_z(self.Zcal, m), ALPHA); self.models[t] = m
        return m

    def score_z(self, Z, m):
        if not len(Z):
            return np.zeros(0)
        return 0.5 * (maha_sq(Z, m['mub'], m['P']) - np.min([maha_sq(Z, mu, m['P']) for mu in m['means'].values()], axis=0))

    def flags(self, X, t):
        m = self.model(t); s = self.score_z(self.std(X), m); return s > m['tau'], m['tau']


def remove_rows(m, drop):
    keep = np.setdiff1d(np.arange(m['size']), drop)
    for k in ('x', 'y', 'ids', 'z', 'width'):
        m[k][:len(keep)] = m[k][keep]
    m['size'] = len(keep)


def audit(method, model, clients, slots, t, auditor, poison_ids, seed):
    rows = []; bslot = slots.index(BENIGN)
    for k, c in enumerate(clients):
        m = c['replay']
        if m is None or m['size'] == 0:
            rows.append(dict(client=k, size=0)); continue
        n = m['size']; y, ids = m['y'][:n], m['ids'][:n]
        is_p = np.isin(ids, poison_ids); ben = np.flatnonzero(y == BENIGN)
        flag, tau = auditor.flags(m['x'][:n][ben], t) if len(ben) else (np.zeros(0, bool), None)
        budget = int(flag.sum())
        if method == 'BRMA':
            drop = ben[flag]
        elif method == 'RAND':
            drop = rng(44, seed, t, k).choice(ben, budget, replace=False) if budget else np.zeros(0, int)
        elif method == 'LOSS':
            model.eval()
            with torch.no_grad():
                loss = F.cross_entropy(model(torch.from_numpy(m['x'][:n][ben].copy())), torch.full((len(ben),), bslot), reduction='none').numpy()
            drop = ben[np.lexsort((ids[ben], -loss))[:budget]]
        elif method == 'ORACLE':
            drop = np.flatnonzero(is_p)
        else:
            drop = np.zeros(0, int)
        drop = np.asarray(drop, np.int64)
        rows.append(dict(client=k, size=int(n), benign_labelled=int(len(ben)), poison_before=int(is_p.sum()), brma_flagged=budget,
                         brma_flagged_poison=int(is_p[ben][flag].sum()) if len(ben) else 0, removed=int(len(drop)),
                         removed_poison=int(is_p[drop].sum()), removed_genuine=int(len(drop) - is_p[drop].sum()), tau=tau))
        remove_rows(m, drop)
    return rows


def census(clients, poison_ids):
    out = []
    for c in clients:
        m = c['replay']
        if m is None:
            out.append(None); continue
        n = m['size']; out.append(dict(size=int(n), poison=int(np.isin(m['ids'][:n], poison_ids).sum()), benign_labelled=int((m['y'][:n] == BENIGN).sum())))
    return out


# ---------------------------------------------------------------- jobs
def key(seed, arm, method):
    return f's{seed}-{arm}-{method}'


def done(out, seed, arm, method):
    return (Path(out) / 'runs' / key(seed, arm, method) / 'result.json').exists()


def save(out, k, payload):
    f = Path(out) / 'runs' / k; f.mkdir(parents=True, exist_ok=True)
    tmp = f / 'result.json.tmp'; tmp.write_text(json.dumps(payload)); tmp.replace(f / 'result.json')


def cfg(job):
    c = dict(job.get('cfg') or {}); c.pop('features', None)
    if 'hidden' in c:
        c['hidden'] = tuple(c['hidden'])
    return c


def worker(job):
    setup()
    folder, out, seed, arm = job['data'], job['out'], job['seed'], job['arm']
    t0 = time.perf_counter()
    data, attack = inputs(folder, seed, arm); d = data[0].x.shape[1]
    pid = np.asarray(attack['poison_ids'], np.uint64)
    base = dict(seed=seed, arm=arm, arm_spec=ARMS[arm], attack=dict(attackers=attack['attackers'], k=attack['k'], dose=attack['dose'], n_poison=len(pid)),
                alpha=ALPHA, dataset='TON_IoT train_test_network', target=CLASSES[TARGET])
    if job['kind'] == 'full':
        mth = job['method']
        s = Settings(**cfg(job), seed=seed, dose=ARMS[arm][1:], learner=mth if mth in ('FT', 'EI') else 'DR',
                     admission=mth if mth in ('DC', 'DG') else 'NONE', aggregation='median' if mth == 'MEDIAN' else 'fedavg')
        clients = [client_state(s, i, d) for i in range(5)]
        model, stages = run_stages(folder, None, clients, [], data, s, 1, 4)
        save(out, key(seed, arm, mth), dict(base, method=mth, settings=asdict(s), stages=stages, memory=census(clients, pid), audits=[],
                                            elapsed_seconds=time.perf_counter() - t0))
        return [key(seed, arm, mth)]
    s = Settings(**cfg(job), seed=seed, dose=ARMS[arm][1:])
    clients = [client_state(s, i, d) for i in range(5)]; slots = []
    model, prefix = run_stages(folder, None, clients, slots, data, s, 1, 2)
    c2 = census(clients, pid); pre = time.perf_counter() - t0
    auditor = Auditor(data, seed); keys = []
    for mth in job['methods']:
        if done(out, seed, arm, mth):
            continue
        t1 = time.perf_counter()
        bm, bc, bs = copy.deepcopy(model), copy.deepcopy(clients), list(slots)
        audits = [dict(after_stage=2, clients=audit(mth, bm, bc, bs, 2, auditor, pid, seed))]
        bm, st3 = run_stages(folder, bm, bc, bs, data, s, 3, 3)
        audits.append(dict(after_stage=3, clients=audit(mth, bm, bc, bs, 3, auditor, pid, seed)))
        bm, st4 = run_stages(folder, bm, bc, bs, data, s, 4, 4)
        save(out, key(seed, arm, mth), dict(base, method=mth, settings=asdict(s), stages=prefix + st3 + st4, memory_after_stage2=c2,
                                            memory=census(bc, pid), audits=audits, elapsed_seconds=pre + time.perf_counter() - t1, shared_prefix_seconds=pre))
        keys.append(key(seed, arm, mth))
    return keys


def build_jobs(folder, out, seeds, arms, branch, full, config=None):
    jobs = []
    for seed in seeds:
        for arm in arms:
            if branch and not all(done(out, seed, arm, m) for m in branch):
                jobs.append(dict(kind='tree', seed=seed, arm=arm, methods=list(branch), data=str(folder), out=str(out), cfg=config))
            for m in full:
                if not done(out, seed, arm, m):
                    jobs.append(dict(kind='full', seed=seed, arm=arm, method=m, data=str(folder), out=str(out), cfg=config))
    jobs.sort(key=lambda j: (seeds.index(j['seed']), j['kind'] != 'tree'))   # seed by seed: a time-out leaves whole seeds
    return jobs


def run(a):
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    config = json.loads(a.cfg) if a.cfg else None
    (out / 'protocol.json').write_text(json.dumps(dict(arms={k: ARMS[k] for k in a.arms}, branch_methods=a.branch, full_methods=a.full, alpha=ALPHA,
        seeds=a.seeds, classes=CLASSES, arrival=ARRIVAL, target=CLASSES[TARGET], dirichlet=DIRICHLET,
        learner=asdict(Settings(**cfg(dict(cfg=config))))), indent=1))
    jobs = build_jobs(a.data, out, a.seeds, a.arms, a.branch, a.full, config); t0 = time.time(); fin = 0
    print(f'{len(jobs)} jobs pending', flush=True)
    deadline = t0 + a.hours * 3600; stopped = False
    with futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        pend = {pool.submit(worker, j): j for j in jobs}
        for f in futures.as_completed(pend):
            j = pend[f]
            if not stopped and time.time() > deadline:
                stopped = True; n = sum(x.cancel() for x in pend)
                print(f'time limit of {a.hours} h reached: {n} queued jobs cancelled, waiting for running jobs; re-run to resume', flush=True)
            if f.cancelled():
                continue
            try:
                ks = f.result()
            except Exception:
                (out / f"failure-{j['kind']}-s{j['seed']}-{j['arm']}-{j.get('method', 'tree')}.txt").write_text(traceback.format_exc())
                print('FAILED', j['kind'], j['seed'], j['arm'], j.get('method', ''), flush=True); continue
            fin += 1; el = time.time() - t0
            print(f"[{fin}/{len(jobs)}] {j['kind']} s{j['seed']} {j['arm']} {j.get('method', '')} -> {len(ks)} results | "
                  f"{el / 60:.1f} min elapsed, ~{el / fin * (len(jobs) - fin) / 60:.0f} min left", flush=True)
    print('complete' if not build_jobs(a.data, out, a.seeds, a.arms, a.branch, a.full, config) else 'incomplete: re-run to resume', flush=True)


# ---------------------------------------------------------------- tuning (development seeds, dev split only)
TUNE_SEEDS = (101, 102, 103)          # disjoint from the 12 evaluation seeds
GRID = {}                             # learner options x feature set; selection is restricted to cost <= MAX_COST x base
for _f in ('base', 'ports'):
    for _n, _c in (('base', {}), ('mem2000', dict(capacity=2000)), ('mem2000_wide', dict(capacity=2000, hidden=[256, 128])),
                   ('mem5000_wide', dict(capacity=5000, hidden=[256, 128])),
                   ('ep2_mem2000_wide', dict(epochs=2, capacity=2000, hidden=[256, 128])),
                   ('ep2_mem2000_wide_weighted', dict(epochs=2, capacity=2000, hidden=[256, 128], class_weight=True)),
                   ('r24_mem2000', dict(rounds=24, capacity=2000)),
                   ('ep2_mem5000_wide_lr2e-3', dict(epochs=2, capacity=5000, hidden=[256, 128], learning_rate=.002))):
        GRID[f'{_f}__{_n}'] = dict(_c, features=_f)
MAX_COST = 3.0


def tune(a):
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, c in GRID.items():
        for seed in a.seeds:
            if not done(out / name, seed, 'C', 'NONE'):
                folder = a.data_ports if c.get('features') == 'ports' else a.data
                if c.get('features') == 'ports' and not folder:
                    continue
                jobs.append(dict(kind='full', seed=seed, arm='C', method='NONE', data=str(folder), out=str(out / name), cfg=c))
    print(f'tuning: {len(GRID)} configurations x {len(a.seeds)} development seeds, {len(jobs)} jobs pending', flush=True)
    t0 = time.time(); fin = 0
    with futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        pend = {pool.submit(worker, j): j for j in jobs}
        for f in futures.as_completed(pend):
            j = pend[f]
            try:
                f.result()
            except Exception:
                (out / f"failure-{Path(j['out']).name}-s{j['seed']}.txt").write_text(traceback.format_exc()); print('FAILED', j['out'], j['seed'], flush=True); continue
            fin += 1; el = time.time() - t0
            print(f"[{fin}/{len(jobs)}] {Path(j['out']).name} s{j['seed']} | {el / 60:.1f} min elapsed, ~{el / fin * (len(jobs) - fin) / 60:.0f} min left", flush=True)
    rows = []
    for name, c in GRID.items():
        rs = [json.loads(p.read_text()) for p in sorted((out / name).glob('runs/*/result.json'))]
        if not rs:
            continue
        m = [result_row(r) for r in rs]
        rows.append(dict(config=name, cfg=c, seeds=len(m), dev_macro_f1=float(np.mean([x['dev_macro_f1'] for x in m])),
                         dev_sd=float(np.std([x['dev_macro_f1'] for x in m])), minutes=float(np.mean([r['elapsed_seconds'] for r in rs]) / 60)))
    rows.sort(key=lambda r: -r['dev_macro_f1'])
    base = next(r for r in rows if r['config'] == 'base__base')
    for r in rows:
        r['cost_vs_base'] = r['minutes'] / base['minutes']
    best = next(r for r in rows if r['cost_vs_base'] <= MAX_COST)
    (out / 'tuning.json').write_text(json.dumps(dict(selection=f'highest mean dev macro-F1 over development seeds 101-103 (clean, no defence) among settings costing <= {MAX_COST}x the base run time; test split not used',
                                                      best=best, table=rows), indent=1))
    print()
    print('configuration                          dev macro-F1 (%)   sd    min/run  cost')
    for r in rows:
        print(f"{r['config']:38s} {100 * r['dev_macro_f1']:8.2f}      {100 * r['dev_sd']:5.2f}  {r['minutes']:6.1f}   {r['cost_vs_base']:.1f}x")
    print()
    print('selected:', best['config'], json.dumps(best['cfg']))


# ---------------------------------------------------------------- metrics
def scores(M, eligible):
    M = np.asarray(M, np.float64); f1 = {}
    for c in eligible:
        tp = M[c, c]; p = tp / M[:, c].sum() if M[:, c].sum() else 0.; r = tp / M[c].sum() if M[c].sum() else 0.
        f1[c] = 2 * p * r / (p + r) if p + r else 0.
    ben = M[BENIGN].sum()
    return dict(f1=f1, macro_f1=float(np.mean([f1[c] for c in eligible])), accuracy=float(np.trace(M) / M.sum()),
                benign_far=float((ben - M[BENIGN, BENIGN]) / ben) if ben else 0.,
                target_recall=float(M[TARGET, TARGET] / M[TARGET].sum()) if M[TARGET].sum() else np.nan,
                target_asr=float(M[TARGET, BENIGN] / M[TARGET].sum()) if M[TARGET].sum() else np.nan)


def result_row(r):
    row = dict(seed=r['seed'], arm=r['arm'], method=r['method'])
    for split in ('dev', 'test'):
        st = [scores(s[split], [c for c in range(K) if ARRIVAL[c] <= s['stage']]) for s in r['stages']]
        last = st[-1]
        row.update({f'{split}_macro_f1': last['macro_f1'], f'{split}_accuracy': last['accuracy'], f'{split}_far': last['benign_far'],
                    f'{split}_old_f1': float(np.mean([last['f1'][c] for c in range(K) if ARRIVAL[c] == 1])),
                    f'{split}_new_f1': float(np.mean([last['f1'][c] for c in range(K) if ARRIVAL[c] > 1])),
                    f'{split}_A': float(np.mean([np.mean([st[t - 1]['f1'][c] for c in range(K) if ARRIVAL[c] == t and c != TARGET]) for t in (2, 3, 4)])),
                    f'{split}_e2_recall': st[1]['target_recall'], f'{split}_e3_asr': st[2]['target_asr'], f'{split}_e4_asr': st[3]['target_asr']})
        for c in range(K):
            row[f'{split}_f1_{CLASSES[c]}'] = last['f1'][c]
    return row


def summarize(a):
    import pandas as pd
    out = Path(a.out)
    rows = [result_row(json.loads(p.read_text())) for p in sorted(out.glob('runs/*/result.json'))]
    df = pd.DataFrame(rows)
    clean = df[df.arm == 'C'][['seed', 'method', 'dev_e3_asr', 'dev_e4_asr', 'test_e3_asr', 'test_e4_asr']]
    clean = clean.rename(columns={c: 'clean_' + c for c in clean.columns if c not in ('seed', 'method')})
    df = df.merge(clean, on=['seed', 'method'], how='left')
    for split in ('dev', 'test'):
        df[f'{split}_H'] = .5 * ((df[f'{split}_e3_asr'] - df[f'clean_{split}_e3_asr']) + (df[f'{split}_e4_asr'] - df[f'clean_{split}_e4_asr']))
    df.to_csv(out / 'metrics.csv', index=False)
    pd.set_option('display.width', 200)
    print(f'runs: {len(df)}, seeds: {sorted(df.seed.unique())}')
    print('\nTest macro-F1 (%) after stage 4, all 10 classes  [Carillo et al. TNSM 2026, clean clients: F1 all = 74.49]')
    print((100 * df.pivot_table(index='method', columns='arm', values='test_macro_f1')).round(2).to_string())
    print('\nTest accuracy (%) after stage 4')
    print((100 * df.pivot_table(index='method', columns='arm', values='test_accuracy')).round(2).to_string())
    print('\nH (test): rise of DDoS-predicted-as-benign rate over the clean run, lower is better')
    print(df.pivot_table(index='method', columns='arm', values='test_H').round(4).to_string())
    c = df[(df.arm == 'C')].groupby('method')[['test_macro_f1', 'test_old_f1', 'test_new_f1']].mean() * 100
    print('\nClean arm, test F1 (%): all / old (stage-1 classes) / new (later classes)   [Carillo: 74.49 / 79.05 / 68.10]')
    print(c.round(2).to_string())
    best = df[(df.arm == 'C') & (df.method == 'NONE')].test_macro_f1.mean() * 100
    print(f'\nOur clean replay learner (DR, no defence): {best:.2f}% vs Carillo 74.49%  ->  {"HIGHER" if best > CARILLO_F1_ALL else "LOWER"} by {abs(best - CARILLO_F1_ALL):.2f} points')
    return df


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--csv', required=True); p.add_argument('--data', required=True); p.add_argument('--ports', action='store_true')
    r = sub.add_parser('run'); r.add_argument('--data', required=True); r.add_argument('--out', required=True)
    r.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS)); r.add_argument('--workers', type=int, default=4)
    r.add_argument('--arms', nargs='+', default=list(ARMS)); r.add_argument('--branch', nargs='*', default=list(BRANCH_METHODS))
    r.add_argument('--full', nargs='*', default=list(FULL_METHODS))
    r.add_argument('--hours', type=float, default=1000.); r.add_argument('--cfg', default='')
    t = sub.add_parser('tune'); t.add_argument('--data', required=True); t.add_argument('--out', required=True)
    t.add_argument('--seeds', type=int, nargs='+', default=list(TUNE_SEEDS)); t.add_argument('--workers', type=int, default=4)
    t.add_argument('--data-ports', default='')
    s = sub.add_parser('summarize'); s.add_argument('--data', required=True); s.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cmd == 'prepare':
        print(json.dumps(prepare(a.csv, a.data, a.ports), indent=1))
    elif a.cmd == 'run':
        run(a)
    elif a.cmd == 'tune':
        tune(a)
    else:
        summarize(a)


if __name__ == '__main__':
    main()
