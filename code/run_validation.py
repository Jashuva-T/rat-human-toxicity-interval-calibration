"""Step 2: 200 repeated 70/30 splits; all methods; held-out metrics only."""
import json, time, numpy as np, pandas as pd
from lib import *
A = pd.read_csv('../out/analysis_set_289.csv'); x = A.x.values; y = A.y.values; N = len(A)
S = CFG['n_splits']; ntr = int(N * CFG['train_fraction']); nte = N - ntr
METHODS = ['bayes_normal', 'bayes_t', 'fixed_width', 'split_conformal', 'bayes_quadratic', 'bayes_normal_V1', 'bayes_normal_V100']
L = len(LEVELS)
R = {m: dict(cover=np.zeros((S, L, nte), bool), width=np.zeros((S, L, nte)), IS=np.zeros((S, L, nte)),
             wis=np.zeros((S, nte)), ae=np.zeros((S, nte)), pred=np.zeros((S, nte))) for m in METHODS}
for m in ['bayes_normal', 'bayes_t', 'bayes_quadratic']:
    R[m]['lpd'] = np.zeros((S, nte)); R[m]['pit'] = np.zeros((S, nte))
test_idx = np.zeros((S, nte), int); stratum = np.zeros((S, nte), int); t_nu_med = np.zeros(S)
t0 = time.time()
for s in range(S):
    seed = CFG['base_seed'] + s
    rng = np.random.default_rng(seed); perm = rng.permutation(N); tr, te = perm[:ntr], perm[ntr:]
    xtr, ytr, xte, yte = x[tr], y[tr], x[te], y[te]; test_idx[s] = te
    d = np.abs(xte - xtr.mean()) / (0.5 * (xtr.max() - xtr.min()))
    ranks = np.empty(nte, int); ranks[np.argsort(d, kind='stable')] = np.arange(nte)
    stratum[s] = np.minimum(ranks * 3 // nte, 2)          # 0 near, 1 intermediate, 2 far (equal-size, rank-based; ties by index order)
    out = {}
    out['bayes_normal'] = bayes_normal(xtr, ytr, xte, yte)
    out['bayes_quadratic'] = bayes_normal(xtr, ytr, xte, yte, degree=2)
    out['bayes_normal_V1'] = bayes_normal(xtr, ytr, xte, v0=1.0)
    out['bayes_normal_V100'] = bayes_normal(xtr, ytr, xte, v0=100.0)
    out['fixed_width'] = fixed_width(xtr, ytr, xte)
    out['split_conformal'] = split_conformal(xtr, ytr, xte, np.random.default_rng(seed + 100000))
    rg = np.random.default_rng(seed + 200000)
    B, sg, nu = gibbs_t(design(xtr), ytr, rg, CFG['student_t_model']['n_iter_splits'], CFG['student_t_model']['burn_splits'])
    out['bayes_t'] = t_predictive(design(xte), yte, B, sg, nu, rg); t_nu_med[s] = np.median(nu)
    for m in METHODS:
        o = out[m]; c, w, IS, wis = interval_metrics(o['lo'], o['hi'], yte, o['med'])
        R[m]['cover'][s] = c; R[m]['width'][s] = w; R[m]['IS'][s] = np.where(np.isinf(IS), np.nan, IS); R[m]['wis'][s] = wis
        R[m]['ae'][s] = np.abs(yte - o['pred']); R[m]['pred'][s] = o['pred']
        for k in ('lpd', 'pit'):
            if k in R[m]: R[m][k][s] = o[k]
    if s % 20 == 0: print(s, round(time.time() - t0, 1), flush=True)
flat = {}
for m in METHODS:
    for k, v in R[m].items(): flat[f'{m}__{k}'] = v
np.savez_compressed('../out/validation_results.npz', test_idx=test_idx, stratum=stratum, t_nu_med=t_nu_med, **flat)
print('done', time.time() - t0)
