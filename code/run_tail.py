"""Step 3: out-of-support (tail-holdout) evaluations; primary 15-85 band plus additional pre-specified partitions."""
import json, time, numpy as np, pandas as pd
from lib import *
A = pd.read_csv('../out/analysis_set_289.csv'); x = A.x.values; y = A.y.values; N = len(A)
parts = {}
for lo_p, hi_p in CFG['tail_central_bands_percentile']:
    ql, qh = np.percentile(x, [lo_p, hi_p]); tr = (x >= ql) & (x <= qh)
    reg = np.where(x < ql, 'lower', np.where(x > qh, 'upper', 'central'))
    parts[f'central_{lo_p}_{hi_p}'] = (np.where(tr)[0], np.where(~tr)[0], reg)
f = CFG['one_sided_fraction']; q = np.quantile(x, f)
tr = x <= q; parts['upper_only'] = (np.where(tr)[0], np.where(~tr)[0], np.where(x > q, 'upper', 'central'))
q = np.quantile(x, 1 - f); tr = x >= q; parts['lower_only'] = (np.where(tr)[0], np.where(~tr)[0], np.where(x < q, 'lower', 'central'))
METHODS = ['bayes_normal', 'bayes_t', 'fixed_width', 'split_conformal', 'bayes_quadratic']
out = {}; info = {}
t0 = time.time()
for pi, (name, (tr, te, reg)) in enumerate(parts.items()):
    xtr, ytr, xte, yte = x[tr], y[tr], x[te], y[te]; m = len(te)
    info[name] = dict(n_train=int(len(tr)), n_test=int(m), n_lower=int((reg[te] == 'lower').sum()), n_upper=int((reg[te] == 'upper').sum()),
                      train_x_range=[float(xtr.min()), float(xtr.max())])
    res = {}
    res['bayes_normal'] = bayes_normal(xtr, ytr, xte, yte); res['bayes_quadratic'] = bayes_normal(xtr, ytr, xte, yte, degree=2)
    res['fixed_width'] = fixed_width(xtr, ytr, xte)
    seed = CFG['base_seed'] + 900000 + pi
    rg = np.random.default_rng(seed)
    B, sg, nu = gibbs_t(design(xtr), ytr, rg, CFG['student_t_model']['n_iter_tail'], CFG['student_t_model']['burn_tail'])
    res['bayes_t'] = t_predictive(design(xte), yte, B, sg, nu, rg)
    reps = [split_conformal(xtr, ytr, xte, np.random.default_rng(seed + 1000 + r)) for r in range(CFG['conformal']['tail_repeats'])]
    res['split_conformal'] = {k: np.mean([r[k] for r in reps], axis=0) for k in reps[0]}   # placeholder, metrics computed per replicate below
    for mth in METHODS:
        if mth == 'split_conformal':
            ms = [interval_metrics(r['lo'], r['hi'], yte, r['med']) for r in reps]
            cover = np.mean([a[0] for a in ms], axis=0); width = np.mean([a[1] for a in ms], axis=0)
            IS = np.mean([np.where(np.isinf(a[2]), np.nan, a[2]) for a in ms], axis=0); wis = np.mean([a[3] for a in ms], axis=0)
            ae = np.mean([np.abs(yte - r['pred']) for r in reps], axis=0)
        else:
            o = res[mth]; cover, width, IS, wis = interval_metrics(o['lo'], o['hi'], yte, o['med']); ae = np.abs(yte - o['pred'])
        for k, v in dict(cover=cover.astype(float), width=width, IS=IS, wis=wis, ae=ae).items(): out[f'{name}__{mth}__{k}'] = v
    out[f'{name}__region'] = reg[te].astype(str); out[f'{name}__te'] = te
    if name == 'central_15_85':
        info[name]['t_nu_median'] = float(np.median(nu)); info[name]['t_nu_ci'] = [float(v) for v in np.percentile(nu, [2.5, 97.5])]
    print(name, info[name], round(time.time() - t0, 1), flush=True)
np.savez_compressed('../out/tail_results.npz', **out); json.dump(info, open('../out/tail_info.json', 'w'), indent=1)
