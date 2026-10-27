"""Step 5: summaries, cluster (compound-level) bootstrap, paired method differences, tail summaries."""
import json, numpy as np, pandas as pd
from lib import *
Z = np.load('../out/validation_results.npz'); TL = np.load('../out/tail_results.npz'); tinfo = json.load(open('../out/tail_info.json'))
A = pd.read_csv('../out/analysis_set_289.csv'); C = len(A); B = CFG['bootstrap_B']
rng = np.random.default_rng(CFG['base_seed'] + 9000000)
W = rng.multinomial(C, np.ones(C) / C, size=B).astype(float)
METH = ['bayes_normal', 'bayes_t', 'fixed_width', 'split_conformal', 'bayes_quadratic', 'bayes_normal_V1', 'bayes_normal_V100']
MAIN = ['bayes_normal', 'bayes_t', 'fixed_width', 'split_conformal']
tidx = Z['test_idx']; S, m = tidx.shape; flat_idx = tidx.reshape(-1); strat = Z['stratum'].reshape(-1)
ncomp = np.bincount(flat_idx, minlength=C).astype(float)
def tot(arr_Sxm):   # (S,m) -> (C,)
    return np.bincount(flat_idx, weights=arr_Sxm.reshape(-1), minlength=C)
def tot_L(arr_SLm):  # (S,L,m) -> (C,L)
    return np.stack([tot(arr_SLm[:, k, :]) for k in range(arr_SLm.shape[1])], 1)
def ci(v): return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
def mace(cov): return np.abs(cov - LEVELS).mean(-1)
R = dict(meta=dict(n_compounds=C, n_splits=S, n_test_per_split=m, evals_per_level=S * m, evals_per_compound=dict(min=int(ncomp.min()), mean=float(ncomp.mean()), max=int(ncomp.max())),
                   n_unique_evaluated=int((ncomp > 0).sum())))
T = {}; res = {}
for mth in METH:
    cov = Z[f'{mth}__cover'].astype(float); wd = Z[f'{mth}__width']; IS = Z[f'{mth}__IS']; wis = Z[f'{mth}__wis']; ae = Z[f'{mth}__ae']
    Tc = tot_L(cov); Tw = tot_L(np.nan_to_num(wd, posinf=np.nan)); Tis = tot_L(np.nan_to_num(IS)); Tw_ = tot(wis); Tae = tot(ae)
    T[mth] = dict(cov=Tc, wd=Tw, IS=Tis, wis=Tw_, ae=Tae)
    est_cov = Tc.sum(0) / ncomp.sum(); bc = (W @ Tc) / (W @ ncomp)[:, None]
    spl = cov.mean(-1)    # (S,L)
    d = dict(coverage=est_cov.tolist(), coverage_ci=[ci(bc[:, k]) for k in range(len(LEVELS))],
             split_sd=spl.std(0, ddof=1).tolist(), split_ci=[[float(np.percentile(spl[:, k], 2.5)), float(np.percentile(spl[:, k], 97.5))] for k in range(len(LEVELS))],
             abs_err=np.abs(est_cov - LEVELS).tolist(), MACE=float(np.abs(est_cov - LEVELS).mean()), MACE_ci=ci(mace(bc)),
             MACE_split_mean=float(mace(spl).mean()), MACE_split_sd=float(mace(spl).std(ddof=1)),
             width=(Tw.sum(0) / ncomp.sum()).tolist(), width_ci90=ci(((W @ Tw) / (W @ ncomp)[:, None])[:, -1]),
             IS90=float(Tis[:, -1].sum() / ncomp.sum()), IS90_ci=ci((W @ Tis[:, -1]) / (W @ ncomp)),
             WIS=float(Tw_.sum() / ncomp.sum()), WIS_ci=ci((W @ Tw_) / (W @ ncomp)))
    spl_ae = ae
    d['MAE_split_mean'] = float(spl_ae.mean(1).mean()); d['MAE_split_sd'] = float(spl_ae.mean(1).std(ddof=1)); d['MAE_split_ci'] = [float(np.percentile(spl_ae.mean(1), q)) for q in (2.5, 97.5)]
    rm = np.sqrt((spl_ae ** 2).mean(1)); d['RMSE_split_mean'] = float(rm.mean()); d['RMSE_split_sd'] = float(rm.std(ddof=1)); d['RMSE_split_ci'] = [float(np.percentile(rm, q)) for q in (2.5, 97.5)]
    md = np.median(spl_ae, 1); d['MedAE_split_mean'] = float(md.mean()); d['MedAE_split_sd'] = float(md.std(ddof=1)); d['MedAE_split_ci'] = [float(np.percentile(md, q)) for q in (2.5, 97.5)]
    d['MAE_pooled'] = float(Tae.sum() / ncomp.sum()); d['MAE_pooled_ci'] = ci((W @ Tae) / (W @ ncomp))
    if f'{mth}__lpd' in Z.files:
        lpd = Z[f'{mth}__lpd']; Tl = tot(lpd); T[mth]['lpd'] = Tl
        d['LPD'] = float(Tl.sum() / ncomp.sum()); d['LPD_ci'] = ci((W @ Tl) / (W @ ncomp))
        pit = Z[f'{mth}__pit'].reshape(-1); d['pit_hist'] = np.histogram(pit, bins=10, range=(0, 1))[0].tolist()
    # distance strata at 90%
    st = {}
    for k, nm in enumerate(['near', 'intermediate', 'far']):
        sel = (strat == k).astype(float)
        c9 = cov[:, -1, :].reshape(-1) * sel; w9 = np.nan_to_num(wd[:, -1, :], posinf=np.nan).reshape(-1) * sel
        Tc9 = np.bincount(flat_idx, weights=c9, minlength=C); Tw9 = np.bincount(flat_idx, weights=w9, minlength=C); nk = np.bincount(flat_idx, weights=sel, minlength=C)
        T[mth][f'strat{k}'] = (Tc9, Tw9, nk)
        st[nm] = dict(n=int(nk.sum()), coverage=float(Tc9.sum() / nk.sum()), coverage_ci=ci((W @ Tc9) / (W @ nk)), width=float(Tw9.sum() / nk.sum()), width_ci=ci((W @ Tw9) / (W @ nk)))
    d['strata'] = st
    res[mth] = d
R['methods'] = res
# paired differences vs. bayes_normal (cluster bootstrap)
diffs = {}
for mth in [x for x in METH if x != 'bayes_normal']:
    a, b = T[mth], T['bayes_normal']
    cb = lambda M: (W @ M) / (W @ ncomp)[:, None]
    cov_a, cov_b = cb(a['cov']), cb(b['cov'])
    dd = dict(cov90=dict(est=float((a['cov'][:, -1].sum() - b['cov'][:, -1].sum()) / ncomp.sum()), ci=ci(cov_a[:, -1] - cov_b[:, -1])),
              MACE=dict(est=res[mth]['MACE'] - res['bayes_normal']['MACE'], ci=ci(mace(cov_a) - mace(cov_b))),
              width90=dict(est=res[mth]['width'][-1] - res['bayes_normal']['width'][-1], ci=ci(cb(a['wd'])[:, -1] - cb(b['wd'])[:, -1])),
              WIS=dict(est=res[mth]['WIS'] - res['bayes_normal']['WIS'], ci=ci((W @ (a['wis'] - b['wis'])) / (W @ ncomp))),
              MAE=dict(est=res[mth]['MAE_pooled'] - res['bayes_normal']['MAE_pooled'], ci=ci((W @ (a['ae'] - b['ae'])) / (W @ ncomp))))
    if 'lpd' in a and 'lpd' in b: dd['LPD'] = dict(est=res[mth]['LPD'] - res['bayes_normal']['LPD'], ci=ci((W @ (a['lpd'] - b['lpd'])) / (W @ ncomp)))
    for k, nm in enumerate(['near', 'intermediate', 'far']):
        ca, na = a[f'strat{k}'][0], a[f'strat{k}'][2]; cb_, nb = b[f'strat{k}'][0], b[f'strat{k}'][2]
        dd[f'cov90_{nm}'] = dict(est=float(ca.sum() / na.sum() - cb_.sum() / nb.sum()), ci=ci((W @ ca) / (W @ na) - (W @ cb_) / (W @ nb)))
    diffs[mth] = dd
R['diffs_vs_bayes_normal'] = diffs
# fixed-width vs conformal difference too
a, b = T['split_conformal'], T['fixed_width']
R['diff_conformal_minus_fixed'] = dict(cov90=dict(est=float((a['cov'][:, -1].sum() - b['cov'][:, -1].sum()) / ncomp.sum()), ci=ci((W @ a['cov'][:, -1]) / (W @ ncomp) - (W @ b['cov'][:, -1]) / (W @ ncomp))))
# t - normal log score
# --- tail summaries
Wt_cache = {}
def tail_summary():
    out = {}
    for part in tinfo:
        te = TL[f'{part}__te']; reg = TL[f'{part}__region']; n_t = len(te); out[part] = dict(info=tinfo[part], methods={})
        Wb = rng.multinomial(n_t, np.ones(n_t) / n_t, size=B).astype(float)
        base = {}
        for mth in ['bayes_normal', 'bayes_t', 'fixed_width', 'split_conformal', 'bayes_quadratic']:
            cov = TL[f'{part}__{mth}__cover']; wd = TL[f'{part}__{mth}__width']; wis = TL[f'{part}__{mth}__wis']; ae = TL[f'{part}__{mth}__ae']
            d = {}
            for rname, sel in [('all', np.ones(n_t, bool)), ('lower', reg == 'lower'), ('upper', reg == 'upper')]:
                if sel.sum() == 0: continue
                ws = Wb * sel[None, :]; nsel = ws.sum(1)
                c9 = cov[-1][sel].mean(); bc = (ws @ cov[-1]) / nsel; wd9 = wd[-1][sel].mean()
                lo_w, hi_w = wilson(c9 * sel.sum(), sel.sum())
                d[rname] = dict(n=int(sel.sum()), cov90=float(c9), cov90_ci=ci(bc), cov90_wilson=[lo_w, hi_w], width90=float(wd9), width90_ci=ci((ws @ wd[-1]) / nsel),
                                WIS=float(wis[sel].mean()), WIS_ci=ci((ws @ wis) / nsel), MAE=float(ae[sel].mean()), MAE_ci=ci((ws @ ae) / nsel))
                if rname == 'all':
                    cl = (ws @ cov.T) / nsel[:, None]; d[rname]['coverage_levels'] = cov[:, sel].mean(1).tolist(); d[rname]['MACE'] = float(np.abs(cov[:, sel].mean(1) - LEVELS).mean()); d[rname]['MACE_ci'] = ci(mace(cl))
            d['_cov'] = cov[-1]; d['_wis'] = wis; d['_ae'] = ae; d['_wd'] = wd[-1]
            base[mth] = d
        for mth in base:
            dd = {}
            if mth != 'bayes_normal':
                for rname in ['all', 'lower', 'upper']:
                    if rname not in base[mth]: continue
                    sel = np.ones(n_t, bool) if rname == 'all' else (reg == rname); ws = Wb * sel[None, :]; nsel = ws.sum(1)
                    dc = base[mth]['_cov'] - base['bayes_normal']['_cov']
                    dd[rname] = dict(cov90_diff=float(dc[sel].mean()), cov90_diff_ci=ci((ws @ dc) / nsel),
                                     WIS_diff=float((base[mth]['_wis'] - base['bayes_normal']['_wis'])[sel].mean()), WIS_diff_ci=ci((ws @ (base[mth]['_wis'] - base['bayes_normal']['_wis'])) / nsel))
            base[mth]['diff_vs_bayes_normal'] = dd
        for mth in base:
            for kk in ['_cov', '_wis', '_ae', '_wd']: base[mth].pop(kk)
        out[part]['methods'] = base
    return out
R['tail'] = tail_summary()
# per-compound tail table for the primary partition for figure
json.dump(R, open('../out/results.json', 'w'), indent=1, default=float)
print('unique evaluated', R['meta'])
for mth in MAIN + ['bayes_quadratic', 'bayes_normal_V1', 'bayes_normal_V100']:
    d = res[mth]; print(f"{mth:18s} cov90 {d['coverage'][-1]:.3f} {d['coverage_ci'][-1]} MACE {100*d['MACE']:.2f} {[round(100*v,2) for v in d['MACE_ci']]} w90 {d['width'][-1]:.3f} WIS {d['WIS']:.3f} IS90 {d['IS90']:.2f} MAE {d['MAE_split_mean']:.3f}±{d['MAE_split_sd']:.3f} RMSE {d['RMSE_split_mean']:.3f} LPD {d.get('LPD','-')}")
print(json.dumps({k: {kk: (vv['est'], vv['ci']) for kk, vv in v.items() if kk in ('cov90','MACE','WIS','width90','LPD')} for k, v in diffs.items()}, default=float, indent=0)[:3000])
