"""Step 4: full-data fit, diagnostics, prior predictive check, posterior predictive checks, sensitivity analyses."""
import json, numpy as np, pandas as pd
from scipy import stats, special
from lib import *
A = pd.read_csv('../out/analysis_set_289.csv'); x = A.x.values; y = A.y.values; yfirst = A.y_first.values; n = len(A)
X = design(x); D = {}
rng = np.random.default_rng(CFG['base_seed'] + 5000000)

# ---- full-data normal-error posterior
mn, Vn, an, bn = nig_posterior(X, y, M0, V0, A0, B0); df = 2 * an
se = np.sqrt(bn / an * np.diag(Vn)); tq = stats.t.ppf([.025, .975], df)
sig_draw = np.sqrt(bn / rng.gamma(an, size=200000))
D['normal_fit'] = dict(beta0=mn[0], beta1=mn[1], ci0=list(mn[0] + tq * se[0]), ci1=list(mn[1] + tq * se[1]),
                       sigma_mean=float(sig_draw.mean()), sigma_ci=list(np.percentile(sig_draw, [2.5, 97.5])), df=df)
pred_in = X @ mn; ae = np.abs(y - pred_in)
D['in_sample'] = dict(MAE=ae.mean(), RMSE=np.sqrt(((y - pred_in) ** 2).mean()), MedAE=np.median(ae),
                      r=float(np.corrcoef(x, y)[0, 1]), r2=float(np.corrcoef(x, y)[0, 1] ** 2))
# ---- OLS diagnostics
bo, res = ols_fit(X, y); sw = stats.shapiro(res)
e2 = res ** 2; lr = stats.linregress(x, e2)
R2 = lr.rvalue ** 2; bp_lm = n * R2; bp_p = 1 - stats.chi2.cdf(bp_lm, 1)
Xq = design(x, 2); bq, rq = ols_fit(Xq, y); s2q = rq @ rq / (n - 3); covq = s2q * np.linalg.inv(Xq.T @ Xq)
tq_stat = bq[2] / np.sqrt(covq[2, 2]); pq = 2 * (1 - stats.t.cdf(abs(tq_stat), n - 3))
# also quadratic in centred x for a well-conditioned test
xc = x - x.mean(); Xc = design(xc, 2); bc, rc = ols_fit(Xc, y); s2c = rc @ rc / (n - 3); covc = s2c * np.linalg.inv(Xc.T @ Xc)
D['diagnostics'] = dict(shapiro_W=sw.statistic, shapiro_p=sw.pvalue, skew=float(stats.skew(res)), excess_kurtosis=float(stats.kurtosis(res)),
                        bp_slope=lr.slope, bp_slope_p=lr.pvalue, bp_LM=bp_lm, bp_LM_p=bp_p,
                        curv_corr=float(np.corrcoef(res, xc ** 2)[0, 1]), curv_corr_p=float(stats.pearsonr(res, xc ** 2)[1]),
                        quad_coef_t=float(tq_stat), quad_coef_p=float(pq), quad_coef_centred=float(bc[2]), quad_coef_centred_ci=[float(bc[2] + s * 1.9681 * np.sqrt(covc[2, 2])) for s in (-1, 1)],
                        n_abs_std_resid_gt3=int((np.abs(res / res.std(ddof=2)) > 3).sum()),
                        resid_quantiles=np.quantile(res / res.std(ddof=2), [0, .01, .05, .5, .95, .99, 1]).tolist())
# QQ data for figure
D['qq'] = dict(theoretical=stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n).tolist(), sample=np.sort(res / res.std(ddof=2)).tolist())

# ---- Student-t model, full data, 4 chains
chains = []
for c in range(4):
    r = np.random.default_rng(CFG['base_seed'] + 6000000 + c); chains.append(gibbs_t(X, y, r, 60000, 10000))
def rhat_ess(a):   # a: (chains, draws) split-Rhat and a simple ESS
    m, k = a.shape; h = k // 2; sp = np.concatenate([a[:, :h], a[:, h:2 * h]], 0)
    W = sp.var(1, ddof=1).mean(); Bn = h * sp.mean(1).var(ddof=1); rh = np.sqrt(((h - 1) / h * W + Bn / h) / W)
    z = a - a.mean(1, keepdims=True); ac = []
    for ch in z:
        f = np.fft.rfft(ch, 2 * k); acf = np.fft.irfft(f * np.conj(f))[:k] / (ch @ ch); ac.append(acf)
    ac = np.mean(ac, 0); s = 0
    for t in range(1, k - 1, 2):
        pair = ac[t] + ac[t + 1]
        if pair < 0: break
        s += pair
    return float(rh), float(m * k / (1 + 2 * s))
Bs = np.stack([c[0] for c in chains]); Ss = np.stack([c[1] for c in chains]); Ns = np.stack([c[2] for c in chains])
D['t_fit'] = dict(beta0=Bs[:, :, 0].mean(), beta1=Bs[:, :, 1].mean(),
                  beta0_ci=list(np.percentile(Bs[:, :, 0], [2.5, 97.5])), beta1_ci=list(np.percentile(Bs[:, :, 1], [2.5, 97.5])),
                  sigma_mean=Ss.mean(), sigma_ci=list(np.percentile(Ss, [2.5, 97.5])),
                  nu_median=float(np.median(Ns)), nu_ci=list(np.percentile(Ns, [2.5, 97.5])), nu_prob_le10=float((Ns <= 10).mean()), nu_prob_le30=float((Ns <= 30).mean()),
                  nu_posterior={str(int(g)): float((Ns == g).mean()) for g in NU_GRID},
                  rhat_beta1=rhat_ess(Bs[:, :, 1])[0], ess_beta1=rhat_ess(Bs[:, :, 1])[1], rhat_sigma=rhat_ess(Ss)[0], ess_sigma=rhat_ess(Ss)[1],
                  rhat_nu=rhat_ess(Ns.astype(float))[0], ess_nu=rhat_ess(Ns.astype(float))[1], n_draws_total=int(Bs.shape[0] * Bs.shape[1]))

# ---- Prior predictive check
Np = 100000; rp = np.random.default_rng(CFG['base_seed'] + 7000000)
s2p = B0 / rp.gamma(A0, size=Np); bp_ = M0 + np.sqrt(s2p)[:, None] * rp.standard_normal((Np, 2)) * np.sqrt(np.diag(V0))
xi = rp.choice(x, Np); yp = bp_[:, 0] + bp_[:, 1] * xi + np.sqrt(s2p) * rp.standard_normal(Np)
xm = x.mean(); ypm = bp_[:, 0] + bp_[:, 1] * xm + np.sqrt(s2p) * rp.standard_normal(Np)
scale_prior_at_mean = float(np.sqrt(B0 / A0 * (1 + xm ** 2 * 10 + 10)))
D['prior_predictive'] = dict(frac_in_0_12=float(((yp >= 0) & (yp <= 12)).mean()), frac_in_obs_range=float(((yp >= y.min()) & (yp <= y.max())).mean()),
                             q_2p5_97p5_at_xmean=list(np.percentile(ypm, [2.5, 97.5])), q_25_75_at_xmean=list(np.percentile(ypm, [25, 75])),
                             frac_abs_gt_100=float((np.abs(yp) > 100).mean()), obs_y_range=[float(y.min()), float(y.max())],
                             prior_slope_q=list(np.percentile(bp_[:, 1], [2.5, 50, 97.5])), analytic_t_df=2 * A0, analytic_scale_at_xmean=scale_prior_at_mean,
                             obs_y_mean_prior_pctile=float(100 * (ypm < y.mean()).mean()))
D['prior_predictive_plot'] = dict(y=np.clip(ypm[:20000], -60, 60).tolist())
# alternative prior predictive for the V0=1 prior
s2q_ = B0 / rp.gamma(A0, size=Np); bq_ = M0 + np.sqrt(s2q_)[:, None] * rp.standard_normal((Np, 2)) * 1.0
yq = bq_[:, 0] + bq_[:, 1] * xi + np.sqrt(s2q_) * rp.standard_normal(Np)
D['prior_predictive']['V1_frac_in_0_12'] = float(((yq >= 0) & (yq <= 12)).mean()); D['prior_predictive']['V1_q_2p5_97p5'] = list(np.percentile(yq, [2.5, 97.5]))

# ---- Posterior predictive checks (full data): discrepancy on standardised residuals (Gelman, Meng & Stern 1996)
def disc(z):
    return dict(skew=stats.skew(z, axis=-1), kurt=stats.kurtosis(z, axis=-1), maxabs=np.abs(z).max(-1), tail=(np.abs(z) > 1.645).mean(-1))
def ppc(draws_beta, draws_sig, nu_draws):
    D_ = len(draws_sig); mu = draws_beta @ X.T
    zobs = (y[None, :] - mu) / draws_sig[:, None]
    eps = rng.standard_t(nu_draws[:, None] * np.ones((1, n))) if nu_draws is not None else rng.standard_normal((D_, n))
    zrep_ = eps if nu_draws is None else eps  # standardised by sigma (t has scale sigma)
    to, tr_ = disc(zobs), disc(zrep_)
    return {k: dict(obs_mean=float(to[k].mean()), rep_mean=float(tr_[k].mean()), ppp=float((tr_[k] >= to[k]).mean())) for k in to}
nd = CFG['ppc_draws']; r2 = np.random.default_rng(CFG['base_seed'] + 8000000)
s2d = bn / r2.gamma(an, size=nd); bd = mn + np.sqrt(s2d)[:, None] * (r2.standard_normal((nd, 2)) @ np.linalg.cholesky(Vn).T)
D['ppc_normal'] = ppc(bd, np.sqrt(s2d), None)
idx = r2.choice(Bs.shape[0] * Bs.shape[1], nd, replace=False)
D['ppc_t'] = ppc(Bs.reshape(-1, 2)[idx], Ss.reshape(-1)[idx], Ns.reshape(-1)[idx])

# ---- Prior sensitivity (five specifications) and aggregation sensitivity
sens = []
for nm, m0, v in [('V0=1 (centre 0,1)', [0, 1], 1.0), ('V0=10 (primary)', [0, 1], 10.0), ('V0=100 (centre 0,1)', [0, 1], 100.0),
                  ('centre slope 0.5, intercept 0 (V0=10)', [0, .5], 10.0), ('centre intercept 1, slope 1 (V0=10)', [1, 1], 10.0)]:
    p = nig_posterior(X, y, np.array(m0, float), np.diag([v, v]), A0, B0); sens.append(dict(spec=nm, intercept=p[0][0], slope=p[0][1]))
D['prior_sensitivity'] = sens
pf = nig_posterior(X, yfirst, M0, V0, A0, B0)
D['aggregation_sensitivity'] = dict(mean=[mn[0], mn[1]], first=[pf[0][0], pf[0][1]])
# restricting to compounds with exactly one human label (no aggregation needed)
m1 = (A.n_human_labels == 1).values; p1 = nig_posterior(X[m1], y[m1], M0, V0, A0, B0)
D['single_label_only'] = dict(n=int(m1.sum()), intercept=p1[0][0], slope=p1[0][1])
json.dump(D, open('../out/diagnostics.json', 'w'), indent=1, default=float)
for k in ['normal_fit', 'in_sample', 'diagnostics', 't_fit', 'prior_predictive', 'ppc_normal', 'ppc_t', 'prior_sensitivity', 'aggregation_sensitivity', 'single_label_only']:
    print(k, json.dumps(D[k], default=float)[:1500])
