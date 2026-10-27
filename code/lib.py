"""Shared modelling and scoring functions."""
import json, math, numpy as np
from scipy import stats, special

CFG = json.load(open('config.json'))
LEVELS = np.array(CFG['levels'])
M0 = np.array(CFG['prior']['m0']); V0 = np.diag(CFG['prior']['V0_diag'])
A0 = CFG['prior']['a0']; B0 = CFG['prior']['b0_rate']
NU_GRID = np.array(CFG['student_t_model']['nu_grid'], float)
QPROBS = np.concatenate([(1 - LEVELS) / 2, (1 + LEVELS) / 2, [0.5]])

def design(x, degree=1):
    return np.column_stack([x ** k for k in range(degree + 1)])

def nig_posterior(X, y, m0, V0, a0, b0):
    V0i = np.linalg.inv(V0); Vn_inv = V0i + X.T @ X; Vn = np.linalg.inv(Vn_inv)
    mn = Vn @ (V0i @ m0 + X.T @ y)
    an = a0 + len(y) / 2
    bn = b0 + 0.5 * (y @ y + m0 @ V0i @ m0 - mn @ Vn_inv @ mn)
    return mn, Vn, an, bn

def nig_predictive(Xn, post):
    mn, Vn, an, bn = post
    loc = Xn @ mn
    scale = np.sqrt(bn / an * (1 + np.einsum('ij,jk,ik->i', Xn, Vn, Xn)))
    return loc, scale, 2 * an

def central_from_t(loc, scale, df):
    q = stats.t.ppf((1 + LEVELS) / 2, df)[:, None]
    return loc - q * scale, loc + q * scale

def gibbs_t(X, y, rng, n_iter, burn, m0=M0, V0=V0, a0=A0, b0=B0):
    """Bayesian linear regression with Student-t errors (scale-mixture Gibbs sampler, nu on a discrete grid)."""
    n, p = X.shape; V0i = np.linalg.inv(V0); m0V = V0i @ m0; m0q = m0 @ m0V
    lam = np.ones(n); nu = 10.0; an = a0 + n / 2
    B = np.empty((n_iter, p)); S = np.empty(n_iter); N = np.empty(n_iter)
    h = NU_GRID / 2; const = n * (h * np.log(h) - special.gammaln(h))
    for it in range(n_iter):
        Xw = X * lam[:, None]; Vn_inv = V0i + X.T @ Xw; Vn = np.linalg.inv(Vn_inv)
        mn = Vn @ (m0V + Xw.T @ y)
        bn = b0 + 0.5 * (lam @ (y * y) + m0q - mn @ Vn_inv @ mn)
        s2 = bn / rng.gamma(an)
        beta = mn + np.linalg.cholesky(s2 * Vn) @ rng.standard_normal(p)
        r = y - X @ beta
        lam = rng.gamma((nu + 1) / 2, 2.0 / (nu + r * r / s2))
        ll = const + (h - 1) * np.log(lam).sum() - h * lam.sum()
        pr = np.exp(ll - ll.max()); pr /= pr.sum()
        nu = rng.choice(NU_GRID, p=pr)
        B[it] = beta; S[it] = s2; N[it] = nu
    return B[burn:], np.sqrt(S[burn:]), N[burn:]

def t_predictive(Xn, y_new, B, sig, nu, rng):
    """Predictive quantiles / logscore / PIT from Gibbs draws."""
    mu = B @ Xn.T                                    # (D, m)
    eps = rng.standard_t(nu[:, None] * np.ones((1, Xn.shape[0])))
    yrep = mu + sig[:, None] * eps
    Q = np.quantile(yrep, QPROBS, axis=0)           # (2L+1, m)
    L = len(LEVELS); lo, hi, med = Q[:L], Q[L:2 * L], Q[-1]
    pred = mu.mean(0)
    out = dict(lo=lo, hi=hi, med=med, pred=pred)
    if y_new is not None:
        z = (y_new[None, :] - mu) / sig[:, None]; v = nu[:, None]
        lp = (special.gammaln((v + 1) / 2) - special.gammaln(v / 2) - 0.5 * np.log(v * np.pi) - np.log(sig)[:, None]
              - (v + 1) / 2 * np.log1p(z * z / v))
        out['lpd'] = special.logsumexp(lp, axis=0) - np.log(len(sig))
        out['pit'] = special.stdtr(v, z).mean(0)
    return out

def ols_fit(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); return beta, y - X @ beta

def fixed_width(x_tr, y_tr, x_te):
    Xtr, Xte = design(x_tr), design(x_te); beta, res = ols_fit(Xtr, y_tr)
    pred = Xte @ beta; q = np.quantile(np.abs(res), LEVELS)[:, None]
    return dict(lo=pred - q, hi=pred + q, med=pred, pred=pred)

def split_conformal(x_tr, y_tr, x_te, rng, frac=None):
    frac = CFG['conformal']['proper_train_fraction'] if frac is None else frac
    n = len(x_tr); perm = rng.permutation(n); npr = int(round(frac * n))
    ip, ic = perm[:npr], perm[npr:]
    beta, _ = ols_fit(design(x_tr[ip]), y_tr[ip])
    sc = np.sort(np.abs(y_tr[ic] - design(x_tr[ic]) @ beta)); nc = len(sc)
    k = np.ceil((nc + 1) * LEVELS).astype(int)
    q = np.where(k > nc, np.inf, sc[np.minimum(k, nc) - 1])[:, None]
    pred = design(x_te) @ beta
    return dict(lo=pred - q, hi=pred + q, med=pred, pred=pred)

def bayes_normal(x_tr, y_tr, x_te, y_te=None, degree=1, v0=None):
    p = degree + 1
    m0 = np.array([0., 1.] + [0.] * (p - 2)) if p >= 2 else M0
    V = np.diag([10.0] * p) if v0 is None else np.diag([v0] * p)
    post = nig_posterior(design(x_tr, degree), y_tr, m0, V, A0, B0)
    loc, scale, df = nig_predictive(design(x_te, degree), post)
    lo, hi = central_from_t(loc, scale, df)
    out = dict(lo=lo, hi=hi, med=loc, pred=loc)
    if y_te is not None:
        out['lpd'] = stats.t.logpdf(y_te, df, loc, scale); out['pit'] = stats.t.cdf(y_te, df, loc, scale)
    return out

def interval_metrics(lo, hi, y, med):
    cover = (y >= lo) & (y <= hi); width = hi - lo; a = (1 - LEVELS)[:, None]
    IS = width + (2 / a) * np.maximum(lo - y, 0) + (2 / a) * np.maximum(y - hi, 0)
    wis = (0.5 * np.abs(y - med) + (a / 2 * IS).sum(0)) / (len(LEVELS) + 0.5)
    return cover, width, IS, wis

def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
