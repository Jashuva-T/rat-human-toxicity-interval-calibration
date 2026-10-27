import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats
from lib import *
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8, 'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'legend.frameon': False, 'savefig.dpi': 300, 'figure.dpi': 150})
R = json.load(open('../out/results.json')); Dg = json.load(open('../out/diagnostics.json')); sel = pd.read_csv('../out/selection_comparison.csv')
A = pd.read_csv('../out/analysis_set_289.csv'); x = A.x.values; y = A.y.values
COL = {'bayes_normal': '#0072B2', 'bayes_t': '#009E73', 'fixed_width': '#D55E00', 'split_conformal': '#CC79A7', 'bayes_quadratic': '#E69F00'}
LAB = {'bayes_normal': 'Bayesian (normal errors)', 'bayes_t': 'Bayesian (Student-t errors)', 'fixed_width': 'Fixed-width residual', 'split_conformal': 'Split conformal', 'bayes_quadratic': 'Bayesian quadratic'}
MK = {'bayes_normal': 'o', 'bayes_t': 's', 'fixed_width': '^', 'split_conformal': 'D', 'bayes_quadratic': 'v'}
MAIN = ['bayes_normal', 'bayes_t', 'fixed_width', 'split_conformal']
mm = 1 / 25.4
def lab(ax, s): ax.text(-0.14, 1.05, s, transform=ax.transAxes, fontsize=10, fontweight='bold')

# Fig 1 reliability
fig, ax = plt.subplots(1, 2, figsize=(190 * mm, 80 * mm))
ax[0].plot([0, 1], [0, 1], color='0.5', lw=0.8, ls='--', label='Ideal')
for m in MAIN:
    d = R['methods'][m]; c = np.array(d['coverage']); ci = np.array(d['coverage_ci'])
    ax[0].plot(LEVELS, c, color=COL[m], marker=MK[m], ms=3.5, lw=1, label=LAB[m])
    ax[1].plot(LEVELS, 100 * (c - LEVELS), color=COL[m], marker=MK[m], ms=3.5, lw=1)
    if m in ('bayes_normal',):
        ax[0].fill_between(LEVELS, ci[:, 0], ci[:, 1], color=COL[m], alpha=0.15, lw=0)
        ax[1].fill_between(LEVELS, 100 * (ci[:, 0] - LEVELS), 100 * (ci[:, 1] - LEVELS), color=COL[m], alpha=0.15, lw=0)
ax[1].axhline(0, color='0.5', lw=0.8, ls='--')
ax[0].set_xlabel('Nominal interval level'); ax[0].set_ylabel('Empirical held-out coverage'); ax[0].legend(loc='upper left', fontsize=7)
ax[1].set_xlabel('Nominal interval level'); ax[1].set_ylabel('Coverage − nominal (percentage points)')
lab(ax[0], 'a'); lab(ax[1], 'b'); ax[0].set_aspect('equal'); fig.tight_layout(); fig.savefig('../out/figs/Fig1_reliability.png'); plt.close(fig)

# Fig 2 fit
mn, Vn, an, bn = nig_posterior(design(x), y, M0, V0, A0, B0)
g = np.linspace(0.6, 6.8, 300); loc, sc, df = nig_predictive(design(g), (mn, Vn, an, bn))
fig, ax = plt.subplots(figsize=(120 * mm, 90 * mm))
for lvl, al in [(0.9, 0.15), (0.5, 0.30)]:
    q = stats.t.ppf((1 + lvl) / 2, df); ax.fill_between(g, loc - q * sc, loc + q * sc, color=COL['bayes_normal'], alpha=al, lw=0, label=f'{int(lvl*100)}% posterior predictive interval')
ax.scatter(x, y, s=8, color='0.25', alpha=0.7, lw=0, label='Compounds (n = 289)', zorder=3)
ax.plot(g, loc, color=COL['bayes_normal'], lw=1.6, label='Posterior mean'); ax.plot(g, g, color='0.4', ls='--', lw=0.9, label='1:1 reference')
ax.set_xlabel('Rat oral LD50 (−log10 mol/kg)'); ax.set_ylabel('Human oral TDLo (−log10 mol/kg)'); ax.legend(loc='upper left', fontsize=7); ax.set_xlim(0.6, 6.8)
fig.tight_layout(); fig.savefig('../out/figs/Fig2_fit.png'); plt.close(fig)

# Fig 3 distance strata
fig, ax = plt.subplots(1, 2, figsize=(190 * mm, 75 * mm)); st = ['near', 'intermediate', 'far']; xs = np.arange(3)
for i, m in enumerate(MAIN):
    s = R['methods'][m]['strata']; off = (i - 1.5) * 0.17
    cv = np.array([s[k]['coverage'] for k in st]); ci = np.array([s[k]['coverage_ci'] for k in st])
    ax[0].errorbar(xs + off, cv, yerr=[cv - ci[:, 0], ci[:, 1] - cv], fmt=MK[m], color=COL[m], ms=4, lw=1, capsize=2, label=LAB[m])
    wv = np.array([s[k]['width'] for k in st]); wc = np.array([s[k]['width_ci'] for k in st])
    ax[1].errorbar(xs + off, wv, yerr=[wv - wc[:, 0], wc[:, 1] - wv], fmt=MK[m], color=COL[m], ms=4, lw=1, capsize=2)
ax[0].axhline(0.9, color='0.5', ls='--', lw=0.8); ax[0].set_ylabel('Coverage of nominal 90% interval'); ax[1].set_ylabel('Mean 90% interval width (log10 units)')
for a in ax: a.set_xticks(xs); a.set_xticklabels(['Near', 'Intermediate', 'Far']); a.set_xlabel('Distance from training-data centre (tercile)')
h_, l_ = ax[0].get_legend_handles_labels(); lab(ax[0], 'a'); lab(ax[1], 'b'); fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.legend(h_, l_, loc='lower center', ncol=4, fontsize=7); fig.savefig('../out/figs/Fig3_distance.png'); plt.close(fig)

# Fig 4 tail
fig, ax = plt.subplots(1, 2, figsize=(190 * mm, 80 * mm)); P = R['tail']['central_15_85']['methods']; regs = ['all', 'lower', 'upper']; xs = np.arange(3)
for i, m in enumerate(MAIN):
    cv = np.array([P[m][r]['cov90'] for r in regs]); ci = np.array([P[m][r]['cov90_ci'] for r in regs]); off = (i - 1.5) * 0.17
    ax[0].errorbar(xs + off, cv, yerr=[cv - ci[:, 0], ci[:, 1] - cv], fmt=MK[m], color=COL[m], ms=4, lw=1, capsize=2, label=LAB[m])
ax[0].axhline(0.9, color='0.5', ls='--', lw=0.8); ax[0].set_xticks(xs); ax[0].set_xticklabels([f"All tails\n(n = {P['bayes_normal']['all']['n']})", f"Lower tail\n(n = {P['bayes_normal']['lower']['n']})", f"Upper tail\n(n = {P['bayes_normal']['upper']['n']})"])
ax[0].set_ylabel('Coverage of nominal 90% interval'); ax[0].set_ylim(0.6, 1.02); ax[0].set_xlabel('Primary partition (central 15th–85th percentile training band)')
parts = ['central_25_75', 'central_20_80', 'central_15_85', 'central_10_90', 'lower_only', 'upper_only']; pl = ['25–75', '20–80', '15–85\n(primary)', '10–90', 'Lower\nonly', 'Upper\nonly']
for m in MAIN:
    ax[1].plot(range(6), [R['tail'][p]['methods'][m]['all']['cov90'] for p in parts], color=COL[m], marker=MK[m], ms=4, lw=1, label=LAB[m])
ax[1].legend(fontsize=7, loc='lower left')
ax[1].axhline(0.9, color='0.5', ls='--', lw=0.8); ax[1].set_xticks(range(6)); ax[1].set_xticklabels(pl); ax[1].set_ylabel('Coverage of nominal 90% interval'); ax[1].set_xlabel('Training band (percentile) / one-sided partition'); ax[1].set_ylim(0.6, 1.02)
lab(ax[0], 'a'); lab(ax[1], 'b'); fig.tight_layout(); fig.savefig('../out/figs/Fig4_tail.png'); plt.close(fig)

# Supp S1: prior predictive, QQ, residual hist, nu posterior
fig, ax = plt.subplots(2, 2, figsize=(190 * mm, 150 * mm)); yp = np.array(Dg['prior_predictive_plot']['y']); yp = yp[np.abs(yp) < 59.9]
ax[0, 0].hist(yp, bins=np.linspace(-60, 60, 121), color='0.6'); ax[0, 0].axvspan(0, 12, color=COL['bayes_normal'], alpha=0.2, label='Plausible range (0–12)'); ax[0, 0].axvspan(y.min(), y.max(), color=COL['fixed_width'], alpha=0.25, label='Observed range')
ax[0, 0].set_xlabel('Prior predictive human TDLo at mean rat LD50'); ax[0, 0].set_ylabel('Draws'); ax[0, 0].legend(fontsize=7); ax[0, 0].set_title('Prior predictive (truncated at ±60)', fontsize=8)
qq = Dg['qq']; ax[0, 1].scatter(qq['theoretical'], qq['sample'], s=6, color='0.25'); ax[0, 1].plot([-3, 3], [-3, 3], color=COL['fixed_width'], lw=0.9); ax[0, 1].set_xlabel('Normal quantiles'); ax[0, 1].set_ylabel('Standardised OLS residuals'); ax[0, 1].set_title('Residual normal Q–Q plot', fontsize=8)
bo, res = ols_fit(design(x), y); z = res / res.std(ddof=2); ax[1, 0].hist(z, bins=24, density=True, color='0.6'); gg = np.linspace(-3.5, 3.5, 200); ax[1, 0].plot(gg, stats.norm.pdf(gg), color=COL['fixed_width'], lw=1.2)
ax[1, 0].set_xlabel('Standardised residual'); ax[1, 0].set_ylabel('Density'); ax[1, 0].set_title(f"Residual distribution (skewness {Dg['diagnostics']['skew']:.2f})", fontsize=8)
nu = Dg['t_fit']['nu_posterior']; ax[1, 1].bar(range(len(nu)), list(nu.values()), color=COL['bayes_t']); ax[1, 1].set_xticks(range(len(nu))); ax[1, 1].set_xticklabels(list(nu.keys()), fontsize=7); ax[1, 1].set_xlabel('Degrees of freedom ν (grid)'); ax[1, 1].set_ylabel('Posterior probability'); ax[1, 1].set_title('Student-t model: posterior of ν', fontsize=8)
for a, s in zip(ax.ravel(), 'abcd'): lab(a, s)
fig.tight_layout(); fig.savefig('../out/figs/FigS1_diagnostics.png'); plt.close(fig)

# Supp S2: PIT
fig, ax = plt.subplots(1, 2, figsize=(190 * mm, 70 * mm))
for a, m in zip(ax, ['bayes_normal', 'bayes_t']):
    h = np.array(R['methods'][m]['pit_hist']); a.bar(np.arange(10) / 10 + 0.05, h / h.sum() * 10, width=0.09, color=COL[m]); a.axhline(1, color='0.4', ls='--', lw=0.8); a.set_xlabel('Probability integral transform (held-out)'); a.set_ylabel('Relative frequency'); a.set_title(LAB[m], fontsize=8)
lab(ax[0], 'a'); lab(ax[1], 'b'); fig.tight_layout(); fig.savefig('../out/figs/FigS2_PIT.png'); plt.close(fig)

# Supp S3: selection forest
s2 = sel[sel.kind.isin(['continuous', 'binary']) & ~sel.variable.isin(['Multi-fragment (salt/mixture)', 'Contains carbon'])].dropna(subset=['smd']).iloc[::-1]
fig, ax = plt.subplots(figsize=(150 * mm, 90 * mm)); ax.barh(range(len(s2)), s2.smd, color=np.where(np.abs(s2.smd) >= 0.2, COL['fixed_width'], '0.6')); ax.set_yticks(range(len(s2))); ax.set_yticklabels(s2.variable, fontsize=7)
ax.axvline(0, color='k', lw=0.7); [ax.axvline(v, color='0.5', ls=':', lw=0.7) for v in (-0.2, 0.2)]; ax.set_xlabel('Standardised mean difference (analysis set − other rat oral LD50 compounds)'); fig.tight_layout(); fig.savefig('../out/figs/FigS3_selection.png'); plt.close(fig)
print('ok')
