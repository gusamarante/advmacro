from aiyagari import Aiyagari
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import getpass


tau_options = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65,  0.7, 0.8, 0.85, 0.9, 0.92, 0.93, 0.94, 0.95, 0.96,0.97, 0.98, 0.99]


results = pd.DataFrame()
for tl in tqdm(tau_options):
    ag = Aiyagari(ns=3, tau_l=tl)
    pol_a, pol_c, stat_dist, k, r, w, coh_gross, coh_net = ag.solve_equilibrium()

    results.loc[tl, "Interest Rate"] = r
    results.loc[tl, "Wages"] = w
    results.loc[tl, "Capital"] = k
    results.loc[tl, "Share of Constrained"] = stat_dist.sum(axis=1)[0]

    mean_a = np.sum(ag.grid_a * stat_dist.sum(axis=1))
    var_a = np.sum(((ag.grid_a - mean_a)**2) * stat_dist.sum(axis=1))
    results.loc[tl, "Var of a"] = var_a

    mean_coh_gross = np.sum(coh_gross * stat_dist)
    var_coh_gross = np.sum(((coh_gross - mean_coh_gross)**2) * stat_dist)
    results.loc[tl, "Var of gross income"] = var_coh_gross

    mean_coh_net = np.sum(coh_net * stat_dist)
    var_coh_net = np.sum(((coh_net - mean_coh_net)**2) * stat_dist)
    results.loc[tl, "Var of net income"] = var_coh_net


# ===== Plot Variance =====
size = 5
fig = plt.figure(figsize=(size * (16 / 9), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("Variance of Income for Different Levels of Tax Rates")
ax.plot(results.index, results['Var of gross income'], label="Gross Income")
ax.plot(results.index, results['Var of net income'], label="Net Income")
ax.set_xlabel(r"$\tau_l$")
ax.set_ylabel("Variance (log scale)")
ax.set_yscale('log')
ax.axvline(0, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

plt.tight_layout()

plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/PSET 1/figures/q2 variance evolution.pdf')
plt.show()
plt.close()


