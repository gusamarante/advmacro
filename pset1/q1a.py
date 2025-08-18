"""
Discretization of AR(1) Process
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import getpass
from numerical import DiscreteAR1
from scipy.stats import norm


# ======================================
# ===== Different Parametrizations =====
# ======================================
options_ns = [3, 5, 7, 11, 15, 21, 41]
options_rho = [0, 0.4, 0.8, 0.99]
options_sigma = [0.1, 1]
options_methods = ['tauchen', 'rouwenhorst']

results = pd.DataFrame(columns=['ns', 'rho', 'sigma', 'method', 'var', 'true var'])
idx_df = 0
for ns in options_ns:
    for rho in options_rho:
        for sigma in options_sigma:
            for method in options_methods:
                ar = DiscreteAR1(ns, rho, sigma, method)

                var = np.sum((ar.grid ** 2) * ar.inv_dist)  # We can ommit the mean, as there is no intercept

                results.loc[idx_df, "ns"] =  ns
                results.loc[idx_df, "rho"] = rho
                results.loc[idx_df, "sigma"] = sigma
                results.loc[idx_df, "method"] = method
                results.loc[idx_df, "var"] = var
                results.loc[idx_df, "true var"] = (sigma ** 2) / (1 - (rho ** 2))

                idx_df += 1

results.pivot(columns='method', index=['rho', 'sigma', 'ns']).to_clipboard()

results = results.drop('true var', axis=1)

# --- Plot of the trajectory ---
size = 6
fig = plt.figure(figsize=(size * (16 / 9), size))

rhop, sigmap = 0, 0.1
df2plot = results[(results['sigma'] == sigmap) & (results['rho'] == rhop)].pivot(index='ns', columns='method', values='var')
ax = plt.subplot2grid((2, 2), (0, 0))
ax.set_title(rf"$\rho={rhop}$")
ax = df2plot.plot(ax=ax, kind='bar')
ax.axhline(0, color='black', lw=0.5)
ax.axhline((sigmap ** 2) / (1 - (rhop ** 2)), color='red', lw=2, ls='--', label='true variance')
ax.set_xlabel(r"Grid Size $n_S$")
ax.set_ylabel(r"Variance")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc='lower left')

rhop, sigmap = 0.4, 0.1
df2plot = results[(results['sigma'] == sigmap) & (results['rho'] == rhop)].pivot(index='ns', columns='method', values='var')
ax = plt.subplot2grid((2, 2), (0, 1))
ax.set_title(rf"$\rho={rhop}$")
ax = df2plot.plot(ax=ax, kind='bar')
ax.axhline(0, color='black', lw=0.5)
ax.axhline((sigmap ** 2) / (1 - (rhop ** 2)), color='red', lw=2, ls='--', label='true variance')
ax.set_xlabel(r"Grid Size $n_S$")
ax.set_ylabel(r"Variance")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc='lower left')

rhop, sigmap = 0.8, 0.1
df2plot = results[(results['sigma'] == sigmap) & (results['rho'] == rhop)].pivot(index='ns', columns='method', values='var')
ax = plt.subplot2grid((2, 2), (1, 0))
ax.set_title(rf"$\rho={rhop}$")
ax = df2plot.plot(ax=ax, kind='bar')
ax.axhline(0, color='black', lw=0.5)
ax.axhline((sigmap ** 2) / (1 - (rhop ** 2)), color='red', lw=2, ls='--', label='true variance')
ax.set_xlabel(r"Grid Size $n_S$")
ax.set_ylabel(r"Variance")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc='lower left')

rhop, sigmap = 0.99, 0.1
df2plot = results[(results['sigma'] == sigmap) & (results['rho'] == rhop)].pivot(index='ns', columns='method', values='var')
ax = plt.subplot2grid((2, 2), (1, 1))
ax.set_title(rf"$\rho={rhop}$")
ax = df2plot.plot(ax=ax, kind='bar')
ax.axhline(0, color='black', lw=0.5)
ax.axhline((sigmap ** 2) / (1 - (rhop ** 2)), color='red', lw=2, ls='--', label='true variance')
ax.set_xlabel(r"Grid Size $n_S$")
ax.set_ylabel(r"Variance")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(frameon=True, loc='upper right')

plt.tight_layout()

plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/PSET 1/figures/q1a variances.pdf')
plt.show()
plt.close()


# ==================================
# ===== Invariant Distribution =====
# ==================================
options_ns = [5, 21, 51]
rho = 0.99
sigma = 0.1

size = 6
fig = plt.figure(figsize=(size * (16 / 9), size))


for idx, ns in enumerate(options_ns):
    tar = DiscreteAR1(ns, rho, sigma, 'tauchen', m=4)
    rar = DiscreteAR1(ns, rho, sigma, 'rouwenhorst')

    ax = plt.subplot2grid((3, 2), (idx, 0))
    ax.set_title(rf"Tauchen and $n_S={ns}$")
    omega = np.mean(np.diff(tar.grid))
    ax.bar(tar.grid, tar.inv_dist / omega, edgecolor='white', width=omega)
    normal_grid = np.linspace(min(tar.grid), max(tar.grid), 1000)
    ax.plot(
        normal_grid,
        norm.pdf(normal_grid, 0, sigma / ((1 - rho ** 2) ** 0.5)),
        color='red',
        ls='--',
        lw=2,
    )
    ax.set_xlim(-3, 3)


    ax = plt.subplot2grid((3, 2), (idx, 1))
    ax.set_title(rf"Rouwenhorst and $n_S={ns}$")
    omega = np.mean(np.diff(rar.grid))
    ax.bar(rar.grid, rar.inv_dist / omega, edgecolor='white', width=omega)
    normal_grid = np.linspace(min(rar.grid), max(rar.grid), 1000)
    ax.plot(
        normal_grid,
        norm.pdf(normal_grid, 0, sigma / ((1 - rho ** 2) ** 0.5)),
        color='red',
        ls='--',
        lw=2,
    )
    ax.set_xlim(-3, 3)

plt.tight_layout()

plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/PSET 1/figures/q1a invariant distribution.pdf')
plt.show()
plt.close()


# ================================
# ===== Simulated Timeseries =====
# ================================
options_ns = [5, 21, 51]
rho = 0.9
sigma = 0.1
n_periods = 800
seed = 666

size = 6
fig = plt.figure(figsize=(size * (16 / 9), size))


for idx, ns in enumerate(options_ns):
    tar = DiscreteAR1(ns, rho, sigma, 'tauchen', m=4)
    rar = DiscreteAR1(ns, rho, sigma, 'rouwenhorst')

    ax = plt.subplot2grid((3, 1), (idx, 0))
    ax.set_title(rf"$n_S={ns}$")
    ax.plot(tar.simulate(n_periods, seed), color="tab:blue", label='Tauchen')
    ax.plot(rar.simulate(n_periods, seed), color="tab:orange", label='Rouwenhorst')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_ylabel(r"$z_t$")
    ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.legend(loc='best', frameon=True)

plt.tight_layout()

plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/PSET 1/figures/q1a simulated trajectories.pdf')
plt.show()
plt.close()
