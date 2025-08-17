"""
Discretization of AR(1)
"""
from numerical import DiscreteAR1
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import numpy as np
import getpass

# TODO Implement compare both methods
#  histogram with theoretical
#  rho close to 1


n_grid = 10
rho = 0.99
sigma_eps = 0.1
n_periods = 1000
random_seed = 666

art = DiscreteAR1(n=n_grid, rho=rho, sigma_eps=sigma_eps, method="tauchen", m=3)
arr = DiscreteAR1(n=n_grid, rho=rho, sigma_eps=sigma_eps, method="rouwenhorst")

print("--- Grid ---")
print("tauchen \n", pd.Series(art.grid).round(3))
print("rouwenhorst \n", pd.Series(arr.grid).round(3))

print("--- Transition Matrix ---")
print("tauchen \n", pd.DataFrame(art.transmat).round(3).round(3))
print("rouwenhorst \n", pd.DataFrame(arr.transmat).round(3))

data_t = pd.Series(art.simulate(n_periods=n_periods, seed=random_seed))
data_r = pd.Series(arr.simulate(n_periods=n_periods, seed=random_seed))

print(f"The correlation of the two series is {data_t.corr(data_r)}")



# ===== Plot Simulated Series =====
size = 4
fig = plt.figure(figsize=(size * (16 / 9), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.plot(data_t, label="Tauchen")
ax.plot(data_r, label="Rouwenhorst")
ax.axvline(0, color='black', lw=0.5)
ax.axhline(0, color='black', lw=0.5)
ax.set_xlabel(r"Periods $t$")
ax.set_ylabel(r"$z_t$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

plt.tight_layout()

plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/figures/Discrete AR1 Simulated Series.pdf')
plt.show()
plt.close()

# TODO make better distributions
# count, bins, ignored = plt.hist(
#     data_r, bins=n_grid, density=True,
#     alpha=0.6, color='g', edgecolor='black'
# )
# plt.plot(
#     np.linspace(min(bins), max(bins), 1000),
#     norm.pdf(
#         np.linspace(min(bins), max(bins), 1000),
#         0,
#         sigma_eps / ((1 - rho ** 2) ** 0.5),
#     ),
#     'r-',
#     lw=2,
# )
# plt.show()
