"""
Discretization of AR(1)

The output chart shows a histogram of the simulated values from the discretized
AR(1) process and the theoretical normal distribution for comparison.
"""
from numerical import DiscreteAR1
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import numpy as np

# TODO Implement compare both methods
#  simulated series with same seed
#  histogram with theoretical
#  rho close to 1


n_grid = 20
rho = 0.9
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


# ===== Plot Simulated Series =====
size = 4
fig = plt.figure(figsize=(size * (16 / 9), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("Simulated Series")
ax.plot(data_t, label="Tauchen")
ax.plot(data_r, label="Rouwenhorst")
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel(r"$a$")
ax.set_ylabel(r"$V\left(a,s\right)$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

plt.tight_layout()

# plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/figures/Aiyagari VFI Value Functions.pdf')  # TODO save and add to notes
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
