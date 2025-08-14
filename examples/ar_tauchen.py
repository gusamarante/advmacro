"""
Discretization of AR(1) process using Tauchen's method.

The output chart shows a histogram of the simulated values from the discretized
AR(1) process and the theoretical normal distribution for comparison.
"""
from numerical import Tauchen
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt


n_grid = 100
rho = 0.7
sigma_eps = 1.5**0.5

tar = Tauchen(n=n_grid, rho=rho, sigma_eps=sigma_eps, m=4)
print(tar.grid)
print(tar.transition_matrix)

data = pd.Series(tar.simulate(n_periods=50000))

count, bins, ignored = plt.hist(
    data, bins=n_grid, density=True,
    alpha=0.6, color='g', edgecolor='black'
)
plt.plot(
    np.linspace(min(bins), max(bins), 1000),
    norm.pdf(
        np.linspace(min(bins), max(bins), 1000),
        0,
        sigma_eps / ((1 - rho ** 2) ** 0.5),
    ),
    'r-',
    lw=2,
)
plt.show()
