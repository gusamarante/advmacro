"""
Discretization of AR(1)

The output chart shows a histogram of the simulated values from the discretized
AR(1) process and the theoretical normal distribution for comparison.
"""
from numerical import DicreteAR1
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import numpy as np

# TODO Implement compare both methods
#  simulated series with same seed
#  histogram with theoretical
#  rho close to 1


n_grid = 5
rho = 0.9
sigma_eps = 0.1

tar = DicreteAR1(n=n_grid, rho=rho, sigma_eps=sigma_eps, method="rouwenhorst", m=3)
print("Grid")
print(pd.Series(tar.grid).round(3))

print("Transition Matrix")
print(pd.DataFrame(tar.transmat).round(3))

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
