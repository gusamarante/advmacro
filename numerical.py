"""
Numerical methods modules
"""
import numpy as np
from scipy.stats import norm


class Tauchen:
    """
    Tauchen's method for discretizing a continuous state space.

    z_t = rho * z_{t-1} + sigma * e_t, where e_t ~ N(0, 1).
    """

    def __init__(self, n, rho, sigma_eps, m=3):
        """
        Initialize the Tauchen object.

        Parameters
        ----------
        n: int
            Number of discrete states for z

        rho: float
            Autoregressive coefficient

        sigma_eps: float
            Standard deviation of the shock

        m: float
            Number of standard deviations of z to consider for the coverage of
            the grid (default is 3)
        """
        self.n = n
        self.rho = rho
        self.sigma_eps = sigma_eps
        self.sigma_z = sigma_eps / ((1 - rho ** 2) ** 0.5)
        self.grid = np.linspace(
            start=-m * self.sigma_z,
            stop=m * self.sigma_z,
            num=n,
        )
        self.omega = np.diff(self.grid)[0]
        self.transition_matrix = self._compute_probs()

    def simulate(self, n_periods):
        """
        Simulate the discretized process for a given number of periods.

        Parameters
        ----------
        n_periods: int
            Number of periods to simulate

        Returns
        -------
        z_simul: np.ndarray
            Simulated values of the discretized process
        """
        r0 = np.random.randint(0, self.n)
        rands = np.random.random(size=n_periods)
        cdf = np.cumsum(self.transition_matrix, axis=1)
        z_simul = np.zeros(n_periods)

        for count, r in enumerate(rands):
            z_simul[count] = self.grid[cdf[r0] >= r][0]
            r0 = self.n - (cdf[r0] >= r).sum()

        return z_simul

    def _compute_probs(self):
        pi = np.zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):
                z_j = self.grid[j]
                z_i = self.grid[i]

                if j == 0:
                    pi[i, j] = norm.cdf((z_j - self.rho * z_i + self.omega / 2) / self.sigma_eps)

                elif j == self.n - 1:
                    pi[i, j] = 1 - norm.cdf((z_j - self.rho * z_i - self.omega / 2) / self.sigma_eps)

                else:
                    pi[i, j] = (
                        norm.cdf((z_j - self.rho * z_i + self.omega / 2) / self.sigma_eps)
                        - norm.cdf((z_j - self.rho * z_i - self.omega / 2) / self.sigma_eps)
                    )
        return pi
