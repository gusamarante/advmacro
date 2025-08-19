"""
Numerical methods modules
"""
from scipy.stats import norm
import numpy as np


class DiscreteAR1:
    """
    Discretization a continuous state space.

    z_t = rho * z_{t-1} + sigma * e_t, where e_t ~ N(0, 1).
    """

    def __init__(self, n, rho, sigma_eps, method="tauchen", m=3):
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

        method: str
            Determines the method used for discretization. Either "tauchen" or
            "rouwenhorst".

        m: float
            Number of standard deviations of z to consider for the coverage of
            the grid (default is 3). Only used if method is "tauchen".

        Attributes
        ----------
        grid: numpy.ndarry
            discrete grid for z

        transmat: numpy.ndarray
            transition matrix of the markov chain

        inv_dist: numpy.ndarray
            invariant/stationary distribution of z
        """
        self.n = n
        self.rho = rho
        self.sigma_eps = sigma_eps
        self.sigma_z = sigma_eps / ((1 - rho ** 2) ** 0.5)

        if method == "tauchen":
            self.grid, self.transmat = self._tauchen_probs(m)
        elif method == "rouwenhorst":
            self.grid, self.transmat = self._rouwenhorst_probs()
        else:
            raise ValueError("Method must be either 'tauchen' or 'rouwenhorst'")

        self.inv_dist = self._get_inv_dist()

    def simulate(self, n_periods, seed=None):
        """
        Simulate the discretized process for a given number of periods.

        Parameters
        ----------
        n_periods: int
            Number of periods to simulate

        seed: int
            Random seed for numpy

        Returns
        -------
        z_simul: np.ndarray
            Simulated values of the discretized process
        """
        rng = np.random.default_rng(seed)
        r0 = rng.integers(0, self.n)
        rands = rng.random(size=n_periods)
        cdf = np.cumsum(self.transmat, axis=1)
        z_simul = np.zeros(n_periods)

        for count, r in enumerate(rands):
            z_simul[count] = self.grid[cdf[r0] >= r][0]
            r0 = self.n - (cdf[r0] >= r).sum()

        return z_simul

    def _get_inv_dist(self):
        eigvals, eigvecs = np.linalg.eig(self.transmat.T)  # TODO this transpose is confusing
        idx = np.argmin(np.abs(eigvals - 1))  # Find index of eigenvalue 1
        v = np.real(eigvecs[:, idx])  # Get the corresponding eigenvector and correct for possible numerical values
        v = v / v.sum()  # Normalize to sum to 1
        return v

    def _tauchen_probs(self, m):
        grid = np.linspace(
            start=-m * self.sigma_z,
            stop=m * self.sigma_z,
            num=self.n,
        )
        omega = np.diff(grid)[0]
        pi = np.zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):
                z_j = grid[j]
                z_i = grid[i]

                if j == 0:
                    pi[i, j] = norm.cdf((z_j - self.rho * z_i + omega / 2) / self.sigma_eps)

                elif j == self.n - 1:
                    pi[i, j] = 1 - norm.cdf((z_j - self.rho * z_i - omega / 2) / self.sigma_eps)

                else:
                    pi[i, j] = (
                        norm.cdf((z_j - self.rho * z_i + omega / 2) / self.sigma_eps)
                        - norm.cdf((z_j - self.rho * z_i - omega / 2) / self.sigma_eps)
                    )
        return grid, pi

    def _rouwenhorst_probs(self):
        p = (1 + self.rho) / 2
        pi = np.array([
            [p, 1 - p],
            [1 - p, p],
        ])
        for i in range(2, self.n):
            pi = (
                    p * np.r_[np.c_[pi, np.zeros(i)] , [np.zeros(i + 1)]]
                    + (1 - p) * np.r_[np.c_[np.zeros(i), pi] , [np.zeros(i + 1)]]
                    + p * np.r_[[np.zeros(i+1)] ,  np.c_[np.zeros(i), pi]]
                    + (1 - p) * np.r_[[np.zeros(i+1)] ,  np.c_[pi, np.zeros(i)]]
            )
            pi[1:-1, :] = pi[1:-1, :] / 2

        m = np.sqrt(self.n - 1)
        grid = np.linspace(
            start=-m * self.sigma_z,
            stop=m * self.sigma_z,
            num=self.n,
        )

        return grid, pi


def create_grid(n, min_val, max_val, grid_growth=0.01):
    """
    Creat a grid of n points between min_val and max_val with a specified
    growth rate.

    Parameters
    ----------
    n: int
        Number of points in the grid

    min_val: float
        Minimum value of the grid

    max_val: float
        Maximum value of the grid

    grid_growth: float
        Growth rate of the grid. If 0, the grid is linear. If > 0, the grid is
        exponentially spaced.
    """
    if grid_growth == 0.0:
        grid =  np.linspace(min_val, max_val, n)

    elif grid_growth > 0.0:
        grid = np.zeros(n)
        for i in range(n):
            grid[i] = min_val + (max_val - min_val) * ((1 + grid_growth)**i -1) / ((1 + grid_growth)**(n - 1.0) - 1)
    else:
        raise ValueError("grid_growth must be non-negative")

    return grid
