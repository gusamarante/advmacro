"""
Numerical methods modules
"""

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
        # TODO Implement
        #  the grid
        #  the probabilities
        #  the simulation of the discretized process
