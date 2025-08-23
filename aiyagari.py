from numerical import DiscreteAR1, create_grid
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from numba import njit
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class Aiyagari:

    def __init__(
            self,
            beta=0.96,
            gamma=2.0,
            alpha=0.33,
            delta=0.05,
            phi=0,
            tau_l=0,
            rho=0.9,
            sigma=0.1,
            na=300,
            ns=7,
            amax=250,
            grid_growth=0.025,
            ar1_method='rouwenhorst',
            maxiter=50_000,
            tol=1e-6,
            verbose=False,
    ):
        """
        Class to solve the Aiyagari model from

            "Uninsured Idiosyncratic Risk and Aggregate Saving" by S. Rao Aiyagari
            The Quarterly Journal of Economics
            Vol. 109, No. 3 (Aug., 1994), pp. 659-684 (26 pages)
            https://www.jstor.org/stable/2118417

        Includes a mechanism for income redistribution towards a universal basic income

        Parameters
        ----------
        beta: float
            Househould time discount factor

        gamma: float
            Intertemporal of elasticity of substitution / inverse of relative risk aversion

        alpha: float
            Cobb-Douglas production function parameter

        delta: float
            Capital depreciation rate

        phi: float
            Credit constraint / lower bound of the asset grid

        tau_l: float
            Tax rate on wages

        rho: float
            AR(1) coefficient for the labor income process

        sigma: float
            Standard deviation of the AR(1) shock of the labor income process

        na: int
            Size of the asset grid

        ns: int
            Number of states in the discretized AR(1) Process

        amax: float
            Upper bound of the asset grid

        grid_growth: float
            Parameter passed to the `create_grid` function to adjust the
            concentration of points closer to the origin

        ar1_method: str
            'tauchen` or 'rouwenhorst' methods to discretize the AR(1)
            process of labor income.

        maxiter: int
            Maximum number of iterations for ALL the numerical procedures
            of the model (Keep it high, some of the methods need it)

        tol: float
            Tolerance for convergence criterion

        verbose: bool
            If True, prints relevant steps of the numerical procedures
        """

        # Model Parameters
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.delta = delta
        self.phi = phi
        self.tau_l = tau_l
        self.rho = rho
        self.sigma = sigma

        # Numerical Parameters
        self.na = na
        self.ns = ns
        self.amax = amax
        self.grid_growth = grid_growth
        self.ar1_method = ar1_method
        self.maxiter = maxiter
        self.tol = tol
        self.verbose = verbose

        # Discrete grid for AR(1) of labor supply
        dar = DiscreteAR1(n=ns, rho=rho, sigma_eps=sigma, method=ar1_method, tol=tol, maxiter=maxiter)
        self.grid_s, self.inv_dist_s, self.transmat_s = np.exp(dar.grid), dar.inv_dist, dar.transmat
        self.l_bar = np.sum(self.inv_dist_s * self.grid_s)  # aggregate labor supply

        # Discrete grid for assets
        self.grid_a = create_grid(n=na, min_val=-phi, max_val=amax, grid_growth=grid_growth)

    def household(self, w, r):
        """
        Solves the household problem given the wage and the interest rate using the endogenous grid method.

        Parameters
        ----------
        w: float
            Wages

        r: float
            Interest Rate

        Returns
        -------
        pa: numpy.ndarray
            Policy function for saving, to be used in conjuntction with `grid_a`

        pc: numpy.ndarray
            Policy function for consumption, to be used in conjuntction with `grid_a`

        grid_coh_gross: numpy.ndarray
            Grid of gross income

        grid_coh_net: numpy.ndarray
            Grid of net income
        """

        # Universal basic income
        w_net = w * (1 - self.tau_l)
        T = self.tau_l * w * self.l_bar  # Total tax revenue
        ubi = T / self.l_bar  # Equally distributed ammount

        # Policy functions
        grid_a_2d = np.repeat(self.grid_a, self.ns).reshape(self.na, self.ns)
        grid_s_2d = np.repeat(self.grid_s, self.na).reshape(self.ns, self.na).T
        grid_coh_gross = grid_s_2d * w + (1 + r) * grid_a_2d
        grid_coh = grid_s_2d * w_net + (1 + r) * grid_a_2d + ubi

        pc = grid_coh - grid_a_2d

        # ===== Endogenous Grid Method =====
        for ii in range(self.maxiter):

            # RHS of euler equation
            expect = self.beta * (1 + r) * ((pc ** (-self.gamma)) @ self.transmat_s.T)

            # Invert marginal utility to find current consumption
            current_c = expect ** (- 1 / self.gamma)

            # Endogenous asset grid
            endog_ap = (current_c + grid_a_2d - w_net * grid_s_2d - ubi) / (1 + r)

            # Interpolate to find the correct asset policy function
            pa_new = np.zeros((self.na, self.ns))

            for s_idx in range(self.ns):
                pa_new[:, s_idx] = interp1d(
                    x=endog_ap[:, s_idx],  # Sample x, the endogenous grid of assets
                    y=self.grid_a,  # Sample y - the asset grid
                    kind='linear',
                    fill_value="extrapolate",
                )(self.grid_a)

                # Check if credit constraint is valid
                for a_idx in range(self.na):
                    if pa_new[a_idx, s_idx] < - self.phi:
                        pa_new[a_idx, s_idx] = - self.phi
                    else:
                        break  # we can stop searching after the kink due to monotonicity

            pc_new = grid_coh - pa_new

            # Check convergence
            diff = np.max(np.abs(pc_new - pc))
            pc = pc_new
            pa = pa_new

            if (ii % 100 == 0) and self.verbose:
                print(f"Household EGM - Iteration {ii} with diff = {diff}")

            if diff < self.tol:
                if self.verbose:
                    print(f'Household EGM - Convergence achieved after {ii + 1} iteations', "\n")
                break
        else:
            raise ArithmeticError('Household EGM: Maximum iterations reached. Convergence not achieved')

        return pa, pc, grid_coh_gross, grid_coh

    def invariant_dist(self, pa):
        """
        Finds the stationary distribution of wealth and labor income (a, s)

        Parameters
        ----------
        pa: numpy.ndarray
            Savings policy function from the househould

        Returns
        -------
        stat_dist: numpy.ndarray
            stationary distribution of (a, s)
        """
        # Allocating the policy function to the grid of asset (non-stochastic simulation)
        nus = np.zeros((self.na, self.ns), dtype=int)  # Indexes of the lower bound of the intervals for the non-stochastic simulation
        ps = np.zeros((self.na, self.ns))  # "probability" / mass to be assined to the lower bound of the interval of the non-stochastic simulation

        for s in range(self.ns):
            nus[:, s] = np.searchsorted(self.grid_a, pa[:, s], side='right') - 1  # index of the upper bound of the interval
            a_low = self.grid_a[nus[:, s]]
            a_high = self.grid_a[np.minimum(nus[:, s] + 1, self.na - 1)]
            ps[:, s] = (a_high - pa[:, s]) / (a_high - a_low)

        ps = np.maximum(np.minimum(ps, 1), 0)  # Adjust for numerical errors

        # ===== Stationary / Invariant Distribution =====
        # We could iterate on every point of the transition matrix, but since its size
        # is (na*ns X na*ns) and it is sparse, we can do something smarter. We iterate
        # the stationary distribution directly and update only the values of its
        # relevant indexes

        stat_dist = np.ones((self.na, self.ns)) / (self.na * self.ns)  # Initial guess with unit mass
        stat_dist = self._find_stat_dist(stat_dist, nus, ps, self.transmat_s, self.na, self.ns, self.tol, self.maxiter, self.verbose)
        return stat_dist

    def capital_demand(self, r):
        """
        Total capital demand from firms

        Parameters
        ----------
        r: float
            Interest rate

        Returns
        -------
        kd: float
            Capital demand
        """
        kd = (self.alpha / (r + self.delta)) ** (1 / (1 - self.alpha)) * self.l_bar
        w = (1 - self.alpha) * (kd / self.l_bar) ** self.alpha
        return kd, w

    def capital_supply(self, w, r):
        """
        Net supply of capital that households are willing to supply, given
        wages (w) and interest rates (r)

        Parameters
        ----------
        w: float
            Wages

        r: float
            Interest Rates

        Returns
        -------
        ks: float
            Net capital supply
        """
        # Solve household
        pol_a, pol_c, coh_gross, coh_net = self.household(w, r)

        # Compute Invariant distribution
        stat_dist = self.invariant_dist(pol_a)

        # Compute Excess Demand of asset market
        grid_a_2d = np.repeat(self.grid_a[None, :], self.ns, axis=0).T  # repeat columns of gA

        # Aggregate asset supply
        ks = (stat_dist * grid_a_2d).sum()
        return ks

    def solve_equilibrium(self):
        """
        Given the parameters, solves the model to find all the endogenous eleements

        Returns
        -------
        pol_a: numpy.ndarray
            Policy function for household savings. To be used in conjunction with self.grid_a

        pol_c: numpy.ndarray
            Policy function for household consumption. To be used in conjunction with self.grid_a

        stat_dist: numpy.ndarray
            Stationary distribution of the wealth and labor income (a, s)

        kd: float
            Equilibrium level of capital
        r: float
            Equilibrium interest rate

        w: float
            Equilibrium wage

        grid_coh_gross: numpy.ndarray
            Grid of gross income

        grid_coh_net: numpy.ndarray
            Grid of net income
        """
        # "brackets" for the Brent root-finding
        r0 = 0.001  # lower bound guess (make sure excess demand is negative)
        r1 = 1 / self.beta - 1  # upper bound guess (make sure excess demand is positive)
        r = brentq(self._excess_demand, r0, r1, xtol=self.tol, maxiter=self.maxiter)

        kd, w = self.capital_demand(r)
        pol_a, pol_c, coh_gross, coh_net = self.household(w, r)
        stat_dist = self.invariant_dist(pol_a)

        return pol_a, pol_c, stat_dist, kd, r, w, coh_gross, coh_net

    @staticmethod
    @njit
    def _find_stat_dist(stat_dist_init, a_idx, p_vals, transmat, na, ns, tol, maxiter, verbose):
        # There is no point in building the full transition function. We
        # iterate the stationary distribution directly, only in the relevant
        # indexes
        for ii in range(maxiter):
            stat_dist_new = np.zeros((na, ns))
            for s in range(ns):
                for a in range(na):
                    if stat_dist_init[a, s] > 0:  # If the stationary distribution already converged to zero, do not waste time on these
                        stat_dist_new[a_idx[a, s], s] += p_vals[a, s] * stat_dist_init[a, s]
                        stat_dist_new[np.minimum(a_idx[a, s] + 1, na - 1), s] += (1 - p_vals[a, s]) * stat_dist_init[a, s]
            stat_dist_new = stat_dist_new @ transmat
            stat_dist_new = np.minimum(np.maximum(stat_dist_new, 0), 1)

            d = np.max(np.abs(stat_dist_init - stat_dist_new))
            stat_dist_init = stat_dist_new

            if (ii % 100 == 0) and verbose:
                print(f"Stationary Distribution - Iteration {ii}")  # `d` is not printable inside numba-decorated function

            if d < tol:
                if verbose:
                    print(f'Stationary Distribution - Convergence achieved after {ii + 1} iteations', "\n")
                break
        else:
            raise ArithmeticError('Maximum iterations reached. Convergence not achieved')

        return stat_dist_init

    def _excess_demand(self, r):
        kd, w = self.capital_demand(r)
        ks = self.capital_supply(w, r)
        return (ks - kd) / ((ks + kd) / 2)  # Excess demand in percentage
