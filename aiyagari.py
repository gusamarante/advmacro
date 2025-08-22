from numerical import DiscreteAR1, create_grid
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from numba import njit
import numpy as np


class Aiyagari:
    # TODO Documentation
    #  Atributes

    def __init__(
            self,
            beta=0.96,
            gamma=0.2,
            alpha=0.33,
            delta=0.05,
            phi=0,
            rho=0.9,
            sigma=0.1,
            na=300,
            ns=7,
            amax=250,
            grid_growth=0.025,
            ar1_method='rouwenhorst',
            maxiter=50_000,
            tol=1e-8,
            verbose=False,
    ):
        # TODO Documentation
        # Model Parameters
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.delta = delta
        self.phi = phi
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
        # TODO documentation

        # Policy functions
        grid_a_2d = np.repeat(self.grid_a, self.ns).reshape(self.na, self.ns)
        grid_s_2d = np.repeat(self.grid_s, self.na).reshape(self.ns, self.na).T
        grid_coh = grid_s_2d * w + (1 + r) * grid_a_2d

        pc = grid_coh - grid_a_2d
        pa = np.zeros((self.na, self.ns))

        # ===== Endogenous Grid Method =====
        for ii in range(self.maxiter):

            # RHS of euler equation
            expect = self.beta * (1 + r) * ((pc ** (-self.gamma)) @ self.transmat_s.T)

            # Invert marginal utility to find current consumption
            current_c = expect ** (- 1 / self.gamma)

            # Endogenous asset grid
            endog_ap = (current_c + grid_a_2d - w * grid_s_2d) / (1 + r)

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
                print(f'Household EGM - Convergence achieved after {ii + 1} iteations', "\n")
                break
        else:
            raise ArithmeticError('Household EGM: Maximum iterations reached. Convergence not achieved')

        return pa, pc

    def invariant_dist(self, pa):
        # TODO documentation

        # Non-stochastic simulation
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

    @staticmethod
    @njit
    def _find_stat_dist(stat_dist_init, a_idx, p_vals, transmat, na, ns, tol, maxiter, verbose):

        # There is no point in building the full transition function, we iterate the stationary distribution directly, only in the relevant indexes
        for ii in range(maxiter):  # TODO speed this up

            stat_dist_new = np.zeros((na, ns))
            for s in range(ns):
                for a in range(na):
                    if stat_dist_init[a, s] > 0:  # If the stationary distribution already converged to zero, do not waste time on these  # TODO test this
                        stat_dist_new[a_idx[a, s], s] += p_vals[a, s] * stat_dist_init[a, s]
                        stat_dist_new[np.minimum(a_idx[a, s] + 1, na - 1), s] += (1 - p_vals[a, s]) * stat_dist_init[a, s]
            stat_dist_new = stat_dist_new @ transmat

            d = np.max(np.abs(stat_dist_init - stat_dist_new))
            stat_dist_init = stat_dist_new

            if (ii % 100 == 0) and verbose:
                print(f"Stationary Distribution - Iteration {ii}")  # `d` is not printable inside numba-decorated function

            if d < tol:
                print(f'Stationary Distribution - Convergence achieved after {ii + 1} iteations', "\n")
                break
        else:
            raise ArithmeticError('Maximum iterations reached. Convergence not achieved')

        return stat_dist_init

    def solve_equilibrium(self):
        # TODO Documentation
        #  find r and w that clears all the market and stat dist

        # "brackets" for the Brent root-finding
        r0 = 0.001  # lower bound guess (make sure excess demand is negative)
        r1 = 1 / self.beta - 1  # upper bound guess (make sure excess demand is positive)

        def _obj_func(r):
            res = self._excess_demand(r)
            return res[0]

        # TODO tol may be too fine here
        r = brentq(_obj_func, r0, r1, xtol=0.0001)

        # TODO compute output
        return r

    def _excess_demand(self, r):
        # TODO Documentation
        #  excess demand

        # Compute Capital Demand and implied wage from the production function
        kd = (self.alpha / (r + self.delta)) ** (1 / (1 - self.alpha)) * self.l_bar  # capital demand
        w = (1 - self.alpha) * (kd / self.l_bar) ** self.alpha

        # Solve household
        pol_a, pol_c = self.household(w, r)

        # Compute Invariant distribution
        stat_dist = self.invariant_dist(pol_a)

        # Compute Excess Demand of asset market
        grid_a_2d = np.repeat(self.grid_a[None, :], self.ns, axis=0).T  # repeat columns of gA

        # Aggregate asset supply
        ks = (stat_dist * grid_a_2d).sum()

        # Excess demand in percentage
        exc_dem = (ks - kd) / ((ks + kd) / 2)

        return exc_dem, pol_a, pol_c, stat_dist, w, kd, ks







# ===== Example =====  # TODO encontrar condições para convergência
ag = Aiyagari(verbose=False)
print(ag.solve_equilibrium())

# print(ag._excess_demand(0.04)[0])

# pa, pc = ag.household(1, 0.04)
# stat_distri = ag.invariant_dist(pa)
# print(stat_distri)
# print(stat_distri.sum())
