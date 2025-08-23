"""
Solve the Aiyagari Model using the endogenous grid method
"""
from numerical import DiscreteAR1, create_grid, stationary_dist
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numba import njit
import pandas as pd
import numpy as np
import getpass


# Model Parameters
beta = 0.96
gamma = 2
phi = 0
rho = 0.9
sigma = 0.1
r = 0.04
w = 1

# Numerical Parameters
na = 300
ns = 7
amax = 250
grid_growth = 0.025
maxiter = 50_000
tol = 1e-11

# Discrete grid for the AR(1)
dar = DiscreteAR1(n=ns, rho=rho, sigma_eps=sigma, method='rouwenhorst', tol=tol, maxiter=maxiter)
grid_s, inv_dist_s, transmat_s = np.exp(dar.grid), dar.inv_dist, dar.transmat

# Discrete grid for assets
grid_a = create_grid(n=na, min_val=-phi, max_val=amax, grid_growth=grid_growth)

# Policy functions
grid_a_2d = np.repeat(grid_a, ns).reshape(na, ns)
grid_s_2d = np.repeat(grid_s, na).reshape(ns, na).T
grid_coh = grid_s_2d * w + (1 + r) * grid_a_2d

pc = grid_coh - grid_a_2d
pa = np.zeros((na, ns))


# ===== EGM =====
for ii in range(maxiter):

    # RHS of euler equation
    expect = beta * (1 + r) * ((pc ** (-gamma)) @ transmat_s.T)

    # Invert marginal utility to find current consumption
    current_c = expect ** (- 1 / gamma)

    # Endogenous asset grid
    endog_ap = (current_c + grid_a_2d - w * grid_s_2d) / (1 + r)

    # Interpolate to find the correct asset policy function
    pa_new = np.zeros((na, ns))

    for s_idx in range(ns):
        pa_new[:, s_idx] = interp1d(
            x=endog_ap[:, s_idx],  # Sample x, the endogenous grid of assets
            y=grid_a,  # Sample y - the asset grid
            kind='linear',
            fill_value="extrapolate",
        )(grid_a)

        # Check if credit constraint is valid
        for a_idx in range(na):
            if pa_new[a_idx, s_idx] < - phi:
                pa_new[a_idx, s_idx] = - phi
            else:
                break # we can stop searching after the kink due to monotonicity

    pc_new = grid_coh - pa_new

    # Check convergence
    diff = np.max(np.abs(pc_new - pc))
    pc = pc_new
    pa = pa_new

    if ii % 50 == 0:
        print(f"Iteration {ii} with diff = {diff}")

    if diff < tol:
        print(f'Convergence achieved after {ii + 1} iteations')
        break

else:
    raise ArithmeticError('Maximum iterations reached. Convergence not achieved')


# maximum contrained asset level for each state
for ii in range(ns):
    try:
        print(f"a_bar_{ii+1}", np.max(grid_a[np.isclose(pa[:, ii], 0)]))
    except ValueError:
        continue



# ===== Non-Stochastic Simulation =====
nus = np.zeros((na, ns), dtype=int)  # Indexes of the lower bound of the intervals for the non-stochastic simulation
ps = np.zeros((na, ns))  # "probability" to be assined to the lower bound of the interval of the non-stochastic simulation

for s in range(ns):
    nus[:, s] = np.searchsorted(grid_a, pa[:, s], side='right') - 1  # index of the upper bound of the interval
    a_low = grid_a[nus[:, s]]
    a_high = grid_a[np.minimum(nus[:, s] + 1, na - 1)]
    ps[:, s] = (a_high - pa[:, s]) / (a_high - a_low)

ps = np.maximum(np.minimum(ps, 1), 0)

# ===== Stationary Distribution =====
# We could iterate on every point of the transition matrix, but since its size
# is (na*ns X na*ns) and it is sparse, we can do something smarter. We iterate
# the stationary distribution directly, and not the full matrix, and update only the values of its
# relevant indexes

stat_dist = np.ones((na, ns)) / (na * ns)

@njit
def find_stat_dist(stat_dist_init, a_idx, p_vals, transmat, maxiter, tol):

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

        if d < tol:
            print(f'Convergence achieved after {ii + 1} iteations')
            break

    else:
        raise ArithmeticError('Maximum iterations reached. Convergence not achieved')

    return stat_dist_init


stat_dist = find_stat_dist(stat_dist, nus, ps, transmat_s, maxiter, tol)


# ===== Plot Policy Functions =====
size = 5
fig = plt.figure(figsize=(size * (16 / 5), size))

ax = plt.subplot2grid((1, 2), (0, 0))
ax.set_title("Savings Policy $a^\prime=g_a(a,s)$")
for ii in range(ns):
    ax.plot(grid_a, pa[:, ii], label=fr"s={round(grid_s[ii], 2)}")
ax.set_xlabel(r"$a$")
ax.set_ylabel(r"$a^\prime$")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='upper left', frameon=True)

ax = plt.subplot2grid((1, 2), (0, 1))
ax.set_title(r"Policy Function $c=g_c(a,s)$")
for ii in range(ns):
    ax.plot(grid_a, pc[:, ii], label=fr"s={round(grid_s[ii], 2)}")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel(r"$a$")
ax.set_ylabel(r"$c$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='lower right', frameon=True)

plt.tight_layout()

plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/PSET 1/figures/q1b policy functions rho {rho} sigma {sigma}.pdf')
plt.show()
plt.close()


# ===== Plot A condtional on S =====
size = 5
fig = plt.figure(figsize=(size * (16 / 5), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("Distributions of $a$ conditional on $s$")
for ii in range(ns):
    ax.plot(grid_a, stat_dist[:, ii] / stat_dist[:, ii].sum(), label=fr"s={round(grid_s[ii], 2)}")
ax.set_xlabel(r"$a$")
ax.set_ylabel("Density")
ax.set_xlim(None, 100)
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

plt.tight_layout()

plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/PSET 1/figures/q1b distribution of a conditional on s.pdf')
plt.show()
plt.close()


# ===== Plot Stationary Distribution =====
size = 5
fig = plt.figure(figsize=(size * (16 / 5), size))

ax = plt.subplot2grid((1, 2), (0, 0))
ax.set_title("Distribution of $s$")
ax.plot(grid_s, inv_dist_s, label="from AR(1)")
ax.plot(grid_s, stat_dist.sum(axis=0), label="from invariant distribution")
ax.set_xlabel(r"$s$")
ax.set_ylabel("Density")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

ax = plt.subplot2grid((1, 2), (0, 1))
ax.set_title("Distribution of $a$")
ax.plot(grid_a, stat_dist.sum(axis=1), label="from invariant distribution")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel(r"$a$")
ax.set_ylabel("Density")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

plt.tight_layout()

plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/PSET 1/figures/q1b Stationary distribution.pdf')
plt.show()
plt.close()
