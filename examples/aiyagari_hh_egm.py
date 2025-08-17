"""
Solve the Aiyagari Model using the endogenous grid method
"""
from numerical import DiscreteAR1, create_grid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np


# Model Parameters
beta = 0.96
gamma = 2
phi = 0
rho = 0.9
sigma_eps = 0.1
r = 0.04
w = 1

# Numerical Parameters
na = 300
ns = 7
amax = 250
grid_growth = 0.025
maxiter = 2000
tol = 1e-8

# Discrete grid for the AR(1)
dar = DiscreteAR1(n=ns, rho=rho, sigma_eps=sigma_eps, method='rouwenhorst')
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
    ap_new = np.zeros((na, ns))

    for s_idx in range(ns):
        # ap_new[:, s_idx] = np.interp(
        #     x=grid_a,  # queried x
        #     xp=endog_ap[:, s_idx],  # Sample x, the endogenous grid of assets
        #     fp=grid_a,  # Sample y - the asset grid
        # )

        ap_new[:, s_idx] = interp1d(
            x=endog_ap[:, s_idx],  # Sample x, the endogenous grid of assets
            y=grid_a,  # Sample y - the asset grid
            kind='linear',
            fill_value="extrapolate",
        )(grid_a)

        # Check if credit constraint is valid
        for a_idx in range(na):
            if ap_new[a_idx, s_idx] < - phi:
                ap_new[a_idx, s_idx] = - phi
            else:
                break # we can stop searching after the kink due to monotonicity

    cp_new = grid_coh - ap_new

    # Check convergence
    diff = np.max(np.abs(cp_new - pc))
    pc = cp_new
    pa = ap_new

    if ii % 10 == 0:
        print(f"Iteration {ii} with diff = {diff}")

    if diff < tol:
        print(f'Convergence achieved after {ii + 1} iteations')
        break

else:
    raise ArithmeticError('Maximum iterations reached. Convergence not achieved')


# ===== Plot Policy Functions =====
size = 4
fig = plt.figure(figsize=(size * (16 / 9), size))

ax = plt.subplot2grid((1, 2), (0, 0))
ax.set_title("Savings Policy $a^\prime=g_a(a,s)$")
for ii in range(ns):
    ax.plot(grid_a, pa[:, ii], label=fr"s={round(grid_s[ii], 2)}")
ax.set_xlabel(r"$a$")
ax.set_ylabel(r"$a^\prime$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

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
ax.legend(loc='best', frameon=True)

plt.tight_layout()

# plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/figures/Aiyagari VFI Policy Functions.pdf')
plt.show()
plt.close()
