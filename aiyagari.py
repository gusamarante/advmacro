"""
Solve the Aiyagari Model
"""
from numerical import DicreteAR1
import matplotlib.pyplot as plt
import numpy as np


# Structural Parameters
gamma = 1.3  # Intertemporal elasticity of substitution
phi = 0  # credit constraint
rho = 0.7  # Persistence of income shock
sigma_eps = 1.5  # Standard deviation of income shock process
beta = 0.98  # Impatience discount factor
r = 0.04  # Interest rate  # TODO remove from here
w = 2  # Wage  # TODO remove from here


# Solution method parameters
na = 100  # Number of points in the asset grid
ns = 10  # Number of points in the AR(1) grid
a_max = 300  # Upper bound of the asset grid
maxiter_vfi = 1000  # Break point for the value function iteration
tol_vfi = 1e-4  # Convergence toletance for the value function iteration


def utility(c, g):
    u = np.full_like(c, -1.0e12, dtype=float)
    mask = c > 0

    if g == 1:
        u[mask] = np.log(c[mask])
    else:
        u[mask] = (c[mask] ** (1 - g)) / (1 - g)

    return u


# Discrete grid for assets
grid_a = np.linspace(start=-phi, stop=a_max, num=na)

# Discrete grid for the AR(1)
dar = DicreteAR1(
    n=ns,
    rho=rho,
    sigma_eps=sigma_eps,
    method='tauchen',
    m=3,
)
grid_s, inv_dist_s, transmat_s = dar.grid, dar.inv_dist, dar.transmat

# initial guess of the value function
V = np.zeros((na, ns))

# initial guess of the policy function (indexes)
policy_idx = np.full((na, ns), 0, dtype=int)

# ===== Value Function Iteration =====
for ii in range(maxiter_vfi):  # ii-th iteration of the value function

    V_new = np.empty_like(V)

    cont_val = beta * (V @ transmat_s.T)  # Continuation value (s_j, a_k)

    for sj_idx, sj in enumerate(grid_s):  # Iterate on every possible state
        for ai_idx, ai in enumerate(grid_a):
            c_choices = (1 + r) * ai + w * np.exp(sj) - grid_a  # Possibilities of consumption over the grid of a
            u_choices = utility(c_choices, gamma)
            rhs = u_choices + cont_val[ai_idx, sj_idx] # RHS of the discrete bellman equation
            V_new[ai_idx, sj_idx] = np.max(rhs)  # Get the maximum value of rhs
            policy_idx[ai_idx, sj_idx] = np.argmax(rhs)

    diff = np.abs(V_new - V).max()
    V = V_new

    if ii % 10 == 0:
        print(f"Iteration {ii} with diff = {diff}")

    if diff < tol_vfi:
        print(f'Convergence achieved after {ii + 1} iteations')
        break
else:
    # Max iterations reached
    raise ArithmeticError('Maximum iterations reached. No convergence of the value function')

print(grid_a[policy_idx])







# ===== Plot of the Functions =====
size = 4
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 2), (0, 0))
ax.set_title("Value Function")
ax.plot(grid_a, V[:, 0], label=fr"s={round(grid_s[0], 2)}")
ax.plot(grid_a, V[:, int(ns/2)], label=fr"s={round(grid_s[int(ns/2)])}")
ax.plot(grid_a, V[:, -1], label=fr"s={round(grid_s[-1], 2)}")
# ax.axhline(0, color='black', lw=0.5)
# ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel(r"$a$")
ax.set_ylabel(r"$V\left(a,s\right)$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='upper left', frameon=True)

# ax = plt.subplot2grid((1, 2), (0, 1))
# ax.set_title("Policy Function")
# ax.plot(gk, policy_function, color="tab:green")
# ax.axhline(0, color='black', lw=0.5)
# ax.axvline(0, color='black', lw=0.5)
# ax.axline((0, 0), (1, 1), color="grey", ls='--', lw=0.5, label="45-degree line")
# ax.scatter(k_ss, k_ss, color="tab:red", label="steady state")
# ax.set_xlabel(r"$k_t$")
# ax.set_ylabel(r"$k_{t+1}$")
# ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
# ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
# ax.legend(loc='upper left', frameon=True)

plt.tight_layout()

# plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/PSET 0/figures/VFI Functions.pdf')
plt.show()
plt.close()





