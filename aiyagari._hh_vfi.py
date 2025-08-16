"""
Solve the Aiyagari Model using the method of value function iteration
"""
from numerical import DiscreteAR1, create_grid
import matplotlib.pyplot as plt
import numpy as np


# Structural Parameters
gamma = 2  # Intertemporal elasticity of substitution
phi = 0  # credit constraint
rho = 0.9  # Persistence of income shock
sigma_eps = 0.1  # Standard deviation of income shock process
beta = 0.96  # Impatience discount factor
r = 0.04  # Interest rate
w = 1.0  # Wage


# Solution method parameters
na = 3000  # Number of points in the asset grid
ns = 5  # Number of points in the AR(1) grid
a_max = 20  # Upper bound of the asset grid
maxiter_vfi = 1000  # Break point for the value function iteration
tol_vfi = 1e-6  # Convergence tolerance for the value function iteration


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
# grid_a = create_grid(na, -phi, a_max, grid_growth=0.01)

# Discrete grid for the AR(1)
dar = DiscreteAR1(
    n=ns,
    rho=rho,
    sigma_eps=sigma_eps,
    method='rouwenhorst',
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

    cont_val =  (V @ transmat_s.T)  # Continuation value (s_j, a_k)

    for sj_idx, sj in enumerate(grid_s):  # Iterate on every possible state
        for ai_idx, ai in enumerate(grid_a):
            c_choices = (1 + r) * ai + w * np.exp(sj) - grid_a  # Possibilities of consumption over the grid of a
            u_choices = utility(c_choices, gamma)
            rhs = u_choices + beta * cont_val[:, sj_idx] # RHS of the discrete bellman equation
            V_new[ai_idx, sj_idx] = np.max(rhs)  # Get the maximum value of rhs
            policy_idx[ai_idx, sj_idx] = np.argmax(rhs)  # TODO this can be done only once, after V converges

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

# Assets policy function
pa = grid_a[policy_idx]
# assert np.all(np.diff(pa, axis=0) >= 0), "Asset policy function not monotone"
# assert pa[-1, -1] < a_max, "Asset policy function binding on the upper grid. `a_max` may be too small"
print(f"borrowing should bind for low income state, and it is {pa[0,0]} (Should be equal to phi)")  # TODO may not be true fir high base wages

# Cash on hand grid
coh = (1 + r) * grid_a[:, None] + w * np.exp(grid_s[None, :])

# Consumption policy function
pc = coh - pa
# assert np.all(np.diff(pc, axis=0) >= 0), "Resulting policy function for consumption is not monotone"
# assert np.all(pc >= 0), "Resulting policy function for consumption is not always positive"


# ===== Plot Value Functions =====
size = 4
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("Value Function")
for ii in range(ns):
    ax.plot(grid_a, V[:, ii], label=fr"s={round(np.exp(grid_s[ii]), 2)}")
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel(r"$a$")
ax.set_ylabel(r"$V\left(a,s\right)$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

plt.tight_layout()

# plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/figures/Aiyagari VFI Value Functions.pdf')
plt.show()
plt.close()


# ===== Plot Policy Functions =====
size = 4
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 2), (0, 0))
ax.set_title("Savings Policy $a^\prime=g_a(a,s)$")
for ii in range(ns):
    ax.plot(grid_a, pa[:, ii], label=fr"s={round(np.exp(grid_s[ii]), 2)}")
ax.set_xlabel(r"$a$")
ax.set_ylabel(r"$a^\prime$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

ax = plt.subplot2grid((1, 2), (0, 1))
ax.set_title(r"Policy Function $c=g_c(a,s)$")
for ii in range(ns):
    ax.plot(grid_a, pc[:, ii], label=fr"s={round(np.exp(grid_s[ii]), 2)}")
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





