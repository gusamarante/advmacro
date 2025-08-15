"""
Value Function Iteration for the Neoclassical Growth Model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import getpass
from scipy.interpolate import CubicSpline


# parameters
alpha = 0.3
beta = 0.96
delta = 0.1

# number of points in the grid of k
nk = 5000

# number of points in the trajectory of k
n_trajectory = 50

# Convergence parameters
maxiter = 1000
tol = 1e-8

# Steady states
k_ss = (alpha / ((1 / beta) - (1 - delta))) ** (1 / (1 - alpha))  # capital
y_ss = k_ss ** alpha
i_ss = delta * k_ss
c_ss = y_ss - i_ss

# grid of capital
gk = np.linspace(2*k_ss/nk, 2*k_ss, num=nk)

# log utility; very negative for infeasible consumption (c <= 0)
def utility(c):
    u = np.full_like(c, -1.0e12, dtype=float)
    mask = c > 0
    u[mask] = np.log(c[mask])
    return u

# initial guess of the value function
V = np.zeros(nk)

# Saves the iterations of V
V_iters = pd.DataFrame(columns=gk)

# initial guess of the policy function (indexes)
policy_idx = np.full(nk, 0, dtype=int)


# ===== Value Function Iteration =====
for ii in range(maxiter):  # ii-th iteration of the value function

    V_new = np.empty_like(V)

    for k0_idx, k0 in enumerate(gk):  # Iterate on every possible state

        # grid of possible consumptions given state k0, computed over the grid of possible kp (gk)
        c_choices = k0 ** alpha + (1 - delta) * k0 - gk

        # get the utility of these consumption options
        u_choices = utility(c_choices)

        # RHS of the Bellman equation
        rhs = u_choices + beta * V

        # Get the maximum value of rhs
        V_new[k0_idx] = np.max(rhs)

        # Get the INDEX of rhs that maximizes it (for policy function)
        policy_idx[k0_idx] = np.argmax(rhs)

    # Check convergence
    diff = np.abs(V_new - V).max()

    if ii % 10 == 0:
        print(f"Iteration {ii} with diff = {diff}")

    if diff < tol:
        print(f'Convergence achieved after {ii + 1} iteations')
        break

    V_iters.loc[ii] = V
    V = V_new

else:
    # Max iterations reached
    raise ArithmeticError('Maximum iterations reached. No convergence of the value function')


# Compute the policy function and interpolate
policy_function = gk[policy_idx]  # k-prime as a function of state k
p_func = CubicSpline(gk, policy_function)

# Compute the simulated trajectories
k_trajectory = pd.Series(data={0: gk[0]})
for ii in range(1, n_trajectory):
    k_trajectory.loc[ii] = p_func(k_trajectory.loc[ii - 1])

y_trajectory = k_trajectory ** alpha
i_trajectory = k_trajectory - (1 - delta) * k_trajectory.shift(1)
c_trajectory = y_trajectory - i_trajectory


# ===== Plot of the Functions =====
size = 4
fig = plt.figure(figsize=(size * (16 / 7.3), size))

ax = plt.subplot2grid((1, 2), (0, 0))
ax.set_title("Value Function")
ax.plot(gk, V, color="tab:blue")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel(r"$k_t$")
ax.set_ylabel(r"$V\left(k_t\right)$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

ax = plt.subplot2grid((1, 2), (0, 1))
ax.set_title("Policy Function")
ax.plot(gk, policy_function, color="tab:green")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.axline((0, 0), (1, 1), color="grey", ls='--', lw=0.5, label="45-degree line")
ax.scatter(k_ss, k_ss, color="tab:red", label="steady state")
ax.set_xlabel(r"$k_t$")
ax.set_ylabel(r"$k_{t+1}$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='upper left', frameon=True)

plt.tight_layout()

plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/PSET 0/figures/VFI Functions.pdf')
plt.show()
plt.close()


# ===== Plot the convergence of V =====
size = 5
fig = plt.figure(figsize=(size * (16 / 9), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("Convergence of the Value Function")

for ii in [0, 1, 2, 10, 50]:
    ax.plot(gk, V_iters.loc[ii].values, label=f"Iteration {ii}")

ax.plot(gk, V_iters.iloc[-1].values, label=f"Converged", ls='--')
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel(r"$k_t$")
ax.set_ylabel(r"$V\left(k_t\right)$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='upper left', frameon=True)

plt.tight_layout()

plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/PSET 0/figures/VFI Convergence.pdf')
plt.show()
plt.close()


# ===== Plot of the trajectory =====
size = 6
fig = plt.figure(figsize=(size * (16 / 9), size))

ax = plt.subplot2grid((2, 2), (0, 0))
ax.set_title("Capital")
ax.plot(k_trajectory, color="tab:blue", label="Trajectory")
ax.axhline(0, color='black', lw=0.5)
ax.axhline(k_ss, color='tab:red', lw=1, ls='--', label='Steady State')
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$k_t$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)

ax = plt.subplot2grid((2, 2), (0, 1))
ax.set_title("Output")
ax.plot(y_trajectory, color="tab:blue", label="Trajectory")
ax.axhline(0, color='black', lw=0.5)
ax.axhline(y_ss, color='tab:red', lw=1, ls='--', label='Steady State')
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$y_t$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

ax = plt.subplot2grid((2, 2), (1, 0))
ax.set_title("Investment")
ax.plot(i_trajectory, color="tab:blue", label="Trajectory")
ax.axhline(0, color='black', lw=0.5)
ax.axhline(i_ss, color='tab:red', lw=1, ls='--', label='Steady State')
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$i_t$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

ax = plt.subplot2grid((2, 2), (1, 1))
ax.set_title("Consumption")
ax.plot(c_trajectory, color="tab:blue", label="Trajectory")
ax.axhline(0, color='black', lw=0.5)
ax.axhline(c_ss, color='tab:red', lw=1, ls='--', label='Steady State')
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$c_t$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

plt.tight_layout()

plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/PSET 0/figures/VFI Trajectories.pdf')
plt.show()
plt.close()
