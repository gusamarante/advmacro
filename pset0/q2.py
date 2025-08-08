import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import getpass


# parameters
alpha = 0.3
beta = 0.96
delta = 0.1

# number of points in the grid of k
nk = 1000

# Convergence parameters
maxiter = 1000
tol = 1e-6

# Steady state of capital
k_ss = (alpha / ((1 / beta) - (1 - delta))) ** (1 / (1 - alpha))

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
    if diff < tol:
        print(f'Convergence achieved after {ii + 1} iteations')
        break

    # Check max iterations
    if ii > maxiter:
        print('Maximum iterations reached. No convergence of the value function')
        break

    V = V_new
    # TODO save iterations to DF here

policy_function = gk[policy_idx]  # k-prime as a function of state k


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


# TODO chart of the convergence
# TODO chart of the trajectory




