"""
Lucas Span-of-control (1978)

Consider a simplified version of Lucas (1978) span-ofcontrol model. Every
period, agents decide whether operate as an entrepreneur or work for the
market wage, w. In case the agent decides to be an entrepreneur, she runs a
business and hires workers to produce the final good using the following
production function:
    y = z * n^alpha    0 < alpha < 1
where z is the managerial ability and n is the number of workers hired.

Agents are heterogenous in their managerial ability, z, which has F(Z) as its
cdf, with support [z_min, inf).
"""
import matplotlib.pyplot as plt
from scipy.stats import pareto
from scipy.integrate import quad_vec
from scipy.optimize import root_scalar
import numpy as np
import getpass


# Parameters
alpha = 0.2  # Production function (0<alpha<1)
zmin = 1  # Minimum of the pareto distribution (>0)
gamma = 3 # Pareto distribution parameter (>2 to have variance)
assert gamma * (1 - alpha) - 1 > 0, "parameters not valid"

pdist = pareto(b=gamma, scale=zmin)  # Since we are working with the numerical solution, we can only use the pdf mehtod

def labor_demand(z, w):
    return ((alpha * z) / w) ** (1 / (1 - alpha))

def profit(z, w):
    return (1 - alpha) * ((alpha / w) ** (alpha / (1 - alpha))) * (z ** (1 / (1 - alpha)))

def inv_profit(i, w):
    return (i ** (1 - alpha)) / (((1 - alpha) ** (1 - alpha)) * ((alpha / w) ** alpha))

def inv_profit_deriv(i, w):
    return ((w * (1 - alpha)) / (alpha * i)) ** alpha

def cutoff(w):  # zbar
    return w / ((1 - alpha) ** (1 - alpha)*(alpha ** alpha))

def total_demand(zbar, w):  # LHS of the equilibrium condition
    if zbar < zmin:
        zbar = zmin

    ld = lambda z: labor_demand(z, w) * pdist.pdf(z)  # individual labor demand
    td, err, info = quad_vec(ld, zbar, np.inf, epsrel=1e-5, quadrature='gk21', full_output=True)

    if not info.success:
        raise ArithmeticError("Numerical integral did not converge")

    return td

def excess_demand(w):
    return total_demand(cutoff(w), w) - pdist.cdf(cutoff(w))

def income(z):
    if z <= zstar:
        return wstar
    else:
        return profit(z, wstar)

def income_cdf(i):
    if i < wstar:
        return 0
    else:
        return pdist.cdf(inv_profit(i, wstar))

def income_pdf(i):
    # TODO integral not adding up to 1, adding up to the share of enterpreneurs
    if i < wstar:
        return 0
    else:
        return pdist.pdf(inv_profit(i, wstar)) * inv_profit_deriv(i, wstar) / share_entre


res = root_scalar(excess_demand, x0=0.5, rtol=1e-5)

if not res.converged:
    raise ArithmeticError("Equilibrium wage not found")


wstar = res.root
zstar = cutoff(res.root)
share_work = pdist.cdf(zstar)
share_entre = 1 - share_work

print(f"Equilibrium wage is {wstar}")
print(f"Managerial ability cutoff is {zstar}")
print(f"Share of workers is {share_work}")
print(f"Share of enterpreneurs is {share_entre}")

avg_income = share_work * wstar + share_entre * quad_vec(lambda i: income_pdf(i) * i, 0, np.inf, epsrel=1e-5, quadrature='gk21', full_output=True)[0]
print(f"Average Income is {avg_income}")

second_moment = share_work * (wstar**2) + share_entre * quad_vec(lambda i: income_pdf(i) * (i ** 2), 0, np.inf, epsrel=1e-2, quadrature='gk21', full_output=True)[0]
stdev_income = np.sqrt(second_moment - avg_income**2)
print(f"Standard Deviation of Income is {stdev_income}")


# ===== Plot - Income as a function of ability =====
ngrid = 1000
ent_grid = np.linspace(zstar, pdist.mean() + 3 * np.sqrt(pdist.var()), ngrid)

size = 4
fig = plt.figure(figsize=(size * (16 / 9), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("Income as a function of ability")
ax.plot([zmin, zstar], [wstar, wstar], color="tab:blue", label="Workers")
ax.plot(ent_grid, [profit(ze, wstar) for ze in ent_grid], color="tab:orange", label="Entrepreneurs")
ax.axvline(zstar, color="tab:red", ls='--', label="$z^{\star}$")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel(r"$z$")
ax.set_ylabel(r"Income")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='upper left', frameon=True)

plt.tight_layout()

plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/PSET 0/figures/SOC Ability-Income.pdf')
plt.show()
plt.close()


# ===== Plot - Income distribution =====
ngrid = 1000
igrid = np.linspace(
    start=0,
    stop=profit(
        z=pdist.mean() + 3 * np.sqrt(pdist.var()),
        w=wstar,
    ),
    num=ngrid,
)

size = 4
fig = plt.figure(figsize=(size * (16 / 9), size))

ax = plt.subplot2grid((1, 2), (0, 0))
ax.set_title("Density")
ax.plot(igrid, [income_pdf(ii) for ii in igrid], color="tab:blue", label="PDF")
ax.axvline(wstar, color="tab:orange", ls='--', label="Equilibrium Wage")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel(r"Income")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='upper right', frameon=True)

ax = plt.subplot2grid((1, 2), (0, 1))
ax.set_title("Cumulative Distribution")
ax.plot(igrid, [income_cdf(ii) for ii in igrid], color="tab:blue", label="CDF")
ax.axvline(wstar, color="tab:orange", ls='--', label="Equilibrium Wage")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel(r"Income")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='lower right', frameon=True)

plt.tight_layout()

plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/PSET 0/figures/SOC Income Distribution.pdf')
plt.show()
plt.close()


# ===== Plot - Gini Index =====
# TODO still wrong
zgini = np.linspace(pdist.support()[0], pdist.mean() + 3 * np.sqrt(pdist.var()), ngrid)
igini = [income(z) for z in zgini]
zcdf = pdist.cdf(zgini)
icdf = [income_cdf(i) for i in igini]

size = 4
fig = plt.figure(figsize=(size * (16 / 9), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.set_title("Gini Index")
ax.plot(zcdf, icdf, color="tab:blue")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel(r"Income")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
# ax.legend(loc='upper right', frameon=True)

plt.tight_layout()

plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/PSET 0/figures/SOC Gini.pdf')
plt.show()
plt.close()
