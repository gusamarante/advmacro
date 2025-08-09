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
alpha = 0.4  # Production function (0<alpha<1)
zmin = 1  # Minimum of the pareto distribution (>0)
gamma = 3 # Pareto distribution parameter (>2 to have variance)
assert gamma * (1 - alpha) - 1 > 0, "parameters not valid"

pdist = pareto(b=gamma, scale=zmin)  # Since we are working with the numerical solution, we can only use the pdf mehtod

def labor_demand(z, w):
    return ((alpha * z) / w) ** (1 / (1 - alpha))

def profit(z, w):
    return (1 - alpha) * ((alpha / w) ** (alpha / (1 - alpha))) * (z ** (1 / (1 - alpha)))

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


res = root_scalar(excess_demand, x0=0.5, rtol=1e-5)

if not res.converged:
    raise ArithmeticError("Equilibrium wage not found")

print(f"Equilibrium wage is {res.root}")
print(f"Managerial ability cutoff is {cutoff(res.root)}")
