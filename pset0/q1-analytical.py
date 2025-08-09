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
from scipy.stats import pareto
import numpy as np
import matplotlib.pyplot as plt
import getpass


# Parameters
alpha = 0.4  # Production function (0<alpha<1)
zmin = 1  # Minimum of the pareto distribution (>0)
gamma = 3 # Pareto distribution parameter (>2 to have variance)


assert gamma * (1 - alpha) - 1 > 0, "parameters not valid"


ngrid = 1000

zstar = (((gamma - 1) / ((1 - alpha) * gamma - 1)) ** (1 / gamma)) * zmin


pdist = pareto(b=gamma, scale=zmin)

zgrid = np.linspace(pdist.support()[0], pdist.mean() + 3 * np.sqrt(pdist.var()), ngrid)
density = pdist.pdf(zgrid)

share_entre = 1 - pdist.cdf(zstar)


# ===== Plot of the trajectory =====
size = 5
fig = plt.figure(figsize=(size * (16 / 9), size))

ax = plt.subplot2grid((1, 1), (0, 0))
ax.plot(zgrid, density, color="tab:blue", label="Density")
ax.plot((zmin, zmin), (0, density[0]), color="tab:blue", ls='--', label=r"$z_{min}=$"+str(round(zmin, 2)))
ax.plot((zstar, zstar), (0, pdist.pdf(zstar)), color="tab:red", ls='--', label=r"$z^{\star}=$"+str(round(zstar, 2)))
ax.fill_between(zgrid[zgrid >= zstar], 0, pdist.pdf(zgrid[zgrid >= zstar]), color="tab:red", alpha=0.5, label=f"Share of Entrepreneurs = {share_entre.round(2)}")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel(r"$z$")
ax.set_ylabel(r"$f(z)$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='upper right', frameon=True)

plt.tight_layout()

plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/PSET 0/figures/SOC Analytical.pdf')
plt.show()
plt.close()
