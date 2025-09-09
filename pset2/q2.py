from sequence_jacobian.hetblocks.hh_sim import hh, make_grids
import matplotlib.pyplot as plt
import sequence_jacobian as sj
import pandas as pd
import numpy as np
import getpass


# =======================
# ===== Calibration =====
# =======================
calibration = {
    'eis': 0.5,  # EIS
    'rho_e': 0.92,  # Persistence of idiosyncratic productivity shocks
    'sd_e': 0.92,  # Standard deviation of idiosyncratic productivity shocks
    'Y': 1.,  # Output
    'r': 0.01,  # target real interest rate
    'min_a': -1,  # Minimum asset level on the grid
    'max_a': 1_000,  # Maximum asset level on the grid
    'n_a': 500,  # Number of asset grid points
    'n_e': 11,  # Number of productivity grid points
}


# ==============================
# ===== HA Household Block =====
# ==============================
def income(Y, e_grid):  # Post-tax labor income
    y = Y * e_grid
    return y

# join the grid funtion together with HA block
hh_simple = hh.add_hetinputs([make_grids, income])


# ===========================
# ===== Market Clearing =====
# ===========================
@sj.simple
def mkt_clearing(A, Y, C):
    asset_mkt = A
    goods_mkt = Y - C
    return asset_mkt, goods_mkt


ha_simple = sj.create_model([hh_simple, mkt_clearing])
# ===== Find the Betas =====
# Note that we are fixing Y and choosing Beta such that the asset market clears
unknowns_ss = {'beta': (0.7, 0.999)}  # provide bounds on beta for the solver
targets_ss = ['asset_mkt']  # set the ss target
ss_ha = ha_simple.solve_steady_state(calibration, unknowns_ss, targets_ss, solver='hybr')


# ===== RA Model =====
@sj.solved(unknowns={'C': 1, 'A': 1}, targets=["euler", "budget_constraint"], solver="broyden_custom")
def household_ra_simple(C, A, Y, eis, beta, r):
    euler = (beta * (1 + r(1))) ** (-eis) * C(1) - C
    budget_constraint = (1 + r) * A(-1) + Y - C - A
    return euler, budget_constraint

ra_model = sj.create_model([household_ra_simple, mkt_clearing], name="Representative Agent Model")

calibration_ra = calibration.copy()
calibration_ra['beta'] = 1 / (1 + calibration_ra['r']) # Different interest rate
ss_ra = ra_model.solve_steady_state(calibration_ra, {'C': 1., 'A': 0.8}, {'budget_constraint': 0., 'asset_mkt': 0.}, dissolve=['household_ra_simple'])

print(r'Beta in the HA model:', ss_ha['beta'], ' ; Beta in the RA model: ', ss_ra['beta'])


# ======================================
# ===== Item (b) - IRF to MP shock =====
# ======================================
T = 300
dr = -0.01 * (0.7 ** np.arange(T)) # This is the shock
shock_r = {'r': dr}

irf_ha = ha_simple.solve_impulse_linear(ss_ha, ['Y'], ['asset_mkt'], shock_r)
irf_ra = ra_model.solve_impulse_linear(ss_ra, ['Y'], ['asset_mkt'], shock_r)

# --- plot ---
size = 5
fig = plt.figure(figsize=(size * (16 / 7), size))

ax = plt.subplot2grid((1, 2), (0, 0))
ax.set_title(r"Output $Y$")
ax.plot(irf_ha['Y'][:30], label='HA', color='tab:blue')
ax.plot(irf_ra['Y'][:30], label=r'RA', color='tab:orange')
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_ylabel(r"% deviation from steady-state")
ax.set_xlabel(r"Periods $t$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

ax = plt.subplot2grid((1, 2), (0, 1))
ax.set_title(r"Interest Rate $r$")
ax.plot(irf_ha['r'][:30], label='HA', color='tab:blue')
ax.plot(irf_ra['r'][:30], label=r'RA', color='tab:orange')
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_ylabel(r"% deviation from steady-state")
ax.set_xlabel(r"Periods $t$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

plt.tight_layout()
plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/PSET 2/figures/q2b irfs.pdf')
plt.show()
plt.close()


# ===== Itam (c) - Jacobian =====
J = ha_simple['hh'].jacobian(ss_ha, inputs=['Y', 'r'], T=T)


dC, dC_dr, dC_dY = {}, {}, {}  # we include that in dictionaries since we will also do for the RA model.

# HA model
dC['ha'] = irf_ha['C']   # Total Effect
dC_dr['ha'] = J['C']['r'] @ dr          # Direct Effect
dC_dY['ha'] = J['C']['Y'] @ dC['ha']    # Indirect Effect

# RA Model
dC['ra'] = irf_ra['C']
beta = calibration_ra['beta']
Mra = (1 - beta) * beta ** (np.tile(np.arange(T), (T, 1)))
dC_dY['ra'] = Mra @ dC['ra']
dC_dr['ra'] = dC['ra'] - dC_dY['ra']

df_ha = pd.DataFrame({'Direct': dC_dr['ha'], 'Indirect': dC_dY['ha']})
df_ra = pd.DataFrame({'Direct': dC_dr['ra'], 'Indirect': dC_dY['ra']})

# --- plot ---
size = 6
fig = plt.figure(figsize=(size * (16 / 7), size))

ax = plt.subplot2grid((1, 2), (0, 0))
ax.set_title(r"Output $Y$ (HA Model)")
ax = df_ha.iloc[:30].plot(kind='bar', stacked=True, ax=ax, width=0.9, alpha=0.7)
ax.plot(dC['ha'][:30], label='Total', color="tab:green", lw=2)
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_ylabel(r"% deviation from steady-state")
ax.set_xlabel(r"Periods $t$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

ax = plt.subplot2grid((1, 2), (0, 1), sharey=ax)
ax.set_title(r"Output $Y$ (RA Model)")
ax = df_ra.iloc[:30].plot(kind='bar', stacked=True, ax=ax, width=0.9, alpha=0.7)
ax.plot(dC['ra'][:30], label='Total', color="tab:green", lw=2)
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_ylabel(r"% deviation from steady-state")
ax.set_xlabel(r"Periods $t$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

plt.tight_layout()
plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/PSET 2/figures/q2c irf decomposition.pdf')
plt.show()
plt.close()