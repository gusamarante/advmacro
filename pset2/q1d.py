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
    "r": 0.03,  # Fixed interest rate
    "eis": 0.5,  # EIS
    "rho_e": 0.9,  # Persistence of idiosyncratic productivity shocks
    "sd_e": 0.92,  # Standard deviation of idiosyncratic productivity shocks
    "G": 0.2,  # Government spending
    "B": 0.8,  # Government debt
    "Y": 1.0,  # Output
    "min_a": 0.0,  # Minimum asset level on the grid
    "max_a": 1_000,  # Maximum asset level on the grid
    "n_a": 200,  # Number of asset grid points
    "n_e": 10,  # Number of productivity grid points
}

# ==============================
# ===== RA Household Block =====
# ==============================
def income(Z, e_grid):  # Post-tax labor income
    y = Z * e_grid
    return y

# join the grid funtion together with HA block
hh_extended = hh.add_hetinputs([make_grids, income])


# =================================
# ===== Gov Budget Constraint =====
# =================================
@sj.simple
def fiscal(B, r, G, Y):
    T = (1 + r) * B(-1) + G - B
    Z = Y - T
    deficit = G - T
    return T, Z, deficit


# ===========================
# ===== Market Clearing =====
# ===========================
@sj.simple
def mkt_clearing(A, B, Y, C, G):
    asset_mkt = A - B
    goods_mkt = Y - C - G
    return asset_mkt, goods_mkt


# ========================
# ===== Create Model =====
# ========================
ha = sj.create_model([hh_extended, fiscal, mkt_clearing], name="Simple HA Model")

# We want to fix the steady state interest rate r at a pre-specified value, to clear the asset market
# We will calibrate beta. All the other parameters come from the calibration above.
unknowns_ss = {'beta': (0.75, 0.9999)}  # provide bounds on beta for the solver
targets_ss = ['asset_mkt']  # set the ss target
ss = ha.solve_steady_state(calibration, unknowns_ss, targets_ss, solver='hybr')


# ====================
# ===== RA Model =====
# ====================
@sj.solved(unknowns={'C': 1, 'A': 1}, targets=["euler", "budget_constraint"])
def hh_ra(C, A, Z, eis, beta, r):
    euler = (beta * (1 + r(+1))) ** (- eis) * C(+1) - C
    budget_constraint = (1 + r) * A(-1) + Z - C - A
    return euler, budget_constraint

ra = sj.create_model([hh_ra, fiscal, mkt_clearing], name="Representative agent model")
calibration_ra = calibration.copy()
calibration_ra['beta'] = 1 / (1 + calibration_ra['r'])
calibration_ra['B'] = ss['B']
unknowns_ra_ss = {'C': 1., 'A': 0.8}
targets_ra_ss = {'budget_constraint': 0., 'asset_mkt': 0.}
ss_ra = ra.solve_steady_state(calibration_ra, unknowns_ra_ss, targets_ra_ss, dissolve=['hh_ra'])


# ====================
# ===== TA Model =====
# ====================
@sj.solved(unknowns={'C_RA': 1, 'A': 1}, targets=["euler", "budget_constraint"])
def hh_ta(C_RA, A, Z, eis, beta, r, lam):
    euler = (beta * (1 + r(+1))) ** (-eis) * C_RA(+1) - C_RA  # consumption of infinitely lived household
    C_H2M = Z   # consumption of an hand to mouth agent
    C = (1 - lam) * C_RA + lam * C_H2M   # aggregate consumption
    budget_constraint = (1 + r) * A(-1) + Z - C - A
    return euler, budget_constraint, C_H2M, C

ta = sj.create_model([hh_ta, fiscal, mkt_clearing], name="Two agent model")
unknowns_ta_ss = {'C_RA': 1., 'A': 0.8}
targets_ta_ss = {'budget_constraint': 0., 'asset_mkt': 0.}

calibration_ta_low = calibration_ra.copy()
calibration_ta_low['lam'] = 0.25  # Calibration of TA model: share of hand-to-mouth agents
ss_ta_low = ta.solve_steady_state(calibration_ta_low, unknowns_ta_ss, targets_ta_ss , dissolve=['hh_ta'])

calibration_ta_high = calibration_ra.copy()
calibration_ta_high['lam'] = 0.75  # Calibration of TA model: share of hand-to-mouth agents
ss_ta_high = ta.solve_steady_state(calibration_ta_high, unknowns_ta_ss, targets_ta_ss , dissolve=['hh_ta'])


# ================================
# ===== IRFs - Debt Financed =====
# ================================
T = 300  # length of IRF

rho_G = 0.8
dG = 0.01 * (rho_G ** np.arange(T))  # Sequence of shocks

rho_B = 0.8 # high persistence shock
dB = np.cumsum(dG) * rho_B ** np.arange(T) # Note the cumsum! The fiscal shock accumulates.
shocks_B = {'G': dG, 'B': dB}

irfs_ha = ha.solve_impulse_linear(ss=ss, unknowns=['Y'], targets=['asset_mkt'], inputs=shocks_B)
irfs_ra = ra.solve_impulse_linear(ss=ss_ra, unknowns=['Y'], targets=['asset_mkt'], inputs=shocks_B)
irfs_ta_low = ta.solve_impulse_linear(ss=ss_ta_low, unknowns=['Y'], targets=['asset_mkt'], inputs=shocks_B)
irfs_ta_high = ta.solve_impulse_linear(ss=ss_ta_high, unknowns=['Y'], targets=['asset_mkt'], inputs=shocks_B)


# --- plot ---
size = 6
fig = plt.figure(figsize=(size * (16 / 7), size))

ax = plt.subplot2grid((2, 2), (0, 0))
ax.set_title(r"Output $Y$")
ax.plot(irfs_ha['Y'][:30], label='HA', color='tab:blue')
ax.plot(irfs_ra['Y'][:30], label=r'RA', color='tab:orange')
ax.plot(irfs_ta_low['Y'][:30], label=r'TA (Low $\lambda$)', color='tab:green', ls='dashed')
ax.plot(irfs_ta_high['Y'][:30], label=r'TA (High $\lambda$)', color='tab:green', ls='dotted')
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_ylabel(r"% deviation from steady-state")
ax.set_xlabel(r"Periods $t$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

ax = plt.subplot2grid((2, 2), (0, 1))
ax.set_title(r"Consumption $C$")
ax.plot(irfs_ha['C'][:30], label='HA', color='tab:blue')
ax.plot(irfs_ra['C'][:30], label=r'RA', color='tab:orange')
ax.plot(irfs_ta_low['C'][:30], label=r'TA (Low $\lambda$)', color='tab:green', ls='dashed')
ax.plot(irfs_ta_high['C'][:30], label=r'TA (High $\lambda$)', color='tab:green', ls='dotted')
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_ylabel(r"% deviation from steady-state")
ax.set_xlabel(r"Periods $t$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

ax = plt.subplot2grid((2, 2), (1, 0))
ax.set_title(r"Government Spending $G$")
ax.plot(irfs_ha['G'][:30], label='HA', color='tab:blue')
ax.plot(irfs_ra['G'][:30], label=r'RA', color='tab:orange')
ax.plot(irfs_ta_low['G'][:30], label=r'TA (Low $\lambda$)', color='tab:green', ls='dashed')
ax.plot(irfs_ta_high['G'][:30], label=r'TA (High $\lambda$)', color='tab:green', ls='dotted')
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_ylabel(r"% deviation from steady-state")
ax.set_xlabel(r"Periods $t$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

ax = plt.subplot2grid((2, 2), (1, 1))
ax.set_title(r"Tax Revenue $T$")
ax.plot(irfs_ha['T'][:30], label='HA', color='tab:blue')
ax.plot(irfs_ra['T'][:30], label=r'RA', color='tab:orange')
ax.plot(irfs_ta_low['T'][:30], label=r'TA (Low $\lambda$)', color='tab:green', ls='dashed')
ax.plot(irfs_ta_high['T'][:30], label=r'TA (High $\lambda$)', color='tab:green', ls='dotted')
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.set_ylabel(r"% deviation from steady-state")
ax.set_xlabel(r"Periods $t$")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.legend(loc='best', frameon=True)

plt.tight_layout()
plt.savefig(f'/Users/{getpass.getuser()}/Dropbox/PhD/Advanced Macro/PSET 2/figures/q1d irfs.pdf')
plt.show()
plt.close()
