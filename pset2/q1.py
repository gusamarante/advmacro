from sequence_jacobian.hetblocks.hh_sim import hh, make_grids
import matplotlib.pyplot as plt
import sequence_jacobian as sj
import pandas as pd
import numpy as np


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

# ========================
# ===== How it works =====
# ========================
print(hh.outputs)
# Notice that the outputs of the `hh` function are ['A', 'C'] by default, so we
# have to maintain that notation. Do not substitute notation for lowercase letters.


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
calibration_ta = calibration_ra.copy()
calibration_ta['lam'] = 0.25  # Calibration of TA
unknowns_ta_ss = {'C_RA': 1., 'A': 0.8}
targets_ta_ss = {'budget_constraint': 0., 'asset_mkt': 0.}
ss_ta = ta.solve_steady_state(calibration_ta, unknowns_ta_ss, targets_ta_ss , dissolve=['hh_ta'])

# Alternative calibration
calibration_ta_n = calibration_ra.copy()
calibration_ta_n['lam'] = 0.50  # Calibration of TA
ss_ta_n = ta.solve_steady_state(calibration_ta_n, unknowns_ta_ss, targets_ta_ss , dissolve=['hh_ta'])


# ===================
# ===== Outputs =====
# ===================
out = pd.concat(
    [
        pd.DataFrame(ss.toplevel.items()).set_index(0),
        pd.DataFrame(ss_ra.toplevel.items()).set_index(0),
        pd.DataFrame(ss_ta.toplevel.items()).set_index(0),
        pd.DataFrame(ss_ta_n.toplevel.items()).set_index(0),
    ],
    axis=1,
)
out.columns = ['HA', 'RA', 'TA', 'TA n']
out.index.name = 'parameters'

print(out)