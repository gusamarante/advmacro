import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq  # root-finding routine
from getpass import getuser

def tauchen(N, rho, sigma, mu=0.0, m=3.0):
    s1 = mu/(1 - rho) - m * np.sqrt(sigma**2 / (1 - rho**2))
    sN = mu/(1 - rho) + m * np.sqrt(sigma**2 / (1 - rho**2))
    s = np.linspace(s1, sN, N) # grid values
    step = (s[N-1] - s[0]) / (N - 1)  # evenly spaced grid
    P = np.zeros((N, N))

    for i in range(np.ceil(N/2).astype(int)):
        P[i, 0] = norm.cdf((s[0] - mu - rho*s[i] + step/2) / sigma)
        P[i, N-1] = 1 - norm.cdf((s[N-1] - mu - rho*s[i] - step/2) / sigma)
        for j in range(1, N-1):
            P[i, j] = norm.cdf((s[j] - mu - rho*s[i] + step/2) / sigma) - \
                      norm.cdf((s[j] - mu - rho*s[i] - step/2) / sigma)
    P[np.floor((N-1)/2+1).astype(int):, :] = P[0:np.ceil((N-1)/2).astype(int), :][::-1, ::-1]

    ps = np.sum(P, axis=1)
    P = P / ps[:, np.newaxis] # transition matrix

    # compute invariant distribution of labor process (by iteration)
    inv = np.ones(N) / N # initial guess
    tol=1E-11; maxit=100^4
    for it in range(maxit):
        inv_new = inv @ P
        if np.max(np.abs(inv_new - inv)) < tol: break
        inv = inv_new # invariant distribution

    return s, P, inv


def setPar(
    beta = 0.8,   # discount factor
    rho = 0.9,    # persistence
    sigma = 0.2,  # std. deviation
    mu_z = 1.0,   # mean of stochastic proces.
    alpha = 2/3,  # returns to scale
    c_e = 20.0,   # entry cost
    c_f = 20.0,   # fixed cost
    A = 0.01,      # Related to labor supply. Inverse of Dbar in Hopenhayn.
    tau = 0.1,	   # Adjustment costs
    nZ = 101,    # Grid of Z
    w = 1.0      # wages
	#nN = 500      # Will define the employment grid "by hand".
    ):

	# === SHOCK DISCRETIZATION
    mu_z = mu_z*(1-rho)  # Force mu of the stochastic process to be always mu_z
    gZ, F_trans, invZ = tauchen(nZ, rho, sigma, mu = mu_z, m = 4.0)
    gZ = np.exp(gZ)

	# === ENTRANTS DISTRIBUTION: Assume they draw from the invariant distribution.
    Gprob = invZ
	# I will set the top of the invariant distribution to zero, so entrants have lower productivity on average. The adjfactor controls how many points I set to zero.
	#adjfactor = 1.0
	#index_cutoff = ceil(Int, nZ*adjfactor) # everything above this index will be zero
	#Gprob = invZ
	#Gprob[index_cutoff+1:end] .= 0.0
	#Gprob = Gprob/sum(Gprob) # normalize to one

	# === EMPLOYMENT GRID
    gN = np.hstack([np.arange(0,21,1), np.arange(22,102,2), np.arange(105,505,5), np.arange(550,1050,50), np.arange(1100,5100,100), np.arange(5500,10500,500) ])
    nN = len(gN)

    # Adjustment Grid
    adjust = np.zeros((nN, nN))  # 1st is n 2nd is nprime
    for i in range(nN):
        for iprime in range(nN):
            adjust[i, iprime] = tau * max(0, gN[i] - gN[iprime])

    # Production given grid nodes
    gY = np.zeros((nN, nZ))
    for jn in range(nN):
        for jz in range(nZ):
            gY[jn, jz] = gZ[jz] * gN[jn]**alpha

	# create dictionary with parameters
    param = {}
    param['rho'] = rho
    param['tau'] = tau
    param['alpha'] = alpha; param['beta'] = beta; param['F_trans'] = F_trans; param['gZ'] = gZ; param['nZ'] = nZ; param['tau'] = tau;
    param['c_e'] = c_e; param['c_f'] = c_f; param['A'] = A; param['Gprob'] = Gprob; param['w'] = w; param['nN'] = nN; param['gN'] = gN;
    param['adjust'] = adjust; param['gY'] = gY

    return param


def SolveBellmanIncumbents(p_guess, param, print_it=False):
    # unpacking parameters
    beta = param["beta"];
    F_trans = param["F_trans"];
    gZ = param["gZ"];
    c_f = param["c_f"];
    nZ = param["nZ"];
    nN = param["nN"];
    gN = param["gN"];
    adjust = param["adjust"];
    gY = param["gY"];

    # housekeeping
    tol = 10 ** -5
    max_iter = 1000
    iter_count = 10

    # initialize VF
    V = -np.ones((nN, nZ))
    Vnext = -np.ones((nN, nZ))

    # auxiliary matrices
    scrap_value = -np.repeat(adjust[:, 0][:, np.newaxis], nZ,
                             axis=1)  # nNprime x nZ matrix
    profit_mat = np.zeros((nN, nZ, nN))  # 1st is n 2nd is nprime
    for iN in range(nN):
        for iNp in range(nN):
            for iZ in range(nZ):
                profit_mat[iN, iZ, iNp] = p_guess * gY[iNp, iZ] - gN[
                    iNp] - p_guess * c_f - adjust[iN, iNp]

    # ============== VF ITERATION ====
    for it in range(max_iter):
        ExpV = V @ F_trans.T
        ConV = beta * np.maximum(ExpV, scrap_value)  # nPrime x Z matrix

        for iZ in range(nZ):
            for iN in range(nN):  # loop over the state
                Vnext[iN, iZ] = np.max(profit_mat[iN, iZ, :] + ConV[:, iZ])

        sup = np.max(np.abs(V - Vnext))  # check tolerance
        V[:] = Vnext[:]
        if sup < tol:
            if print_it: print(f"Iter: {it}. Tol. achieved: {sup:.2E}")
            break
        if (it == max_iter) and print_it: print(
            f"Max iterations achieved. VF did not converge: {sup:.2E}")
        if (it % iter_count == 0) and print_it: print(
            f"Iter: {it}. Tol: {sup:.2E}")

    # ============== RECOVER POLICY FUNCTIONS =======
    ExpV = V @ F_trans.T
    ConV = beta * np.maximum(ExpV, scrap_value)  # nPrime x Z matrix

    exiter = (scrap_value > ExpV).astype(int)  # exit policy (1 == exit)

    npi = np.zeros((nN, nZ), dtype=int)  # recover index
    profit = np.zeros((nN, nZ))  # optimal profits
    tax_r = np.zeros((nN, nZ))  # taxes

    for iZ in range(nZ):
        for iN in range(nN):
            npi[iN, iZ] = np.argmax(profit_mat[iN, iZ, :] + ConV[:, iZ])
            profit[iN, iZ] = profit_mat[iN, iZ, npi[iN, iZ]]
            tax_r[iN, iZ] = adjust[iN, npi[iN, iZ]] * (1 - exiter[iN, iZ]) + \
                            exiter[iN, iZ] * adjust[iN, 1]

    npol = gN[npi]  # employment values

    return V, npi, exiter, npol, profit, tax_r


def solve_price(param, solvefor_ce=False, print_it=False):
    if not solvefor_ce:  # === SOLVE FOR EQ. USING PRICES
        def entry(p_guess):
            V, _, _, _, _, _ = SolveBellmanIncumbents(p_guess, param)
            excess_entry = sum(V[1,:] * param['Gprob']) - param['c_e']
            if print_it: print("Excess entry: ", excess_entry)
            return excess_entry

        p0 = 0.1; p1 = 4.0 # guess: lower and upper bound. Might have to change for diff. parameters
        p = brentq(entry, p0, p1, xtol = 0.01)
        c_e = param['c_e'] # not important, but must define c_e otherwise it throws an error
        V, npi, exiter, npol, profit, tax_r = SolveBellmanIncumbents(p, param)

    elif solvefor_ce:  # === ASSUME P=1 AND SOLVE FOR COST OF ENTRY -> this is faster and impose entry/exit but ce is chosen endogenously
        p = 1
        V, npi, exiter, np, profit, tax_r = SolveBellmanIncumbents(p, param)
        def entry_ce(c_e):
            excess_entry = sum(V[1,:] * param['Gprob']) - c_e
            if print_it: print("Excess entry: ", excess_entry)
            return excess_entry

        c0 = -120; c1 = 120 # guess: lower and upper bound. Might have to change for diff. parameters
        c_e = brentq(entry_ce, c0, c1, xtol = 0.001)

    return p, V, exiter, npi, npol, tax_r, profit, c_e


def invariant_dist(param, solution, print_it=False):
    Gprob = param['Gprob'];
    F_trans = param['F_trans'];
    nN = param['nN'];
    nZ = param['nZ']
    exiter = solution[2];
    npi = solution[3]

    # housekeeping
    tol = 10 ** -6
    max_iter = 1000
    iter_count = 20

    inv_dist = np.zeros((nN, nZ))
    inv_dist[0, :] = Gprob  # guess
    dsn_next = np.zeros((nN, nZ))

    for iter in range(max_iter):
        dsn_next[:] = 0.0

        for jn in range(nN):
            for jz in range(nZ):
                dsn_next[npi[jn, jz], :] += inv_dist[jn, jz] * F_trans[jz,
                                                               :] * (
                                                        1 - exiter[jn, jz])
        dsn_next[0, :] += Gprob

        sup = np.max(np.abs(dsn_next - inv_dist))  # check tolerance
        inv_dist[:] = dsn_next

        if sup < tol:
            if print_it: print(f"Iter: {iter}. Tol. achieved: {sup:.2E}")
            break
        if (iter == max_iter) & print_it: print(
            f"Max iterations achieved. Inv. dist did not converge: {sup:.2E}")
        if (iter % iter_count == 0) & print_it: print(
            f"Iter: {iter}. Tol: {sup:.2E}")

    return inv_dist

def solve_m(param, solution, print_it=False):
    gZ = param['gZ']; nN = param['nN'];
    p = solution[0]; exiter = solution[2]; npol = solution[4]; tax_r = solution[5]; profit = solution[6]

    # Solve for the Invariant Distribution
    mu = invariant_dist(param, solution, print_it=print_it)

    # Solve for the mass of entrants: M
    Ynz = npol**param['alpha'] * gZ.T
    Yone = np.sum((Ynz - param['c_f']) * mu)  # aggregate suppy (without M)
    demand = 1 / (param['A'] * p)
    M = demand / Yone

    # Recover true mu
    mu = M * mu

    # Aggregate variables
    Y = Yone * M
    N = np.sum(npol * mu) + param['c_e'] * M
    T = np.sum(tax_r * mu)
    Pi = np.sum(profit * mu) - param['c_e'] * M
    Pi2 = p * Y - N - T  # compute profit in a different way to check

    # Exit productivity:
    zcut = np.zeros(nN)
    for i in range(nN):
        cut_index = np.argmax(exiter[i, :] == 0)
        zcut[i] = gZ[cut_index]

    return M, mu, zcut, Y, N, T, Pi, Pi2


def ModelStats(param, sol_price, sol_dist, Printa=True):
    nN = param['nN'];
    gZ = param["gZ"];
    gN = param["gN"];
    alpha = param["alpha"]

    p = sol_price[0];
    exiter = sol_price[2];
    npol = sol_price[4]
    mu = sol_dist[1];
    M = sol_dist[0]

    # productivity distribution
    Mu = np.sum(mu)  # mass of operating firms
    pdf_dist = np.sum(mu, axis=0) / Mu
    cdf_dist = np.cumsum(pdf_dist)

    # employment distribution by productivity
    agg_emp = np.sum(npol * mu)
    emp_dist = np.sum(npol * mu, axis=0) / agg_emp

    # misallocation
    MPNs = alpha * gZ * npol ** (alpha - 1)
    indx_inf = MPNs != np.inf
    MPNv = MPNs[indx_inf]
    muv = mu[indx_inf]

    avg_misalloc = np.sum(np.abs(MPNv - 1 / p) * p * muv) / np.sum(
        muv) * 100  # divide by 1/p so we get %
    # max_misalloc = max(np.abs(MPNv - 1/p)*p)

    # stats
    avg_firm_size = agg_emp / Mu
    exit_rate = M / np.sum(mu)
    avg_prod = np.sum(gZ * np.sum(mu, axis=0)) / Mu

    # firm and employment share by bins
    size_array = np.array([10, 20, 50, 100, 1000])
    share_array = np.zeros((3, len(size_array)))
    share_array[0, :] = size_array

    for i in range(len(size_array)):
        if i == 0:
            indx = (0.0 <= npol) & (npol <= size_array[i])
        elif i > 0:
            indx = (size_array[i - 1] < npol) & (npol <= size_array[i])

        share_array[1, i] = np.sum(mu[indx]) / Mu  # firm share
        share_array[2, i] = np.sum(mu[indx] * npol[indx]) / agg_emp  # emp share
    share_array[1, :] = np.cumsum(share_array[1, :])
    share_array[2, :] = np.cumsum(share_array[2, :])

    # print
    if Printa == True:
        print("Model Stats")
        print("Price: ", p)
        print("Avg. Firm Size: ", avg_firm_size)
        print("Exit/entry Rate: ", exit_rate)
        print("Avg. Productivity: ", avg_prod)
        print("Avg. Misallocation (%): ", avg_misalloc)
        print("Agg. Output: ", sol_dist[3])
        print("Agg. Labor Supply: ", sol_dist[4])
        print("Agg. Tax Revenue: ", sol_dist[5])
        print("Agg. Profits: ", sol_dist[6])
        print("Mass of Firms: ", Mu)
        print("Mass of Entrants: ", M)
        print("")
        print("Size           ", share_array[0, 0], share_array[0, 1],
              share_array[0, 2], share_array[0, 3], share_array[0, 4])
        print("Firm Share     ", share_array[1, 0], share_array[1, 1],
              share_array[1, 2], share_array[1, 3], share_array[1, 4])
        print("Emp. Share     ", share_array[2, 0], share_array[2, 1],
              share_array[2, 2], share_array[2, 3], share_array[2, 4])

    return (Mu, pdf_dist, emp_dist, avg_firm_size, exit_rate, avg_prod,
            avg_misalloc, share_array)


def SolveModel(param):
    # Solve the model
    print("Solving for the price...")
    sol_price = solve_price(param, print_it=True)
    print("Solving for the distribution...")

    sol_dist = solve_m(param, sol_price, print_it=True)

    if sol_dist[0] <= 0: print("Warning: No entry, eq. not found.")

    # Compute Stats
    stats = ModelStats(param, sol_price, sol_dist)

    return (sol_price, sol_dist, stats)



# ===== Testing the Functions =====
# param = setPar()
# testVF = SolveBellmanIncumbents(1.0, param, print_it=True)
# sol_ptest = solve_price(param, print_it=True)
# mu_test = invariant_dist(param, sol_ptest, print_it=True)
# dist_test = solve_m(param, sol_ptest, print_it = True)
# test = ModelStats(param, sol_ptest, dist_test)

# ===== ACTUAL SOLUTION =====



count = 0
rho_list = [0.5, 0.9]
tau_list = [0.5, 0.9]
df = pd.DataFrame()

for rho_i in rho_list:
    for tau_i in tau_list:
        param = setPar(rho=rho_i, tau=tau_i)
        solution = SolveModel(param)

        df.loc[count, "Rho"] = param['rho']
        df.loc[count, "Tau"] = param['tau']
        df.loc[count, "Average Firm Size"] = solution[2][3]
        df.loc[count, "Average Firm Productivity"] = solution[2][5]
        df.loc[count, "Entry / Exit Rate"] = solution[2][4]
        df.loc[count, "Average Misallocation"] = solution[2][6]

        count += 1


with pd.ExcelWriter(fr"/Users/{getuser()}/Dropbox/PhD/Advanced Macro/PSET 3/Q1 b c.xlsx") as writer:
    df.to_excel(writer)
