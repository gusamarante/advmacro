import numpy as np
import matplotlib.pyplot as plt
from numba import njit, jit, guvectorize # NUMBA speed up quite a lot, see the functions that have the decorator just above
from scipy.optimize import brentq  # root-finding routine

# Matplotlib par
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({"axes.grid" : True, "grid.color": "black", "grid.alpha":"0.25", "grid.linestyle": "--"})
plt.rcParams.update({'font.size': 12})

def rouwenhorst(N, rho, sigma, mu=0.0):
    """Rouwenhorst method to discretize AR(1) process"""

    q = (rho + 1)/2
    nu = ((N-1.0)/(1.0-rho**2))**(1/2)*sigma
    s = np.linspace(mu/(1.0-rho)-nu, mu/(1.0-rho)+nu, N) # states

    # implement recursively transition matrix
    P = np.array([[q, 1-q], [1-q, q]])

    for i in range(2,N): # add one row/column one by one
        P = q*np.r_[np.c_[P, np.zeros(i)] , [np.zeros(i+1)]] + (1-q)*np.r_[np.c_[np.zeros(i), P] , [np.zeros(i+1)]] + (1-q)*np.r_[[np.zeros(i+1)] ,  np.c_[P, np.zeros(i)]] + q*np.r_[[np.zeros(i+1)] ,  np.c_[np.zeros(i), P]]
        P[1:i,:]=P[1:i,:]/2  # adjust the middle rows of the matrix

    return s, P

# I took this function from state-space Jacobian package
@guvectorize(['void(float64[:], float64[:], float64[:], float64[:])'], '(n),(nq),(n)->(nq)')
def interpolate_y(x, xq, y, yq):
    """Efficient linear interpolation exploiting monotonicity.
    Complexity O(n+nq), so most efficient when x and xq have comparable number of points.
    Extrapolates linearly when xq out of domain of x.
    Parameters
    ----------
    x  : array (n), ascending data points
    xq : array (nq), ascending query points
    y  : array (n), data points
    Returns
    ----------
    yq : array (nq), interpolated points
    """
    nxq, nx = xq.shape[0], x.shape[0]

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi_cur = (x_high - xq_cur) / (x_high - x_low)
        yq[xqi_cur] = xqpi_cur * y[xi] + (1 - xqpi_cur) * y[xi + 1]

@guvectorize(['void(float64[:], float64[:], uint32[:], float64[:])'], '(n),(nq)->(nq),(nq)')
def interpolate_coord(x, xq, xqi, xqpi):
    """Get representation xqi, xqpi of xq interpolated against x:
    xq = xqpi * x[xqi] + (1-xqpi) * x[xqi+1]
    Parameters
    ----------
    x    : array (n), ascending data points
    xq   : array (nq), ascending query points
    Returns
    ----------
    xqi  : array (nq), indices of lower bracketing gridpoints
    xqpi : array (nq), weights on lower bracketing gridpoints
    """
    nxq, nx = xq.shape[0], x.shape[0]

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi[xqi_cur] = (x_high - xq_cur) / (x_high - x_low)
        xqi[xqi_cur] = xi
        xqpi[xqi_cur] =  min(max(xqpi[xqi_cur],0.0), 1.0) # if the weight is outside 0 or 1, this will catch it

def setPar(
        nA=300,  # Asset grid size
        nS=7,  # Labor endowment grid size
        alpha=0.33,
        delta=0.05,
        beta=0.96,
        gamma=2.0,  # CRRA parameter
        phi=0.0,  # borrowing constraint
        rho=0.9,  # autocorrelation
        sigma=0.1,  # std deviation
        amax=250.0,  # maximum grid point
        grid_growth=0.025  # asset grid curvature (growth rate between points)
):
    #### Define the labor grid
    gS, transS = rouwenhorst(nS, rho, sigma)
    gS = np.exp(gS)

    # compute invariant distribution of labor process (by iteration)
    invS = np.ones(nS) / nS  # initial guess
    tol = 1E-11;
    maxit = 100 ^ 4
    for it in range(maxit):
        invS_new = invS @ transS
        if np.max(np.abs(invS_new - invS)) < tol: break
        invS = invS_new

    Lbar = sum(invS * gS)  # aggregate labor supply

    #### Define the asset grid:
    if grid_growth == 0.0:
        gA = np.linspace(-phi, amax, nA)
    elif grid_growth > 0.0:
        gA = np.zeros(nA)
        for i in range(nA):
            gA[i] = -phi + (amax - (-phi)) * ((1 + grid_growth) ** i - 1) / (
                        (1 + grid_growth) ** (nA - 1.0) - 1)

    # create dictionary with parameters
    param = {}
    param['alpha'] = alpha;
    param['delta'] = delta;
    param['beta'] = beta;
    param['gamma'] = gamma;
    param['beta'] = beta;
    param['phi'] = phi
    param['nA'] = nA;
    param['gA'] = gA
    param['nS'] = nS;
    param['gS'] = gS;
    param['transS'] = transS;
    param['Lbar'] = Lbar;
    param['invS'] = invS

    return param

def solveHHproblemEGM(param, r, w):
    # unpacking parameters
    nA = param['nA'];
    gA = param['gA'];
    nS = param['nS'];
    gS = param['gS']
    transS = param['transS'];
    beta = param['beta'];
    gamma = param['gamma']

    # Define parameters
    maxiter = 2000
    tol = 1E-11  # tolerance

    # Useful matrices
    gAA = np.repeat(gA, nS).reshape((nA, nS))  # nA x nS matrix of asset grid.
    gSS = np.repeat(gS, nA).reshape((nS, nA)).T  # nA x nS matrix of labor grid. Note the transpose.
    gYY = w * gSS + (1.0 + r) * gAA  # nA x nS matrix of cash on hand grid.

    # Pre-allocate value and policy funcctions
    cp = gYY - gAA  # consumption policy
    ap = np.zeros((nA, nS))  # asset policy
    endogAA = np.zeros((nA, nS))  # endogenous asset grid

    for iter in range(maxiter):
        # ===== Perform EGM ===== #
        expect = beta * (1.0 + r) * np.matmul(cp ** (-gamma),
                                              transS.T)  # Right hand side of Euler Eq.
        c = expect ** (-1.0 / gamma)  # Invert marginal util to get contemporaneous C
        endogAA = (c + gAA - w * gSS) / (1 + r)  # compute asset state on endogenous grid (note that gAA is the policy function, which is on-grid)

        # === Interpolate to find the correct asset policy function
        apNew = np.zeros((nA, nS))  # here we store the policy function that we interpolate

        for iS in range(nS):
            interpolate_y(endogAA[:, iS], gA, gA, apNew[:, iS])  # perform interpolation
            # Must account for the binding constraint!
            for ia in range(nA):
                if apNew[ia, iS] < gA[0]:
                    apNew[ia, iS] = gA[0]
                else:
                    break  # exploit monotinicity of ia.

        cpNew = gYY - apNew  # get updated consumption policy

        # ===== Check if policuufunction has converged =====
        d = np.amax(np.abs(cp - cpNew))
        cp = cpNew
        ap = apNew

        # print("Iter: ", iter, "Tol. ", d)

        if d < tol:
            print(f"Tol. achieved (EGM): {d}")
            break  # break the loop in case we found the Policy Function!

        if iter == maxiter:
            print("Max iterations achieved. VF did not converge")

    return ap, cp, endogAA

# @njit  # ps. using numba here is crucial, it speeds up quite a lot
def iterationDist(dsn, dsnNew, ibelow, iweight, transS):
    nA, nS = dsn.shape
    for iA in range(nA):
        for iS in range(nS):
            if dsn[iA, iS] > 0:  # this speed a bit in some problems
                dsnNew[ibelow[iA, iS], iS] += iweight[iA, iS] * dsn[iA, iS]
                dsnNew[ibelow[iA, iS] + 1, iS] += (1 - iweight[iA, iS]) * dsn[iA, iS]
    dsnNew = dsnNew @ transS  # apply the markov-chain of labor process
    return dsnNew

def solveInvariant(param, decisions):
    nA = param['nA'];
    gA = param['gA'];
    nS = param['nS'];
    transS = param['transS'];
    ap = decisions[0]

    # Define aux parameters:
    maxiterInv = 50000
    tolInv = 1e-10

    # === 1. RETRIEVE GRID AND WEIGHT OF INTERPOLATED POLICIES ============== #
    # ps. the interpolation assume that policy is monotone
    ibelow = np.zeros((nA, nS), dtype=int)
    iweight = np.zeros((nA, nS))

    for iS in range(nS):
        interpolate_coord(gA, ap[:, iS], ibelow[:, iS], iweight[:, iS])
        # iweight is probability agent ends in grid "ibelow".

    # TODO BATIDO ATÉ AQUI
    # ================ 2. ITERATE FORWARD DISTRIBUTION ======================== #
    dsn = np.ones((nA, nS)) / (nS * nA)  # initial guess, positive mass everywhere must sum to one

    for iter in range(maxiterInv):
        # compute next distribution
        dsnNew = np.zeros((nA, nS))
        dsnNew = iterationDist(dsn, dsnNew, ibelow, iweight, transS)

        # ===== Check if distribution has converged =====
        d = np.amax(np.abs(dsn - dsnNew))
        dsn = dsnNew  # update distribution

        if d < tolInv:
            print("Tol. achieved (Inv. dist.): ", d)
            break

        if iter == maxiterInv - 1:
            print(
                "Max iterations achieved. Invariant distribution did not converge")

    return dsn

def ExcessDemand(param, r):
    #unpack parameters
    Lbar = param['Lbar']; gA = param['gA']; alpha = param['alpha']; nS = param['nS']; delta = param['delta']

    # Compute Capital Demand and implied wage from the production function
    Kd = (alpha/(r+delta))**(1/(1-alpha))*Lbar # capital demand
    w = (1-alpha)*(Kd/Lbar)**alpha

    # Solve HH problem
    decisions = solveHHproblemEGM(param, r, w)

    # Compute Invariant distribution
    dsn = solveInvariant(param, decisions)

    # Compute Excess Demand of asset market
    gAA = np.repeat(gA[None,:], nS, axis=0).T # repeat columns of gA
    Ea = sum(sum(dsn*gAA)) # Aggregate asset supply
    excDem = (Ea -Kd)/((Ea+Kd)/2) # excess demand in percentage

    return (excDem, decisions, dsn, w, Kd, Ea)

def model_solution(param):
    # Parameters
    r0 = 0.001  # lower bound guess (make sure excess demand is negative)
    r1 = 1/param['beta'] - 1  # upper bound guess (make sure excess demand is positive)

    tol_eq = 0.0001

    def obj_fct(r_guess):
        print("Interest Rate Guess: ", r_guess)
        results = ExcessDemand(param, r_guess)
        print("Excess Demand: ", results[0])
        return results[0]

    r = brentq(obj_fct, r0, r1, xtol=tol_eq)
    _, decisions, dsn, w, Kd, Ea = ExcessDemand(param, r)  # get the stuff from the model

    return decisions, dsn, w, r, Kd, Ea  # model stats

def ModelStats(param, decisions, dsn, w, r, Kd, Ea):
    alpha = param['alpha'];
    Lbar = param['Lbar'];
    nS = param['nS'];
    gS = param['gS']
    gA = param['gA']
    cp = decisions[1]  # policy consumption
    ap = decisions[0]  # policy savings

    # Compute useful stats
    Y = Kd ** alpha * Lbar ** (1 - alpha)
    const_hh = sum(dsn[0, :])

    # Marginal propensity to consume
    mg_c = (cp[1:, :] - cp[0:-1, :])
    mg_wealth = ((1 + r) * (gA[1:] - gA[0:-1]))
    mg_wealth = np.repeat(mg_wealth[None, :], param['nS'],
                          axis=0).T  # repeat columns
    mpc = mg_c / mg_wealth

    # Wealth distribution:
    dsn_a = np.sum(dsn, axis=1)
    cdf_a = np.cumsum(dsn_a)
    mean_a = np.sum(dsn_a * gA)
    std_a = np.sqrt(np.sum(dsn_a * (gA - mean_a) ** 2))

    def percentile(gA, cdf_a, p):  # got this function from Jeppe Druedahl
        nA = len(gA)

        # a. check first
        if p < cdf_a[0]: return gA[0]

        # b. find with loop
        i = 0
        while True:
            if p > cdf_a[i + 1]:
                if i + 1 >= nA: raise Exception()
                i += 1
                continue
            else:
                w = (p - cdf_a[i]) / (cdf_a[i + 1] - cdf_a[i])
                diff = gA[i + 1] - gA[i]
                return gA[i] + w * diff

    p25_a = percentile(gA, cdf_a, 0.25)
    p50_a = percentile(gA, cdf_a, 0.50)
    p95_a = percentile(gA, cdf_a, 0.95)
    p99_a = percentile(gA, cdf_a, 0.99)

    print("\nModel Stats:")
    print(f'\nEq. wage and interest rate: {w:6.4f}  {r:6.4f}')
    print(f'Aggregate Capital and Asset Supply: {Kd:6.4f}  {Ea:6.4f}')
    print(f'Labor Supply: {Lbar:6.3f}')
    print(f'K/L: {Kd / Lbar:6.3f}')
    print(f'Agg. Output: {Y:6.3f}')
    print(f'A/Y: {Ea / Y:6.3f}')
    print(f'Fraction of constrained households:: {const_hh:6.3f}')
    # print("Gini of Wealth:")
    # print("Gini of Income:")
    # print("Gini of Consumption:")
    print("\nWealth Distribution:")
    print(f'Avg. a: {mean_a:6.3f}')
    print(f'Std. a: {std_a:6.3f}')
    print(f'p25  a: {p25_a:6.3f}')
    print(f'p50  a: {p50_a:6.3f}')
    print(f'p95  a: {p95_a:6.3f}')
    print(f'p99  a: {p99_a:6.3f}')

    # Plot policy function:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,
                                                  5))  # Ax1 is consumption ax2 is savings

    for s in range(nS):
        ax1.plot(gA, cp[:, s], label=f's = {gS[s]:.2f}')
        ax2.plot(gA, ap[:, s], label=f's = {gS[s]:.2f}')

    ax1.legend(frameon=True)
    ax1.set_xlabel('$a$')
    ax1.set_ylabel('$c$')
    ax1.set_title('Consumption Policy Function $g_c(a,s)$')

    ax2.legend(frameon=True)
    ax2.set_xlabel('$a$')
    ax2.set_ylabel('$a\prime$')
    ax2.set_title('Savings Policy Function $g_a(a,s)$')
    # fig.tight_layout(pad=0.5)
    # plt.savefig(path + "/policy_func.png", bbox_inches='tight')

    # Plot MPCs:
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    ax1.plot(param['gA'][0:-1], mpc[:, 0], label="s min")
    ax1.plot(param['gA'][0:-1], mpc[:, -1], label="s max")
    ax1.set_xlim([param['gA'][0] - 1.0, 50])
    # ax1.set_ylim([0.0, 0.001])

    ax1.set_xlabel('$a$')
    ax1.set_ylabel('MPC')
    ax1.legend(frameon=True)
    ax1.set_title('Marginal Propensities to Consume')

    # Plot densities:
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    ax1.plot(param['gA'], dsn[:, 0], label="s min")
    ax1.plot(param['gA'], dsn[:, -1], label="s max")
    ax1.set_xlim([param['gA'][0] - 1.0, 50])
    ax1.set_ylim([0.0, 0.001])

    ax1.set_xlabel('$a$')
    ax1.set_ylabel('density')
    ax1.legend(frameon=True)
    ax1.set_title('Invariant Distribution')

    plt.show()


param = setPar()
dec = solveHHproblemEGM(param, 0.04, 1.0)
dsn = solveInvariant(param, dec)

# param = setPar()
# (decisions, dsn, w, r, Kd, Ea) = model_solution(param)
# ModelStats(param, decisions, dsn, w, r, Kd, Ea)

a = 1