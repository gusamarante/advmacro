from aiyagari import Aiyagari
import numpy as np


ag = Aiyagari(ns=3, phi=0)
pol_a, pol_c, stat_dist, k, r, w = ag.solve_equilibrium()

print('Interest Rate', r)
print('Wages', w)
print('Capital Stock', k)
print('Fraction of Constrained Investors', stat_dist.sum(axis=1)[0])

mean_a = np.sum(ag.grid_a * stat_dist.sum(axis=1))
var_a = np.sum(((ag.grid_a - mean_a)**2) * stat_dist.sum(axis=1))
print("Variance of asset distribution", var_a)