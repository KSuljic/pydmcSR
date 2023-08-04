#%%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pydmc
from pydmc import *

import importlib
importlib.reload(pydmc)
# --------------------



# %%
Data = pydmc.flanker_data()
Data.head()

# %%
res_ob = Ob(Data, n_caf=9)


# %%
fit_diff = Fit(res_ob, n_caf=9)
fit_diff.fit_data('differential_evolution', maxiter=30)


# %%
# TODO: extract parameters with lowest cost


# %%
fit_diff.table_summary()


# %%
result_para = fit_diff.res_th.prms
print(result_para)

# amp:17.7 tau:278.2 drc:0.53 bnds:49.8 res_mean: 331 res_sd:25.6 aa_shape: 1.6 sp_shape: 2.5 sp_shape: 0.0 sp_shape: 3.0 sp_shape: 4.0 | cost=6.57


# %%
fit_plot = PlotFit(fit_diff).summary()

# fit_diff.sim_prms
# returns
# Prms(amp=20, tau=30, drc=0.5, bnds=75, res_mean=300, res_sd=30, aa_shape=2, sp_shape=3, sigma=4, res_dist=1, t_max=1000, sp_dist=1, sp_lim=(-75, 75), sp_bias=0.0, dr_dist=0, dr_lim=(0.1, 0.7), dr_shape=3)
# TODO: wrong t_max=1000

# %%
result_para = fit_diff.res_th.prms
sim = Sim(result_para, n_caf=9, full_data=True, n_trls_data=150)
# %%

# %%
Plot(sim).summary()
# %%
Fit.calculate_cost_value_rmse(sim, res_ob)


# %%
