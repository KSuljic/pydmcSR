'''
B010 - pydmcSR Fitting

Author: Kenan Suljic
Date: 16.08.2023

'''

#%%
import os

from dataclasses import asdict
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pydmc
from pydmc import *

import importlib
importlib.reload(pydmc)

# %% Parameters
seed = 18

# -------------- #

# %%
df = 

# %%
res_ob = Ob(df, n_caf=9)

# %%
prmsfit = PrmsFit()
prmsfit.set_random_start_values(seed_value=seed)
prmsfit

# -------------- #
# %%
fit_diff = Fit(res_ob, n_trls=10000, start_vals=prmsfit, n_caf=9)

# %%
fit_diff.fit_data('differential_evolution', maxiter=10, disp=True, seed=seed)

# %%
print(f'Best Parameters: {fit_diff.best_prms_out}')
print(f'Best cost: {fit_diff.best_cost}')

prmsfit_adv = set_best_parameters(fit_diff)

# %%
fit_vals_x = fit_diff.fit['x']
fit_vals_x


# %%
fit_diff_adv = Fit(res_ob, n_trls=10000, start_vals=prmsfit_adv, n_caf=9, search_grid=False) 
fit_diff_adv.fit_data('differential_evolution', x0=fit_vals_x, mutation=(0.1,0.4), maxiter=10, disp=True, seed=seed)

