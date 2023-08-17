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
df = pd.read_csv('..\data\Adults_B010_LH2cross_150Cutoff_wo9-10-11-13-45-62.csv', usecols=['Subject', 'Condition', 'Reaction Time [ms]', 'Accuracy'])


# %%
df.columns = ['Subject', 'RT', 'condition', 'Error']

# %%
df.head()


# %%
condis = df.condition.unique()
condis

# %%
condi_new = ['exHULU', 'exHCLU', 'exHULC', 'exHCLC', 'anHULU', 'anHCLU', 'anHULC', 'anHCLC']

# %%
condi_dict = dict(zip(condis, condi_new))
condi_dict

# %%
df['condition'] = df['condition'].replace(condi_dict)

# %%
conditions_mapping = {
        "exHULU": ("comp", "comp"),
        "exHCLU": ("comp", "comp"),
        "exHULC": ("incomp", "comp"),
        "exHCLC": ("incomp", "comp"),
        "anHULU": ("comp", "comp"),
        "anHCLU": ("comp", "incomp"),
        "anHULC": ("incomp", "incomp"),
        "anHCLC": ("incomp", "comp"),
    }



# %%
# Add sens_comp and resp_comp columns based on the condition column
df["sens_comp"] = df["condition"].map(lambda x: conditions_mapping[x][0])
df["resp_comp"] = df["condition"].map(lambda x: conditions_mapping[x][1])



# %%
df.tail()


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
fit_diff.fit_data('differential_evolution', maxiter=75, disp=True, seed=seed)

# sens_amp:35.9 sens_tau:205.5 sens_drc:0.51 sens_bnds:80.0 sens_aa_shape: 2.2 sens_res_mean: 102 sens_res_sd:36.5 resp_amp:19.2 resp_tau:90.6 resp_drc:0.31 resp_bnds:115.5 resp_aa_shape: 1.1 resp_res_mean:  70 resp_res_sd:27.6 sp_shape: 3.3 sigma: 4.0 sp_bias: 0.0 dr_shape: 3.0 | cost=277.83


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

