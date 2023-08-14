'''
2_diff_dmc Parameter recovery test.


Author: Kenan Suljic
Date: 14.08.2023

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
# --------------------
# %%
prms = Prms()
# %%
prms
# %%
sim = Sim(prms=prms, n_caf=9, full_data=False)

# %% 
df = sim2data(sim)
df.head(15)


# %%
prmsfit = PrmsFit()
#prmsfit.dmc_prms()
prmsfit.set_random_start_values(seed_value=12)
#prmsfit.set_start_values()
prmsfit

# %%
res_ob = Ob(df, n_caf=9)

# %%
fit_diff = Fit(res_ob, start_vals=prmsfit, n_caf=9)


# %%
fit_diff.fit_data(maxiter=100)

# %%
prmsfit_adv = set_best_parameters(fit_diff)

# %%
fit_diff_adv = Fit(res_ob, start_vals=prmsfit_adv, n_caf=9)

# %%
fit_diff_adv.fit_data(maxiter=100)

# %%
prmsfit_adv2 = set_best_parameters(fit_diff_adv)
fit_diff_adv2 = Fit(res_ob, start_vals=prmsfit_adv2, n_caf=9) 
fit_diff_adv2.fit_data(maxiter=100)

# %%
prms

# %%
prmsfit_adv3 = set_best_parameters(fit_diff_adv3)
fit_diff_adv3 = Fit(res_ob, start_vals=prmsfit_adv3, n_caf=9) 
fit_diff_adv3.fit_data(maxiter=200)


# %%
prmsfit_adv3 = set_best_parameters(fit_diff_adv3)
fit_diff_adv4 = Fit(res_ob, start_vals=prmsfit_adv3, n_caf=9) 
fit_diff_adv4.fit_data('differential_evolution', maxiter=3)


# %%
prmsfit_adv5 = set_best_parameters(fit_diff_adv5)
fit_diff_adv5 = Fit(res_ob, start_vals=prmsfit_adv5, search_grid=True, n_caf=9) 
fit_diff_adv5.fit_data(maxiter=100)


# %%
sns.displot(data=df,
             x='rt',
             hue='response',
             row='condition',
             )





# %%
df_caf = generate_cafs(sim)

# %%
df_caf.head(20)
# %%
sns.pointplot(data=df_caf,
              x='bin',
              y='error',
              hue='condition')

