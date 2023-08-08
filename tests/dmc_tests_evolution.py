'''
Fitting Data via DMCfun package adaptation.

Author: Kenan Suljic
Date: 08.08.2023

'''

#%%
import os

from dataclasses import asdict

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
Data = pydmc.flanker_data()
Data.head()

# %%
res_ob = Ob(Data, n_caf=9)

# %%
prmsfit = PrmsFit()
#prmsfit.dmc_prms()
prmsfit.set_random_start_values(seed_value=18)
#prmsfit.set_start_values()
prmsfit


# %%
fit_diff = Fit(res_ob, start_vals=prmsfit, n_caf=9)


# %%
fit_diff.fit_data('differential_evolution', maxiter=100)

# %%
best_prms_dict = asdict(fit_diff.best_prms_out)
fit_diff.table_summary()


# %%
prmsfit.set_start_values(**best_prms_dict)
prmsfit

# %%
fit_diff_adv = Fit(res_ob, start_vals=prmsfit, search_grid=False, n_caf=9)

# %%
#fit_diff.fit_data('differential_evolution', maxiter=10)
fit_diff_adv.fit_data(maxiter=300)



# %%
fit_diff_adv.table_summary()

# %%
'''
Paper Values (flanker)

prms_instance = Prms(
    amp=19.42,
    tau=84.22,
    drc=0.6,
    bnds=56.47,
    res_mean=325.14,
    res_sd=28.28,
    aa_shape=2.24,
    sp_shape=2.8,
    sigma=4
)

'''


# %%
#result_para = fit_diff_adv.res_th.prms
#print(result_para)


# %%
fit_plot = PlotFit(fit_diff_adv).summary()

# %%
fit_diff_adv.res_th.prms

# %%
#result_para = fit_diff_adv.best_prms_out
result_para = fit_diff_adv.res_th.prms


# %%
sim = Sim(result_para, n_caf=9, full_data=True, n_trls_data=20)
# sim = Sim(prms_instance, n_caf=9, full_data=True, n_trls_data=20)

# %%
Plot(sim).summary()

# %%
Fit.calculate_cost_value_rmse(sim, res_ob)


# %%
df1 = pd.DataFrame(data=sim.data[0].T, columns=['RT', 'Response'])
df2 = pd.DataFrame(data=sim.data[1].T, columns=['RT', 'Response'])
# %%
sns.histplot(data=df1,
             x='RT',
             hue='Response')
# %%
sns.histplot(data=df2,
             x='RT',
             hue='Response')
# %%
sns.histplot(data=Data,
             x='RT',
             hue='Error')