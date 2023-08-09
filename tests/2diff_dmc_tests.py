'''
Fitting Data via DMCfun package adaptation.

Author: Kenan Suljic
Date: 08.08.2023

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


# %% Sample Data
# Creating sample data


# Creating sample data
np.random.seed(42)
data = {
    "Subject": np.tile(np.arange(1, 51), 2),
    "Comp": ["comp" if i % 2 == 0 else "incomp" for i in range(100)],
    "RT": np.random.randint(300, 500, 100),
    "Error": np.random.choice([0, 1], 100),
}

# Creating a Pandas DataFrame
df_observed = pd.DataFrame(data)

print(df_observed)


# %%
#Data = pydmc.flanker_data()
#Data.head()

# %%
res_ob = Ob(df_observed, n_caf=9)

# %%
prmsfit = PrmsFit()
#prmsfit.dmc_prms()
prmsfit.set_random_start_values(seed_value=18)
#prmsfit.set_start_values()
prmsfit


# %%
fit_diff = Fit(res_ob, start_vals=prmsfit, n_caf=9)


# %%
fit_diff.fit_data('differential_evolution', maxiter=1)

# %%
fit_diff.print_summary()


# %%
PlotFit(fit_diff).summary()


# %%
#prms_res = fit_diff.return_result_prms()
#prms_res


# %%
fit_diff.res_th.prms

# %%
fit_diff.res_th.plot.summary()

# %%
best_prms = fit_diff.res_th.prms


# %%
sim = Sim(prms=best_prms, n_caf=9, full_data=False)

# %%
print(sim.prms)
Plot(sim).summary()

# %% 
Fit.calculate_cost_value_rmse(sim, res_ob)


# %%
prmsfit_adv = PrmsFit()
best_prms_dict = asdict(best_prms)
prmsfit_adv.set_start_values(**best_prms_dict)
prmsfit_adv


# %%
fit_diff_adv = Fit(res_ob, start_vals=prmsfit_adv, search_grid=False, n_caf=9)

# %%
#fit_diff.fit_data('differential_evolution', maxiter=10)
fit_diff_adv.fit_data(maxiter=200)



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
PlotFit(fit_diff_adv).summary()


# %%
best_prms_adv = fit_diff_adv.res_th.prms



# %%
sim = Sim(best_prms_adv, n_caf=9, full_data=False)
# sim = Sim(prms_instance, n_caf=9, full_data=True, n_trls_data=20)

# %%
sim.plot.summary()

# %%
print(f'RMSE: {Fit.calculate_cost_value_rmse(sim, res_ob)}')
print(f'SPE (%): {Fit.calculate_cost_value_spe(sim, res_ob)*100:4.2f}')


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
# %%
