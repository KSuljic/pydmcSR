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
sim = Sim(prms=fit_diff.res_th.prms, n_caf=9, full_data=True)

# %%
print(sim.prms)
Plot(sim).summary()



# %%
prmsfit_res_adv = PrmsFit()
best_prms_dict = asdict(prms_res)
prmsfit_res_adv.set_start_values(**best_prms_dict)
prmsfit_res_adv


# %%
fit_diff_adv = Fit(res_ob, start_vals=prmsfit_res_adv, search_grid=False, n_caf=9)

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
PlotFit(fit_diff_adv).summary()


# %%
prms_res_adv = fit_diff_adv.return_result_prms()



# %%
sim = Sim(prms_res_adv, n_caf=9, full_data=True, n_trls_data=20)
# sim = Sim(prms_instance, n_caf=9, full_data=True, n_trls_data=20)

# %%
Plot(sim).summary()

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
