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

# %% Parameters
seed = 18


# %%
prms = Prms()
# %%
prms
# %%
sim = Sim(prms=prms, n_caf=9, full_data=False)

# %%
# sim_full = Sim(prms=prms, n_caf=9, full_data=True)


# %% 
df = sim2data(sim)
# df_full = sim2data(sim_full)

# %%
res_ob = Ob(df, n_caf=9)


# -----------------------------------------


# %%
prmsfit = PrmsFit()
prmsfit.set_random_start_values(seed_value=seed)
prmsfit

# %%
fit_diff = Fit(res_ob, n_trls=10000, start_vals=prmsfit, n_caf=9)


# %%
fit_diff.fit_data('differential_evolution', maxiter=20, disp=True, seed=seed)

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










# %%
fit_diff_adv = Fit(res_ob, start_vals=prmsfit_adv, n_caf=9, search_grid=True)

# %%
fit_diff_adv.fit_data(maxiter=100)



# %%
print(f'Best Parameters: {fit_diff_adv.best_prms_out}')
print(f'Best cost: {fit_diff_adv.best_cost}')

#prmsfit_adv2 = set_best_parameters(fit_diff_adv)


# %%
fit_diff_adv2 = Fit(res_ob, start_vals=prmsfit_adv2, n_caf=9, search_grid=True) 
fit_diff_adv2.fit_data(maxiter=100)

# %%
prms

# %%
print(f'Best Parameters: {fit_diff_adv2.best_prms_out}')
print(f'Best cost: {fit_diff_adv2.best_cost}')

prmsfit_adv2 = set_best_parameters(fit_diff_adv2)




# %%
prmsfit_adv3 = set_best_parameters(fit_diff_adv2)
fit_diff_adv3 = Fit(res_ob, start_vals=prmsfit_adv3, n_caf=9, search_grid=True) 
fit_diff_adv3.fit_data(maxiter=100)


# %%
print(f'Best Parameters: {fit_diff_adv3.best_prms_out}')
print(f'Best cost: {fit_diff_adv3.best_cost}')


# %%
fit_diff_adv3 = Fit(res_ob, start_vals=prmsfit_adv3, n_caf=9, search_grid=True) 
fit_diff_adv3.fit_data(maxiter=100)



# %%

print(f'Best Parameters: {fit_diff_adv.best_prms_out}')
print(f'Best cost: {fit_diff_adv.best_cost}')
fit_vals_x = fit_diff_adv.fit['x']
prmsfit_adv2 = set_best_parameters(fit_diff_adv)
prmsfit_adv2.__dict__

# %%
fit_diff_adv.best_prms_out

# %%
fit_diff_adv.fit_data('differential_evolution', x0=fit_vals_x, maxiter=20, disp=True, seed=seed)

# %%
fit_diff_adv2 = Fit(res_ob, n_trls=10000, start_vals=prmsfit_adv2, n_caf=9, search_grid=True) 
fit_diff_adv2.fit_data(maxiter=100)


# %%
print(f'Best Parameters: {fit_diff_adv2.best_prms_out}')
print(f'Best cost: {fit_diff_adv2.best_cost}')
prmsfit_adv3 = set_best_parameters(fit_diff_adv2)


# %%
fit_diff_adv3 = Fit(res_ob, n_trls=10000, start_vals=prmsfit_adv3, n_caf=9, search_grid=False) 
fit_diff_adv3.fit_data(maxiter=100)


# %%
print(f'Best Parameters: {fit_diff_adv3.best_prms_out}')
print(f'Best cost: {fit_diff_adv3.best_cost}')
prmsfit_adv4 = set_best_parameters(fit_diff_adv3)

fit_vals_x = fit_diff_adv3.fit['x']
fit_vals_x
#fit_diff_adv3.__dict__

# %%
fit_diff_adv4 = Fit(res_ob, n_trls=10000, start_vals=prmsfit_adv4, n_caf=9, search_grid=False) 
fit_diff_adv4.fit_data('differential_evolution', x0=fit_vals_x, mutation=(0,0.2), maxiter=5, disp=True, seed=seed)



# %%
print(f'Best Parameters: {fit_diff_adv4.best_prms_out}')
print(f'Best cost: {fit_diff_adv4.best_cost}')
prmsfit_adv5 = set_best_parameters(fit_diff_adv4)

fit_vals_x = fit_diff_adv4.fit['x']
fit_vals_x
#fit_diff_adv3.__dict__

# %%
fit_diff_adv5 = Fit(res_ob, n_trls=10000, start_vals=prmsfit_adv5, n_caf=9, search_grid=False) 
fit_diff_adv5.fit_data('differential_evolution', x0=fit_vals_x, mutation=(0.1,0.3), maxiter=10, disp=True, seed=seed)



# %%
print(f'Best Parameters: {fit_diff_adv5.best_prms_out}')
print(f'Best cost: {fit_diff_adv5.best_cost}')
prmsfit_adv6 = set_best_parameters(fit_diff_adv5)

fit_vals_x = fit_diff_adv5.fit['x']
fit_vals_x
#fit_diff_adv3.__dict__

# %%
fit_diff_adv6 = Fit(res_ob, n_trls=10000, start_vals=prmsfit_adv6, n_caf=9, search_grid=False) 
fit_diff_adv6.fit_data('differential_evolution', x0=fit_vals_x, mutation=(0.1,0.3), maxiter=5, disp=True, seed=seed)




# %%
sns.displot(data=df,
             x='RT',
             hue='Error',
             row='condition',
             )


# %%
df_fit = sim2data(fit_diff_adv5.res_th)



# %%
sns.displot(data=df_fit,
             x='RT',
             hue='Error',
             row='condition',
             )


# %%
df.insert(len(df.columns), 'Source', 'Sim')

# %%
df_fit.insert(len(df_fit.columns), 'Source', 'Fit')
df_comp = pd.concat([df, df_fit])


# %%
sns.displot(data=df_comp,
             x='RT',
             hue='Source',
             col='Error',
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

