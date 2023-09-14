'''
B010 - pydmcSR Fitting - E299 Version

Author: Kenan Suljic
Date: 05.09.2023

'''

#%%
import os

from datetime import datetime

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

import pickle

import ipywidgets as widgets
from IPython.display import display, clear_output

# %% Parameters
seed = 42

# %%
pd.options.display.max_columns = 999

# -------------- #

# %%
df = pd.read_csv('..\data\B010_pydmcVersion_150Cutoff_wo9-10-11-13-45-62.csv', usecols=lambda col: col not in ['Unnamed: 0'])


# %%
res_ob = Ob(df, n_caf=9, error_coding=(1,0))


# %%
prmsfit = PrmsFit()
prmsfit.set_random_start_values(seed_value=seed)
prms_dict = prmsfit.dict()
prms_dict


# %%
prms = Prms(**prms_dict)
sim = Sim(prms)
df = sim2data(sim)
res_ob = Ob(df, n_caf=9)



# %%
Plot(res_ob).caf()

# %%
Plot(res_ob).delta()

# %%
Plot(res_ob).pdf()
# ----------------- #


# -------------- #
# %%
fit_diff = Fit(res_ob, n_trls=2000, start_vals=prmsfit, n_caf=9)

# %%
fit_diff.fit_data('differential_evolution', disp=True, maxiter=300, seed=seed)

# | cost=0.7610 amp_ana: 95.1 tau_ana: 428.0 aa_shape_ana:  2.7 sens_drc_comp: 0.79 sens_drc_incomp: 0.77 sens_bnds: 101.9 sp_lim_sens: (-101.92728040317043, 101.92728040317043) amp_ext: 87.0 tau_ext: 490.8 aa_shape_ext:  4.2 amp_anaS2extR:  4.0 tau_anaS2extR: 81.6 aa_shape_anaS2extR:  3.7 amp_extS2anaR: 93.3 tau_extS2anaR: 457.8 aa_shape_extS2anaR:  2.9 resp_drc: 0.57 resp_bnds: 78.3 sp_lim_resp: (-78.31679232747805, 78.31679232747805) drc_sd:  0.4 res_dist: 1 res_mean: 253.3 res_sd: 28.3 sp_sd: 42.2 sp_dist: 0 sp_shape:  3.0 sp_bias:  0.0 dr_dist: 1 dr_lim: (0.1, 0.7) dr_shape:  3.0

'''
{
    'amp_ana': 95.1,
    'tau_ana': 428.0,
    'aa_shape_ana': 2.7,
    'sens_drc_comp': 0.79,
    'sens_drc_incomp': 0.77,
    'sens_bnds': 101.9,
    'sp_lim_sens': (-101.92728040317043, 101.92728040317043),
    'amp_ext': 87.0,
    'tau_ext': 490.8,
    'aa_shape_ext': 4.2,
    'amp_anaS2extR': 4.0,
    'tau_anaS2extR': 81.6,
    'aa_shape_anaS2extR': 3.7,
    'amp_extS2anaR': 93.3,
    'tau_extS2anaR': 457.8,
    'aa_shape_extS2anaR': 2.9,
    'resp_drc': 0.57,
    'resp_bnds': 78.3,
    'sp_lim_resp': (-78.31679232747805, 78.31679232747805),
    'drc_sd': 0.4,
    'res_dist': 1,
    'res_mean': 253.3,
    'res_sd': 28.3,
    'sp_sd': 42.2,
    'sp_dist': 0,
    'sp_shape': 3.0,
    'sp_bias': 0.0,
    'dr_dist': 1,
    'dr_lim': (0.1, 0.7),
    'dr_shape': 3.0
}


'''

# %%
prmsfit = PrmsFit()
prmsfit.set_start_values(**para_f)

fit_vals_x = prmsfit.array()

# ----------------- #


# %%
print(f'Best Parameters: {fit_diff.best_prms_out}')
print(f'Best cost: {fit_diff.best_cost}')

prmsfit = set_best_parameters(fit_diff)

# %%
fit_vals_x = fit_diff.fit['x']
fit_vals_x


# %%
para_dict = fit_diff.best_prms_out.__dict__
df_para = pd.DataFrame(para_dict)
df_para



# %%
fit_diff = Fit(res_ob, n_trls=5000, start_vals=prmsfit, n_caf=9)

# %%
fit_diff.fit_data('differential_evolution', x0=fit_vals_x, maxiter=100, disp=True, seed=seed)




# ----------------- #

# %%
fit_diff_adv = Fit(res_ob, n_trls=11000, start_vals=prmsfit, n_caf=9, search_grid=True) 
fit_vals_x = fit_diff_adv.start_vals.array()
fit_vals_x

# %%
fit_diff_adv.fit_data('differential_evolution', x0=fit_vals_x, mutation=(0.1,0.3), maxiter=50, disp=True, seed=seed)
# fit_diff_adv.fit_data('differential_evolution', x0=fit_vals_x, maxiter=30, disp=True, seed=seed)

# | cost=0.1623 amp_ana: 102.3 tau_ana: 346.8 aa_shape_ana:  4.0 sens_drc_comp: 0.59 sens_drc_incomp: 0.66 sens_bnds: 88.5 sp_lim_sens: (-88.54448453455268, 88.54448453455268) amp_ext: 88.4 tau_ext: 457.9 aa_shape_ext:  2.7 amp_anaS2extR: 34.7 tau_anaS2extR: 51.9 aa_shape_anaS2extR:  1.8 amp_extS2anaR: 96.1 tau_extS2anaR: 156.7 aa_shape_extS2anaR:  4.9 resp_drc: 0.43 resp_bnds: 78.1 sp_lim_resp: (-78.07396424553717, 78.07396424553717) drc_sd:  0.3 res_dist: 1 res_mean: 229.5 res_sd: 59.7 sp_sd: 101.3 sp_dist: 0 sp_shape:  3.0 sp_bias:  0.0 dr_dist: 1 dr_lim: (0.1, 0.7) dr_shape:  3.0
# | cost=0.1567 amp_ana: 102.1 tau_ana: 344.6 aa_shape_ana:  4.0 sens_drc_comp: 0.59 sens_drc_incomp: 0.66 sens_bnds: 88.6 sp_lim_sens: (-88.55784980714024, 88.55784980714024) amp_ext: 88.3 tau_ext: 458.0 aa_shape_ext:  2.7 amp_anaS2extR: 35.2 tau_anaS2extR: 52.0 aa_shape_anaS2extR:  1.8 amp_extS2anaR: 96.1 tau_extS2anaR: 156.6 aa_shape_extS2anaR:  4.9 resp_drc: 0.43 resp_bnds: 78.0 sp_lim_resp: (-78.01843941631826, 78.01843941631826) drc_sd:  0.3 res_dist: 1 res_mean: 229.3 res_sd: 60.2 sp_sd: 99.8 sp_dist: 0 sp_shape:  3.0 sp_bias:  0.0 dr_dist: 1 dr_lim: (0.1, 0.7) dr_shape:  3.0
# | cost=0.1397 amp_ana: 102.2 tau_ana: 342.6 aa_shape_ana:  4.0 sens_drc_comp: 0.59 sens_drc_incomp: 0.66 sens_bnds: 88.7 sp_lim_sens: (-88.71719919291417, 88.71719919291417) amp_ext: 88.4 tau_ext: 458.5 aa_shape_ext:  2.7 amp_anaS2extR: 35.2 tau_anaS2extR: 52.1 aa_shape_anaS2extR:  1.8 amp_extS2anaR: 96.1 tau_extS2anaR: 156.9 aa_shape_extS2anaR:  4.9 resp_drc: 0.43 resp_bnds: 78.4 sp_lim_resp: (-78.40013224178334, 78.40013224178334) drc_sd:  0.3 res_dist: 1 res_mean: 229.5 res_sd: 60.2 sp_sd: 99.6 sp_dist: 0 sp_shape:  3.0 sp_bias:  0.0 dr_dist: 1 dr_lim: (0.1, 0.7) dr_shape:  3.0


# ----------------- #


# %%
print(f'Best Parameters: {fit_diff_adv.best_prms_out}')
print(f'Best cost: {fit_diff_adv.best_cost}')

prmsfit_adv = set_best_parameters(fit_diff_adv)

# %%
fit_vals_x = fit_diff_adv.fit['x']
fit_vals_x

# ----------------- #

# %%
fit_nelder = Fit(res_ob, n_trls=10000, start_vals=prmsfit, n_caf=9, search_grid=False) 
fit_nelder.start_vals

# %%
fit_nelder.fit_data(maxiter=1000, disp=True)


# %%
print(f'Best Parameters: {fit_nelder.best_prms_out}')
print(f'Best cost: {fit_nelder.best_cost}')

prmsfit = set_best_parameters(fit_nelder)

# %%
fit_vals_x = fit_nelder.fit['x']
fit_vals_x




# %%
def activation(tim, amp, tau, aa_shape, drc, comp):

    dr_con = (drc * tim)

    eq4 = (
        amp
        * np.exp(-tim / tau)
        * (np.exp(1) * tim / (aa_shape - 1) / tau)
        ** (aa_shape - 1)
    ) # Expected automatic evidence accumulation trajectory
    
    dr_auto = (
        comp
        * eq4
        * ((aa_shape - 1) / tim - 1 / tau)
    ) # corresponding automatic drift rate

    super_comp = (eq4 + dr_con)
    super_incomp = (-eq4 + dr_con)
    
    return dr_con, eq4, dr_auto, super_comp, super_incomp


def plot_activation(tim, activation, title):
    plt.figure(figsize=(10,6))
    plt.plot(tim, activation[0], color='black', label='controlled')
    plt.plot(tim, activation[1], color='green', label='dr_auto, compatible')
    plt.plot(tim, -activation[1], color='red', label='dr_auto, compatible')
    plt.plot(tim, activation[2], color='pink', label='automatic drift rate')
    plt.plot(tim, activation[3], color='darkgreen', label='super_comp')
    plt.plot(tim, activation[4], color='darkred', label='super_incomp')



    plt.title(title)
    plt.xlabel('Time (ms)')
    plt.ylim((-1000, 1000))
    plt.ylabel('Activation')
    plt.grid(True)
    plt.show()



# %%
# Example usage
tim = np.linspace(0.01, 1000, 2000)  # This is just an example time range; you might want to adjust this
plot_activation(tim, activation(tim, 150, 150, 1.5, 0.489, -1), 'Sensory Activation')
plot_activation(tim, activation(tim, 140, 126, 1.6, 0.91, -1), 'Response Activation')
plot_activation(tim, activation(tim, 139, 158, 1.34, 0.91, -1), 'Ana Activation')



# %%
# ----------------- #

# ----------------- #

# %%
#df.insert(len(df.columns), 'Source', 'Og')
df = res_ob.data

#%%
df_fit = sim2data(fit_diff.res_th)

# %%
df_fit.insert(len(df_fit.columns), 'Source', 'Fit')

# %%
df_fit_sub = df_fit.groupby('condition').sample(n=11982)

# %%
df_comp = pd.concat([df, df_fit])


# %%
df_fit.groupby('condition').count()


# %%
axes = sns.displot(data=df_comp,
             x='RT',
             hue='Source',
             col='Error',
             row='condition',
             kind='kde',
             )

for ax in axes.axes.flat:
    ax.set_xticks(np.arange(0, 2500, 100))
    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    ax.grid(True, axis='x')

# ----------------- #

# %%
palette_condition = ['silver', 'limegreen', 'dodgerblue', 'tomato',  
              'grey', 'green', 'darkblue', 'darkred']

axes = sns.displot(data=df,
             x='RT',
             hue='condition',
             col='Error',
             kind='kde',
             palette=palette_condition,
             )

for ax in axes.axes.flat:
    ax.set_xticks(np.arange(0, 2500, 100))
    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    ax.grid(True, axis='x');


# ------------- #




# %%
prmsfit = PrmsFit()
prmsfit.set_start_values(**parameters)
prmsfit

# -------------- #
# %%
fit = Fit(res_ob, n_trls=10000, start_vals=prmsfit, n_caf=9, search_grid=False)

# %%
fit.fit_data(maxiter=5000, disp=True)


# %%
para = fit_diff_adv.best_prms_out
fit_diff_adv.best_cost

# %%
model_check(res_ob, para)


# ----------------- #

# %%
sim = Sim(fit_diff_adv.best_prms_out, n_caf=9)
Fit.calculate_cost_value_rmse(sim, res_ob)

# %%
# import inspect

para_dict = fit_diff.best_prms_out.__dict__
para_dict



# %%
Plot(res_ob).caf()
Plot(res_ob).delta()

# %%
Plot(sim).caf()
Plot(sim).delta()



# ------------------------------------------- #

# %%
# Get the current date and time
now = datetime.now()

# Format the date and time as a string
date_string = now.strftime("%Y-%m-%d")  # For date (year, month, day)
time_string = now.strftime("%Hh_%Mmin")     # For time (hour, minute)


fname = 'Fit_V2_E299_dr1_sp0_'+date_string+'_'+time_string+'_5Effect_R3.pkl'

# Saving the objects:
with open('../Fits/'+fname, 'wb') as f: 
    pickle.dump([fit_diff_adv, seed], f)


# %% 

fname = 'Fit_V2_E299_dr0_sp0_2023-09-09_06h_35min.pkl'

# Getting back the objects:
with open('../Fits/'+fname, 'rb') as f: 
    fit_diff, seed = pickle.load(f)



# %%
'''
comp = 1

res_dist_type = 1
t_max = 2000
n_trls = 5

res_mean = 135
res_sd = 45

bnds = 22

sp_lim = (22, 22)
sp_shape = 3.6
sp_bias = 0

data = np.vstack((np.ones(n_trls) * t_max, np.zeros(n_trls)))
data[1, 1] = 1
data[1, 3] = 1
sens_results = data

sigma = 4.0

sp = sim._sp()
dr = sim._sens_dr()

 # Sensory Process (SP)
drc = (
    comp
    * sim.sens_eq4 # Expected automatic evidence accumulation trajectory (https://link.springer.com/article/10.3758/s13423-023-02288-0)
    * ((sim.prms.sens_aa_shape - 1) / sim.tim - 1 / sim.prms.sens_tau)
) # corresponding automatic drift rate



# Initializes arrays to store data and random residuals based on the specified distribution.

if res_dist_type == 1:
    res_dist = np.random.normal(res_mean, res_sd, n_trls)
else:
    width = max([0.01, np.sqrt((res_sd * res_sd / (1 / 12)))])
    res_dist = np.random.uniform(res_mean - width, res_mean + width, n_trls)
#


# For each trial:
for trl in range(n_trls):

    trl_xt = sp[trl]

    # Checks sensory results (if provided) to determine the outcome direction.
    reverse_outcome = False
    if sens_results is not None:
        reverse_outcome = sens_results[1, trl] == 1 # resp boundaries flipped if sens == 1

    for tp in range(0, t_max):

        trl_xt += drc[tp] + dr[trl] + (sigma * np.random.randn()) # evidence accumulation can be modeled as a single combined Wiener process
        
        # Determines the RT and outcome based on the accumulated evidence and boundary.
        if np.abs(trl_xt) > bnds:
            data[0, trl] = tp + max(0, res_dist[trl])
            data[1, trl] = (trl_xt < 0.0) if not reverse_outcome else (trl_xt >= 0.0)
            break
'''