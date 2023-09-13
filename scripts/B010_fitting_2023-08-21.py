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
seed = 40

# %%
pd.options.display.max_columns = 999

# -------------- #

# %%
df = pd.read_csv('data\B010_pydmcVersion_150Cutoff_wo9-10-11-13-45-62.csv', usecols=lambda col: col not in ['Unnamed: 0'])


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



# # %%
# prmsfit = PrmsFit()
# prmsfit.set_start_values(sens_amp=17.81197539507317, sens_tau=339.43962817696405, sens_drc=0.7150909968781219, sens_bnds=112.15466147140911, sens_aa_shape=3.276941985912356, sp_lim_sens=(-112.15466147140911, 112.15466147140911), resp_amp=144.2996073267513, resp_tau=341.8996840532543, resp_drc=0.9979278012297952, resp_bnds=26.343192995405516, resp_aa_shape=1.3340081032175015, resp_amp_ana=113.13666504598645, resp_tau_ana=117.22705484742178, resp_aa_shape_ana=1.4981306675266353, sp_lim_resp=(-26.343192995405516, 26.343192995405516), res_dist=1, res_mean=154.84377389599183, res_sd=62.14378633126033, sp_shape=3, sigma=4, t_max=2000, sp_dist=0, sp_bias=0.0, dr_dist=0, dr_lim=(0.1, 0.7), dr_shape=3)
# fit_vals_x = prmsfit.array()


# -------------- #
# %%
fit_diff = Fit(res_ob, n_trls=10000, start_vals=prmsfit, n_caf=9)

# %%
fit_diff.fit_data('differential_evolution', disp=True, maxiter=300, seed=seed)

# | cost=0.1155 amp_ana: 98.8 tau_ana: 253.4 aa_shape_ana:  2.5 sens_drc_comp: 0.38 sens_drc_incomp: 0.12 sens_bnds: 61.6 sp_lim_sens: (-61.62525113683415, 61.62525113683415) amp_ext: 109.8 tau_ext: 203.0 aa_shape_ext:  2.3 amp_anaS2extR: 108.4 tau_anaS2extR: 209.6 aa_shape_anaS2extR:  2.2 amp_extS2anaR: 98.1 tau_extS2anaR: 314.1 aa_shape_extS2anaR:  3.8 resp_drc: 0.72 resp_bnds: 131.2 sp_lim_resp: (-131.17845491535212, 131.17845491535212) drc_sd:  0.2 res_dist: 1 res_mean: 209.9 res_sd: 72.2 sp_sd: 29.1 sp_dist: 1 sp_shape:  3.0 sp_bias:  0.0 dr_dist: 1 dr_lim: (0.1, 0.7) dr_shape:  3.0

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
fit_diff = Fit(res_ob, n_trls=10000, start_vals=prmsfit, n_caf=9)

# %%
fit_diff.fit_data('differential_evolution', x0=fit_vals_x, maxiter=500, disp=True, seed=seed)




# ----------------- #

# %%
fit_diff_adv = Fit(res_ob, n_trls=10000, start_vals=prmsfit, n_caf=9, search_grid=False) 
fit_vals_x = fit_diff_adv.start_vals.array()
fit_vals_x

# %%
fit_diff_adv.fit_data('differential_evolution', x0=fit_vals_x, mutation=(0.1,0.3), maxiter=50, disp=True, seed=seed)
# fit_diff_adv.fit_data('differential_evolution', x0=fit_vals_x, maxiter=30, disp=True, seed=seed)


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
# Define the widget function
def model_check(para):
    # Rest of the function remains the same, assuming Prms and other functions are defined elsewhere

    df = res_ob.data
    df['Source'] = 'Og'
       
    sim = Sim(para, n_trls=11000, n_caf=9)
    df_fit = sim2data(sim)
    print(Fit.calculate_cost_value_rmse(sim, res_ob))


    df_fit.insert(len(df_fit.columns), 'Source', 'Fit')

    # df_fit_sub = df_fit.groupby('condition').sample(n=11982)
    df_comp = pd.concat([df, df_fit])

    axes = sns.displot(data=df_comp,
                       x='RT',
                       hue='Source',
                       col='Error',
                       row='condition',
                       kind='kde')
    
    for ax in axes.axes.flat:
        ax.set_xticks(np.arange(0, 2500, 100))
        ax.set_xticklabels(ax.get_xticks(), rotation=90)
        ax.grid(True, axis='x')

    plt.show()

    Plot(sim).caf()
    Plot(sim).delta()

# %%
# Create widgets for each parameter in Prms
sens_amp_slider = widgets.FloatSlider(value=71.3, min=0, max=300, step=0.1, description='sens_amp:')
sens_tau_slider = widgets.FloatSlider(value=395, min=0, max=500, step=1, description='sens_tau:')
sens_drc_slider = widgets.FloatSlider(value=0.87, min=0, max=1, step=0.01, description='sens_drc:')
sens_bnds_slider = widgets.FloatSlider(value=101.68, min=0, max=200, step=0.1, description='sens_bnds:')
sens_aa_shape_slider = widgets.FloatSlider(value=4.6, min=0, max=10, step=0.1, description='sens_aa_shape:')

resp_amp_slider = widgets.FloatSlider(value=100, min=0, max=300, step=0.1, description='resp_amp:')
resp_tau_slider = widgets.FloatSlider(value=168, min=0, max=500, step=1, description='resp_tau:')
resp_drc_slider = widgets.FloatSlider(value=0.95, min=0, max=1, step=0.01, description='resp_drc:')
resp_bnds_slider = widgets.FloatSlider(value=26.7, min=0, max=200, step=0.1, description='resp_bnds:')
resp_aa_shape_slider = widgets.FloatSlider(value=1.4, min=0, max=10, step=0.1, description='resp_aa_shape:')

resp_amp_ana_slider = widgets.FloatSlider(value=117, min=0, max=300, step=0.1, description='resp_amp_ana:')
resp_tau_ana_slider = widgets.FloatSlider(value=305, min=0, max=500, step=1, description='resp_tau_ana:')
resp_aa_shape_ana_slider = widgets.FloatSlider(value=1.3, min=0, max=10, step=0.1, description='resp_aa_shape_ana:')

res_mean_slider = widgets.FloatSlider(value=145, min=0, max=300, step=0.1, description='res_mean:')
res_sd_slider = widgets.FloatSlider(value=16, min=0, max=100, step=0.1, description='res_sd:')

sp_shape_slider = widgets.FloatSlider(value=3, min=0, max=10, step=0.1, description='sp_shape:')
sigma_slider = widgets.FloatSlider(value=4.0, min=0, max=10, step=0.1, description='sigma:')
sp_bias_slider = widgets.FloatSlider(value=0.0, min=-10, max=10, step=0.1, description='sp_bias:')
dr_shape_slider = widgets.FloatSlider(value=3.0, min=0, max=10, step=0.1, description='dr_shape:')


# Define the user interface with all the sliders
ui = widgets.VBox([
    sens_amp_slider, sens_tau_slider, sens_drc_slider, sens_bnds_slider, sens_aa_shape_slider, 
    resp_amp_slider, resp_tau_slider, resp_drc_slider, resp_bnds_slider, resp_aa_shape_slider,
    resp_amp_ana_slider, resp_tau_ana_slider, resp_aa_shape_ana_slider,
    res_mean_slider, res_sd_slider, sp_shape_slider, 
    sigma_slider, sp_bias_slider, dr_shape_slider
])


# Link the sliders to the widget function
out = widgets.interactive_output(widget_function, {
    'sens_amp': sens_amp_slider, 
    'sens_tau': sens_tau_slider, 
    'sens_drc': sens_drc_slider,
    'sens_bnds': sens_bnds_slider,
    'sens_aa_shape': sens_aa_shape_slider,
 
    'resp_amp': resp_amp_slider,
    'resp_tau': resp_tau_slider,
    'resp_drc': resp_drc_slider,
    'resp_bnds': resp_bnds_slider,
    'resp_aa_shape': resp_aa_shape_slider,
    
    'resp_amp_ana': resp_amp_ana_slider,
    'resp_tau_ana': resp_tau_ana_slider,
    'resp_aa_shape_ana': resp_aa_shape_ana_slider,
    
    'res_mean': res_mean_slider,
    'res_sd': res_sd_slider,
    'sp_shape': sp_shape_slider,
    'sigma': sigma_slider,
    'sp_bias': sp_bias_slider,
    'dr_shape': dr_shape_slider
})

display(ui, out)


# %%
parameters = {
    'sens_amp': sens_amp_slider.value,
    'sens_tau': sens_tau_slider.value,
    'sens_drc': sens_drc_slider.value,
    'sens_bnds': sens_bnds_slider.value,
    'sens_aa_shape': sens_aa_shape_slider.value,
    'resp_amp': resp_amp_slider.value,
    'resp_tau': resp_tau_slider.value,
    'resp_drc': resp_drc_slider.value,
    'resp_bnds': resp_bnds_slider.value,
    'resp_aa_shape': resp_aa_shape_slider.value,
    'res_mean': res_mean_slider.value,
    'res_sd': res_sd_slider.value,
    'sp_shape': sp_shape_slider.value,
    'sigma': sigma_slider.value,
    'sp_bias': sp_bias_slider.value,
    'dr_shape': dr_shape_slider.value,
}

print(parameters)


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
para = fit_diff.best_prms_out
fit_diff.best_cost

# %%
model_check(para)

# Parameters = Prms(**parameters)
# para_dict = {'sens_amp': 71.18560180328221,
#  'sens_tau': 330.4488888153414,
#  'sens_drc': 0.6997694240123543,
#  'sens_bnds': 38.53804588029781,
#  'sens_aa_shape': 1.0434417723602718,
#  'resp_amp': 56.21317380950466,
#  'resp_tau': 140.65382904177403,
#  'resp_drc': 0.8293132470115754,
#  'resp_bnds': 109.777044895029,
#  'resp_aa_shape': 3.4058281549109326,
#  'resp_amp_ana': 83.7285071630022,
#  'resp_tau_ana': 332.3539375066405,
#  'resp_aa_shape_ana': 1.581211178840412,
#  'res_mean': 160.01369798119245,
#  'res_sd': 20.41539562694442,
#  'sp_shape': 3,
#  'sigma': 4,
#  'sp_bias':0,
#  'dr_shape':3}

# para_dict = fit_diff.best_prms_out.__dict__

# Parameters = Prms(**para_dict)
# sim = Sim(Parameters, n_caf=9)
# Fit.calculate_cost_value_rmse(sim, res_ob)

# ----------------- #

# %%
sim = Sim(fit_diff.best_prms_out, n_caf=9)
Fit.calculate_cost_value_rmse(sim, res_ob)

# %%
# import inspect

para_dict = fit_diff.best_prms_out.__dict__
para_dict

# para_dict = {'sens_amp': 20,
#  'sens_tau': 100,
#  'sens_drc': 0.31590874978190203,
#  'sens_bnds': 71.63633510731007,
#  'sens_aa_shape': 1.8,
#  'sp_lim_sens': (-71.63633510731007, 71.63633510731007),
#  'resp_amp': 150,
#  'resp_tau': 450,
#  'resp_drc': 0.4,
#  'resp_bnds': 75.66747024929097,
#  'resp_aa_shape': 3.5,
#  'resp_amp_ana': 15,
#  'resp_tau_ana': 200,
#  'resp_aa_shape_ana': 2.2,
#  'sp_lim_resp': (-75.66747024929097, 75.66747024929097),
#  'res_dist': 1,
#  'res_mean': 180,
#  'res_sd': 27.43828746055531,
#  'sp_shape': 3,
#  'sigma': 4,
#  't_max': 2000,
#  'sp_dist': 0,
#  'sp_bias': 0.0,
#  'dr_dist': 0,
#  'dr_lim': (0.1, 0.7),
#  'dr_shape': 3}

# function_parameters = inspect.signature(model_check).parameters
# filtered_args = {k: para_dict[k] for k in function_parameters if k in para_dict}

# model_check(**filtered_args)


# %%
Plot(res_ob).caf()
Plot(res_ob).delta()

# %%
Plot(sim).caf()
Plot(sim).delta()



# %%
# Get the current date and time
now = datetime.now()

# Format the date and time as a string
date_string = now.strftime("%Y-%m-%d")  # For date (year, month, day)
time_string = now.strftime("%Hh_%Mmin")     # For time (hour, minute)


fname = 'Fit_V2_E299_dr1_sp0_'+date_string+'_'+time_string+'.pkl'

# Saving the objects:
with open('Fits/'+fname, 'wb') as f: 
    pickle.dump([fit_diff, seed], f)


# %% 

fname = 'Fit_V2_E299_dr0_sp0_2023-09-09_06h_35min.pkl'

# Getting back the objects:
with open('Fits/'+fname, 'rb') as f: 
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