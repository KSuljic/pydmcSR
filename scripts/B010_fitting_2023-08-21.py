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


import ipywidgets as widgets
from IPython.display import display, clear_output

# %% Parameters
seed = 12

# -------------- #

# %%
df = pd.read_csv('..\data\B010_pydmcVersion_150Cutoff_wo9-10-11-13-45-62.csv', usecols=lambda col: col not in ['Unnamed: 0'])



# %%
res_ob = Ob(df, n_caf=9, error_coding=(1,0))



# %%
Plot(res_ob).caf()

# %%
Plot(res_ob).delta()

# %%
Plot(res_ob).pdf()
# ----------------- #


# %%
prmsfit = PrmsFit()
prmsfit.set_random_start_values(seed_value=seed)
prmsfit


# # %%
# prmsfit = PrmsFit()
# prmsfit.set_start_values(sens_amp=31.548172023227444, sens_tau=103.05265775452429, sens_drc=0.681503332423484, sens_bnds=67.41466283935907, sens_aa_shape=1.2973868891908833, sens_res_mean=88.7467731111973, sens_res_sd=10.483958752969002, resp_amp=44.31823917665024, resp_tau=36.27356063321143, resp_drc=0.30592880998664884, resp_bnds=91.31821857596368, resp_aa_shape=1.921883910034707, resp_res_mean=175.50203117507124, resp_res_sd=84.69724860156873, resp_amp_ana=48.248678508946824, resp_tau_ana=266.19490748860835, resp_aa_shape_ana=1.5902979548578722, sp_shape=3.623193780450819, sigma=4, res_dist=1, t_max=2000, sp_dist=1, sp_lim=(-75, 75), sp_bias=0.0, dr_dist=0, dr_lim=(0.1, 0.7), dr_shape=3, sp_lim_sens=(-67.41466283935907, 67.41466283935907), sp_lim_resp=(-91.31821857596368, 91.31821857596368))
# fit_vals_x = prmsfit.array()



# -------------- #
# %%
fit_diff = Fit(res_ob, n_trls=10000, start_vals=prmsfit, n_caf=9)

# %%
fit_diff.fit_data('differential_evolution', maxiter=100, disp=True, seed=seed)


# ----------------- #


# %%
print(f'Best Parameters: {fit_diff.best_prms_out}')
print(f'Best cost: {fit_diff.best_cost}')

prmsfit = set_best_parameters(fit_diff)

# %%
fit_vals_x = fit_diff.fit['x']
fit_vals_x



# %%
fit_diff = Fit(res_ob, n_trls=10000, start_vals=prmsfit, n_caf=9)

# %%
fit_diff.fit_data('differential_evolution', x0=fit_vals_x, maxiter=200, disp=True, seed=seed)

# sens_amp:28.2 sens_tau:55.0 sens_drc:0.73 sens_bnds:70.3 sens_aa_shape: 2.1 sens_res_mean:  80 sens_res_sd: 6.9 resp_amp:47.2 resp_tau:37.8 resp_drc:0.29 resp_bnds:92.0 resp_aa_shape: 2.0 resp_res_mean: 158 resp_res_sd:82.8 resp_amp_ana:44.4 resp_tau_ana:258.7 resp_aa_shape_ana: 1.7 sp_shape: 3.8 sigma: 4.0 sp_bias: 0.0 dr_shape: 3.0 | cost=1.63
# 1.63

# sens_amp:27.0 sens_tau:90.7 sens_drc:0.64 sens_bnds:65.8 sens_aa_shape: 1.3 sens_res_mean:  87 sens_res_sd:20.2 resp_amp:41.7 resp_tau:40.2 resp_drc:0.32 resp_bnds:95.2 resp_aa_shape: 2.0 resp_res_mean: 172 resp_res_sd:74.5 resp_amp_ana:47.5 resp_tau_ana:259.1 resp_aa_shape_ana: 1.7 sp_shape: 3.3 sigma: 4.0 sp_bias: 0.0 dr_shape: 3.0 | cost=1.62
# 1.62

# sens_amp:35.5 sens_tau:57.5 sens_drc:0.76 sens_bnds:63.6 sens_aa_shape: 1.3 sens_res_mean:  67 sens_res_sd:97.8 resp_amp:46.2 resp_tau:27.1 resp_drc:0.31 resp_bnds:98.2 resp_aa_shape: 2.3 resp_res_mean: 186 resp_res_sd:13.2 resp_amp_ana:49.8 resp_tau_ana:297.4 resp_aa_shape_ana: 1.5 sp_shape: 3.6 sigma: 4.0 sp_bias: 0.0 dr_shape: 3.0 | cost=1.59
# 1.59

# sens_amp:28.8 sens_tau:71.7 sens_drc:0.72 sens_bnds:61.3 sens_aa_shape: 1.3 sens_res_mean:  53 sens_res_sd:94.2 resp_amp:49.2 resp_tau:68.1 resp_drc:0.34 resp_bnds:112.7 resp_aa_shape: 1.7 resp_res_mean: 164 resp_res_sd:11.2 resp_amp_ana:49.6 resp_tau_ana:288.2 resp_aa_shape_ana: 1.7 sp_shape: 3.7 sigma: 4.0 sp_bias: 0.0 dr_shape: 3.0 | cost=1.58
# 1.58

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

# Best Parameters: Prms(sens_amp=19.20885340290588, sens_tau=165.85223982491962, sens_drc=0.4730848156841636, sens_bnds=77.97299209408345, sens_aa_shape=2.018024089618249, sens_res_mean=92.19991174745076, sens_res_sd=22.96390204007147, resp_amp=39.90639062945641, resp_tau=141.72950391399982, resp_drc=0.38950050621674165, resp_bnds=57.28400867989853, resp_aa_shape=1.4354113748695227, resp_res_mean=153.54745300580504, resp_res_sd=12.449202040418001, sp_shape=2.8805178957173716, sigma=4, res_dist=1, t_max=2000, sp_dist=1, sp_lim=(-75, 75), sp_bias=0.0, dr_dist=0, dr_lim=(0.1, 0.7), dr_shape=3, sp_lim_sens=(-77.97299209408345, 77.97299209408345), sp_lim_resp=(-57.28400867989853, 57.28400867989853))
# Best cost: 2.3616139756573293

# Prms(sens_amp=31.548172023227444, sens_tau=103.05265775452429, sens_drc=0.681503332423484, sens_bnds=67.41466283935907, sens_aa_shape=1.2973868891908833, sens_res_mean=88.7467731111973, sens_res_sd=10.483958752969002, resp_amp=44.31823917665024, resp_tau=36.27356063321143, resp_drc=0.30592880998664884, resp_bnds=91.31821857596368, resp_aa_shape=1.921883910034707, resp_res_mean=175.50203117507124, resp_res_sd=84.69724860156873, resp_amp_ana=48.248678508946824, resp_tau_ana=266.19490748860835, resp_aa_shape_ana=1.5902979548578722, sp_shape=3.623193780450819, sigma=4, res_dist=1, t_max=2000, sp_dist=1, sp_lim=(-75, 75), sp_bias=0.0, dr_dist=0, dr_lim=(0.1, 0.7), dr_shape=3, sp_lim_sens=(-67.41466283935907, 67.41466283935907), sp_lim_resp=(-91.31821857596368, 91.31821857596368))
# 1.6069973724887472
# ----------------- #

# %%
fit_nelder = Fit(res_ob, n_trls=10000, start_vals=prmsfit, n_caf=9, search_grid=True) 
fit_nelder.start_vals

# %%
fit_nelder.fit_data(maxiter=500, disp=True)


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
def widget_function(sens_amp, sens_tau, sens_drc, sens_bnds, sens_aa_shape, 
                    resp_amp, resp_tau, resp_drc, resp_bnds, resp_aa_shape,
                    resp_amp_ana, resp_tau_ana, resp_aa_shape_ana,
                    res_mean, res_sd,
                    sp_shape, sigma, sp_bias, dr_shape):
    
    para = Prms(
        sens_amp=sens_amp, sens_tau=sens_tau, sens_drc=sens_drc, sens_bnds=sens_bnds, sens_aa_shape=sens_aa_shape,
        resp_amp=resp_amp, resp_tau=resp_tau, resp_drc=resp_drc, resp_bnds=resp_bnds, resp_aa_shape=resp_aa_shape, 
        resp_amp_ana=resp_amp_ana, resp_tau_ana=resp_tau_ana, resp_aa_shape_ana=resp_aa_shape_ana,
        res_mean=res_mean, res_sd=res_sd,
        sp_shape=sp_shape, sigma=sigma, sp_bias=sp_bias, dr_shape=dr_shape
    )

    df = res_ob.data
    df['Source'] = 'Og'
       
    sim = Sim(para, n_trls=10000, n_caf=9)
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
sens_amp_slider = widgets.FloatSlider(value=70, min=0, max=100, step=0.1, description='sens_amp:')
sens_tau_slider = widgets.FloatSlider(value=440, min=0, max=300, step=1, description='sens_tau:')
sens_drc_slider = widgets.FloatSlider(value=0.49, min=0, max=1, step=0.01, description='sens_drc:')
sens_bnds_slider = widgets.FloatSlider(value=144, min=0, max=200, step=0.1, description='sens_bnds:')
sens_aa_shape_slider = widgets.FloatSlider(value=3.7, min=0, max=10, step=0.1, description='sens_aa_shape:')

resp_amp_slider = widgets.FloatSlider(value=139, min=0, max=100, step=0.1, description='resp_amp:')
resp_tau_slider = widgets.FloatSlider(value=126, min=0, max=500, step=1, description='resp_tau:')
resp_drc_slider = widgets.FloatSlider(value=0.9, min=0, max=1, step=0.01, description='resp_drc:')
resp_bnds_slider = widgets.FloatSlider(value=31, min=0, max=200, step=0.1, description='resp_bnds:')
resp_aa_shape_slider = widgets.FloatSlider(value=1.6, min=0, max=10, step=0.1, description='resp_aa_shape:')

resp_amp_ana_slider = widgets.FloatSlider(value=138, min=0, max=100, step=0.1, description='resp_amp_ana:')
resp_tau_ana_slider = widgets.FloatSlider(value=158, min=0, max=500, step=1, description='resp_tau_ana:')
resp_aa_shape_ana_slider = widgets.FloatSlider(value=1.34, min=0, max=10, step=0.1, description='resp_aa_shape_ana:')

res_mean_slider = widgets.FloatSlider(value=180, min=0, max=300, step=0.1, description='res_mean:')
res_sd_slider = widgets.FloatSlider(value=0, min=0, max=100, step=0.1, description='res_sd:')

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
para = fit.best_prms_out
fit.best_cost

# %%
Parameters = Prms(**parameters)
sim = Sim(Parameters, n_caf=9)
Fit.calculate_cost_value_rmse(sim, res_ob)

# ----------------- #

# %%
sim = Sim(fit_diff.best_prms_out, n_caf=9)
Fit.calculate_cost_value_rmse(sim, res_ob)

# %%
import inspect

para_dict = fit_diff_adv.best_prms_out.__dict__
function_parameters = inspect.signature(widget_function).parameters
filtered_args = {k: para_dict[k] for k in function_parameters if k in para_dict}

widget_function(**filtered_args)


# %%
Plot(res_ob).caf()
Plot(res_ob).delta()

# %%
Plot(sim).caf()
Plot(sim).delta()







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