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
seed = 18

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


# -------------- #
# %%
fit_diff = Fit(res_ob, n_trls=10000, start_vals=prmsfit, n_caf=9)

# %%
fit_diff.fit_data('differential_evolution', maxiter=10, disp=True, seed=seed)


# Best Parameters: Prms(sens_amp=29.62708004095449, sens_tau=231.9884182080549, sens_drc=0.6145516480773158, sens_bnds=81.24575319252381, sens_aa_shape=1.6356387824131913, sens_res_mean=138.48768180106424, sens_res_sd=56.96543849604238, resp_amp=31.262200562204143, resp_tau=231.60736107205798, resp_drc=0.36983104461375127, resp_bnds=58.26011880699383, resp_aa_shape=1.7676762844869327, resp_res_mean=127.34667997398945, resp_res_sd=39.19809057684322, sp_shape=3.1958983107817365, sigma=4, res_dist=1, t_max=2000, sp_dist=1, sp_lim=(-75, 75), sp_bias=0.0, dr_dist=0, dr_lim=(0.1, 0.7), dr_shape=3, sp_lim_sens=(-81.24575319252381, 81.24575319252381), sp_lim_resp=(-58.26011880699383, 58.26011880699383))
# Best cost: 2.545066108399058
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
fit_diff.fit_data('differential_evolution', x0=fit_vals_x, maxiter=50, disp=True, seed=seed)

# sens_amp:22.9 sens_tau:172.2 sens_drc:0.70 sens_bnds:68.5 sens_aa_shape: 1.8 sens_res_mean: 116 sens_res_sd:47.4 resp_amp:36.3 resp_tau:246.7 resp_drc:0.38 resp_bnds:59.5 resp_aa_shape: 1.4 resp_res_mean: 175 resp_res_sd:32.1 sp_shape: 2.7 sigma: 4.0 sp_bias: 0.0 dr_shape: 3.0 | cost=2.44
# 2.44

# sens_amp:33.8 sens_tau:212.2 sens_drc:0.66 sens_bnds:72.3 sens_aa_shape: 2.0 sens_res_mean:  78 sens_res_sd:18.0 resp_amp:35.8 resp_tau:225.9 resp_drc:0.37 resp_bnds:65.8 resp_aa_shape: 1.4 resp_res_mean: 185 resp_res_sd:17.3 sp_shape: 2.3 sigma: 4.0 sp_bias: 0.0 dr_shape: 3.0 | cost=2.39
# 2.39
# Best Parameters: Prms(sens_amp=33.76135167483847, sens_tau=212.23852121228498, sens_drc=0.663677458251879, sens_bnds=72.25296921868068, sens_aa_shape=2.0059233372334075, sens_res_mean=78.46839576832889, sens_res_sd=17.975469918185986, resp_amp=35.78299328536764, resp_tau=225.8759207866558, resp_drc=0.3709202395349319, resp_bnds=65.84273447650459, resp_aa_shape=1.3849635066316799, resp_res_mean=185.16443049263196, resp_res_sd=17.34761946850987, sp_shape=2.304085898281109, sigma=4, res_dist=1, t_max=2000, sp_dist=1, sp_lim=(-75, 75), sp_bias=0.0, dr_dist=0, dr_lim=(0.1, 0.7), dr_shape=3, sp_lim_sens=(-72.25296921868068, 72.25296921868068), sp_lim_resp=(-65.84273447650459, 65.84273447650459))


# ----------------- #

# %%
fit_diff_adv = Fit(res_ob, n_trls=10000, start_vals=prmsfit, n_caf=9, search_grid=False) 
fit_diff_adv.start_vals

# %%
fit_diff_adv.fit_data('differential_evolution', x0=fit_vals_x, mutation=(0.1,0.4), maxiter=70, disp=True, seed=seed)


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
fit_nelder.fit_data(maxiter=300, disp=True)


# %%
print(f'Best Parameters: {fit_nelder.best_prms_out}')
print(f'Best cost: {fit_nelder.best_cost}')

prmsfit = set_best_parameters(fit_nelder)

# %%
fit_vals_x = fit_nelder.fit['x']
fit_vals_x




# %%
def activation(tim, amp, tau, aa_shape, comp):
    eq4 = (
        amp
        * np.exp(-tim / tau)
        * (np.exp(1) * tim / (aa_shape - 1) / tau)
        ** (aa_shape - 1)
    )
    
    drc = (
        comp
        * eq4
        * ((aa_shape - 1) / tim - 1 / tau)
    )
    
    return drc, eq4


def plot_activation(tim, activation, title):
    plt.figure(figsize=(10,6))
    plt.plot(tim, activation[0], color='black')
    plt.plot(tim, activation[1])
    plt.title(title)
    plt.xlabel('Time (ms)')
    plt.ylabel('Activation')
    plt.grid(True)
    plt.show()



# %%
# Example usage
tim = np.linspace(0.01, 2000, 2000)  # This is just an example time range; you might want to adjust this
plot_activation(tim, activation(tim, 33.8, 212, 2, 10), 'Sensory Activation')
plot_activation(tim, activation(tim, 35.8, 225, 1.38, 10), 'Response Activation')


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
df_comp = pd.concat([df, df_fit_sub])


# %%
df_fit_sub.groupby('condition').count()


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
def widget_function(sens_amp, sens_tau, sens_drc, sens_bnds, sens_aa_shape, sens_res_mean, sens_res_sd,
                    resp_amp, resp_tau, resp_drc, resp_bnds, resp_aa_shape, resp_res_mean, resp_res_sd,
                    sp_shape, sigma, sp_bias, dr_shape):
    
    para = Prms(
        sens_amp=sens_amp, sens_tau=sens_tau, sens_drc=sens_drc, sens_bnds=sens_bnds, sens_aa_shape=sens_aa_shape,
        sens_res_mean=sens_res_mean, sens_res_sd=sens_res_sd, resp_amp=resp_amp, resp_tau=resp_tau, resp_drc=resp_drc,
        resp_bnds=resp_bnds, resp_aa_shape=resp_aa_shape, resp_res_mean=resp_res_mean, resp_res_sd=resp_res_sd,
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

# %%
# Create widgets for each parameter in Prms
sens_amp_slider = widgets.FloatSlider(value=25, min=0, max=100, step=0.1, description='sens_amp:')
sens_tau_slider = widgets.FloatSlider(value=150, min=0, max=300, step=0.1, description='sens_tau:')
sens_drc_slider = widgets.FloatSlider(value=0.4, min=0, max=1, step=0.01, description='sens_drc:')
sens_bnds_slider = widgets.FloatSlider(value=72.25, min=0, max=200, step=0.1, description='sens_bnds:')
sens_aa_shape_slider = widgets.FloatSlider(value=2.00, min=0, max=10, step=0.1, description='sens_aa_shape:')
sens_res_mean_slider = widgets.FloatSlider(value=78.5, min=0, max=200, step=0.1, description='sens_res_mean:')
sens_res_sd_slider = widgets.FloatSlider(value=18, min=0, max=100, step=0.1, description='sens_res_sd:')
resp_amp_slider = widgets.FloatSlider(value=24, min=0, max=100, step=0.1, description='resp_amp:')
resp_tau_slider = widgets.FloatSlider(value=200, min=0, max=300, step=0.1, description='resp_tau:')
resp_drc_slider = widgets.FloatSlider(value=0.4, min=0, max=1, step=0.01, description='resp_drc:')
resp_bnds_slider = widgets.FloatSlider(value=80, min=0, max=200, step=0.1, description='resp_bnds:')
resp_aa_shape_slider = widgets.FloatSlider(value=1.5, min=0, max=10, step=0.1, description='resp_aa_shape:')
resp_res_mean_slider = widgets.FloatSlider(value=150, min=0, max=200, step=0.1, description='resp_res_mean:')
resp_res_sd_slider = widgets.FloatSlider(value=17.35, min=0, max=100, step=0.1, description='resp_res_sd:')
sp_shape_slider = widgets.FloatSlider(value=2.3, min=0, max=10, step=0.1, description='sp_shape:')
sigma_slider = widgets.FloatSlider(value=4.0, min=0, max=10, step=0.1, description='sigma:')
sp_bias_slider = widgets.FloatSlider(value=0.0, min=-10, max=10, step=0.1, description='sp_bias:')
dr_shape_slider = widgets.FloatSlider(value=3.0, min=0, max=10, step=0.1, description='dr_shape:')


# Define the user interface with all the sliders
ui = widgets.VBox([
    sens_amp_slider, sens_tau_slider, sens_drc_slider, sens_bnds_slider, sens_aa_shape_slider, 
    sens_res_mean_slider, sens_res_sd_slider, resp_amp_slider, resp_tau_slider, resp_drc_slider, 
    resp_bnds_slider, resp_aa_shape_slider, resp_res_mean_slider, resp_res_sd_slider, sp_shape_slider, 
    sigma_slider, sp_bias_slider, dr_shape_slider
])


# Link the sliders to the widget function
out = widgets.interactive_output(widget_function, {
    'sens_amp': sens_amp_slider, 
    'sens_tau': sens_tau_slider, 
    'sens_drc': sens_drc_slider,
    'sens_bnds': sens_bnds_slider,
    'sens_aa_shape': sens_aa_shape_slider,
    'sens_res_mean': sens_res_mean_slider,
    'sens_res_sd': sens_res_sd_slider,
    'resp_amp': resp_amp_slider,
    'resp_tau': resp_tau_slider,
    'resp_drc': resp_drc_slider,
    'resp_bnds': resp_bnds_slider,
    'resp_aa_shape': resp_aa_shape_slider,
    'resp_res_mean': resp_res_mean_slider,
    'resp_res_sd': resp_res_sd_slider,
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
    'sens_res_mean': sens_res_mean_slider.value,
    'sens_res_sd': sens_res_sd_slider.value,
    'resp_amp': resp_amp_slider.value,
    'resp_tau': resp_tau_slider.value,
    'resp_drc': resp_drc_slider.value,
    'resp_bnds': resp_bnds_slider.value,
    'resp_aa_shape': resp_aa_shape_slider.value,
    'resp_res_mean': resp_res_mean_slider.value,
    'resp_res_sd': resp_res_sd_slider.value,
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
fit = Fit(res_ob, n_trls=10000, start_vals=prmsfit, n_caf=9, search_grid=True)

# %%
fit.fit_data(maxiter=2, disp=True)


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

para_dict = fit_nelder.best_prms_out.__dict__
function_parameters = inspect.signature(widget_function).parameters
filtered_args = {k: para_dict[k] for k in function_parameters if k in para_dict}

widget_function(**filtered_args)


# %%
Plot(res_ob).caf()
Plot(res_ob).delta()

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