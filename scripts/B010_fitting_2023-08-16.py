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



# ----------------- #


# %%
print(f'Best Parameters: {fit_diff.best_prms_out}')
print(f'Best cost: {fit_diff.best_cost}')

prmsfit_adv = set_best_parameters(fit_diff)

# %%
fit_vals_x = fit_diff.fit['x']
fit_vals_x


# ----------------- #

# %%
fit_diff_adv = Fit(res_ob, n_trls=10000, start_vals=prmsfit_adv, n_caf=9, search_grid=False) 
fit_diff_adv.start_vals

# %%
fit_diff_adv.fit_data('differential_evolution', x0=fit_vals_x, mutation=(0.1,0.4), maxiter=10, disp=True, seed=seed)


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
fit_diff_adv = Fit(res_ob, n_trls=10000, start_vals=prmsfit_adv, n_caf=9, search_grid=False) 
fit_diff_adv.start_vals

# %%
fit_diff_adv.fit_data('differential_evolution', x0=fit_vals_x, mutation=(0.1,0.4), maxiter=60, disp=True, seed=seed)



# sens_amp:39.3 sens_tau:143.1 sens_drc:0.70 sens_bnds:74.4 sens_aa_shape: 2.0 sens_res_mean: 106 sens_res_sd:26.2 resp_amp:20.1 resp_tau:168.6 resp_drc:0.31 resp_bnds:117.3 resp_aa_shape: 1.0 resp_res_mean: 100 resp_res_sd:24.0 sp_shape: 3.0 sigma: 4.0 sp_bias: 0.0 dr_shape: 3.0 | cost=256.25
# sens_amp:37.2 sens_tau:136.9 sens_drc:0.71 sens_bnds:68.6 sens_aa_shape: 2.1 sens_res_mean:  88 sens_res_sd:28.2 resp_amp:18.6 resp_tau:47.8 resp_drc:0.30 resp_bnds:117.4 resp_aa_shape: 1.0 resp_res_mean: 121 resp_res_sd:24.2 sp_shape: 3.1 sigma: 4.0 sp_bias: 0.0 dr_shape: 3.0 | cost=245.96









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
plot_activation(tim, activation(tim, 37.2, 136.9, 2.1, 10), 'Sensory Activation')
plot_activation(tim, activation(tim, 18.6, 47.8, 1.0, -10), 'Response Activation')


# %%
# ----------------- #
para = fit_diff_adv.best_prms_out


sim = Sim(Prms())



# ----------------- #

# %%
df.insert(len(df.columns), 'Source', 'Og')

#%%
df_fit = sim2data(fit_diff_adv.res_th)

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



# %%
fit_diff.best_prms_out

# %%
dat = pydmc.Sim(fit_diff.best_prms_out, n_caf=9, full_data=True)

# %%
dat.plot.summary() 


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

    sim = Sim(para, n_trls=10000, n_caf=9)
    df_fit = sim2data(sim)
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

# Create widgets for each parameter in Prms
sens_amp_slider = widgets.FloatSlider(value=30, min=0, max=100, step=0.1, description='sens_amp:')
sens_tau_slider = widgets.FloatSlider(value=140, min=0, max=200, step=0.1, description='sens_tau:')
sens_drc_slider = widgets.FloatSlider(value=0.4, min=0, max=1, step=0.01, description='sens_drc:')
sens_bnds_slider = widgets.FloatSlider(value=70, min=0, max=200, step=0.1, description='sens_bnds:')
sens_aa_shape_slider = widgets.FloatSlider(value=2.1, min=0, max=10, step=0.1, description='sens_aa_shape:')
sens_res_mean_slider = widgets.FloatSlider(value=50, min=0, max=200, step=0.1, description='sens_res_mean:')
sens_res_sd_slider = widgets.FloatSlider(value=28, min=0, max=100, step=0.1, description='sens_res_sd:')
resp_amp_slider = widgets.FloatSlider(value=30, min=0, max=100, step=0.1, description='resp_amp:')
resp_tau_slider = widgets.FloatSlider(value=100, min=0, max=200, step=0.1, description='resp_tau:')
resp_drc_slider = widgets.FloatSlider(value=0.40, min=0, max=1, step=0.01, description='resp_drc:')
resp_bnds_slider = widgets.FloatSlider(value=110, min=0, max=200, step=0.1, description='resp_bnds:')
resp_aa_shape_slider = widgets.FloatSlider(value=3, min=0, max=10, step=0.1, description='resp_aa_shape:')
resp_res_mean_slider = widgets.FloatSlider(value=50, min=0, max=200, step=0.1, description='resp_res_mean:')
resp_res_sd_slider = widgets.FloatSlider(value=24, min=0, max=100, step=0.1, description='resp_res_sd:')
sp_shape_slider = widgets.FloatSlider(value=3.1, min=0, max=10, step=0.1, description='sp_shape:')
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
fit = Fit(res_ob, n_trls=10000, start_vals=prmsfit, n_caf=9, search_grid=False)

# %%
fit.fit_data(maxiter=500, disp=True, seed=seed)


# %%
para = fit.best_prms_out
fit.best_cost

# %%
Parameters = Prms(**parameters)
sim = Sim(Parameters, n_caf=9)
Fit.calculate_cost_value_rmse(sim, res_ob)