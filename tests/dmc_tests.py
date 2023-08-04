#%%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pydmc
from pydmc import Ob, Fit, Sim, PlotFit, Plot
#%%%
%matplotlib widget

# %%
Data = pydmc.flanker_data()
Data.head()

# %%
res_ob = Ob(Data, n_caf=9)

# %%
fit = Fit(res_ob, n_caf=9)


# %% fits the data 
fit.fit_data()



# %%
# fit_diff = Fit(res_ob, n_caf=9)
# fit_diff.fit_data('differential_evolution', maxiter=20)




# %%
fit.table_summary()


# %%
# fit_diff.table_summary()



# %%
fit_plot = PlotFit(fit).summary()

# %%
sim = Sim(fit.dmc_prms, n_caf=9, full_data=True, n_trls_data=100)
# %%

# %%
Plot(sim).summary()
# %%
Fit.calculate_cost_value_rmse(sim, res_ob)