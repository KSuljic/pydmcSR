#%%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
fit = Fit(res_ob, n_caf=9)

# %% fits the data 
fit.fit_data(maxiter=1500)



# %%
fit.table_summary()

# %%
fit_plot = PlotFit(fit).summary()



# %%
result_para = fit.res_th.prms
sim = Sim(result_para, n_caf=9, full_data=True, n_trls_data=150)
# %%

# %%
Plot(sim).summary()
# %%
Fit.calculate_cost_value_rmse(sim, res_ob)


# %%
