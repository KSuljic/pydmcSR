#%%
import pydmc

dmc_sim = pydmc.Sim()
dmc_sim.plot.summary()      # Fig 2
dmc_sim = pydmc.Sim(full_data=True)
dmc_sim.plot.summary()      # Fig 3
dmc_sim = pydmc.Sim(pydmc.Prms(tau = 150))
dmc_sim.plot.summary()      # Fig 4
dmc_sim = pydmc.Sim(pydmc.Prms(tau = 90))
dmc_sim.plot.summary()      # Fig 5
dmc_sim = pydmc.Sim(pydmc.Prms(sp_dist = 1))
dmc_sim.plot.summary()      # Fig 6
dmc_sim = pydmc.Sim(pydmc.Prms(dr_dist = 1))
dmc_sim.plot.summary()      # Fig 7