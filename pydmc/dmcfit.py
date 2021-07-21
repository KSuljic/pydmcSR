import numpy as np
from dataclasses import dataclass, fields
from scipy.optimize import fmin
from pydmc.dmcsim import DmcSim, DmcParameters
from pydmc.dmcplot import DmcPlotFit


@dataclass
class DmcParameterBounds:
    amp: tuple = (0, 40)
    tau: tuple = (5, 300)
    drc: tuple = (0.1, 1.0)
    bnds: tuple = (20, 150)
    res_mean: tuple = (200, 800)
    res_sd: tuple = (5, 100)
    aa_shape: tuple = (1, 3)
    sp_shape: tuple = (2, 4)
    sigma: tuple = (4, 4)


class DmcFit:
    def __init__(
        self,
        res_ob,
        n_trls=100000,
        start_vals=DmcParameters(),
        bound_vals=DmcParameterBounds(),
        n_delta=19,
        p_delta=None,
        t_delta=1,
        n_caf=5,
        var_sp=True,
    ):
        """
        Parameters
        ----------
        res_ob
        n_trls
        start_vals
        bound_vals
        n_delta
        p_delta
        t_delta
        n_caf
        var_sp
        """
        self.res_ob = res_ob
        self.res_th = DmcSim
        self.n_trls = n_trls
        self.start_vals = start_vals
        self.bound_vals = bound_vals
        self._min_vals = self._fieldvalues(0)
        self._max_vals = self._fieldvalues(1)
        self.n_delta = n_delta
        self.p_delta = p_delta
        self.t_delta = t_delta
        self.n_caf = n_caf
        self.var_sp = var_sp
        self.cost_value = np.Inf

    def _fieldvalues(self, idx):
        return [getattr(self.bound_vals, f.name)[idx] for f in fields(self.bound_vals)]

    def fit_data(self, **kwargs):
        self.start_vals.var_sp = True
        self.res_th = DmcSim(self.start_vals)
        self.fit = fmin(
            self._function_to_minimise,
            [getattr(self.start_vals, f.name) for f in fields(self.start_vals)][:9],
            **kwargs,
        )

    def summary(self):
        """Print summary of DmcFit."""
        print(
            f"amp:{self.res_th.prms.amp:4.1f}",
            f"tau:{self.res_th.prms.tau:4.1f}",
            f"drc:{self.res_th.prms.drc:4.2f}",
            f"bnds:{self.res_th.prms.bnds:4.1f}",
            f"res_mean:{self.res_th.prms.res_mean:4.0f}",
            f"res_sd:{self.res_th.prms.res_sd:4.1f}",
            f"aa_shape:{self.res_th.prms.aa_shape:4.1f}",
            f"sp_shape:{self.res_th.prms.sp_shape:4.1f}",
            f"| cost={self.cost_value:.2f}",
        )

    def _function_to_minimise(self, x):

        # bounds hack
        x = np.maximum(x, self._min_vals)
        x = np.minimum(x, self._max_vals)

        self.res_th.prms.amp = x[0]
        self.res_th.prms.tau = x[1]
        self.res_th.prms.drc = x[2]
        self.res_th.prms.bnds = x[3]
        self.res_th.prms.res_mean = x[4]
        self.res_th.prms.res_sd = x[5]
        self.res_th.prms.aa_shape = x[6]
        self.res_th.prms.sp_shape = x[7]
        self.res_th.prms.sigma = x[8]
        self.res_th.prms.sp_lim = (-x[3], x[3])

        self.res_th.run_simulation()
        self.cost_value = DmcFit.calculate_cost_value_rmse(self.res_th, self.res_ob)
        self.summary()

        return self.cost_value

    @staticmethod
    def calculate_cost_value_rmse(res_th, res_ob):
        """calculate_cost_value_rmse

        Parameters
        ----------
        res_th
        res_ob
        """
        n_rt = len(res_th.delta) * 2
        n_err = len(res_th.caf)

        cost_caf = np.sqrt(
            (1 / n_err) * np.sum((res_th.caf["Error"] - res_ob.caf["Error"]) ** 2)
        )

        cost_rt = np.sqrt(
            (1 / n_rt)
            * np.sum(
                np.sum(
                    res_th.delta[["mean_comp", "mean_incomp"]]
                    - res_ob.delta[["mean_comp", "mean_incomp"]]
                )
            )
            ** 2
        )

        weight_rt = n_rt / (n_rt + n_err)
        weight_caf = (1 - weight_rt) * 1500

        cost_value = (weight_caf * cost_caf) + (weight_rt * cost_rt)

        return cost_value

    def plot(self, **kwargs):
        """Plot."""
        DmcPlotFit(self.res_th, self.res_ob).plot(**kwargs)

    def plot_rt_correct(self, **kwargs):
        """Plot reaction time correct."""
        DmcPlotFit(self.res_th, self.res_ob).plot_rt_correct(**kwargs)

    def plot_er(self, **kwargs):
        """Plot erorr rate."""
        DmcPlotFit(self.res_th, self.res_ob).plot_er(**kwargs)

    def plot_rt_error(self, **kwargs):
        """Plot reaction time errors."""
        DmcPlotFit(self.res_th, self.res_ob).plot_rt_error(**kwargs)

    def plot_cdf(self, **kwargs):
        """Plot CDF."""
        DmcPlotFit(self.res_th, self.res_ob).plot_cdf(**kwargs)

    def plot_caf(self, **kwargs):
        """Plot CAF."""
        DmcPlotFit(self.res_th, self.res_ob).plot_caf(**kwargs)

    def plot_delta(self, **kwargs):
        """Plot delta."""
        DmcPlotFit(self.res_th, self.res_ob).plot_delta(**kwargs)
