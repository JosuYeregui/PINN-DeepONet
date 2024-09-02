from scr.SPMe import Solid_Phase
from scr.dependencies.pinn import FFNN
from scr.dependencies.utils import load_params

import ipywidgets as widgets
import matplotlib.pyplot as plt

import pybamm
import torch


def plot_concentrations(t):

    bcs_sample_r = torch.linspace(0., 1., 1000)
    bcs_sample_t = torch.ones_like(bcs_sample_r)
    bcs_sample = torch.stack([bcs_sample_t, bcs_sample_r]).t()

    c_pinn_p = PINN_pos(bcs_sample)
    c_pinn_n = PINN_neg(bcs_sample)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    (plot_c_n,) = ax1.plot(
        r_n, c_s_n(r=r_n, t=t * 3600, x=x[0])
    )  # can evaluate at arbitrary x (single representative particle)
    (plot_c_p,) = ax2.plot(
        r_p, c_s_p(r=r_p, t=t * 3600, x=x[-1])
    )  # can evaluate at arbitrary x (single representative particle)

    (plot_c_n,) = ax1.plot(
        bcs_sample_r.detach().numpy(), c_pinn_p
    )  # can evaluate at arbitrary x (single representative particle)
    (plot_c_p,) = ax2.plot(
        bcs_sample_r.detach().numpy(), c_pinn_n
    )  # can evaluate at arbitrary x (single representative particle)
    ax1.set_ylabel("Negative particle concentration")
    ax2.set_ylabel("Positive particle concentration")
    ax1.set_xlabel(r"$r_n$ [m]")
    ax2.set_xlabel(r"$r_p$ [m]")
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    plt.show()


if __name__ == "__main__":

    parameters = load_params()

    model = FFNN(2, 1)
    PINN_pos = Solid_Phase(model, parameters, electrode="pos")
    PINN_pos.load_model("models/positive.pt")
    PINN_neg = Solid_Phase(model, parameters, electrode="neg")
    PINN_neg.load_model("models/negative.pt")

    param = pybamm.ParameterValues("Chen2020")
    PBM_model = pybamm.lithium_ion.SPM()
    experiment = pybamm.Experiment(["Discharge at 1C for 10000 seconds or until 2.5 V"])
    sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
    solution = sim.solve(initial_soc=1)

    x = solution["x [m]"].entries[:, 0]

    c_s_n = solution["Negative particle concentration"]
    c_s_p = solution["Positive particle concentration"]
    r_n = solution["r_n [m]"].entries[:, 0, 0]
    r_p = solution["r_p [m]"].entries[:, 0, 0]

    widgets.interact(
        plot_concentrations, t=widgets.FloatSlider(min=0, max=1, step=10, value=0)
    )

