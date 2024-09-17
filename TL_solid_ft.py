from scr.SPMe import Solid_Phase, Cell
from scr.utils.pinn import FFNN, DeepONet, NN_TL_Diffusion
from scr.utils.sampling import Sampler, Sampler_DONet
from scr.utils.parameters import load_params
from scr.utils.profiles import zheng_current, constant

import pybamm
import torch
import numpy as np
import time

import matplotlib.pyplot as plt

np.set_printoptions(precision=3)


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))


if __name__ == "__main__":

    dr = 0.5

    parameters = load_params()

    training_points = {"PDE": {"type": "PDE", "N": 1000},
                       "IV": {"type": "IV", "N": 100},
                       "BC_Center": {"type": "BC", "N": 100, "BC_pos": 0.},
                       "BC_Surf": {"type": "BC", "N": 100, "BC_pos": 1.}}
    validation_points = {"PDE": {"type": "PDE", "N": 30},
                         "IV": {"type": "IV", "N": 15},
                         "BC_Center": {"type": "BC", "N": 15, "BC_pos": 0.},
                         "BC_Surf": {"type": "BC", "N": 15, "BC_pos": 1.}}

    pos_weights = {"PDE": 1e4, "IV": 10., "BC_Center": 1., "BC_Surf": 1e4}
    neg_weights = {"PDE": 1e4, "IV": 10., "BC_Center": 1., "BC_Surf": 1e4}

    model_p = NN_TL_Diffusion(general_layers=[64, 64, 64, 64], fine_layers=[32, 32], dim_in=3, dim_int=32,
                              dim_out=1, dropout=0.)
    pos_model = Solid_Phase(model_p, parameters, criterion=RMSELoss, electrode="pos", weights=pos_weights)

    model_n = NN_TL_Diffusion(general_layers=[64, 64, 64, 64], fine_layers=[32, 32], dim_in=3, dim_int=32,
                              dim_out=1, dropout=0.)
    neg_model = Solid_Phase(model_n, parameters, criterion=RMSELoss, electrode="neg", weights=neg_weights)

    PINN = Cell(pos_model, neg_model)

    PINN.pos_model.load_model("./models/TL_pos.pt")
    PINN.neg_model.load_model("./models/TL_neg.pt")

    PINN.neg_model.model.freeze_general_model()

    optimizer_n = torch.optim.Adam(PINN.neg_model.parameters(), lr=0.0005)

    PINN.neg_model.params["as_n"] = torch.nn.Parameter(data=torch.tensor(parameters["as_n"]))
    optimizer_param = torch.optim.Adam([PINN.neg_model.params["as_n"]], lr=0.001*parameters["as_n"])
    # as_n = torch.nn.Parameter(data=torch.tensor(parameters["as_n"]))
    # optimizer_param = torch.optim.Adam([as_n], lr=1e10)

    Sampler = Sampler(training_points, constant(1.), mode="uniform")

    # PBM
    param = pybamm.ParameterValues("Chen2020")
    param["Negative electrode active material volume fraction"] *= dr
    print("Target: ", PINN.neg_model.params["as_n"].detach().numpy() * dr * parameters["R_n"] / 3.)

    PBM_model = pybamm.lithium_ion.SPM()

    experiment = pybamm.Experiment(["Discharge at 1C for 100000 seconds or until 2.5 V"])
    sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
    solution = sim.solve(initial_soc=1)

    t = solution["Time [s]"].entries
    V = solution["Terminal voltage [V]"].entries

    cur_fun = constant(1.)

    pinn_sample_t_0 = torch.linspace(0, 1, 100)
    pinn_sample_r = torch.ones_like(pinn_sample_t_0)
    pinn_sample_I = torch.tensor(cur_fun(pinn_sample_t_0.numpy() * 3600.))
    pinn_sample = torch.stack([pinn_sample_t_0, pinn_sample_r, pinn_sample_I]).t()

    V_pinn_0 = PINN.compute_V(pinn_sample)

    eps_n = []

    start = time.time()
    for j in range(1000 + 1):

        optimizer_n.zero_grad()
        optimizer_param.zero_grad()

        t_points = np.sort(np.random.random(size=(1000, 1)) * t[-1], axis=0)
        V_pbm = torch.asarray(np.interp(t_points, t, V), dtype=torch.float32, requires_grad=True).flatten()

        Sampler.update_samples(training_points)

        loss, _ = PINN.neg_model.compute_loss(Sampler)

        pinn_sample_t = torch.asarray(t_points/3600., dtype=torch.float32).flatten()
        pinn_sample_r = torch.ones_like(pinn_sample_t)
        pinn_sample_I = torch.tensor(cur_fun(pinn_sample_t.numpy() * 3600.))
        pinn_sample = torch.stack([pinn_sample_t, pinn_sample_r, pinn_sample_I]).t()

        V_pinn = PINN.compute_V(pinn_sample)

        V_error = RMSELoss(V_pbm, V_pinn)
        loss += V_error
        loss.backward()

        optimizer_n.step()

        optimizer_param.step()

        if j % 100 == 0:
            print(j, "\t\t", V_error.detach().numpy(), "\t\t", loss.detach().numpy(), "\t\t",
                  PINN.neg_model.params["as_n"].detach().numpy() * parameters["R_n"] / 3.)

        eps_n.append(PINN.neg_model.params["as_n"].detach().numpy() * parameters["R_n"] / 3.)

    print(time.time() - start)

    plt.figure()
    plt.grid()
    plt.plot(range(1000 + 1), eps_n, "k-", label="eps_n")
    plt.plot(0, parameters["eps_n"], 'kD', label="Initial")
    plt.axhline(y=parameters["eps_n"] * dr, color='r', linestyle='--', label="Target")
    plt.legend()
    plt.ylim([0.3, 0.8])
    plt.xlabel("Iteration")
    plt.ylabel("eps_n")
    plt.show()

    plt.figure()
    plt.grid()
    plt.plot(t_points / 3600., V_pbm.detach().numpy(), "r-", label="PyBaMM")
    plt.plot(pinn_sample_t_0.detach().numpy(), V_pinn_0.detach().numpy(), "k--", label="PINN_0")
    plt.plot(pinn_sample_t.detach().numpy(), V_pinn.detach().numpy(), "k-", label="PINN")
    plt.legend()
    plt.xlabel("t [h]")
    plt.ylabel("V [V]")
    plt.show()