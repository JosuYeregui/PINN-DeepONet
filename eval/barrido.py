from scr.SPMe import Solid_Phase, Cell
from scr.utils.pinn import FFNN, DeepONet, NN_TL_Diffusion, DeepONet_TL
from scr.utils.sampling import Sampler, Sampler_DONet
from scr.utils.parameters import load_params
from scr.utils.profiles import zheng_current, constant
from scr.utils.optimizers import NTK_Adaptive, Adam_Custom

import pybamm
import torch
import numpy as np
import pickle as pkl
import os
import copy

from time import time

import matplotlib.pyplot as plt

# np.set_printoptions(precision=3)
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print("Device: ", device, "\n")
# print(torch.get_num_threads())
torch.set_num_threads(1)

def iteration(dr_p, dr_n, PINN):
    iterations = 2000

    crate = -1.
    cur_fun = constant(crate)

    # Run PyBaMM simulation
    param = pybamm.ParameterValues("Chen2020")
    param["Positive electrode active material volume fraction"] *= dr_p 
    param["Negative electrode active material volume fraction"] *= dr_n

    PBM_model = pybamm.lithium_ion.SPM()

    experiment = pybamm.Experiment(["Charge at 1C for 100000 seconds or until 4.2 V"])
    sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
    solution = sim.solve(initial_soc=0.)

    t = solution["Time [s]"].entries
    V = solution["Terminal voltage [V]"].entries

    # PybaMM concentrations
    c_s_n = solution["Negative particle concentration"]
    c_s_p = solution["Positive particle concentration"]
    r_n = solution["r_n [m]"].entries[:, 0, 0]
    r_p = solution["r_p [m]"].entries[:, 0, 0]
    t_pbm = solution["Time [s]"].entries
    x = solution["x [m]"].entries[:, 0]

    c_p_pbm = c_s_p(r=r_p[-1], t=t, x=x[-1])
    c_n_pbm = c_s_n(r=r_n[-1], t=t, x=x[0])

    training_points = {"PDE": {"type": "PDE", "N": 1000},
                       "IV": {"type": "IV", "N": 100},
                       "BC_Center": {"type": "BC", "N": 100, "BC_pos": 0.},
                       "BC_Surf": {"type": "BC", "N": 100, "BC_pos": 1.}}
    validation_points = {"PDE": {"type": "PDE", "N": 30},
                         "IV": {"type": "IV", "N": 15},
                         "BC_Center": {"type": "BC", "N": 15, "BC_pos": 0.},
                         "BC_Surf": {"type": "BC", "N": 15, "BC_pos": 1.}}

    optimizer_n = torch.optim.Adam(PINN.neg_model.model.parameters(), lr=0.0001)

    PINN.neg_model.params["as_n"] = torch.nn.Parameter(data=torch.tensor(parameters["as_n"]))
    optimizer_param_n = torch.optim.Adam([PINN.neg_model.params["as_n"]], lr=0.001 * parameters["as_n"])

    optimizer_p = torch.optim.Adam(PINN.pos_model.model.parameters(), lr=0.0001)

    PINN.pos_model.params["as_p"] = torch.nn.Parameter(data=torch.tensor(parameters["as_p"]))
    optimizer_param_p = torch.optim.Adam([PINN.pos_model.params["as_p"]], lr=0.001 * parameters["as_p"])

    PINN.pos_model.params["D_p"] = torch.nn.Parameter(data=torch.tensor(parameters["D_p"]))
    optimizer_param_Dp = torch.optim.Adam([PINN.pos_model.params["D_p"]], lr=20. * parameters["D_p"])
    PINN.neg_model.params["D_n"] = torch.nn.Parameter(data=torch.tensor(parameters["D_n"]))
    optimizer_param_Dn = torch.optim.Adam([PINN.neg_model.params["D_n"]], lr=20. * parameters["D_n"])

    parameters["SOL_p"][0] = c_p_pbm[0]
    parameters["SOL_n"][0] = c_n_pbm[0]

    # as_n = torch.nn.Parameter(data=torch.tensor(parameters["as_n"]))
    # optimizer_param = torch.optim.Adam([as_n], lr=1e10)

    # Sampler = Sampler(training_points, constant(1.), mode="uniform")
    Sampler = Sampler_DONet(training_points, constant(crate), branch_samp=360, mode="quasi", device=device)

    eps_prev = (PINN.pos_model.params["as_p"].detach().numpy() * parameters["R_p"] / 3. +
                PINN.neg_model.params["as_n"].detach().numpy() * parameters["R_n"] / 3.)

    for j in range(iterations + 1):

        PINN.neg_model.params["SOC_0"] = 0.
        PINN.pos_model.params["SOC_0"] = 0.

        t_points = np.sort(np.random.random(size=(1000, 1)) * t[-1], axis=0)
        V_pbm = torch.asarray(np.interp(t_points, t, V), dtype=torch.float32, requires_grad=True).flatten()

        Sampler.update_samples(training_points, constant(crate), 1. / np.abs(crate))

        start = time()

        loss_n = PINN.neg_model.compute_loss(Sampler)
        loss_p = PINN.pos_model.compute_loss(Sampler)

        loss_tot_n = torch.sum(loss_n * torch.tensor(PINN.neg_model.weights))
        loss_tot_p = torch.sum(loss_p * torch.tensor(PINN.pos_model.weights))

        pinn_sample_t = torch.asarray(t_points / 3600., dtype=torch.float32).flatten()
        pinn_sample_r = torch.ones_like(pinn_sample_t)
        pinn_sample_I = torch.tensor(cur_fun(pinn_sample_t.numpy() * 3600.))
        pinn_sample = torch.stack([pinn_sample_t, pinn_sample_r, pinn_sample_I]).t()

        N = torch.tensor(cur_fun(np.arange(0., 3600, 10)))

        V_pinn = PINN.compute_V((pinn_sample, N))

        V_error = RMSELoss(V_pbm, V_pinn)

        # loss_param = losses[3] + V_error

        optimizer_n.zero_grad()
        optimizer_p.zero_grad()
        optimizer_param_n.zero_grad()
        optimizer_param_p.zero_grad()

        loss = loss_tot_n + loss_tot_p + V_error
        loss.backward()

        optimizer_n.step()
        optimizer_p.step()
        optimizer_param_n.step()
        optimizer_param_p.step()

        eps_now = (PINN.pos_model.params["as_p"].detach().numpy() * parameters["R_p"] / 3. +
                   PINN.neg_model.params["as_n"].detach().numpy() * parameters["R_n"] / 3.)

        # if np.abs(eps_now - eps_prev) < 1e-7:
        #    break

        eps_prev = eps_now.copy()

    print("Case: dr_p = ", dr_p, ", dr_n = ", dr_n)
    print("\tConvergence at iteration: ", j, " with error: ", loss.item())
    print("\tEps_n: ", PINN.neg_model.params["as_n"].detach().numpy() * parameters["R_n"] / 3.,
          ", Eps_p: ", PINN.pos_model.params["as_p"].detach().numpy() * parameters["R_p"] / 3.)

    rel_error_p = (np.abs(param["Positive electrode active material volume fraction"] -
                         (PINN.pos_model.params["as_p"].detach().numpy() * parameters["R_p"] / 3.))
                   / param["Positive electrode active material volume fraction"] * 100.)
    rel_error_n = (np.abs(param["Negative electrode active material volume fraction"] -
                            (PINN.neg_model.params["as_n"].detach().numpy() * parameters["R_n"] / 3.))
                     / param["Negative electrode active material volume fraction"] * 100.)

    print("\tRelative error in eps_p: ", rel_error_p, ", eps_n: ", rel_error_n)

    return (PINN.neg_model.params["as_n"].detach().numpy() * parameters["R_n"] / 3.,
            PINN.pos_model.params["as_p"].detach().numpy() * parameters["R_p"] / 3., j)

def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))

if __name__ == "__main__":

    dr_p = np.linspace(0.8, 1., 20)
    dr_n = np.linspace(0.8, 1., 20)

    parameters = load_params()

    file = "2025-03-07_16-00"
    iter = ""
    if iter != "":
        fold = os.path.join("../models", file, iter)
    else:
        fold = os.path.join("../models", file)

    model_p = DeepONet_TL(branch_layers=[64, 64, 64, 64], trunk_layers=[64, 64, 64, 64], fine_layers=[32, 32],
                          dim_branch=360, dim_trunk=3, dim_int=128, dim_out=1, dropout=0.).to(device)
    pos_model = Solid_Phase(model_p, parameters, [1., 1., 1.], criterion=RMSELoss, electrode="pos").to(device)

    model_n = DeepONet_TL(branch_layers=[64, 64, 64, 64], trunk_layers=[64, 64, 64, 64], fine_layers=[32, 32],
                          dim_branch=360, dim_trunk=3, dim_int=128, dim_out=1, dropout=0.).to(device)
    neg_model = Solid_Phase(model_n, parameters, [1., 1., 1.], criterion=RMSELoss, electrode="neg").to(device)

    PINN_m = Cell(pos_model, neg_model)

    PINN_m.pos_model.load_model(os.path.join(fold, "TL_pos.pt"))
    PINN_m.neg_model.load_model(os.path.join(fold, "TL_neg.pt"))

    with open(os.path.join(fold, "Weights.pkl"), "rb") as input_file:
        weights = pkl.load(input_file)
    PINN_m.pos_model.weights = weights["P"]
    PINN_m.neg_model.weights = weights["N"]

    PINN_m.pos_model.model.freeze_general_model()
    PINN_m.neg_model.model.freeze_general_model()

    data = {"dr_p": [], "dr_n": [], "eps_n": [], "eps_p": [], "iterations": []}

    for dp in dr_p:
        for dn in dr_n:
            try:
                eps_n, eps_p, it = iteration(dp, dn, copy.deepcopy(PINN_m))
                data["dr_p"].append(dp)
                data["dr_n"].append(dn)
                data["eps_n"].append(eps_n)
                data["eps_p"].append(eps_p)
                data["iterations"].append(it)
            except:
                print("Error in case: dr_p = ", dp, ", dr_n = ", dn)
                data["dr_p"].append(dp)
                data["dr_n"].append(dn)
                data["eps_n"].append(0)
                data["eps_p"].append(0)
                data["iterations"].append(0)

    with open("data_full.pkl", "wb") as output_file:
        pkl.dump(data, output_file)
