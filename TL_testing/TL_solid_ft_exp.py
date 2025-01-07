from scr.SPMe import Solid_Phase, Cell
from scr.utils.pinn import FFNN, DeepONet, NN_TL_Diffusion, DeepONet_TL
from scr.utils.sampling import Sampler, Sampler_DONet
from scr.utils.parameters import load_params
from scr.utils.profiles import zheng_current, constant
from scr.utils.optimizers import NTK_Adaptive, Adam_Custom

import pybamm
import torch
import numpy as np
import time
import pickle as pkl
import os

import matplotlib.pyplot as plt

np.set_printoptions(precision=3)
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print("Device: ", device, "\n")
# print(torch.get_num_threads())
torch.set_num_threads(1)


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))

# Load excel and return voltage and time from sheet Channel-6_1
def load_excel(file, channel):
    import pandas as pd
    data = pd.read_excel(file, sheet_name="Channel-"+channel+"_1")
    # Filter the data to take only the ones with Step Index = 6 (CHG) or 10 (DCH)
    data = data[data["Step Index"] == 6]
    return data["Voltage (V)"].values, data["Step Time (s)"].values
    # return data["Voltage (V)"].values[:-1000], data["Step Time (s)"].values[:-1000]


if __name__ == "__main__":

    iterations = 5000

    crate = -0.3
    SoC_0 = 0.

    dt_p = 1.25
    dt_n = 0.75

    # file = "2024-12-09_13-53"
    file = "CC_dtp1.25_dtn0.75_2024-12-10_17-04"
    iter = ""
    if iter != "":
        fold = os.path.join("..\\models", file, iter)
    else:
        fold = os.path.join("..\\models", file)

    V, t = load_excel(os.path.join("..\\Data\\BoL_PINN_Channel_7_Wb_1.xlsx"), "7")

    # PybaMM concentrations


    parameters = load_params()

    # parameters["SOL_p"] = [0.8599, 0.2719]
    # parameters["SOL_n"] = [0.0339, 0.9742]
    parameters["D_p"] = 1.22679499e-15  # 1.64852539e-14
    parameters["D_n"] = 1.46457550e-15
    parameters["as_p"] *= 7.555631166180735 / 8.7323 * dt_p
    parameters["eps_p"] *= 7.555631166180735 / 8.7323 * dt_p
    parameters["as_n"] *= 6.014505825096271 / 5.8276 * dt_n
    parameters["eps_n"] *= 6.014505825096271 / 5.8276 * dt_n
    # Pybamm takes stechiometric coefficients differently, so we need to adjust the parameters
    eps_p_0 = parameters["eps_p"]
    eps_n_0 = parameters["eps_n"]
    # parameters["SOL_p"] = [0.8599, 0.2719]
    # parameters["SOL_n"] = [0.0339, 0.9742]
    # parameters["D_p"] = 5.57979526e-13# 1.64852539e-14
    # parameters["D_n"] = 1.91909180e-15

    training_points = {"PDE": {"type": "PDE", "N": 1000},
                       "IV": {"type": "IV", "N": 100},
                       "BC_Center": {"type": "BC", "N": 100, "BC_pos": 0.},
                       "BC_Surf": {"type": "BC", "N": 100, "BC_pos": 1.}}
    validation_points = {"PDE": {"type": "PDE", "N": 30},
                         "IV": {"type": "IV", "N": 15},
                         "BC_Center": {"type": "BC", "N": 15, "BC_pos": 0.},
                         "BC_Surf": {"type": "BC", "N": 15, "BC_pos": 1.}}

    model_p = DeepONet_TL(branch_layers=[64, 64, 64, 64], trunk_layers=[64, 64, 64, 64], fine_layers=[32, 32],
                          dim_branch=360, dim_trunk=3, dim_int=128, dim_out=1, dropout=0.).to(device)
    pos_model = Solid_Phase(model_p, parameters, [1., 1., 1.], criterion=RMSELoss, electrode="pos").to(device)

    model_n = DeepONet_TL(branch_layers=[64, 64, 64, 64], trunk_layers=[64, 64, 64, 64], fine_layers=[32, 32],
                          dim_branch=360, dim_trunk=3, dim_int=128, dim_out=1, dropout=0.).to(device)
    neg_model = Solid_Phase(model_n, parameters, [1., 1., 1.], criterion=RMSELoss, electrode="neg").to(device)

    PINN = Cell(pos_model, neg_model)

    PINN.pos_model.load_model(os.path.join(fold, "TL_pos.pt"))
    PINN.neg_model.load_model(os.path.join(fold, "TL_neg.pt"))

    with open(os.path.join(fold, "Weights.pkl"), "rb") as input_file:
        weights = pkl.load(input_file)
    PINN.pos_model.weights = weights["P"]
    PINN.neg_model.weights = weights["N"]

    PINN.pos_model.model.freeze_general_model()
    PINN.neg_model.model.freeze_general_model()

    optimizer_n = torch.optim.Adam(PINN.neg_model.model.parameters(), lr=0.00001)

    PINN.neg_model.params["as_n"] = torch.nn.Parameter(data=torch.tensor(parameters["as_n"]))
    optimizer_param_n = torch.optim.Adam([PINN.neg_model.params["as_n"]], lr=0.001 * parameters["as_n"])

    optimizer_p = torch.optim.Adam(PINN.pos_model.model.parameters(), lr=0.00001)

    PINN.pos_model.params["as_p"] = torch.nn.Parameter(data=torch.tensor(parameters["as_p"]))
    optimizer_param_p = torch.optim.Adam([PINN.pos_model.params["as_p"]], lr=0.001 * parameters["as_p"])

    PINN.pos_model.params["SOL_p"][0] = torch.nn.Parameter(data=torch.tensor(parameters["SOL_p"][0]))
    optimizer_param_p_S = torch.optim.Adam([PINN.pos_model.params["SOL_p"][0]], lr=0.0001 * parameters["SOL_p"][0])

    PINN.pos_model.params["D_p"] = torch.nn.Parameter(data=torch.tensor(parameters["D_p"]))
    optimizer_param_Dp = torch.optim.Adam([PINN.pos_model.params["D_p"]], lr=1. * parameters["D_p"])
    PINN.neg_model.params["D_n"] = torch.nn.Parameter(data=torch.tensor(parameters["D_n"]))
    optimizer_param_Dn = torch.optim.Adam([PINN.neg_model.params["D_n"]], lr=1. * parameters["D_n"])

    # as_n = torch.nn.Parameter(data=torch.tensor(parameters["as_n"]))
    # optimizer_param = torch.optim.Adam([as_n], lr=1e10)

    # Sampler = Sampler(training_points, constant(1.), mode="uniform")
    Sampler = Sampler_DONet(training_points, constant(crate), branch_samp=360, mode="quasi", device=device)

    cur_fun = constant(crate)

    tar_eps_p = PINN.neg_model.params["as_p"].detach().numpy() * parameters["R_p"] / 3. * (1 - 0.058756)
    tar_eps_n = PINN.pos_model.params["as_n"].detach().numpy() * parameters["R_n"] / 3. * (1 - 0.26054)

    tar_eps_p = parameters["eps_p"] / dt_p
    tar_eps_n = parameters["eps_n"] / dt_n

    print("Target P: ", tar_eps_p)
    print("Target N: ", tar_eps_n)

    PINN.neg_model.params["SOC_0"] = SoC_0
    PINN.pos_model.params["SOC_0"] = SoC_0

    pinn_sample_t_0 = torch.linspace(0, t[-1] / 3600., 100)
    pinn_sample_r = torch.ones_like(pinn_sample_t_0)
    pinn_sample_I = torch.tensor(cur_fun(pinn_sample_t_0.numpy() * 3600.))
    pinn_sample = torch.stack([pinn_sample_t_0, pinn_sample_r, pinn_sample_I]).t()

    N = torch.tensor(cur_fun(np.arange(0., 3600, 10)))

    V_pinn_0 = PINN.compute_V((pinn_sample, N))
    c_p_0 = PINN.pos_model((pinn_sample, N))
    c_n_0 = PINN.neg_model((pinn_sample, N))
    t_pinn_0 = pinn_sample_t_0.detach().numpy()

    # Plot V_pbm and V_pinn
    plt.figure()
    plt.grid()
    plt.plot(t_pinn_0, V_pinn_0.detach().numpy(), "k-", label="PINN")
    plt.plot(t/3600., V, "r-", label="Experimental")
    plt.legend()
    plt.xlabel("t [h]")
    plt.ylabel("V [V]")
    plt.show()

    eps_n = []
    eps_p = []
    D_n = []
    D_p = []

    start = time.time()
    for j in range(iterations + 1):

        PINN.neg_model.params["SOC_0"] = SoC_0
        PINN.pos_model.params["SOC_0"] = SoC_0

        t_points = np.sort(np.random.random(size=(1000, 1)) * t[-1], axis=0)
        V_pbm = torch.asarray(np.interp(t_points, t, V), dtype=torch.float32, requires_grad=True).flatten()
        # Check if there is a nan value in the V_pbm

        Sampler.update_samples(training_points, constant(crate), 1. / np.abs(crate))

        loss_n = PINN.neg_model.compute_loss(Sampler)
        loss_p = PINN.pos_model.compute_loss(Sampler)

        loss_tot_n = torch.sum(loss_n * torch.tensor(PINN.neg_model.weights))
        loss_tot_p = torch.sum(loss_p * torch.tensor(PINN.pos_model.weights))

        pinn_sample_t = torch.asarray(t_points/3600., dtype=torch.float32).flatten()
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
        optimizer_param_p_S.zero_grad()
        # optimizer_param_Dp.zero_grad()
        # optimizer_param_Dn.zero_grad()

        loss = loss_tot_n + loss_tot_p + V_error
        loss.backward()

        optimizer_n.step()
        optimizer_p.step()
        optimizer_param_n.step()
        optimizer_param_p.step()
        # optimizer_param_p_S.step()
        # optimizer_param_Dp.step()
        # optimizer_param_Dn.step()

        if j % 100 == 0:
            print(j, "\t\t", V_error.detach().numpy(), "\t\t", loss_tot_n.detach().numpy(), "\t\t", loss_tot_p.detach().numpy(), "\t\t",
                  PINN.pos_model.params["as_p"].detach().numpy() * parameters["R_p"] / 3.,
                  PINN.neg_model.params["as_n"].detach().numpy() * parameters["R_n"] / 3.,
                  PINN.pos_model.params["D_p"].detach().numpy(), PINN.neg_model.params["D_n"].detach().numpy(), "\t\t",
                  PINN.pos_model.params["SOL_p"][0].detach().numpy(), "\t\t")
                  # PINN.pos_model.params["as_p"].grad.detach().numpy(), PINN.neg_model.params["as_n"].grad.detach().numpy())
                  # torch.autograd.grad(V_error, PINN.pos_model.params["as_p"], retain_graph=True)[0].detach().numpy(),
                  # torch.autograd.grad(V_error, PINN.neg_model.params["as_n"], retain_graph=True)[0].detach().numpy())

            # Plot V_pbm and V_pinn
            # plt.figure()
            # plt.grid()
            # plt.plot(t_points / 3600., V_pbm.detach().numpy(), "r-", label="PyBaMM")
            # plt.plot(pinn_sample_t_0.detach().numpy(), V_pinn_0.detach().numpy(), "k--", label="PINN_0")
            # plt.plot(pinn_sample_t.detach().numpy(), V_pinn.detach().numpy(), "k-", label="PINN")
            # plt.legend()
            # plt.title("Iteration: " + str(j))
            # plt.xlabel("t [h]")
            # plt.ylabel("V [V]")
            # plt.show()

        eps_n.append(PINN.neg_model.params["as_n"].detach().numpy() * parameters["R_n"] / 3.)
        eps_p.append(PINN.pos_model.params["as_p"].detach().numpy() * parameters["R_p"] / 3.)
        D_n.append(PINN.neg_model.params["D_n"].detach().numpy() * 1.)
        D_p.append(PINN.pos_model.params["D_p"].detach().numpy() * 1.)

    print(time.time() - start)

    plt.figure()
    plt.grid()
    plt.plot(range(iterations + 1), eps_n, "r-", label="eps_n")
    plt.plot(0, eps_n_0, 'rD')
    plt.axhline(y=tar_eps_n, color='r', linestyle='--', label="eps_n target")
    plt.plot(range(iterations + 1), eps_p, "g-", label="eps_p")
    plt.plot(0, eps_p_0, 'gD')
    plt.axhline(y=tar_eps_p, color='g', linestyle='--', label="eps_p target")
    plt.legend()
    plt.ylim([0.3, 0.9])
    plt.xlabel("Iteration")
    plt.ylabel("eps")
    plt.show()

    plt.figure()
    plt.grid()
    plt.semilogy(range(iterations + 1), D_n, "m-", label="D_n")
    plt.semilogy(range(iterations + 1), D_p, "c-", label="D_p")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("D [m2.s-1]")
    plt.show()


    plt.figure()
    plt.grid()
    plt.plot(t_points / 3600., V_pbm.detach().numpy(), "r-", label="PyBaMM")
    plt.plot(t_pinn_0, V_pinn_0.detach().numpy(), "k--", label="PINN_0")
    plt.plot(pinn_sample_t.detach().numpy(), V_pinn.detach().numpy(), "k-", label="PINN")
    plt.legend()
    plt.xlabel("t [h]")
    plt.ylabel("V [V]")
    plt.show()

    # Plot concentrations before and after training
    c_p = PINN.pos_model((pinn_sample, N))
    c_n = PINN.neg_model((pinn_sample, N))
    plt.figure()
    plt.grid()
    plt.plot(pinn_sample_t_0.detach().numpy(), c_p_0.detach().numpy(), "r--", label="c_p_0")
    plt.plot(pinn_sample_t.detach().numpy(), c_p.detach().numpy(), "r-", label="c_p")
    plt.plot(pinn_sample_t_0.detach().numpy(), c_n_0.detach().numpy(), "b--", label="c_n_0")
    plt.plot(pinn_sample_t.detach().numpy(), c_n.detach().numpy(), "b-", label="c_n")
    plt.legend()
    plt.xlabel("t [h]")
    plt.ylabel("c [-]")
    plt.show()