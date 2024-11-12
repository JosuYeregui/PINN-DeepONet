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

def plot_voltage_components(solution, PINN, samples):
    """
    Plot the voltage components of a solution
    """
    t = solution["Time [s]"].entries
    V_0_p = solution["X-averaged positive electrode open-circuit potential [V]"].entries
    V_0_n = solution["X-averaged negative electrode open-circuit potential [V]"].entries
    V_0 = V_0_p - V_0_n
    V_react_p = solution["X-averaged battery positive reaction overpotential [V]"].entries
    V_react_n = -solution["X-averaged battery negative reaction overpotential [V]"].entries

    U_0_p, U_0_n, eta_p, eta_n = PINN.compute_V_comps(samples)
    # Detach the tensors
    U_0_p = U_0_p.detach()
    U_0_n = U_0_n.detach()
    eta_p = eta_p.detach()
    eta_n = eta_n.detach()

    plt.figure()
    plt.plot(t / 3600, V_0, ':k',label="V_0")
    plt.plot(t / 3600, V_react_p + V_0, '--k',label="V_over_p")
    plt.plot(t / 3600, V_0 +  V_react_p + V_react_n, '-k',label="V_over_n")

    t = samples[0][:, 0]

    plt.plot(t, U_0_p-U_0_n, ':r', label="U_0")
    plt.plot(t, U_0_p-U_0_n+eta_p, '--r', label="eta_p")
    plt.plot(t, U_0_p-U_0_n+eta_p-eta_n, '-r', label="eta_n")
    plt.legend()
    plt.xlabel("t [h]")
    plt.ylabel("V [V]")
    plt.show()


if __name__ == "__main__":

    dr_p = 1.1
    dr_n = 0.75
    dr_Dp = 1.
    dr_Dn = 1.

    iterations = 2000

    crate = -1.

    file = "2024-10-28_09-14"
    iter = ""
    if iter != "":
        fold = os.path.join("..\\models", file, iter)
    else:
        fold = os.path.join("..\\models", file)

    # Run PyBaMM simulation
    param = pybamm.ParameterValues("Chen2020")
    param["Positive electrode active material volume fraction"] *= dr_p
    param["Negative electrode active material volume fraction"] *= dr_n
    param["Positive electrode diffusivity [m2.s-1]"] *= dr_Dp
    param["Negative electrode diffusivity [m2.s-1]"] *= dr_Dn

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


    parameters = load_params()
    # Pybamm takes stechiometric coefficients differently, so we need to adjust the parameters
    parameters["SOL_p"][0] = c_p_pbm[0]
    parameters["SOL_n"][0] = c_n_pbm[0]
    eps_p_0 = parameters["eps_p"]
    eps_n_0 = parameters["eps_n"]
    D_p_0 = parameters["D_p"]
    D_n_0 = parameters["D_n"]

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

    # as_n = torch.nn.Parameter(data=torch.tensor(parameters["as_n"]))
    # optimizer_param = torch.optim.Adam([as_n], lr=1e10)

    # Sampler = Sampler(training_points, constant(1.), mode="uniform")
    Sampler = Sampler_DONet(training_points, constant(crate), branch_samp=360, mode="quasi", device=device)

    cur_fun = constant(crate)

    print("Target P: ", PINN.pos_model.params["as_p"].detach().numpy() * dr_p * parameters["R_p"] / 3.)
    print("Target N: ", PINN.neg_model.params["as_n"].detach().numpy() * dr_n * parameters["R_n"] / 3.)

    PINN.neg_model.params["SOC_0"] = 0.
    PINN.pos_model.params["SOC_0"] = 0.

    pinn_sample_t_0 = torch.linspace(0, 1, 100)
    pinn_sample_r = torch.ones_like(pinn_sample_t_0)
    pinn_sample_I = torch.tensor(cur_fun(pinn_sample_t_0.numpy() * 3600.))
    pinn_sample = torch.stack([pinn_sample_t_0, pinn_sample_r, pinn_sample_I]).t()

    N = torch.tensor(cur_fun(np.arange(0., 3600, 10)))

    V_pinn_0 = PINN.compute_V((pinn_sample, N))
    c_p_0 = PINN.pos_model((pinn_sample, N))
    c_n_0 = PINN.neg_model((pinn_sample, N))
    t_pinn_0 = pinn_sample_t_0.detach().numpy()

    # plot_voltage_components(solution, PINN, (pinn_sample, N))

    eps_n = []
    eps_p = []
    D_n = []
    D_p = []

    start = time.time()
    for j in range(iterations + 1):

        PINN.neg_model.params["SOC_0"] = 0.
        PINN.pos_model.params["SOC_0"] = 0.

        t_points = np.sort(np.random.random(size=(1000, 1)) * t[-1], axis=0)
        V_pbm = torch.asarray(np.interp(t_points, t, V), dtype=torch.float32, requires_grad=True).flatten()

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
        # optimizer_param_Dp.zero_grad()
        # optimizer_param_Dn.zero_grad()

        loss = loss_tot_n + loss_tot_p + V_error
        loss.backward()

        optimizer_n.step()
        optimizer_p.step()
        optimizer_param_n.step()
        optimizer_param_p.step()
        # optimizer_param_Dp.step()
        # optimizer_param_Dn.step()

        if j % 100 == 0:
            print(j, "\t\t", V_error.detach().numpy(), "\t\t", loss_tot_n.detach().numpy(), "\t\t",
                  PINN.pos_model.params["as_p"].detach().numpy() * parameters["R_p"] / 3.,
                  PINN.neg_model.params["as_n"].detach().numpy() * parameters["R_n"] / 3.,
                  PINN.pos_model.params["D_p"].detach().numpy(),
                  PINN.neg_model.params["D_n"].detach().numpy(), "\t\t",
                  PINN.pos_model.params["as_p"].grad.detach().numpy(), PINN.neg_model.params["as_n"].grad.detach().numpy())
                  # torch.autograd.grad(V_error, PINN.pos_model.params["as_p"], retain_graph=True)[0].detach().numpy(),
                  # torch.autograd.grad(V_error, PINN.neg_model.params["as_n"], retain_graph=True)[0].detach().numpy())

        eps_n.append(PINN.neg_model.params["as_n"].detach().numpy() * parameters["R_n"] / 3.)
        eps_p.append(PINN.pos_model.params["as_p"].detach().numpy() * parameters["R_p"] / 3.)
        D_n.append(PINN.neg_model.params["D_n"].detach().numpy() * 1.)
        D_p.append(PINN.pos_model.params["D_p"].detach().numpy() * 1.)

    print(time.time() - start)

    plt.figure()
    plt.grid()
    plt.plot(range(iterations + 1), eps_n, "r-", label="eps_n")
    plt.plot(0, eps_n_0, 'rD')
    plt.axhline(y=eps_n_0 * dr_n, color='r', linestyle='--', label="eps_n target")
    plt.plot(range(iterations + 1), eps_p, "g-", label="eps_p")
    plt.plot(0, eps_p_0, 'gD')
    plt.axhline(y=eps_p_0 * dr_p, color='g', linestyle='--', label="eps_p target")
    plt.legend()
    plt.ylim([0.3, 0.9])
    plt.xlabel("Iteration")
    plt.ylabel("eps")
    plt.show()

    # Plot D_n and D_p with semi-log scale in y
    plt.figure()
    plt.grid()
    plt.semilogy(range(iterations + 1), D_n, "m-", label="D_n")
    plt.plot(0, D_n_0, 'mD')
    plt.axhline(y=D_n_0 * dr_Dn, color='m', linestyle='--', label="D_n target")
    plt.semilogy(range(iterations + 1), D_p, "c-", label="D_p")
    plt.plot(0, D_p_0, 'cD')
    plt.axhline(y=D_p_0 * dr_Dp, color='c', linestyle='--', label="D_p target")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("D [m2.s-1]")
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

    # Plot concentrations after training

    # PINN concentrations
    c_p = PINN.pos_model((pinn_sample, N))
    c_n = PINN.neg_model((pinn_sample, N))
    # Plot PBM and PINN concentrations
    plt.figure()
    plt.grid()
    plt.plot(t_pbm / 3600., c_p_pbm, "r-", label="P PyBaMM")
    plt.plot(pinn_sample_t.detach().numpy(), c_p.detach().numpy(), "r--", label="P PINN")
    plt.plot(t_pinn_0, c_p_0.detach().numpy(), "r:", label="P PINN_0")
    plt.plot(t_pbm / 3600., c_n_pbm, "b-", label="N PyBaMM")
    plt.plot(pinn_sample_t.detach().numpy(), c_n.detach().numpy(), "b--", label="N PINN")
    plt.plot(t_pinn_0, c_n_0.detach().numpy(), "b:", label="N PINN_0")
    plt.legend()
    plt.xlabel("t [h]")
    plt.ylabel("c_p [mol.m-3]")
    plt.show()
