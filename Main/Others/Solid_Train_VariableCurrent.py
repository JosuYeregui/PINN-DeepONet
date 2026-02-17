import sys
import os

# sys.path.insert(0, "C:/Users/Josu/MGEP Dropbox/Josu Yeregui Unanue/Josu/1. Tesia/1.7 PINN DeepONet/PINN DeepONet")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(current_dir, '..', '..', 'PINN DeepONet')
# Add the project path to sys.path
sys.path.insert(0, project_path)
# wd = 'C:/Users/malen.etxeberria/OneDrive - Mondragon Unibertsitatea/Documentos/Energia/Master/TFM/VariableCurrent/Kodea'
# os.chdir(wd)

from scr.SPMe import Solid_Phase, Cell
from scr.utils.pinn import FFNN, DeepONet, NN_TL_Diffusion, DeepONet_TL, DeepONet_TL_FF
from scr.utils.sampling import Sampler, Sampler_DONet
from scr.utils.parameters import load_params
from scr.utils.profiles import zheng_current, constant, GRF, drivecycles
from scr.utils.optimizers import NTK_Adaptive, Adam_Custom

import pybamm
import torch
import numpy as np
import pickle
from tqdm import tqdm
from datetime import datetime

import matplotlib.pyplot as plt

np.set_printoptions(precision=3)
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print("Device: ", device, "\n")
# print(torch.get_num_threads())
torch.set_num_threads(1)

import pybamm
import numpy as np
import torch
import matplotlib.pyplot as plt


def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def simulate_spm_pinn(profile_name):
    current_function = drivecycles()

    PINN.neg_model.params["SOC_0"] = 0.9
    PINN.pos_model.params["SOC_0"] = 0.9
    t_eval = np.linspace(0, 3600, 1000)

    current_values = np.array([current_function(t, profile_name) for t in t_eval])

    param = pybamm.ParameterValues("Chen2020")
    current_interpolant = pybamm.Interpolant(t_eval, current_values * param["Nominal cell capacity [A.h]"],
                                             pybamm.t)
    param["Current function [A]"] = current_interpolant

    PBM_model = pybamm.lithium_ion.SPM()
    sim = pybamm.Simulation(PBM_model, parameter_values=param)
    sol = sim.solve(initial_soc=0.9, t_eval=t_eval)

    c_s_n = sol["Negative particle concentration"]
    c_s_p = sol["Positive particle concentration"]
    r_n = sol["r_n [m]"].entries[:, 0, 0]
    r_p = sol["r_p [m]"].entries[:, 0, 0]
    t = sol["Time [s]"].entries
    x = sol["x [m]"].entries[:, 0]

    pos_SPM_r0 = c_s_p(r=r_p[0], t=t, x=x[-1])
    pos_SPM_r1 = c_s_p(r=r_p[-1], t=t, x=x[-1])
    neg_SPM_r0 = c_s_n(r=r_n[0], t=t, x=x[0])
    neg_SPM_r1 = c_s_n(r=r_n[-1], t=t, x=x[0])

    t = t / 3600.

    # ---- PINN ---- #
    bcs_sample_t = torch.linspace(0., 1., 3600)
    N = torch.tensor(current_function(bcs_sample_t.numpy() * 3600., profile_name))
    bcs_sample_r = torch.ones_like(bcs_sample_t)
    bcs_sample_I = torch.tensor(current_function(bcs_sample_t.numpy() * 3600., profile_name))
    bcs_sample = torch.stack([bcs_sample_t, bcs_sample_r, bcs_sample_I]).t()
    c_bcs = PINN.pos_model((bcs_sample, N))

    plt.figure()
    plt.grid(True)
    plt.plot(bcs_sample_t.detach().numpy() * PINN.pos_model.tc / 3600., c_bcs.detach().numpy(), "k", label="PINN")
    plt.plot(t, pos_SPM_r1, "r", label="Pybamm")
    plt.legend()
    plt.xlabel("Tiempo [h]")
    plt.ylabel("Concentración [-]")
    plt.show()

    num = 1000
    t_test = (torch.arange(0, num, dtype=torch.float, requires_grad=True) / num)[::10]
    r_rand = (torch.arange(0, num, dtype=torch.float, requires_grad=True) / num)[::10]

    t_new, r_new = torch.meshgrid(t_test, r_rand)
    pos = np.zeros_like(t_new.detach().numpy())
    pos_dcdt = np.zeros_like(t_new.detach().numpy())
    pos_dcdr = np.zeros_like(t_new.detach().numpy())

    for i, (t, r) in enumerate(zip(t_new, r_new)):
        cur = torch.tensor(current_function(t.detach().numpy() * 3600., profile_name))
        sample = (torch.stack((t, r, cur)).t(), N)
        residuals = PINN.pos_model.compute_residuals(sample)
        pos[i, :] = np.abs(residuals.detach().numpy())
        gradients = PINN.pos_model.compute_gradients(sample)
        pos_dcdt[i, :] = np.abs(gradients[:, 0].detach().numpy())
        pos_dcdr[i, :] = np.abs(gradients[:, 1].detach().numpy())

    def plot_area(points, style, cmap_label):
        plt.subplots(figsize=(7, 3), tight_layout=True)
        plot = plt.pcolormesh(t_new.detach().numpy(), r_new.detach().numpy(), points, cmap=style, shading='gouraud')
        cbar = plt.colorbar(plot)
        plt.contour(t_new.detach().numpy(), r_new.detach().numpy(), points, 10, colors='gray')
        plt.ylabel('$x$')
        plt.xlabel('$t$ [h]')
        cbar.set_label(cmap_label)
        plt.show()

    plot_area(pos, 'RdBu_r', 'Residuos')
    plot_area(pos_dcdt, 'Oranges', 'dcdt')
    plot_area(pos_dcdr, 'Oranges', 'dcdr')


def plot_pybamm_vs_PINN(cur_func, SOC, PINN, folder_path, name, plot=False):
    bcs_sample_t = torch.linspace(0., 1., 1000)
    bcs_sample_r = torch.ones_like(bcs_sample_t)
    bcs_sample_I = torch.tensor(cur_func(bcs_sample_t.numpy()))
    bcs_sample = torch.stack([bcs_sample_t, bcs_sample_r, bcs_sample_I]).t()

    # if isinstance(cur_func, GRF):
    #     print(np.mean(cur_func.u))

    N = torch.tensor(cur_func(np.linspace(0., 1, 3600)))
    PINN.neg_model.params["SOC_0"] = SOC
    PINN.pos_model.params["SOC_0"] = SOC

    V_pinn = PINN.compute_V((bcs_sample, N)).detach().numpy()

    t_eval = bcs_sample_t.detach().numpy() * 3600.

    # cur_func_pybamm = lambda t: cur_func(t.numpy())
    cur_func_pybamm = pybamm.Interpolant(t_eval, cur_func(t_eval / 3600.) * param["Nominal cell capacity [A.h]"],
                                         pybamm.t)
    param["Current function [A]"] = cur_func_pybamm
    sim = pybamm.Simulation(PBM_model, parameter_values=param)
    try:
        solution = sim.solve(t_eval, initial_soc=SOC)
        V = solution["Terminal voltage [V]"].entries
        if t_eval.shape[0] != V.shape[0]:
            V = np.interp(t_eval, solution["Time [s]"].entries, V)
    except:
        V = np.zeros_like(t_eval)

        # Calcular RMSE entre PyBaMM y PINN
    error_rmse = rmse(V_pinn, V)
    print(f"RMSE ({name}): {error_rmse:.6f} V")

    plt.figure()
    plt.grid()
    plt.plot(t_eval / 3600., V, "k-", label="PyBaMM")
    plt.plot(t_eval / 3600., V_pinn, "r-", label="PINN")
    # plt.ylim([2.5, 4.3])
    plt.xlabel("time [h]")
    plt.ylabel("V [V]")
    plt.legend()
    plt.savefig(os.path.join(folder_path, name + ".png"))
    if plot:
        plt.show()
    plt.close()

    # Plot the current profile
    plt.figure()
    plt.grid()
    plt.plot(bcs_sample_t.detach().numpy(), bcs_sample_I.detach().numpy(), "b-")
    plt.xlabel("time [h]")
    plt.ylabel("I [A]")
    plt.savefig(os.path.join(folder_path, name + "_I.png"))
    if plot:
        plt.show()
    plt.close()


def store_and_print(path, j, PINN, plot=False):
    if not isinstance(j, str):
        j = str(j)
    part_path = os.path.join(path, j)
    os.makedirs(part_path, exist_ok=True)

    PINN.pos_model.save_model(os.path.join(part_path, "TL_pos.pt"))
    PINN.neg_model.save_model(os.path.join(part_path, "TL_neg.pt"))

    # Use predefined drive cycles
    drivecycles_func = drivecycles()

    # Fast profile
    SOC = np.clip(np.random.normal(0.75, 0.083), 0, 1)
    plot_pybamm_vs_PINN(lambda t: drivecycles_func(t, "US06"), SOC, PINN, part_path, "US06", plot)

    # Slow profile
    SOC = np.clip(np.random.normal(0.75, 0.083), 0, 1)
    plot_pybamm_vs_PINN(lambda t: drivecycles_func(t, "NYCC"), SOC, PINN, part_path, "NYCC", plot)


def plot_samples(sampler, condition="PDE"):
    if condition not in sampler.points:
        raise ValueError(f"Condition '{condition}' not found in sampler!")

    points = sampler.points[condition]

    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], s=20, alpha=0.8, marker='x', label=f"{condition} - {sampler.mode}",
                color='black')
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(f"{condition}")
    plt.legend()
    plt.grid()
    plt.show()


def get_curfunc():
    drivecycles_func = drivecycles()
    profiles = ["UDDS", "NEDC", "WLTC", "HWFET"]
    rand_num = np.random.rand()
    if rand_num < 0.9:
        profile_name = np.random.choice(profiles)
        SOC = np.clip(np.random.normal(0.75, 0.083), 0, 1)
        return lambda t: drivecycles_func(t, profile_name), SOC  # 15 minutos
    else:
        cur = -np.random.normal(0.01, 0.1)
        SOC = np.clip(np.random.normal(0.25, 0.083), 0, 1)
        high_freq_cur = lambda t: cur * np.sin(2 * np.pi * 1000 * t)  # Corriente de alta frecuencia (50 Hz)
        return high_freq_cur, SOC


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))


if __name__ == "__main__":

    # EPOCH = 100000
    EPOCH = 10000
    header = "VariableCurrent"

    parameters = load_params()
    param = pybamm.ParameterValues("Chen2020")
    PBM_model = pybamm.lithium_ion.SPM()

    training_points = {"PDE": {"type": "PDE", "N": 1000},
                       "IV": {"type": "IV", "N": 100},
                       "BC_Center": {"type": "BC", "N": 100, "BC_pos": 0.},
                       "BC_Surf": {"type": "BC", "N": 100, "BC_pos": 1.}}
    validation_points = {"PDE": {"type": "PDE", "N": 30},
                         "IV": {"type": "IV", "N": 15},
                         "BC_Center": {"type": "BC", "N": 15, "BC_pos": 0.},
                         "BC_Surf": {"type": "BC", "N": 15, "BC_pos": 1.}}
    from torch import nn
    model_p = DeepONet_TL_FF(branch_layers=[64, 64, 64, 64], trunk_layers=[64, 64, 64], fine_layers=[32, 32],
                             dim_branch=3600, dim_trunk=3, dim_int=100, dim_out=1, sigmas_fourier=[0.1, 1.],
                             dropout=0., activation=nn.Tanh, bypass=True).to(device)
    pos_model = Solid_Phase(model_p, parameters, [1., 1., 1.], criterion=RMSELoss, electrode="pos").to(device)
    optimizer_p = NTK_Adaptive(pos_model.model.parameters(), pos_model.weights,
                               adam_param={'lr': 0.00005, 'betas': (0.9, 0.999)}, device=device)

    model_n = DeepONet_TL_FF(branch_layers=[64, 64, 64, 64], trunk_layers=[64, 64, 64], fine_layers=[32, 32],
                             dim_branch=3600, dim_trunk=3, dim_int=100, dim_out=1, sigmas_fourier=[0.1, 1.],
                             dropout=0., activation=nn.Tanh, bypass=True).to(device)
    neg_model = Solid_Phase(model_n, parameters, [1., 1., 1.], criterion=RMSELoss, electrode="neg").to(device)
    optimizer_n = NTK_Adaptive(neg_model.model.parameters(), neg_model.weights,
                               adam_param={'lr': 0.00005, 'betas': (0.9, 0.999)}, device=device)

    PINN = Cell(pos_model, neg_model)

    cur_func, SOC = get_curfunc()
    Sampler_tr = Sampler_DONet(training_points, cur_func, branch_samp=3600, mode="quasi", device=device)
    Sampler_val = Sampler_DONet(validation_points, cur_func, branch_samp=3600, mode="quasi", device=device)

    # Puntuak ploteatu

    # plot_samples(Sampler_tr, condition="PDE")
    # plot_samples(Sampler_tr, condition="BC_Surf")
    # plot_samples(Sampler_tr, condition="BC_Center")
    # plot_samples(Sampler_tr, condition="IV")

    history = {"positive": {"loss_tr": [], "losses_tr": [], "loss_val": [], "losses_val": [], "iteration": []},
               "negative": {"loss_tr": [], "losses_tr": [], "loss_val": [], "losses_val": [], "iteration": []}}

    # Define the directory where you want to create the folder
    directory = '../models/'
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M')
    folder_path = os.path.join(directory, header + "_" + current_date)
    os.makedirs(folder_path, exist_ok=True)

    # print("Iter \t\t PDE \t BC Centre \t BC Surf \t\t\t PDE \t BC Centre \t BC Surf")
    with tqdm(total=EPOCH, desc="Training Progress",
              bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:100}{r_bar}", colour="blue",
              smoothing=0.1) as pbar:
        for j in range(EPOCH):

            cur_func, SOC = get_curfunc()

            PINN.neg_model.params["SOC_0"] = SOC
            PINN.pos_model.params["SOC_0"] = SOC

            Sampler_tr.update_samples(training_points, cur_func, 1.)
            Sampler_tr.update_N(cur_func)
            losses_tr_p = PINN.pos_model.train_step(optimizer_p, Sampler_tr)
            losses_tr_n = PINN.neg_model.train_step(optimizer_n, Sampler_tr)

            if j % 1000 == 0:

                cur_func, SOC = get_curfunc()
                PINN.neg_model.params["SOC_0"] = SOC
                PINN.pos_model.params["SOC_0"] = SOC
                Sampler_val.update_samples(validation_points, cur_func, 1.)
                Sampler_val.update_N(cur_func)
                losses_val_p = PINN.pos_model.evaluate(Sampler_val)
                losses_val_n = PINN.neg_model.evaluate(Sampler_val)

                loss_tot_tr = np.sum([l * w for l, w in zip(losses_tr_p, PINN.pos_model.weights)])
                loss_tot_val = np.sum([l * w for l, w in zip(losses_val_p, PINN.pos_model.weights)])

                history["positive"]["loss_tr"].append(loss_tot_tr)
                history["positive"]["losses_tr"].append(losses_tr_p)
                history["positive"]["loss_val"].append(loss_tot_val)
                history["positive"]["losses_val"].append(losses_val_p)
                history["positive"]["iteration"].append(j)

                tqdm.write(f"\033[3mIteration: {j}\033[0m")
                tqdm.write("\033[1mPositive\033[0m")
                for l, w, n in zip(losses_tr_p, PINN.pos_model.weights, ["PDE", "BC c", "BC s"]):
                    tqdm.write(f"{n} loss: {l:.3E}", end="\t")
                    tqdm.write(f"\033[92m{n} weight: {w:.3E}\033[0m", end="\t")

                tqdm.write(f"\t\033[4mTrain loss: {loss_tot_tr:.3E}", end="\t")
                tqdm.write(f"Val loss: {loss_tot_val:.3E}\033[0m")

                loss_tot_tr = np.sum([l * w for l, w in zip(losses_tr_n, PINN.neg_model.weights)])
                loss_tot_val = np.sum([l * w for l, w in zip(losses_val_n, PINN.neg_model.weights)])

                history["negative"]["loss_tr"].append(loss_tot_tr)
                history["negative"]["losses_tr"].append(losses_tr_n)
                history["negative"]["loss_val"].append(loss_tot_val)
                history["negative"]["losses_val"].append(losses_val_n)
                history["negative"]["iteration"].append(j)

                tqdm.write("\033[1mNegative\033[0m")
                for l, w, n in zip(losses_tr_n, PINN.neg_model.weights, ["PDE", "BC c", "BC s"]):
                    tqdm.write(f"{n} loss: {l:.3E}", end="\t")
                    tqdm.write(f"\033[92m{n} weight: {w:.3E}\033[0m", end="\t")

                tqdm.write(f"\t\033[4mTrain loss: {loss_tot_tr:.3E}", end="\t")
                tqdm.write(f"Val loss: {loss_tot_val:.3E}\033[0m")
                tqdm.write(f"\n")

                store_and_print(folder_path, j, PINN)

            pbar.update(1)

    PINN.pos_model.save_model(os.path.join(folder_path, "TL_pos.pt"))
    PINN.neg_model.save_model(os.path.join(folder_path, "TL_neg.pt"))

    with open(os.path.join(folder_path, 'TL_hist.pkl'), 'wb') as fp:
        pickle.dump(history, fp)

    plt.figure()
    plt.grid()
    plt.semilogy(history["positive"]["iteration"], history["positive"]["loss_tr"], '--r', label="+ Training")
    plt.semilogy(history["positive"]["iteration"], history["positive"]["loss_val"], '-r', label="+ Validation")
    plt.semilogy(history["negative"]["iteration"], history["negative"]["loss_tr"], '--b', label="- Training")
    plt.semilogy(history["negative"]["iteration"], history["negative"]["loss_val"], '-b', label="- Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig(os.path.join(folder_path, "hist.png"))
    # plt.show()

    store_and_print(folder_path, j, PINN, plot=True)

    simulate_spm_pinn("US06")
    simulate_spm_pinn("NYCC")
