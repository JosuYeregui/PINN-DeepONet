import sys
import os
# sys.path.insert(0, "C:/Users/Josu/MGEP Dropbox/Josu Yeregui Unanue/Josu/1. Tesia/1.7 PINN DeepONet/PINN DeepONet")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(current_dir, '..', '..', 'PINN DeepONet')
# Add the project path to sys.path
sys.path.insert(0, project_path)

from scr.SPMe import Solid_Phase, Cell
from scr.utils.pinn import FFNN, DeepONet, NN_TL_Diffusion, DeepONet_TL
from scr.utils.sampling import Sampler, Sampler_DONet
from scr.utils.parameters import load_params
from scr.utils.profiles import zheng_current, constant, GRF
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
    cur_func_pybamm = pybamm.Interpolant(t_eval, cur_func(t_eval / 3600.) * param["Nominal cell capacity [A.h]"], pybamm.t)
    param["Current function [A]"] = cur_func_pybamm
    sim = pybamm.Simulation(PBM_model, parameter_values=param)
    try:
        solution = sim.solve(t_eval, initial_soc=SOC)
        V = solution["Terminal voltage [V]"].entries
        if t_eval.shape[0] != V.shape[0]:
            V = np.interp(t_eval, solution["Time [s]"].entries, V)
    except:
        V = np.zeros_like(t_eval)

    plt.figure()
    plt.grid()
    plt.plot(t_eval/3600., V, "k-", label="PyBaMM")
    plt.plot(t_eval/3600., V_pinn, "r-", label="PINN")
    plt.ylim([2.5, 4.3])
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

    state = np.random.get_state()
    l_s = 0.01
    grf_func.update(1, grf_func.N, l_s)
    SOC = np.clip(np.random.normal(0.75, 0.083), 0, 1)

    plot_pybamm_vs_PINN(grf_func, SOC, PINN, part_path, "Fast", plot)
    np.random.set_state(state)

    l_s = 0.1
    grf_func.update(1, grf_func.N, l_s)
    SOC = np.clip(np.random.normal(0.75, 0.083), 0, 1)

    plot_pybamm_vs_PINN(grf_func, SOC, PINN, part_path, "Slow", plot)
    np.random.set_state(state)

    prof = constant(0.5)
    SOC = np.clip(np.random.normal(0.75, 0.083), 0.5, 1)

    plot_pybamm_vs_PINN(prof, SOC, PINN, part_path, "Constant", plot)
    np.random.set_state(state)

def get_curfunc():

    rand_num = np.random.rand()

    if rand_num < 0.8:
        grf_func.update_u()
        # random SOC with mean in 0.75 and std in 0.2 and clip in 0 and 1.
        SOC = np.clip(np.random.normal(0.75, 0.083), 0, 1)
        return grf_func, SOC
    elif rand_num < 0.9:
        l_s = np.random.uniform(0.01, 0.1)
        grf_func.update(1, grf_func.N, l_s)
        # random SOC with mean in 0.75 and 3 * std in 0.2 and clip in 0 and 1.
        SOC = np.clip(np.random.normal(0.75, 0.083), 0, 1)
        return grf_func, SOC
    elif rand_num < 0.95:
        # random constant current with mean in 0.5 and std in 0.2
        cur = np.random.normal(0.5, 0.1)
        SOC = np.clip(np.random.normal(0.75, 0.083), 0.5, 1)
        return constant(cur), SOC
    else:
        # random Zheng current with mean in 0.5 and std in 0.2
        cur = -np.random.normal(0.5, 0.1)
        SOC = np.clip(np.random.normal(0.25, 0.083), 0, 0.5)
        return zheng_current(cur), SOC


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))


if __name__ == "__main__":

    EPOCH = 100000

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

    # model_p = NN_TL_Diffusion(general_layers=[64, 64, 64, 64], fine_layers=[32, 32], dim_in=3, dim_int=32,
    #                           dim_out=1, dropout=0.).to(device)
    model_p = DeepONet_TL(branch_layers=[64, 64, 64, 64], trunk_layers=[128, 64, 64, 64], fine_layers=[32, 32],
                          dim_branch=3600, dim_trunk=3, dim_int=128, dim_out=1, dropout=0.).to(device)
    pos_model = Solid_Phase(model_p, parameters, [1., 1., 1.], criterion=RMSELoss, electrode="pos").to(device)
    optimizer_p = NTK_Adaptive(pos_model.model.parameters(), pos_model.weigths, adam_param = {'lr': 0.00001, 'betas': (0.9, 0.999)}, device=device)
    # optimizer_p = torch.optim.Adam(pos_model.model.parameters(), lr=0.0005)
    # optimizer_p_w = torch.optim.Adam([pos_model.adj_w], lr=0.0001)
    # model_n = NN_TL_Diffusion(general_layers=[64, 64, 64, 64], fine_layers=[32, 32], dim_in=3, dim_int=32,
    #                           dim_out=1, dropout=0.).to(device)
    model_n = DeepONet_TL(branch_layers=[64, 64, 64, 64], trunk_layers=[64, 64, 64, 64], fine_layers=[32, 32],
                          dim_branch=3600, dim_trunk=3, dim_int=128, dim_out=1, dropout=0.).to(device)
    neg_model = Solid_Phase(model_n, parameters,[1., 1., 1.], criterion=RMSELoss, electrode="neg").to(device)
    optimizer_n = NTK_Adaptive(neg_model.model.parameters(), neg_model.weigths, adam_param = {'lr': 0.00001, 'betas': (0.9, 0.999)}, device=device)

    # optimizer_n = torch.optim.Adam(neg_model.model.parameters(), lr=0.0005)
    # optimizer_n_w = torch.optim.Adam([neg_model.adj_w], lr=0.0001)

    PINN = Cell(pos_model, neg_model)

    l_s = np.random.uniform(0.01, 0.1)
    grf_func = GRF(1, 1000, l_s, mean=0.5, variance=0.1)

    # PINN.neg_model.params["as_n"] = torch.nn.Parameter(torch.tensor([parameters["as_n"]]), requires_grad=False)

    # Sampler_tr = Sampler(training_points, constant(1.), mode="quasi", device=device)
    # Sampler_val = Sampler(validation_points, constant(1.), mode="quasi", device=device)
    cur_func, SOC = get_curfunc()
    Sampler_tr = Sampler_DONet(training_points, cur_func, branch_samp=3600, mode="quasi", device=device)
    Sampler_val = Sampler_DONet(validation_points, cur_func, branch_samp=3600, mode="quasi", device=device)

    history = {"positive":{"loss_tr": [], "losses_tr": [], "loss_val": [], "losses_val": [], "iteration": []},
               "negative":{"loss_tr": [], "losses_tr": [], "loss_val": [], "losses_val": [], "iteration": []}}

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

                loss_tot_tr = np.sum([l * w for l, w in zip(losses_tr_p, PINN.pos_model.weigths)])
                loss_tot_val = np.sum([l * w for l, w in zip(losses_val_p, PINN.pos_model.weigths)])

                history["positive"]["loss_tr"].append(loss_tot_tr)
                history["positive"]["losses_tr"].append(losses_tr_p)
                history["positive"]["loss_val"].append(loss_tot_val)
                history["positive"]["losses_val"].append(losses_val_p)
                history["positive"]["iteration"].append(j)

                tqdm.write(f"\033[3mIteration: {j}\033[0m")
                tqdm.write("\033[1mPositive\033[0m")
                for l, w, n in zip(losses_tr_p, PINN.pos_model.weigths, ["PDE", "BC c", "BC s"]):
                    tqdm.write(f"{n} loss: {l:.3E}", end="\t")
                    tqdm.write(f"\033[92m{n} weight: {w:.3E}\033[0m", end="\t")

                tqdm.write(f"\t\033[4mTrain loss: {loss_tot_tr:.3E}", end="\t")
                tqdm.write(f"Val loss: {loss_tot_val:.3E}\033[0m")

                loss_tot_tr = np.sum([l * w for l, w in zip(losses_tr_n, PINN.neg_model.weigths)])
                loss_tot_val = np.sum([l * w for l, w in zip(losses_val_n, PINN.neg_model.weigths)])

                history["negative"]["loss_tr"].append(loss_tot_tr)
                history["negative"]["losses_tr"].append(losses_tr_n)
                history["negative"]["loss_val"].append(loss_tot_val)
                history["negative"]["losses_val"].append(losses_val_n)
                history["negative"]["iteration"].append(j)

                tqdm.write("\033[1mNegative\033[0m")
                for l, w, n in zip(losses_tr_n, PINN.neg_model.weigths, ["PDE", "BC c", "BC s"]):
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

