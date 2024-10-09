import sys
sys.path.insert(0, "C:/Users/Josu/MGEP Dropbox/Josu Yeregui Unanue/Josu/1. Tesia/1.7 PINN DeepONet/PINN DeepONet")

from scr.SPMe import Solid_Phase, Cell
from scr.utils.pinn import FFNN, DeepONet, NN_TL_Diffusion, DeepONet_TL
from scr.utils.sampling import Sampler, Sampler_DONet
from scr.utils.parameters import load_params
from scr.utils.profiles import zheng_current, constant
from scr.utils.optimizers import NTK_Adaptive, Adam_Custom

import pybamm
import torch
import numpy as np
import pickle
import time
import os
from tqdm import tqdm
from datetime import datetime

import matplotlib.pyplot as plt

np.set_printoptions(precision=3)
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print("Device: ", device, "\n")
# print(torch.get_num_threads())
torch.set_num_threads(1)


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))


if __name__ == "__main__":

    EPOCH = 50000

    parameters = load_params()

    training_points = {"PDE": {"type": "PDE", "N": 1000},
                       "IV": {"type": "IV", "N": 100},
                       "BC_Center": {"type": "BC", "N": 100, "BC_pos": 0.},
                       "BC_Surf": {"type": "BC", "N": 100, "BC_pos": 1.}}
    validation_points = {"PDE": {"type": "PDE", "N": 30},
                         "IV": {"type": "IV", "N": 15},
                         "BC_Center": {"type": "BC", "N": 15, "BC_pos": 0.},
                         "BC_Surf": {"type": "BC", "N": 15, "BC_pos": 1.}}

    betas_tr = [0.2, 0.3, 0.5, 0.7, 0.75, 0.8, 0.85, 0.95, 1.1]
    betas_val = [0.4, 0.6]
    test_beta = 1.

    # model_p = NN_TL_Diffusion(general_layers=[64, 64, 64, 64], fine_layers=[32, 32], dim_in=3, dim_int=32,
    #                           dim_out=1, dropout=0.).to(device)
    model_p = DeepONet_TL(branch_layers=[64, 64, 64, 64], trunk_layers=[64, 64, 64, 64], fine_layers=[32, 32],
                          dim_branch=360, dim_trunk=3, dim_int=100, dim_out=1, dropout=0.).to(device)
    pos_model = Solid_Phase(model_p, parameters, [1., 1., 1.], criterion=RMSELoss, electrode="pos").to(device)
    optimizer_p = NTK_Adaptive(pos_model.model.parameters(), pos_model.weigths, adam_param = {'lr': 0.0005, 'betas': (0.9, 0.999)}, device=device)
    # optimizer_p = torch.optim.Adam(pos_model.model.parameters(), lr=0.0005)
    # optimizer_p_w = torch.optim.Adam([pos_model.adj_w], lr=0.0001)
    # model_n = NN_TL_Diffusion(general_layers=[64, 64, 64, 64], fine_layers=[32, 32], dim_in=3, dim_int=32,
    #                           dim_out=1, dropout=0.).to(device)
    model_n = DeepONet_TL(branch_layers=[64, 64, 64, 64], trunk_layers=[64, 64, 64, 64], fine_layers=[32, 32],
                          dim_branch=360, dim_trunk=3, dim_int=100, dim_out=1, dropout=0.).to(device)
    neg_model = Solid_Phase(model_n, parameters,[1., 1., 1.], criterion=RMSELoss, electrode="neg").to(device)
    optimizer_n = NTK_Adaptive(neg_model.model.parameters(), neg_model.weigths, adam_param = {'lr': 0.0005, 'betas': (0.9, 0.999)}, device=device)

    # optimizer_n = torch.optim.Adam(neg_model.model.parameters(), lr=0.0005)
    # optimizer_n_w = torch.optim.Adam([neg_model.adj_w], lr=0.0001)

    PINN = Cell(pos_model, neg_model)

    # PINN.neg_model.params["as_n"] = torch.nn.Parameter(torch.tensor([parameters["as_n"]]), requires_grad=False)

    # Sampler_tr = Sampler(training_points, constant(1.), mode="quasi", device=device)
    # Sampler_val = Sampler(validation_points, constant(1.), mode="quasi", device=device)
    Sampler_tr = Sampler_DONet(training_points, constant(1.), mode="quasi", device=device)
    Sampler_val = Sampler_DONet(validation_points, constant(1.), mode="quasi", device=device)

    history = {"positive":{"loss_tr": [], "losses_tr": [], "loss_val": [], "losses_val": [], "iteration": []},
               "negative":{"loss_tr": [], "losses_tr": [], "loss_val": [], "losses_val": [], "iteration": []}}

    # print("Iter \t\t PDE \t BC Centre \t BC Surf \t\t\t PDE \t BC Centre \t BC Surf")
    with tqdm(total=EPOCH, desc="Training Progress",
              bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:100}{r_bar}", colour="blue",
              smoothing=0.1) as pbar:
        for j in range(EPOCH):

            # t = time.time()
            rate_tr = np.random.choice(betas_tr)
            rate_val = np.random.choice(betas_val)
            # rate_tr *= -1
            # rate_val *= -1
            # PINN.neg_model.params["SOC_0"] = 0.
            # PINN.pos_model.params["SOC_0"] = 0.

            if np.random.rand() < 0.5:
                rate_tr *= -1
                rate_val *= -1
                PINN.neg_model.params["SOC_0"] = 0.
                PINN.pos_model.params["SOC_0"] = 0.
            else:
                PINN.neg_model.params["SOC_0"] = 1.
                PINN.pos_model.params["SOC_0"] = 1.

            # Sampler_tr.update_current_func(constant(rate_tr))
            # Sampler_tr.update_t(rate_tr)
            # Sampler_val.update_current_func(constant(rate_val))
            # Sampler_val.update_t(rate_val)

            Sampler_tr.update_samples(training_points, constant(rate_tr), 1. / np.abs(rate_tr))
            Sampler_val.update_samples(validation_points, constant(rate_val), 1. / np.abs(rate_val))

            Sampler_tr.update_N(constant(rate_tr))
            Sampler_val.update_N(constant(rate_val))

            # optimizer_p_w.zero_grad()
            # optimizer_n_w.zero_grad()

            losses_tr = PINN.pos_model.train_step(optimizer_p, Sampler_tr)
            losses_val = PINN.pos_model.evaluate(Sampler_val)
            loss_tot_tr = np.sum([l * w for l, w in zip(losses_tr, PINN.pos_model.weigths)])
            loss_tot_val = np.sum([l * w for l, w in zip(losses_val, PINN.pos_model.weigths)])

            if j % 1000 == 0:
                history["positive"]["loss_tr"].append(loss_tot_tr)
                history["positive"]["losses_tr"].append(losses_tr)
                history["positive"]["loss_val"].append(loss_tot_val)
                history["positive"]["losses_val"].append(losses_val)
                history["positive"]["iteration"].append(j)

                tqdm.write(f"\033[3mIteration: {j}\033[0m")
                tqdm.write("\033[1mPositive\033[0m")
                for l, w, n in zip(losses_tr, PINN.pos_model.weigths, ["PDE", "BC c", "BC s"]):
                    tqdm.write(f"{n} loss: {l:.3E}", end="\t")
                    tqdm.write(f"\033[92m{n} weight: {w:.3E}\033[0m", end="\t")

                tqdm.write(f"\t\033[4mTrain loss: {loss_tot_tr:.3E}", end="\t")
                tqdm.write(f"Val loss: {loss_tot_val:.3E}\033[0m")

            losses_tr = PINN.neg_model.train_step(optimizer_n, Sampler_tr)
            losses_val = PINN.neg_model.evaluate(Sampler_val)
            loss_tot_tr = np.sum([l * w for l, w in zip(losses_tr, PINN.neg_model.weigths)])
            loss_tot_val = np.sum([l * w for l, w in zip(losses_val, PINN.neg_model.weigths)])

            # optimizer_p_w.step()
            # optimizer_n_w.step()

            if j % 1000 == 0:
                history["negative"]["loss_tr"].append(loss_tot_tr)
                history["negative"]["losses_tr"].append(losses_tr)
                history["negative"]["loss_val"].append(loss_tot_val)
                history["negative"]["losses_val"].append(losses_val)
                history["negative"]["iteration"].append(j)

                tqdm.write("\033[1mNegative\033[0m")
                for l, w, n in zip(losses_tr, PINN.neg_model.weigths, ["PDE", "BC c", "BC s"]):
                    tqdm.write(f"{n} loss: {l:.3E}", end="\t")
                    tqdm.write(f"\033[92m{n} weight: {w:.3E}\033[0m", end="\t")

                tqdm.write(f"\t\033[4mTrain loss: {loss_tot_tr:.3E}", end="\t")
                tqdm.write(f"Val loss: {loss_tot_val:.3E}\033[0m")
                tqdm.write(f"\n")

            pbar.update(1)

    # Define the directory where you want to create the folder
    directory = '../models/'
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M')
    folder_path = os.path.join(directory, current_date)
    os.makedirs(folder_path, exist_ok=True)

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

    t_eval = np.arange(0, 3600)
    cur_fun = constant(test_beta)

    bcs_sample_t = torch.linspace(0., 1., 1000)
    bcs_sample_r = torch.ones_like(bcs_sample_t)
    bcs_sample_I = torch.tensor(cur_fun(bcs_sample_t.numpy() * 3600.))
    bcs_sample = torch.stack([bcs_sample_t, bcs_sample_r, bcs_sample_I]).t()

    N = torch.tensor(cur_fun(np.arange(0., 3600, 10)))
    PINN.neg_model.params["SOC_0"] = 1.
    PINN.pos_model.params["SOC_0"] = 1.

    V_pinn = PINN.compute_V((bcs_sample, N)).detach().numpy()

    param = pybamm.ParameterValues("Chen2020")
    PBM_model = pybamm.lithium_ion.SPM()

    experiment = pybamm.Experiment(["Discharge at 1C for 100000 seconds or until 2.5 V"])
    sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
    solution = sim.solve(initial_soc=1)

    t = solution["Time [s]"].entries

    V = solution["Terminal voltage [V]"].entries

    plt.figure()
    plt.grid()
    plt.plot(t / 3600. * 5., V, "k-", label="PyBaMM")
    plt.plot(bcs_sample_t * 5., V_pinn, "r-", label="PINN")
    plt.ylim([2.5, 4.3])
    plt.xlabel("Disch. Capacity [Ah]")
    plt.ylabel("V [V]")
    plt.legend()
    plt.savefig(os.path.join(folder_path, "Discharge.png"))
    # plt.show()



    cur_fun = constant(-test_beta)

    bcs_sample_t = torch.linspace(0., 1., 1000)
    bcs_sample_r = torch.ones_like(bcs_sample_t)
    bcs_sample_I = torch.tensor(cur_fun(bcs_sample_t.numpy() * 3600.))
    bcs_sample = torch.stack([bcs_sample_t, bcs_sample_r, bcs_sample_I]).t()
    PINN.neg_model.params["SOC_0"] = 0.
    PINN.pos_model.params["SOC_0"] = 0.

    N = torch.tensor(cur_fun(np.arange(0., 3600, 10)))

    V_pinn = PINN.compute_V((bcs_sample, N)).detach().numpy()

    param = pybamm.ParameterValues("Chen2020")
    PBM_model = pybamm.lithium_ion.SPM()

    experiment = pybamm.Experiment(["Charge at 1C for 100000 seconds or until 4.2 V"])
    sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
    solution = sim.solve(initial_soc=0)

    t = solution["Time [s]"].entries

    V = solution["Terminal voltage [V]"].entries

    plt.figure()
    plt.grid()
    plt.plot(t / 3600. * 5., V, "k-", label="PyBaMM")
    plt.plot(bcs_sample_t * 5., V_pinn, "r-", label="PINN")
    plt.ylim([2.5, 4.3])
    plt.xlabel("Chg. Capacity [Ah]")
    plt.ylabel("V [V]")
    plt.legend()
    plt.savefig(os.path.join(folder_path, "Charge.png"))
    # plt.show()