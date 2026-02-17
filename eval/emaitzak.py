import sys
import os

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
from tqdm import tqdm
from datetime import datetime

import matplotlib.pyplot as plt
import scienceplots as sp

def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))


with open(os.path.join('../models/2025-03-07_16-00/TL_hist.pkl'), 'rb') as fp:
   hist = pickle.load(fp)

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
#%%
parameters = load_params()
# parameters["SOL_p"] = [0.8599, 0.2719]
# parameters["SOL_n"] = [0.0339, 0.9742]
# parameters["D_p"] = 5.57979526e-13  # 1.64852539e-14
# parameters["D_n"] = 1.91909180e-15

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

model_p = DeepONet_TL(branch_layers=[64, 64, 64, 64], trunk_layers=[64, 64, 64, 64], fine_layers=[32, 32],
                      dim_branch=360, dim_trunk=3, dim_int=128, dim_out=1, dropout=0.).to(device)
pos_model = Solid_Phase(model_p, parameters, [1., 1., 1.], criterion=RMSELoss, electrode="pos").to(device)
optimizer_p = NTK_Adaptive(pos_model.model.parameters(), pos_model.weights, adam_param = {'lr': 0.0001, 'betas': (0.9, 0.999)}, device=device)

model_n = DeepONet_TL(branch_layers=[64, 64, 64, 64], trunk_layers=[64, 64, 64, 64], fine_layers=[32, 32],
                      dim_branch=360, dim_trunk=3, dim_int=128, dim_out=1, dropout=0.).to(device)
neg_model = Solid_Phase(model_n, parameters,[1., 1., 1.], criterion=RMSELoss, electrode="neg").to(device)
optimizer_n = NTK_Adaptive(neg_model.model.parameters(), neg_model.weights, adam_param = {'lr': 0.0001, 'betas': (0.9, 0.999)}, device=device)

# optimizer_n = torch.optim.Adam(neg_model.model.parameters(), lr=0.0005)
# optimizer_n_w = torch.optim.Adam([neg_model.adj_w], lr=0.0001)

PINN = Cell(pos_model, neg_model)

PINN.pos_model.load_model("../models/2025-03-07_16-00/TL_pos.pt")
PINN.neg_model.load_model("../models/2025-03-07_16-00/TL_neg.pt")
#%%
cur_fun = constant(-1)

bcs_sample_t = torch.linspace(0., 1., 1000)
bcs_sample_r = torch.ones_like(bcs_sample_t)
bcs_sample_I = torch.tensor(cur_fun(bcs_sample_t.numpy() * 3600.))
bcs_sample = torch.stack([bcs_sample_t, bcs_sample_r, bcs_sample_I]).t()
PINN.neg_model.params["SOC_0"] = 0.
PINN.pos_model.params["SOC_0"] = 0.

N = torch.tensor(cur_fun(np.arange(0., 3600, 10)))

V_pinn = PINN.compute_V((bcs_sample, N)).detach().numpy()

# points = {"PDE": {"type": "PDE", "N": 1000},
#                    "IV": {"type": "IV", "N": 100},
#                    "BC_Center": {"type": "BC", "N": 100, "BC_pos": 0.},
#                    "BC_Surf": {"type": "BC", "N": 100, "BC_pos": 1.}}
# Sampler = Sampler_DONet(points, constant(-1), branch_samp=360, mode="uniform", device=device)
#
# pos_loss = torch.sum(torch.tensor([l*w for l, w in zip(PINN.pos_model.compute_loss(Sampler), PINN.pos_model.weights)]))
# neg_loss = torch.sum(torch.tensor([l*w for l, w in zip(PINN.neg_model.compute_loss(Sampler), PINN.neg_model.weights)]))
#
# print(pos_loss * 1000, neg_loss*1000)

param = pybamm.ParameterValues("Chen2020")
PBM_model = pybamm.lithium_ion.SPM()

experiment = pybamm.Experiment(["Charge at 1C for 100000 seconds or until 4.2 V"])
sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
solution = sim.solve(initial_soc=0)

t = solution["Time [s]"].entries

V = solution["Terminal voltage [V]"].entries

t_pinn = bcs_sample_t.numpy()
t_pinn = t_pinn[V_pinn < 4.2]
V_pinn = V_pinn[V_pinn < 4.2]

with plt.style.context(['science', 'ieee']):
    plt.figure()
    plt.plot(t / 3600., V, label="PyBaMM SPM")
    plt.plot(t_pinn, V_pinn, label="PINN")
    plt.xlabel("Time [h]")
    plt.ylabel("Voltage [V]")
    plt.ylim([2.5, 4.2])
    plt.legend()
    # Error in second axis
    ax2 = plt.twinx()
    ax2.set_ylabel('sin', color='springgreen')
    ax2.tick_params(axis='y', labelcolor='springgreen')
    # Interpolate V so it has the same shape as V_pinn
    V_n = np.interp(t_pinn, t / 3600., V)
    plt.plot(t_pinn, np.abs(V_n - V_pinn) * 1000, label="Voltage difference", color="springgreen", linestyle="-")
    plt.ylabel("Absolute difference [mV]")
    plt.ylim([0, 100])
    # plt.savefig(os.path.join(folder_path, "Voltage.svg"), format="svg")
    plt.savefig(os.path.join("Paper_results", "Charge.svg"), format="svg")
    plt.savefig(os.path.join("Paper_results", "Charge.pdf"), format="pdf")
    # plt.tight_layout()
    plt.show()

# Voltage rmse between PINN and PyBaMM
rmse = np.sqrt(np.mean((V_n - V_pinn) ** 2))
print(rmse)

num = 1000.
PINN.neg_model.params["SOC_0"] = 0.
PINN.pos_model.params["SOC_0"] = 0.

t_test = (torch.arange(0, num, dtype=torch.float, requires_grad=True) / num * t_pinn[-1])[::10]
r_rand = (torch.arange(0, num, dtype=torch.float, requires_grad=True) / num)[::10]

t_new, r_new = torch.meshgrid(t_test, r_rand)
pos = np.zeros_like(t_new.detach().numpy())
neg = np.zeros_like(t_new.detach().numpy())
I_test = torch.tensor(cur_fun(t_new.detach().numpy() * 3600.))
N = torch.tensor(cur_fun(np.arange(0., 3600, 10)))

pos_dcdt = np.zeros_like(t_new.detach().numpy())
neg_dcdt = np.zeros_like(t_new.detach().numpy())
pos_dcdr = np.zeros_like(t_new.detach().numpy())
neg_dcdr = np.zeros_like(t_new.detach().numpy())

pos_c = np.zeros_like(t_new.detach().numpy())
neg_c = np.zeros_like(t_new.detach().numpy())
pos_c_pybamm = np.zeros_like(t_new.detach().numpy())
neg_c_pybamm = np.zeros_like(t_new.detach().numpy())

param = pybamm.ParameterValues("Chen2020")
PBM_model = pybamm.lithium_ion.SPM()
experiment = pybamm.Experiment(["Charge at 1C for 10000 seconds or until 4.2 V"])
sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
solution = sim.solve(initial_soc=0.)

x = solution["x [m]"].entries[:, 0]

c_s_n = solution["Negative particle concentration"]
c_s_p = solution["Positive particle concentration"]
r_n = solution["r_n [m]"].entries[:, 0, 0]
r_p = solution["r_p [m]"].entries[:, 0, 0]

for i, (t, r, I_t) in enumerate(zip(t_new, r_new, I_test)):
    residuals = PINN.pos_model.compute_residuals((torch.stack((t, r, I_t)).t(), N))
    pos[i, :] = np.abs(residuals.detach().numpy())
    gradients = PINN.pos_model.compute_gradients((torch.stack((t, r, I_t)).t(), N))
    pos_dcdt[i, :] = np.abs(gradients[:, 0].detach().numpy())
    pos_dcdr[i, :] = np.abs(gradients[:, 1].detach().numpy())
    pos_c[i, :] = PINN.pos_model((torch.stack((t, r, I_t)).t(), N)).detach().numpy()
    pos_c_pybamm[i, :] = c_s_p(r=r.detach().numpy() * param["Positive particle radius [m]"], t=t.detach().numpy()[0] * 3600, x=x[-1])

    residuals = PINN.neg_model.compute_residuals((torch.stack((t, r, I_t)).t(), N))
    neg[i, :] = np.abs(residuals.detach().numpy())
    gradients = PINN.neg_model.compute_gradients((torch.stack((t, r, I_t)).t(), N))
    neg_dcdt[i, :] = np.abs(gradients[:, 0].detach().numpy())
    neg_dcdr[i, :] = np.abs(gradients[:, 1].detach().numpy())
    neg_c[i, :] = PINN.neg_model((torch.stack((t, r, I_t)).t(), N)).detach().numpy()
    neg_c_pybamm[i, :] = c_s_n(r=r.detach().numpy() * param["Negative particle radius [m]"], t=t.detach().numpy()[0] * 3600, x=x[0])

# COmpute rmse between PINN and PyBaMM
rmse_pos = np.sqrt(np.mean((pos_c - pos_c_pybamm) ** 2))
rmse_neg = np.sqrt(np.mean((neg_c - neg_c_pybamm) ** 2))
print(rmse_pos, rmse_neg)
def plot_area(points, style, cmap_label, contour=True, unit=" [-]"):
    with plt.style.context(['science', 'ieee']):
        plt.subplots(figsize=(3.3,1.25))#, tight_layout=True)
        plot = plt.pcolormesh(t_new.detach().numpy(), r_new.detach().numpy(), points, cmap=style, shading='gouraud')
        cbar = plt.colorbar(plot)
        if contour:
            plt.contour(t_new.detach().numpy(), r_new.detach().numpy(), points, 10, colors='gray')
        plt.ylabel('$r$ [-]')
        plt.xlabel('$t$ [h]')
        cbar.set_label(cmap_label + unit)
        # plt.savefig('/content/drive/MyDrive/Datos/con_PINN_pos.png')
        # plt.savefig(os.path.join("Paper_results", cmap_label+".svg"), format="svg")
        # plt.savefig(os.path.join("Paper_results", cmap_label+".pdf"), format="pdf")
        plt.show()

plot_area(pos_c, 'YlOrRd', '$c_p$')
plot_area(neg_c, 'YlGnBu', '$c_n$')

# plot_area(np.abs(pos_c - pos_c_pybamm)/pos_c_pybamm * 100, 'jet', 'Rel. error $c_p$', contour=False, unit=" [\%]")
# plot_area(np.abs(neg_c - neg_c_pybamm)/neg_c_pybamm * 100, 'jet', 'Rel. error $c_n$', contour=False, unit=" [\%]")


plot_area(np.abs(pos_c - pos_c_pybamm), 'viridis', 'Abs. difference $c_p$', contour=False, unit=" [-]")
plot_area(np.abs(neg_c - neg_c_pybamm), 'viridis', 'Abs. difference $c_n$', contour=False, unit=" [-]")

