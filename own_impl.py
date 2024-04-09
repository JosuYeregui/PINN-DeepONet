from scr.SPM import Solid_Phase
from scr.pinn import FFNN
from scr.utils import load_params

import pybamm
import torch
import numpy as np
import pickle

import matplotlib.pyplot as plt

np.set_printoptions(precision=3)


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))

if __name__ == "__main__":

    parameters = load_params()

    training_points = {"PDE": 1000, "IV": 50, "BC_Center": 50, "BC_Surf": 50}
    validation_points = {"PDE": 20, "IV": 10, "BC_Center": 10, "BC_Surf": 10}

    model = FFNN(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    PINN_neg = Solid_Phase(model, parameters, criterion=RMSELoss, electrode="neg")

    history = {"loss_tr": [], "losses_tr": [], "loss_val": [], "losses_val": [], "iteration": []}

    print("Iter \t\t PDE \t IV \t BC Centre \t BC Surf \t\t\t PDE \t IV \t BC Centre \t BC Surf")
    for j in range(20000 + 1):

        loss_tr, losses_tr, loss_val, losses_val = PINN_neg.train_step(optimizer, training_points, validation_points)

        if j % 1000 == 0:
            print(j, "\t\t", losses_tr, "\t\t", losses_val)
            history["loss_tr"].append(loss_tr)
            history["losses_tr"].append(losses_tr)
            history["loss_val"].append(loss_val)
            history["losses_val"].append(losses_val)
            history["iteration"].append(j)

    # PINN_pos.save_model("/models/previous.pt")
    #
    # with open('models/previous.pkl', 'wb') as fp:
    #     pickle.dump(history, fp)

    # Plot
    plt.figure()
    plt.grid("on")
    plt.semilogy(history["iteration"], history["loss_tr"], label="Training")
    plt.semilogy(history["iteration"], history["loss_val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

    # Evaluation
    # param = pybamm.ParameterValues("ORegan2022")
    param = pybamm.ParameterValues("Chen2020")
    PBM_model = pybamm.lithium_ion.SPM()
    I = -5
    experiment = pybamm.Experiment(["Discharge at 1C for 10000 seconds or until 2.5 V"])
    sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
    sol = sim.solve(initial_soc=1)

    # pos_SPM_r0 = sol['X-averaged positive particle concentration'].entries[0, :]
    # pos_SPM_r1 = sol['X-averaged positive particle concentration'].entries[-1, :]

    c_s_n = sol["Negative particle concentration"]
    c_s_p = sol["Positive particle concentration"]
    r_n = sol["r_n [m]"].entries[:, 0, 0]
    r_p = sol["r_p [m]"].entries[:, 0, 0]
    t = sol["Time [s]"].entries
    x = sol["x [m]"].entries[:, 0]

    pos_SPM_r0 = c_s_n(r=r_n[0], t=t, x=x[0])
    pos_SPM_r1 = c_s_n(r=r_n[-1], t=t, x=x[0])

    t_end = t[-1]

    t = np.linspace(0, t_end, num=len(pos_SPM_r1))/3600.

    bcs_sample_t = torch.linspace(0., 1., 1000)
    bcs_sample_r = torch.ones_like(bcs_sample_t)
    bcs_sample = torch.stack([bcs_sample_t, bcs_sample_r]).t()
    c_bcs = PINN_neg(bcs_sample)

    plt.figure()
    plt.grid("on")
    plt.plot(bcs_sample_t.detach().numpy(), c_bcs.detach().numpy(), "k", label="PINN")
    plt.plot(t, pos_SPM_r1, "r", label="Pybamm")
    plt.legend()
    plt.xlabel("t [h]")
    plt.ylabel("x [-]")
    plt.show()

    bcs_sample_r = torch.zeros_like(bcs_sample_t)
    bcs_sample_r0 = torch.stack([bcs_sample_t, bcs_sample_r]).t()
    c_bcs_r0 = PINN_neg(bcs_sample_r0)

    plt.figure()
    plt.grid("on")
    plt.plot(bcs_sample_t.detach().numpy(), c_bcs_r0.detach().numpy(), "k", label="PINN")
    plt.plot(t, pos_SPM_r0, "r", label="Pybamm")
    plt.legend()
    plt.xlabel("t [h]")
    plt.ylabel("x [-]")
    plt.show()
