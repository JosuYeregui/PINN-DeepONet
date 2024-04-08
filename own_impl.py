from scr.SPM import Solid_Phase
from scr.pinn import FFNN

import pybamm
import torch
import numpy as np

import matplotlib.pyplot as plt

np.set_printoptions(precision=3)


if __name__ == "__main__":

    param = pybamm.ParameterValues("Chen2020")
    PBM_model = pybamm.lithium_ion.SPM()

    parameters = {
        "E_p": lambda sto: -0.8090 * sto + 4.4875 - 0.0428 * torch.tanh(18.5138 * (sto - 0.5542)) -
                           17.7326 * torch.tanh(15.7890 * (sto - 0.3117)) + 17.5842 * torch.tanh(
            15.9308 * (sto - 0.3120)),
        "E_n": lambda sto: 1.9793 * torch.exp(-39.3631 * sto) + 0.2482 - 0.0909 * torch.tanh(29.8538 * (sto - 0.1234)) -
                           0.04478 * torch.tanh(14.9159 * (sto - 0.2769)) - 0.0205 * torch.tanh(
            30.4444 * (sto - 0.6103)),
        "I_typ": 5,
        "SOC_0": 1.,
        "L_p": param["Positive electrode thickness [m]"],
        "L_n": param["Negative electrode thickness [m]"],
        "R_p": param["Positive particle radius [m]"],
        "R_n": param["Negative particle radius [m]"],
        "A": param["Electrode height [m]"] * param["Electrode width [m]"],
        "as_p": 3 * param["Positive electrode active material volume fraction"] / param["Positive particle radius [m]"],
        "as_n": 3 * param["Negative electrode active material volume fraction"] / param["Negative particle radius [m]"],
        "alpha_p": param["Positive electrode charge transfer coefficient"],
        "alpha_n": param["Negative electrode charge transfer coefficient"],
        "c_p_max": param["Maximum concentration in positive electrode [mol.m-3]"],
        "c_n_max": param["Maximum concentration in negative electrode [mol.m-3]"],
        "D_p": param["Positive electrode diffusivity [m2.s-1]"],
        "D_n": param["Negative electrode diffusivity [m2.s-1]"],
        "SOL_neg": [0.002, 0.7619],
        "SOL_pos": [0.9332, 0.3987],
        "F": 96485.33212,
        "R": 8.314462,
        "T": 298.15
    }

    training_points = {"PDE": 100, "IV": 20, "BC_Center": 20, "BC_Surf": 20}
    validation_points = {"PDE": 20, "IV": 10, "BC_Center": 10, "BC_Surf": 10}

    model = FFNN(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    PINN_pos = Solid_Phase(model, parameters)

    print("Iter \t\t PDE \t BC Centre \t BC Surf \t IV \t\t\t PDE \t BC Centre \t BC Surf \t IV")
    for j in range(10000 + 1):

        loss_tr, losses_tr, loss_val, losses_val = PINN_pos.train_step(optimizer, training_points, validation_points)

        if j % 1000 == 0:
            print(j, "\t\t", losses_tr, "\t\t", losses_val)

    # Plot


