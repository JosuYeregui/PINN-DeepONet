import torch
import pybamm
import numpy as np

def load_params():
    param = pybamm.ParameterValues("Chen2020")

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
        "L_s": param["Separator thickness [m]"],
        "L": param["Positive electrode thickness [m]"] + param["Negative electrode thickness [m]"] +
        param["Separator thickness [m]"],
        "R_p": param["Positive particle radius [m]"],
        "R_n": param["Negative particle radius [m]"],
        "A": param["Electrode height [m]"] * param["Electrode width [m]"],
        "eps_p": param["Positive electrode active material volume fraction"],
        "eps_n": param["Negative electrode active material volume fraction"],
        "por_p": param["Positive electrode porosity"],
        "por_n": param["Negative electrode porosity"],
        "por_s": param["Separator porosity"],
        "t_plus": param["Cation transference number"],
        "as_p": 3 * param["Positive electrode active material volume fraction"] / param["Positive particle radius [m]"],
        "as_n": 3 * param["Negative electrode active material volume fraction"] / param["Negative particle radius [m]"],
        "alpha_p": param["Positive electrode charge transfer coefficient"],
        "alpha_n": param["Negative electrode charge transfer coefficient"],
        "c_p_max": param["Maximum concentration in positive electrode [mol.m-3]"],
        "c_n_max": param["Maximum concentration in negative electrode [mol.m-3]"],
        "ce0": param["Initial concentration in electrolyte [mol.m-3]"],
        "D_p": param["Positive electrode diffusivity [m2.s-1]"],
        "D_n": param["Negative electrode diffusivity [m2.s-1]"],
        "D_e": lambda c: 8.794e-11 * torch.pow(c, 2) - 3.972e-10 * torch.pow(c, 2) + 4.862e-10,
        "D_e_np": lambda c: 8.794e-11 * np.power(c, 2) - 3.972e-10 * np.power(c, 2) + 4.862e-10,
        "D_e_const": 4.862e-10,
        "sigma_e": lambda c: 0.1297 * torch.pow(c, 3) - 2.51 * torch.pow(c, 1.5) + 3.329 * c,
        "brug": 1.5,
        "SOL_n": [0.0263473, 0.91061212],
        "SOL_p": [0.9332, 0.252],
        "F": 96485.33212,
        "R": 8.314462,
        "T": 298.15
    }
    return parameters