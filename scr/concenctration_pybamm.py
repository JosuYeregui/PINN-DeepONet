"""Backend supported: tensorflow.compat.v1, pytorch, paddle"""
import deepxde as dde
from deepxde.backend import tf
import matplotlib.pyplot as plt
import numpy as np

# import torch
import pybamm

param = pybamm.ParameterValues("ORegan2022")
PBM_model = pybamm.lithium_ion.SPM()

parameters = {
    "E_p": lambda sto: -0.8090 * sto + 4.4875 - 0.0428 * tf.math.tanh(18.5138 * (sto - 0.5542)) -
                        17.7326 * tf.math.tanh(15.7890 * (sto - 0.3117)) + 17.5842 * tf.math.tanh(15.9308 * (sto - 0.3120)),
    "E_n": lambda sto: 1.9793 * tf.math.exp(-39.3631 * sto) + 0.2482 - 0.0909 * tf.math.tanh(29.8538 * (sto - 0.1234)) -
                        0.04478 * tf.math.tanh(14.9159 * (sto - 0.2769)) - 0.0205 * tf.math.tanh(30.4444 * (sto - 0.6103)),
    "I_typ": 5,
    "L_p": param["Positive electrode thickness [m]"],
    "L_n": param["Negative electrode thickness [m]"],
    "R_p": param["Positive particle radius [m]"],
    "R_n": param["Negative particle radius [m]"],
    "A": param["Electrode height [m]"] * param["Electrode width [m]"],
    "as_p": 3*param["Positive electrode active material volume fraction"]/param["Positive particle radius [m]"],
    "as_n": 3*param["Negative electrode active material volume fraction"]/param["Negative particle radius [m]"],
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

SOC_t0 = 1.
t_end = 3600.
cs_ini_p = parameters["SOL_pos"][0] + ((parameters["SOL_pos"][1] - parameters["SOL_pos"][0]) * SOC_t0)
print(cs_ini_p)
I = -5
experiment = pybamm.Experiment(["Discharge at 1C for 10000 seconds or until 2.5 V"])
experiment = pybamm.Experiment(["Rest for 100 seconds"])
sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
sol = sim.solve(initial_soc=SOC_t0)

pos_SPM = sol['X-averaged positive particle concentration'].entries[-1, :]

t = np.linspace(0, t_end, num=len(pos_SPM))

plt.figure()
plt.plot(t, pos_SPM, "r")
plt.show()

# u_2 = model.predict(torch.tensor(a))
#
# def cs_2(x, y):
#     D = parameters["D_p"]
#     R = parameters["R_p"]
#     dy_t = dde.grad.jacobian(y, x, j=1)
#     dy_x = dde.grad.jacobian(y, x, j=0)
#     N_xx = dde.grad.jacobian(dy_x * torch.pow(x[:, 0], 2), x, j=0)
#     # dy_xx = dde.grad.hessian(y, x, j=0)
#     return dy_t * torch.pow(x[:, 0], 2) - D/np.power(R, 2) * N_xx
#
# print(cs_2(a, u_2))