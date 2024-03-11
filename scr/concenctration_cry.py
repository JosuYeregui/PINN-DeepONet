"""Backend supported: tensorflow.compat.v1, pytorch, paddle"""
import deepxde as dde
from deepxde.backend import tf
import matplotlib.pyplot as plt
import numpy as np

# import torch
import pybamm

param = pybamm.ParameterValues("Chen2020")
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
    "SOL_pos": [0.9332, 0.252],
    "F": 96485.33212,
    "R": 8.314462,
    "T": 298.15
}

SOC_t0 = 1.
t_end = 1.
cs_ini_p = parameters["SOL_pos"][0] + ((parameters["SOL_pos"][1] - parameters["SOL_pos"][0]) * SOC_t0)
print(cs_ini_p)
I = -5
experiment = pybamm.Experiment(["Discharge at 1C for 10000 seconds or until 2.5 V"])
sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
sol = sim.solve(initial_soc=SOC_t0)

pos_SPM_r1 = sol['X-averaged positive particle concentration'].entries[-1, :]
pos_SPM_r0 = sol['X-averaged positive particle concentration'].entries[0, :]
t_pybamm = np.linspace(0, t_end, num=len(pos_SPM_r1))

print(np.power(parameters["R_p"], 2) * I / (
                3 * param["Positive electrode active material volume fraction"] *
                parameters["D_p"] * parameters["L_p"] * parameters["F"] *
                parameters["A"] * parameters["c_p_max"]))


# PDE
def cs(x, y):
    D = parameters["D_p"]
    R = parameters["R_p"]
    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_x = dde.grad.jacobian(y, x, j=0)
    N_xx = dde.grad.jacobian(dy_x * tf.reshape(tf.math.pow(x[:, 0], 2), (-1, 1)), x, j=0)
    return dy_t * tf.reshape(tf.math.pow(x[:, 0], 2), (-1,1)) / 3600. - D/np.power(R, 2) * N_xx
    # N_xx = dde.grad.hessian(y, x, j=0)
    # return dy_t - D / np.power(R, 2) * N_xx



geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, t_end)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


def bc_func(x):
    # return parameters["R_p"] * (-I / (np.power(x[:, 0], 2) * parameters["D_p"] * parameters["L_p"] * parameters["as_p"] * parameters["F"] * parameters["A"] * parameters["c_p_max"])) # parameters["R_p"] *
    return np.power(parameters["R_p"], 2) * I / (
                3 * param["Positive electrode active material volume fraction"] *
                parameters["D_p"] * parameters["L_p"] * parameters["F"] *
                parameters["A"] * parameters["c_p_max"])  # np.ones_like(x[:, 0]) *

def boundary_l(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)

def boundary_r(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)


bc_r = dde.icbc.NeumannBC(geomtime, bc_func, boundary_r)
bc_l = dde.icbc.NeumannBC(geomtime, lambda x: 0, boundary_l)
ic = dde.icbc.IC(geomtime, lambda _: cs_ini_p, lambda _, on_initial: on_initial)

data = dde.data.TimePDE(
    geomtime,
    cs,
    [bc_l, bc_r, ic],
    num_domain=1000,
    num_boundary=20,
    num_initial=20,
    num_test=100,
)

# # Function space
# func_space = dde.data.GRF(length_scale=0.2)
#
# # Data
# eval_pts = np.linspace(0, 1, num=50)[:, None]
# data = dde.data.PDEOperator(
#     pde, func_space, eval_pts, 1000, function_variables=[0], num_test=1000
# )

# Net
# net = dde.nn.DeepONet(
#     [50, 128, 128, 128],
#     [2, 128, 128, 128],
#     "tanh",
#     "Glorot normal",
# )

layer_size = [2] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=20000)
model.compile("L-BFGS-B")
# model.train_step.optimizer_kwargs = {'options': {'maxfun': 1e5, 'ftol': 1e-20, 'gtol': 1e-20, 'eps': 1e-20, 'iprint': -1, 'maxiter': 1e5}}
losshistory, train_state = model.train(callbacks=[dde.callbacks.EarlyStopping(patience=100)])
# dde.utils.plot_loss_history(losshistory)

dde.saveplot(losshistory, train_state, issave=False, isplot=True)
t = np.linspace(0, t_end, num=1000)
x = np.ones(1000)
a = np.array([x, t]).T
u = np.ravel(model.predict(a))

plt.figure()
plt.plot(t, u, "r", label="PINN")
plt.plot(t_pybamm, pos_SPM_r1, "g", label="Pybamm")
plt.xlabel("t [s]")
plt.ylabel("x [-]")
plt.legend()
plt.grid()
plt.show()

t = np.linspace(0, t_end, num=1000)
x = np.zeros(1000)
a = np.array([x, t]).T
u = np.ravel(model.predict(a))

plt.figure()
plt.plot(t, u, "r", label="PINN")
plt.plot(t_pybamm, pos_SPM_r0, "g", label="Pybamm")
plt.xlabel("t [s]")
plt.ylabel("x [-]")
plt.legend()
plt.grid()
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