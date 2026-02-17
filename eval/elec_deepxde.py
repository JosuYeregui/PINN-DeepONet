"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import torch
import deepxde as dde
import numpy as np
# Backend tensorflow.compat.v1 or tensorflow
# Backend pytorch

# Backend jax
# import jax.numpy as jnp
# Backend paddle
# import paddle

# Force cpu execution
torch.set_default_device("cpu")


def pde(x, y):
    # Most backends
    i_app = 1. * 5. / 0.10270000000000001

    # dcdx = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
    #                            create_graph=True)[0]
    dcdt = dde.grad.jacobian(y, x, j=1)
    dcdr = dde.grad.jacobian(y, x, j=0)

    L_n = 8.52e-05 / 0.0001728
    L_s = 1.2e-05 / 0.0001728

    idx_Ln = x[:, 0] < L_n
    idx_Lp = x[:, 0] >= L_n# + L_s
    #idx_Ls = (L_n <= x[:, 0]) & (x[:, 0] <= L_n + L_s)

    left = dcdt * 1000. / 3600. * 0.25# * self.params["por_n"]
    # left[idx_Ln] *= 0.25
    # left[idx_Ls] *= 0.47
    left[idx_Lp] *= 0.15 / 0.25
    # left = torch.ones_like(c)

    De_eff = torch.ones_like(x[:, 0].detach()) * 1.7694e-10 #* 0.25 ** 1.5
    # De_eff = self.params["D_e"](c)
    # De_eff[idx_Ln] *= np.sqrt(0.25) # Arazoa hemen dago
    # De_eff[idx_Ls] *= np.sqrt(0.47)#(0.47 ** .5)
    De_eff[idx_Lp] *= np.power(0.15, 1.5)#(0.335 ** .5)
    De_eff[idx_Ln] *= np.power(0.25, 1.5) # Arazoa hemen dago
    #De_eff[idx_Ls] *= np.power(0.25, 1.5)#(0.47 ** .5)
    # De_eff[idx_Lp] *= np.power(0.25, 1.5)#(0.335 ** .5)
    # right = torch.autograd.grad(De_eff * dcdx[:, 1], x, grad_outputs=torch.ones_like(dcdx[:, 1]),
    #                             create_graph=True)[0][:, 1]
    # right = De_eff * dde.grad.hessian(y, x, j=0)
    right = dde.grad.jacobian(De_eff * dcdr, x, j=0)

    right *= 1000. / 0.0001728 ** 2

    # right = torch.zeros_like(y)

    right[idx_Ln] += (i_app / (96485.33212 * 8.52e-05) * (1 - 0.2594)) #/ 0.25
    right[idx_Lp] -= (i_app / (96485.33212 * 7.56e-05) * (1 - 0.2594)) #/ 0.335

    #right[idx_Ln] /= 0.25
    #right[idx_Ls] /= 0.47
    #right[idx_Lp] /= 0.335
    #right[idx_Ln] /= 0.25
    #right[idx_Ls] /= 0.25
    #right[idx_Lp] /= 0.25
    return (left - right)


geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.NeumannBC(geomtime, lambda x: 0., lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, lambda x: 1., lambda _, on_initial: on_initial)

data = dde.data.TimePDE(geomtime, pde, [bc], num_domain=1000, num_boundary=50)

layer_size = [2] + [32] * 4 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
net.apply_output_transform(
    # Backend tensorflow.compat.v1 or tensorflow
    lambda x, y:  x[:, 1:] * y + 1.#(-1. * x[:,0:1] + 1.5)
    # Backend pytorch
    # lambda x, y: x[:, 1:2] * (1 - x[:, 0:1] ** 2) * y + torch.sin(np.pi * x[:, 0:1])
    # Backend jax
    # lambda x, y: x[..., 1:2] * (1 - x[..., 0:1] ** 2) * y + jnp.sin(np.pi * x[..., 0:1])
    # Backend paddle
    # lambda x, y: x[:, 1:2] * (1 - x[:, 0:1] ** 2) * y + paddle.sin(np.pi * x[:, 0:1])
)

model = dde.Model(data, net)

model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=202000)

dde.saveplot(losshistory, train_state, issave=False, isplot=True)

# plot at t = 1
x = np.linspace(0, 1, 100)
t = np.ones_like(x) * 1.
x = np.stack([x, t]).T
y = model.predict(x)

# Get pybamm reference solution
import pybamm

param = pybamm.ParameterValues("Chen2020")
param['Electrolyte diffusivity [m2.s-1]'] = lambda c, T:  1.7694e-10  # 4.862e-10 #  8.794e-11 * (c*1000)**2 - 3.972e-10 * (c*1000)**2 + 4.862e-10#
# param['Electrolyte diffusivity [m2.s-1]'] = lambda c, T:  8.794e-11 * (c/1000)**2 - 3.972e-10 * (c/1000) + 4.862e-10
param['Positive electrode porosity'] = 0.15
param['Negative electrode porosity'] = 0.25
param['Separator porosity'] = 0.25
PBM_model = pybamm.lithium_ion.SPMe()
experiment = pybamm.Experiment(["Discharge at 1C for 100000 seconds or until 2.5 V"])
sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
sol = sim.solve(initial_soc=1)

ce = sol["Electrolyte concentration [mol.m-3]"]

# pos_SPM_r0 = sol['X-averaged positive particle concentration'].entries[0, :]
# pos_SPM_r1 = sol['X-averaged positive particle concentration'].entries[-1, :]

t = sol["Time [s]"].entries
pos = sol["x [m]"].entries[:, 0]

ce_t0 = ce(t=t[0], x=pos) /1000
ce_tend = ce(t=t[-1], x=pos) /1000

t_end = t[-1]

import matplotlib.pyplot as plt
plt.plot(x[:, 0], y, "r", label="PINN")
plt.plot(pos/0.0001728, ce_tend, "k", label="Pybamm")
plt.xlabel("x")
plt.ylabel("c")
plt.title("t = 1")
plt.grid()
plt.legend()
plt.show()

num = 1000

t_test = (np.arange(0, num) / num)[::10] #* t_end
r_rand = (np.arange(0, num) / num)[::10] #* parameters["L"]

t_new, r_new = np.meshgrid(t_test, r_rand)
res = np.zeros_like(t_new)

for i, (t, r) in enumerate(zip(t_new, r_new)):
    res[i, :] = model.predict(np.stack([r, t]).T).reshape(-1)


def plot_area(points, style, cmap_label):
    plt.subplots(figsize=(7, 3), tight_layout=True)
    plot = plt.pcolormesh(t_new, r_new, points, cmap=style, shading='gouraud')
    cbar = plt.colorbar(plot)
    plt.contour(t_new, r_new, points, 10, colors='gray')
    plt.ylabel('$x$')
    plt.xlabel('$t$ [h]')
    cbar.set_label(cmap_label)
    # plt.savefig('/content/drive/MyDrive/Datos/con_PINN_pos.png')
    plt.show()

plot_area(res, 'RdBu_r', 'res')