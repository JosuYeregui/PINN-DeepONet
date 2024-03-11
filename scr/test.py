"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
# Backend tensorflow.compat.v1 or tensorflow
from deepxde.backend import tf
# Backend pytorch
# import torch
# Backend jax
# import jax.numpy as jnp
# Backend paddle
# import paddle


def pde(x, y):
    # Most backends
    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_xx = dde.grad.hessian(y, x, j=0)
    # dy_x = dde.grad.jacobian(y, x, j=0)
    # Backend tensorflow.compat.v1 or tensorflow
    return (
        dy_t / 3600
        - dy_xx
        + tf.exp(-x[:, 1:]*3600)
        * (tf.sin(np.pi * x[:, 0]) - np.pi ** 2 * tf.sin(np.pi * x[:, 0]))
    )
    # Backend pytorch
    # return (
    #     dy_t
    #     - dy_xx
    #     # + torch.exp(-x[:, 1:])
    #     # * (torch.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * torch.sin(np.pi * x[:, 0:1]))
    # )
    # Backend jax
    # return (
    #     dy_t
    #     - dy_xx
    #     + jnp.exp(-x[:, 1:])
    #     * (jnp.sin(np.pi * x[..., 0:1]) - np.pi ** 2 * jnp.sin(np.pi * x[..., 0:1]))
    # )
    # Backend paddle
    # return (
    #     dy_t
    #     - dy_xx
    #     + paddle.exp(-x[:, 1:])
    #     * (paddle.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * paddle.sin(np.pi * x[:, 0:1]))
    # )
    # D = 1e-14
    # R = 1e-6
    # dy_t = dde.grad.jacobian(y, x, j=1)
    # dy_x = dde.grad.jacobian(y, x, j=0)
    # N_xx = dde.grad.jacobian(dy_x * torch.pow(x[:, 0], 2).view(-1, 1), x, j=0)
    # return dy_t * torch.pow(x[:, 0], 2).view(-1, 1) - D/np.power(R, 2) * N_xx

t_end = 3600.
def func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])

def bc_func(x):
    return -0.

def boundary_l(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)

def boundary_r(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, t_end)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc_r = dde.icbc.NeumannBC(geomtime, bc_func, boundary_r)
bc_l = dde.icbc.NeumannBC(geomtime, lambda x: 0, boundary_l)

bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)
# ic = dde.icbc.IC(geomtime, lambda _: 0.398, lambda _, on_initial: on_initial)
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=3600,
    num_boundary=20,
    num_initial=10,
    # solution=func,
    num_test=10000,
    train_distribution="pseudo"
)

layer_size = [2] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

model.compile("adam", lr=0.01)
losshistory, train_state = model.train(iterations=10000)

dde.saveplot(losshistory, train_state, issave=False, isplot=True)

# dde.saveplot(losshistory, train_state, issave=True, isplot=True)
t = np.linspace(0, t_end, num=1000)
x = np.ones(1000)
a = np.array([x, t]).T
u = np.ravel(model.predict(a))

plt.figure()
plt.plot(t, u, "r")
plt.show()