import torch
import pybamm
import numpy as np
from sklearn import gaussian_process as gp

import matplotlib.pyplot as plt


def zheng_current(beta):
    """
    Variable current function for a given time array. Consists on an hour long test with
    Based on the work of Zheng et.al.
    :param beta:
    :return: Current profile function for a given t array sequence.
    """
    def current(t):
        h = np.ones_like(t.copy(), dtype=np.float32)
        h[t < 1800] = 1 / 1800. * t[t < 1800]
        h[t >= 1800] = -1 / 1800. * t[t >= 1800] + 2
        return -h * beta + 1

    return current

def constant(crate):
    """
    Constant current function for a given time array.
    :param crate: Current C-rate
    :return: Current profile function for a given t array sequence.
    """
    def current(t):
        h = np.ones_like(t.copy(), dtype=np.float32)
        return h * crate

    return current

def grf(T, N, length_scale=1, mean=1, variance=0.25):
    """
    Gaussian random field for a given temperature profile.
    :param T: Temperature profile
    :return: Gaussian random field
    """
    x = np.linspace(0, T, num=N)[:, None]
    K = gp.kernels.RBF(length_scale=length_scale)
    K = K(x)
    L = np.linalg.cholesky(K + 1e-13 * np.eye(N))
    def current(t):
        u = np.random.randn(N)
        return mean + np.sqrt(variance) * np.dot(L, u*T).T

    return current

# convert above grf function to class for better handling
class GRF:
    def __init__(self, T, N, length_scale, mean=1, variance=0.25):
        self.mean = mean
        self.variance = variance
        self.N = N
        self.T = T

        self.update(T, N, length_scale)

    def update(self, T, N, length_scale):
        x = np.linspace(0, T, num=N)[:, None]
        K = gp.kernels.RBF(length_scale=length_scale)
        K = K(x)
        self.L = np.linalg.cholesky(K + 1e-13 * np.eye(N))

        self.N = N
        self.T = T

        self.update_u()

    def update_u(self):
        x = np.random.randn(self.N)
        self.u = self.mean + np.sqrt(self.variance) * np.dot(self.L, x*self.T).T

    def __call__(self, t):
        return np.interp(t, np.linspace(0, self.T, self.N), self.u).astype(np.float32)


# TODO: Add a UDDS variable profile


if __name__ == "__main__":

    param = pybamm.ParameterValues("Chen2020")

    t_eval = np.arange(0, 3600)
    cur_fun = zheng_current(1.)

    plt.figure()
    plt.grid(True)
    plt.plot(t_eval, cur_fun(t_eval), "b")
    plt.xlabel("t [h]")
    plt.ylabel("I [A]")
    plt.show()

    current_interpolant = pybamm.Interpolant(t_eval, cur_fun(t_eval)*5., pybamm.t)
    param["Current function [A]"] = current_interpolant

    PBM_model = pybamm.lithium_ion.SPM()
    sim = pybamm.Simulation(PBM_model, parameter_values=param)
    sol = sim.solve(initial_soc=1., t_eval=t_eval)

    c_s_n = sol["Negative particle concentration"]
    c_s_p = sol["Positive particle concentration"]
    r_n = sol["r_n [m]"].entries[:, 0, 0]
    r_p = sol["r_p [m]"].entries[:, 0, 0]
    t = sol["Time [s]"].entries
    x = sol["x [m]"].entries[:, 0]

    pos_SPM_r0 = c_s_p(r=r_p[0], t=t, x=x[-1])
    pos_SPM_r1 = c_s_p(r=r_p[-1], t=t, x=x[-1])

    neg_SPM_r0 = c_s_n(r=r_n[0], t=t, x=x[0])
    neg_SPM_r1 = c_s_n(r=r_n[-1], t=t, x=x[0])

    plt.figure()
    plt.grid("on")
    plt.plot(t, pos_SPM_r1, "r", label="Pybamm")
    plt.legend()
    plt.xlabel("t [h]")
    plt.ylabel("x [-]")
    plt.show()
