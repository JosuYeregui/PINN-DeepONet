import torch
import pybamm
import numpy as np

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
