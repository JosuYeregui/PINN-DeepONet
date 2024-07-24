from scr.utils import load_params
from scr.profiles import zheng_current
import matplotlib.pyplot as plt

import pybamm
import numpy as np

if __name__ == "__main__":

    test_beta = 1.

    parameters = load_params()

    param = pybamm.ParameterValues("Chen2020")

    t_eval = np.arange(0, 3600)
    # cur_fun = zheng_current(test_beta)
    cur_fun = zheng_current(test_beta)

    current_interpolant = pybamm.Interpolant(t_eval, cur_fun(t_eval) * parameters["I_typ"], pybamm.t)
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

    c = c_s_p(r=r_p, t=t, x=x[-1])

    dcdx = np.transpose(np.transpose(np.diff(c, axis=0)) / np.diff(r_p))
    dcdx = np.concatenate([dcdx, dcdx[-1, :].reshape(1, -1)], axis=0)
    dcdx *= parameters["R_p"]
    dcdt = np.diff(c, axis=-1) / np.diff(t)
    dcdt = np.concatenate([dcdt, dcdt[:, -1].reshape(-1, 1)], axis=1)
    dcdt *= 3600.

    def plot_area(points, style, cmap_label):
        plt.subplots(figsize=(7, 3), tight_layout=True)
        plot = plt.pcolormesh(t/3600, r_p/parameters["R_p"], points, cmap=style, shading='gouraud')
        cbar = plt.colorbar(plot)
        plt.contour(t/3600, r_p/parameters["R_p"], points, 10, colors='gray')
        plt.ylabel('$x$')
        plt.xlabel('$t$ [h]')
        cbar.set_label(cmap_label)
        # plt.savefig('/content/drive/MyDrive/Datos/con_PINN_pos.png')
        plt.show()

    plot_area(dcdt, 'Oranges', 'dcdt')
    plot_area(dcdx, 'Oranges', 'dcdr')

    i_app = - 1. * parameters["I_typ"] / parameters["A"]

    j = - i_app / (parameters["as_p"] * parameters["L_p"] * parameters["F"])

    print(j * parameters["R_p"] / parameters["c_p_max"] / parameters["D_p"])
