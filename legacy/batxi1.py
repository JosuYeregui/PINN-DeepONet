from scr.utils import load_params
import matplotlib.pyplot as plt

import pybamm
import numpy as np


def pde(x, c, t, C_rate, params):

    i_app = C_rate * params["I_typ"] / params["A"]
    tc = 1 / C_rate * 3600.
    t = t/tc

    dcdx = np.transpose( np.transpose(np.diff(c, axis=0)) / np.diff(x) )
    dcdx = np.concatenate([dcdx, dcdx[-1,:].reshape(1,-1)], axis=0)
    dcdt = np.diff(c, axis=-1) / np.diff(t)
    dcdt = np.concatenate([dcdt, dcdt[:,1].reshape(-1,1)], axis=1)

    L_n = params["L_n"] / params["L"]
    L_s = params["L_s"] / params["L"]

    idx_Ln = x < L_n
    idx_Lp = x > L_n + L_s
    idx_Ls = (L_n <= x) & (x <= L_n + L_s)

    left = dcdt * params["ce0"] / tc  # * self.params["por_n"]
    left[idx_Ln] *= params["por_n"]
    left[idx_Ls] *= params["por_s"]
    left[idx_Lp] *= params["por_p"]

    dNdx = dcdx * params["D_e_const"] * params["ce0"] / params["L"]
    dNdx[idx_Ln] *= params["por_n"] ** params["brug"]
    dNdx[idx_Ls] *= params["por_s"] ** params["brug"]
    dNdx[idx_Lp] *= params["por_p"] ** params["brug"]
    right = np.transpose( np.transpose(np.diff(dNdx, axis=0)) / np.diff(x) )
    right = np.concatenate([right, right[-1,:].reshape(1,-1)], axis=0) / params["L"]
    right[idx_Ln] += i_app / (params["F"] * params["L_n"]) * (1 - params["t_plus"])
    right[idx_Lp] -= i_app / (params["F"] * params["L_p"]) * (1 - params["t_plus"])

    return left - right

def pde_ns(x, c, t, C_rate, params, Ne):

    i_app = C_rate * params["I_typ"] / params["A"]

    dcdx = np.transpose(np.transpose(np.diff(c, axis=0)) / np.diff(x))
    dcdx = np.concatenate([dcdx, dcdx[-1,:].reshape(1,-1)], axis=0)
    dcdt = np.diff(c, axis=-1) / np.diff(t)
    dcdt = np.concatenate([dcdt, dcdt[:,-1].reshape(-1,1)], axis=1)

    L_n = params["L_n"]
    L_s = params["L_s"]

    idx_Ln = x < L_n
    idx_Lp = x > L_n + L_s
    idx_Ls = (L_n <= x) & (x <= L_n + L_s)

    left = dcdt.copy()  # * self.params["por_n"]
    left[idx_Ln, :] *= params["por_n"]
    left[idx_Ls, :] *= params["por_s"]
    left[idx_Lp, :] *= params["por_p"]

    dNdx = dcdx.copy() * params["D_e_const"]
    dNdx[idx_Ln, :] *= params["por_n"] ** params["brug"]
    dNdx[idx_Ls, :] *= params["por_s"] ** params["brug"]
    dNdx[idx_Lp, :] *= params["por_p"] ** params["brug"]
    right = np.transpose(np.transpose(np.diff(dNdx, axis=0)) / np.diff(x))
    right = np.concatenate([right, right[-1, :].reshape(1, -1)], axis=0)
    right[idx_Ln, :] += i_app / (params["F"] * params["L_n"]) * (1 - params["t_plus"])
    right[idx_Lp, :] -= i_app / (params["F"] * params["L_p"]) * (1 - params["t_plus"])

    N = -dNdx.copy()
    N[idx_Ln, :] += np.transpose(np.tile(x[idx_Ln] * i_app / (params["F"] * params["L_n"]) * (params["t_plus"]), (N.shape[1], 1)))
    N[idx_Ls, :] += np.transpose(np.tile(i_app / (params["F"]) * (params["t_plus"]), (N.shape[1], 1)))
    N[idx_Lp, :] += np.transpose(np.tile((params["L"] - x[idx_Lp]) * i_app / (params["F"] * params["L_p"]) * (params["t_plus"]), (N.shape[1], 1)))

    right_b = np.transpose(np.transpose(np.diff(N, axis=0)) / np.diff(x))
    right_b = - np.concatenate([right_b, right_b[-1, :].reshape(1, -1)], axis=0)
    right_b[idx_Ln, :] += i_app / (params["F"] * params["L_n"])
    right_b[idx_Lp, :] -= i_app / (params["F"] * params["L_p"])

    right_b[idx_Ln, :] *= 1/params["por_n"]
    right_b[idx_Ls, :] *= 1/params["por_s"]
    right_b[idx_Lp, :] *= 1/params["por_p"]

    res_sum = 1. * np.cumsum(right_b, axis=1) + 1000.

    dN = np.transpose(np.transpose(np.diff(N, axis=0)) / np.diff(x))
    dN = - np.concatenate([dN, dN[-1, :].reshape(1, -1)], axis=0)
    dNe = np.transpose(np.transpose(np.diff(Ne, axis=0)) / np.diff(x))
    dNe = - np.concatenate([dNe, dNe[-1, :].reshape(1, -1)], axis=0)

    result = left - right

    return result


def pde_ns_new(x, c, t, C_rate, params, Ne):

    i_app = C_rate * params["I_typ"] / params["A"]

    dcdx = np.transpose(np.transpose(np.diff(c, axis=0)) / np.diff(x))
    dcdx = np.concatenate([dcdx, dcdx[-1,:].reshape(1,-1)], axis=0)
    dcdt = np.diff(c, axis=-1) / np.diff(t)
    dcdt = np.concatenate([dcdt, dcdt[:,-1].reshape(-1,1)], axis=1)

    dccdxx = np.transpose(np.transpose(np.diff(dcdx, axis=0)) / np.diff(x))
    dccdxx = np.concatenate([dccdxx, dccdxx[-1, :].reshape(1, -1)], axis=0)

    L_n = params["L_n"]
    L_s = params["L_s"]

    idx_Ln = x < L_n
    idx_Lp = x > L_n + L_s
    idx_Ls = (L_n <= x) & (x <= L_n + L_s)

    left = dcdt.copy()  # * self.params["por_n"]
    left[idx_Ln, :] *= params["por_n"]
    left[idx_Ls, :] *= params["por_s"]
    left[idx_Lp, :] *= params["por_p"]

    right = dccdxx.copy()
    right[idx_Ln, :] *= (params["por_n"] ** params["brug"]) * params["D_e_const"]
    right[idx_Ls, :] *= (params["por_s"] ** params["brug"]) * params["D_e_const"]
    right[idx_Lp, :] *= (params["por_p"] ** params["brug"]) * params["D_e_const"]

    right[idx_Ln, :] += i_app / (params["F"] * params["L_n"]) * (1 - params["t_plus"])
    right[idx_Lp, :] -= i_app / (params["F"] * params["L_p"]) * (1 - params["t_plus"])

    result = left - right

    return result


if __name__ == "__main__":

    parameters = load_params()

    test_C = 1.

    param = pybamm.ParameterValues("Chen2020")
    param['Electrolyte diffusivity [m2.s-1]'] = lambda c, T: 4.862e-10
    PBM_model = pybamm.lithium_ion.SPMe()
    experiment = pybamm.Experiment(["Discharge at " + str(test_C) + "C for 100000 seconds or until 2.5 V"])
    # experiment = pybamm.Experiment(["Rest for 4000 seconds"])
    sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
    sol = sim.solve(initial_soc=1.)

    ce = sol["Electrolyte concentration [mol.m-3]"]
    Ne = sol["Electrolyte flux [mol.m-2.s-1]"]
    tor = sol["Electrolyte transport efficiency"]

    t = sol["Time [s]"].entries[:-1]
    t_test = np.arange(0, t[-1], 1)
    x = sol["x [m]"].entries[:, 0]
    x_test = np.arange(0, x[-1], 1e-7)
    ce_tend = ce(t=t_test, x=x)
    Ne_tend = Ne(t=t_test, x=x)

    # result = pde(x/parameters["L"], ce_tend/parameters["ce0"], t_test, test_C, parameters)
    result = pde_ns_new(x, ce_tend, t_test, test_C, parameters, Ne_tend)
    print(result)


