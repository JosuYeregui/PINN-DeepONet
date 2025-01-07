import numpy as np
import pybamm
from scipy import optimize as opt
from time import time
import os
import warnings
warnings.filterwarnings("ignore")

def load_excel(file, sheet_name="Channel-7_1"):
    import pandas as pd
    data = pd.read_excel(file, sheet_name=sheet_name)
    # Filter the data to take only the ones with Step Index = 6 (CHG) or 10 (DCH)
    data = data[data["Step Index"] == 6]
    return data["Voltage (V)"].values, data["Step Time (s)"].values
    # return data["Voltage (V)"].values[:-100], data["Step Time (s)"].values[:-100]

def my_optim(x):

    param["Positive electrode diffusivity [m2.s-1]"] = x[0]
    param["Negative electrode diffusivity [m2.s-1]"] = x[1]

    PBM_model = pybamm.lithium_ion.SPM()

    experiment = pybamm.Experiment(["Charge at 0.3C for 100000 seconds or until 4.2 V"])
    sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
    solution = sim.solve(initial_soc=0.)

    t_new = solution["Time [s]"].entries
    V_new = solution["Terminal voltage [V]"].entries

    # Calculate the objective function
    V_pbm = np.interp(t_new, t, V)

    # return the rmse of the voltage
    return np.sqrt(np.mean((V_pbm - V_new)**2))

if __name__=="__main__":

    # Run PyBaMM simulation
    param = pybamm.ParameterValues("Chen2020")
    # Get initial parameters
    eps_init_p = param["Positive electrode active material volume fraction"]
    eps_init_n = param["Negative electrode active material volume fraction"]
    D_p = param["Positive electrode diffusivity [m2.s-1]"]
    D_n = param["Negative electrode diffusivity [m2.s-1]"]
    param["Positive electrode active material volume fraction"] *= 7.555631166180735/8.7323# (1 - 0.058756) * 7.555631166180735/8.7323
    param["Negative electrode active material volume fraction"] *= 6.014505825096271/5.8276# (1 - 0.26054) * 6.014505825096271/5.8276

    PBM_model = pybamm.lithium_ion.SPM()

    V, t = load_excel(os.path.join("..\\Data\\BoL_PINN_Channel_7_Wb_1.xlsx"), sheet_name="Channel-7_1")

    experiment = pybamm.Experiment(["Charge at 1C for 100000 seconds or until 4.2 V"])
    sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
    solution = sim.solve(initial_soc=0.)

    methods = ['L-BFGS-B', 'Nelder-Mead', 'Powell',]# 'trust-constr', 'COBYLA', 'TNC']
    y0 = np.array([D_p, D_n])
    bounds = [(1e-20, 1e-10), (1e-20, 1e-10)]
    for method in methods:
        print("Method:", method)
        start = time()
        try:
            res = opt.minimize(my_optim, y0, method=method, bounds=bounds)
            if method == "Nelder-Mead":
                res_opt = res
            print("Optimization took", time()-start, "seconds")
            print(res.x)
            print("Optimization error:", res.fun)
        except:
            print("Failed after", time()-start, "seconds")
    print("Starting optimization")

    # plot results with Nelder-Mead method
    param["Positive electrode diffusivity [m2.s-1]"] = res_opt.x[0]
    param["Negative electrode diffusivity [m2.s-1]"] = res_opt.x[1]

    experiment = pybamm.Experiment(["Charge at 0.3C for 100000 seconds or until 4.2 V"])
    sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
    solution = sim.solve(initial_soc=0.)

    t_new = solution["Time [s]"].entries
    V_new = solution["Terminal voltage [V]"].entries

    experiment = pybamm.Experiment(["Charge at 1C for 100000 seconds or until 4.2 V"])
    sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
    solution = sim.solve(initial_soc=0.)

    t_new_1c = solution["Time [s]"].entries
    V_new_1c = solution["Terminal voltage [V]"].entries

    import matplotlib.pyplot as plt
    plt.plot(t_new/3600. * 0.3 * 5., V_new, label="Simulated")
    plt.plot(t/3600. * 0.3 * 5., V, label="Experimental")
    plt.plot(t_new_1c/3600. * 5., V_new_1c, label="Simulated 1C")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.legend()
    plt.show()
