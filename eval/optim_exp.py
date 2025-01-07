import numpy as np
import pybamm
from scipy import optimize as opt
from time import time
import warnings
warnings.filterwarnings("ignore")

def my_optim(x):
    dr_p = x[0]
    dr_n = x[1]

    # Run PyBaMM simulation
    param = pybamm.ParameterValues("Chen2020")
    param["Positive electrode active material volume fraction"] = dr_p
    param["Negative electrode active material volume fraction"] = dr_n
    if len(x) > 2:
        param["Positive electrode diffusivity [m2.s-1]"] = x[2]
        param["Negative electrode diffusivity [m2.s-1]"] = x[3]

    PBM_model = pybamm.lithium_ion.SPM()

    experiment = pybamm.Experiment(["Charge at 1C for 100000 seconds or until 4.2 V"])
    sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
    solution = sim.solve(initial_soc=0.)

    t_new = solution["Time [s]"].entries
    V_new = solution["Terminal voltage [V]"].entries

    # Calculate the objective function
    V_pbm = np.interp(t_new, t, V)

    # return the rmse of the voltage
    return np.sqrt(np.mean((V_pbm - V_new)**2))

if __name__=="__main__":

    dr_p = 1.1
    dr_n = 0.75
    diff_coef = 100.

    # Run PyBaMM simulation
    param = pybamm.ParameterValues("Chen2020")
    # Get initial parameters
    eps_init_p = param["Positive electrode active material volume fraction"]
    eps_init_n = param["Negative electrode active material volume fraction"]
    D_p = param["Positive electrode diffusivity [m2.s-1]"]
    D_n = param["Negative electrode diffusivity [m2.s-1]"]
    param["Positive electrode active material volume fraction"] *= dr_p
    param["Negative electrode active material volume fraction"] *= dr_n
    param["Positive electrode diffusivity [m2.s-1]"] *= diff_coef
    param["Negative electrode diffusivity [m2.s-1]"] *= diff_coef

    print("Initial parameters")
    print(eps_init_p, eps_init_n, D_p, D_n)

    print("Target parameters")
    print(param["Positive electrode active material volume fraction"], param["Negative electrode active material volume fraction"],
          param["Positive electrode diffusivity [m2.s-1]"], param["Negative electrode diffusivity [m2.s-1]"])
    print("")

    PBM_model = pybamm.lithium_ion.SPM()

    experiment = pybamm.Experiment(["Charge at 1C for 100000 seconds or until 4.2 V"])
    sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
    solution = sim.solve(initial_soc=0.)

    t = solution["Time [s]"].entries
    V = solution["Terminal voltage [V]"].entries

    methods = ['L-BFGS-B', 'Nelder-Mead', 'Powell', 'trust-constr', 'COBYLA', 'TNC']
    y0 = np.array([eps_init_p, eps_init_n, D_p, D_n])
    bounds = [(0.1, 10), (0.1, 10), (1e-20, 1e-10), (1e-20, 1e-10)]
    for method in methods:
        print("Method:", method)
        start = time()
        try:
            res = opt.minimize(my_optim, y0, method=method, bounds=bounds)
            print("Optimization took", time()-start, "seconds")
            print(res.x)
        except:
            print("Failed after", time()-start, "seconds")
    print("Starting optimization")
