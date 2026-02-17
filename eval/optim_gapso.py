import numpy as np
import pybamm
from scipy import optimize as opt
from time import time
import warnings

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback

warnings.filterwarnings("ignore")


class BestPopulationCallback(Callback):
    def __init__(self, n_best=3):
        super().__init__()
        self.n_best = n_best
        self.data["best"] = []
        self.history = {"x": [], "fitness": []}

    def notify(self, algorithm):
        pop = algorithm.pop
        F = pop.get("F").flatten()

        sorted_indices = np.argsort(F)
        best_indices = sorted_indices[:self.n_best]

        print(f"\n--- Generation: {algorithm.n_gen} ---")
        print("Best Individuals in Population:")

        for i, idx in enumerate(best_indices):
            print(f"  Rank {i + 1}:")
            print(f"    Variables (X) : {pop[idx].get('X')}")
            print(f"    Objective (F) : {pop[idx].get('F').round(6)}")

        self.data["best"].append(F[sorted_indices[0]])

        self.history["x"].append(pop.get("X")[sorted_indices[0]])
        self.history["fitness"].append(F[sorted_indices[0]])
        print("It from", time() - start, "seconds")

class MyProblem(ElementwiseProblem):

    def __init__(self, n_var=4, n_obj=1, xl=None, xu=None):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
    def _evaluate(self, x, out, *args, **kwargs):
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
        out["F"] = np.sqrt(np.mean((V_pbm - V_new)**2))

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

    methods = [PSO]
    history = {}
    my_callback = BestPopulationCallback(n_best=1)
    y0 = np.array([eps_init_p, eps_init_n, D_p, D_n])
    bounds = [(0.1, 10), (0.1, 10), (1e-20, 1e-10), (1e-20, 1e-10)]
    for method in methods:
        print("Method:", method.__name__)
        start = time()
        algorithm = method(pop_size=50)
        #try:
        res = minimize(MyProblem(n_var=len(y0), n_obj=1, xl=[b[0] for b in bounds], xu=[b[1] for b in bounds]),
                       algorithm,
                       ('n_gen', 200),
                       verbose=False,
                       callback=my_callback)
        print("Optimization took", time()-start, "seconds")

        print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

        history[method.__name__] = my_callback.history
        #except:
        #    print("Failed after", time()-start, "seconds")
        # Comppute rms
        tot_error = 0
        for p, t in zip(res.X, [eps_init_p, eps_init_n, D_p, D_n]):
            tot_error += np.power(np.abs(p - t), 2)

        tot_error = np.sqrt(tot_error)
        print("Total error: ", tot_error*100.)

    # Save the history of the best solutions
    # import pickle
    # with open("optim_history_both.pkl", "wb") as f:
    #     pickle.dump(history, f)
