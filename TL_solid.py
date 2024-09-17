from scr.SPMe import Solid_Phase, Cell
from scr.utils.pinn import FFNN, DeepONet, NN_TL_Diffusion
from scr.utils.sampling import Sampler, Sampler_DONet
from scr.utils.parameters import load_params
from scr.utils.profiles import zheng_current, constant

import pybamm
import torch
import numpy as np
import pickle

import matplotlib.pyplot as plt

np.set_printoptions(precision=3)


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))


if __name__ == "__main__":

    parameters = load_params()

    training_points = {"PDE": {"type": "PDE", "N": 1000},
                       "IV": {"type": "IV", "N": 100},
                       "BC_Center": {"type": "BC", "N": 100, "BC_pos": 0.},
                       "BC_Surf": {"type": "BC", "N": 100, "BC_pos": 1.}}
    validation_points = {"PDE": {"type": "PDE", "N": 30},
                         "IV": {"type": "IV", "N": 15},
                         "BC_Center": {"type": "BC", "N": 15, "BC_pos": 0.},
                         "BC_Surf": {"type": "BC", "N": 15, "BC_pos": 1.}}

    pos_weights = {"PDE": 1e4, "IV": 10., "BC_Center": 1., "BC_Surf": 1e4}
    neg_weights = {"PDE": 1e4, "IV": 10., "BC_Center": 1., "BC_Surf": 1e4}

    betas_tr = [0.2, 0.3, 0.5, 0.7, 0.75, 0.8, 0.85, 0.95, 1.1]
    betas_val = [0.4, 0.6]
    test_beta = 1.

    model_p = NN_TL_Diffusion(general_layers=[64, 64, 64, 64], fine_layers=[32, 32], dim_in=3, dim_int=32,
                            dim_out=1, dropout=0.)
    optimizer_p = torch.optim.Adam(model_p.parameters(), lr=0.0005)
    pos_model = Solid_Phase(model_p, parameters, criterion=RMSELoss, electrode="pos", weights=pos_weights)
    model_n = NN_TL_Diffusion(general_layers=[64, 64, 64, 64], fine_layers=[32, 32], dim_in=3, dim_int=32,
                            dim_out=1, dropout=0.)
    optimizer_n = torch.optim.Adam(model_n.parameters(), lr=0.0005)
    neg_model = Solid_Phase(model_n, parameters, criterion=RMSELoss, electrode="neg", weights=neg_weights)

    PINN = Cell(pos_model, neg_model)

    # PINN.neg_model.params["as_n"] = torch.nn.Parameter(torch.tensor([parameters["as_n"]]), requires_grad=False)

    Sampler_tr = Sampler(training_points, constant(1.), mode="uniform")
    Sampler_val = Sampler(validation_points, constant(1.), mode="uniform")

    history = {"positive":{"loss_tr": [], "losses_tr": [], "loss_val": [], "losses_val": [], "iteration": []},
               "negative":{"loss_tr": [], "losses_tr": [], "loss_val": [], "losses_val": [], "iteration": []}}

    print("Iter \t\t PDE \t IV \t BC Centre \t BC Surf \t\t\t PDE \t IV \t BC Centre \t BC Surf")
    for j in range(50000 + 1):

        Sampler_tr.update_current_func(constant(np.random.choice(betas_tr)))

        Sampler_val.update_current_func(constant(np.random.choice(betas_val)))

        loss_tr, losses_tr = PINN.pos_model.train_step(optimizer_p, Sampler_tr)
        loss_val, losses_val = PINN.pos_model.evaluate(Sampler_val)

        if j % 1000 == 0:
            print(j, "P\t\t", losses_tr, "\t\t", losses_val)
            history["positive"]["loss_tr"].append(loss_tr)
            history["positive"]["losses_tr"].append(losses_tr)
            history["positive"]["loss_val"].append(loss_val)
            history["positive"]["losses_val"].append(losses_val)
            history["positive"]["iteration"].append(j)

        loss_tr, losses_tr = PINN.neg_model.train_step(optimizer_n, Sampler_tr)
        loss_val, losses_val = PINN.neg_model.evaluate(Sampler_val)

        if j % 1000 == 0:
            print(j, "N\t\t", losses_tr, "\t\t", losses_val)
            history["negative"]["loss_tr"].append(loss_tr)
            history["negative"]["losses_tr"].append(losses_tr)
            history["negative"]["loss_val"].append(loss_val)
            history["negative"]["losses_val"].append(losses_val)
            history["negative"]["iteration"].append(j)

    PINN.pos_model.save_model("models/TL_pos.pt")
    PINN.neg_model.save_model("models/TL_neg.pt")

    with open('models/TL.pkl', 'wb') as fp:
        pickle.dump(history, fp)

    t_eval = np.arange(0, 3600)
    cur_fun = constant(test_beta)

    bcs_sample_t = torch.linspace(0., 1., 1000)
    bcs_sample_r = torch.ones_like(bcs_sample_t)
    bcs_sample_I = torch.tensor(cur_fun(bcs_sample_t.numpy() * 3600.))
    bcs_sample = torch.stack([bcs_sample_t, bcs_sample_r, bcs_sample_I]).t()

    V_pinn = PINN.compute_V(bcs_sample).detach().numpy()


    param = pybamm.ParameterValues("Chen2020")
    PBM_model = pybamm.lithium_ion.SPM()

    experiment = pybamm.Experiment(["Discharge at 1C for 100000 seconds or until 2.5 V"])
    sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
    solution = sim.solve(initial_soc=1)

    t = solution["Time [s]"].entries

    V = solution["Terminal voltage [V]"].entries

    plt.figure()
    plt.grid()
    plt.plot(t / 3600. * 5., V, "k-")
    plt.plot(bcs_sample_t * 5., V_pinn, "r-")
    plt.ylim([2.5, 4.3])
    plt.xlabel("Disch. Capacity [Ah]")
    plt.ylabel("V [V]")
    plt.show()