from scr.SPMe import Solid_Phase, Cell
from scr.utils.pinn import FFNN, DeepONet, NN_TL_Diffusion
from scr.utils.sampling import Sampler, Sampler_DONet
from scr.utils.parameters import load_params
from scr.utils.profiles import zheng_current, constant
from scr.utils.optimizers import NTK_Adaptive, Adam_Custom

import pybamm
import torch
import numpy as np
import pickle
import time

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

    pos_weights = {"PDE": 1e4, "IV": 10., "BC_Center": 1., "BC_Surf": 1e5}
    neg_weights = {"PDE": 1e4, "IV": 10., "BC_Center": 1., "BC_Surf": 1e5}

    betas_tr = [0.2, 0.3, 0.5, 0.7, 0.75, 0.8, 0.85, 0.95, 1.1]
    betas_val = [0.4, 0.6]
    test_beta = 1.

    model_p = NN_TL_Diffusion(general_layers=[64, 64, 64, 64], fine_layers=[32, 32], dim_in=3, dim_int=32,
                              dim_out=1, dropout=0.)
    pos_model = Solid_Phase(model_p, parameters, [1., 1., 1.], criterion=RMSELoss, electrode="pos", weights=pos_weights, adjustable_weights=False)
    optimizer_p = NTK_Adaptive(pos_model.model.parameters(), pos_model.weigths, adam_param = {'lr': 0.0005, 'betas': (0.9, 0.999)})
    # optimizer_p = torch.optim.Adam(pos_model.model.parameters(), lr=0.0005)
    # optimizer_p_w = torch.optim.Adam([pos_model.adj_w], lr=0.0001)
    model_n = NN_TL_Diffusion(general_layers=[64, 64, 64, 64], fine_layers=[32, 32], dim_in=3, dim_int=32,
                              dim_out=1, dropout=0.)
    neg_model = Solid_Phase(model_n, parameters,[1., 1., 1.], criterion=RMSELoss, electrode="neg", weights=neg_weights, adjustable_weights=False)
    optimizer_n = NTK_Adaptive(neg_model.model.parameters(), neg_model.weigths, adam_param = {'lr': 0.0005, 'betas': (0.9, 0.999)})

    # optimizer_n = torch.optim.Adam(neg_model.model.parameters(), lr=0.0005)
    # optimizer_n_w = torch.optim.Adam([neg_model.adj_w], lr=0.0001)

    PINN = Cell(pos_model, neg_model)

    # PINN.neg_model.params["as_n"] = torch.nn.Parameter(torch.tensor([parameters["as_n"]]), requires_grad=False)

    Sampler_tr = Sampler(training_points, constant(1.), mode="quasi")
    Sampler_val = Sampler(validation_points, constant(1.), mode="quasi")

    history = {"positive":{"loss_tr": [], "losses_tr": [], "loss_val": [], "losses_val": [], "iteration": []},
               "negative":{"loss_tr": [], "losses_tr": [], "loss_val": [], "losses_val": [], "iteration": []}}

    print("Iter \t\t PDE \t IV \t BC Centre \t BC Surf \t\t\t PDE \t IV \t BC Centre \t BC Surf")
    for j in range(50000 + 1):
        t = time.time()
        rate_tr = np.random.choice(betas_tr)
        rate_val = np.random.choice(betas_val)

        # Sampler_tr.update_current_func(constant(rate_tr))
        # Sampler_tr.update_t(rate_tr)
        # Sampler_val.update_current_func(constant(rate_val))
        # Sampler_val.update_t(rate_val)

        Sampler_tr.update_samples(training_points, constant(rate_tr), 1. / rate_tr)
        Sampler_val.update_samples(validation_points, constant(rate_val), 1. / rate_val)

        # optimizer_p_w.zero_grad()
        # optimizer_n_w.zero_grad()

        losses_tr = PINN.pos_model.train_step(optimizer_p, Sampler_tr)
        losses_val = PINN.pos_model.evaluate(Sampler_val)

        if j % 1000 == 0:
            print(j, "P\t\t", losses_tr, "\t\t", losses_val)
            history["positive"]["loss_tr"].append(np.sum([l * w for l, w in zip(losses_tr, PINN.pos_model.weigths)]))
            history["positive"]["losses_tr"].append(losses_tr)
            history["positive"]["loss_val"].append(np.sum([l * w for l, w in zip(losses_val, PINN.pos_model.weigths)]))
            history["positive"]["losses_val"].append(losses_val)
            history["positive"]["iteration"].append(j)

        losses_tr = PINN.neg_model.train_step(optimizer_n, Sampler_tr)
        losses_val = PINN.neg_model.evaluate(Sampler_val)

        # optimizer_p_w.step()
        # optimizer_n_w.step()

        if j % 1000 == 0:
            print(j, "N\t\t", losses_tr, "\t\t", losses_val)
            print("\t\tWeights\t\tN\t\t", PINN.neg_model.weigths, "\t\tP\t\t", PINN.pos_model.weigths)
            print("\t\tTime per iter: ", (time.time() - t) * 1000, "ms")
            history["negative"]["loss_tr"].append(np.sum([l * w for l, w in zip(losses_tr, PINN.neg_model.weigths)]))
            history["negative"]["losses_tr"].append(losses_tr)
            history["negative"]["loss_val"].append(np.sum([l * w for l, w in zip(losses_val, PINN.neg_model.weigths)]))
            history["negative"]["losses_val"].append(losses_val)
            history["negative"]["iteration"].append(j)

    # PINN.pos_model.save_model("../models/TL_pos_HardIV.pt")
    # PINN.neg_model.save_model("../models/TL_neg_HardIV.pt")
    #
    # with open('../models/TL_HardIV.pkl', 'wb') as fp:
    #     pickle.dump(history, fp)

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
    plt.plot(t / 3600. * 5., V, "k-", label="PyBaMM")
    plt.plot(bcs_sample_t * 5., V_pinn, "r-", label="PINN")
    plt.ylim([2.5, 4.3])
    plt.xlabel("Disch. Capacity [Ah]")
    plt.ylabel("V [V]")
    plt.legend()
    plt.show()

    plt.figure()
    plt.grid()
    plt.semilogy(history["positive"]["iteration"], history["positive"]["loss_tr"], '--r', label="+ Training")
    plt.semilogy(history["positive"]["iteration"], history["positive"]["loss_val"], '-r', label="+ Validation")
    plt.semilogy(history["negative"]["iteration"], history["negative"]["loss_tr"], '--b', label="- Training")
    plt.semilogy(history["negative"]["iteration"], history["negative"]["loss_val"], '-b', label="- Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()