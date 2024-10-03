from scr.SPMe import Solid_Phase, Cell, Electrolyte
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

    # training_points = {"PDE": 1000, "IV": 50, "BC_Center": 50, "BC_Surf": 50}
    training_points = {"PDE": {"type": "PDE", "N": 1000},
                       "IV": {"type": "IV", "N": 50},
                       "BC_Left": {"type": "BC", "N": 50, "BC_pos": 0.},
                       "BC_Right": {"type": "BC", "N": 50, "BC_pos": 1.}}
    # validation_points = {"PDE": 20, "IV": 10, "BC_Center": 10, "BC_Surf": 10}
    validation_points = {"PDE": {"type": "PDE", "N": 20},
                         "IV": {"type": "IV", "N": 10},
                         "BC_Left": {"type": "BC", "N": 10, "BC_pos": 0.},
                         "BC_Right": {"type": "BC", "N": 10, "BC_pos": 1.}}

    betas_tr = [0.2, 0.3, 0.5, 0.7, 0.75, 0.8, 0.85, 0.95, 1.1]
    betas_val = [0.4, 0.6]
    test_beta = 1.

    model_elec = NN_TL_Diffusion(general_layers=[64, 64, 64, 64], fine_layers=[32, 32], dim_in=3, dim_int=32,
                              dim_out=1, dropout=0.)
    elec_model = Electrolyte(model_elec, parameters, [1., 1., 1.], criterion=RMSELoss)
    optimizer = NTK_Adaptive(elec_model.model.parameters(), elec_model.weights, adam_param = {'lr': 0.0005, 'betas': (0.9, 0.999)})

    # PINN.neg_model.params["as_n"] = torch.nn.Parameter(torch.tensor([parameters["as_n"]]), requires_grad=False)

    Sampler_tr = Sampler(training_points, constant(1.),  mode="quasi")
    Sampler_val = Sampler(validation_points, constant(1.),  mode="quasi")

    history = {"electrolyte":{"loss_tr": [], "losses_tr": [], "loss_val": [], "losses_val": [], "iteration": []}}

    print("Iter \t\t PDE \t IV \t BC Centre \t BC Surf \t\t\t PDE \t IV \t BC Centre \t BC Surf")
    for j in range(3000 + 1):
        t = time.time()
        rate_tr = np.random.choice(betas_tr)
        rate_val = np.random.choice(betas_val)

        # Sampler_tr.update_current_func(constant(rate_tr))
        # Sampler_tr.update_t(rate_tr)
        # Sampler_val.update_current_func(constant(rate_val))
        # Sampler_val.update_t(rate_val)
        Sampler_tr.update_samples(training_points, constant(rate_tr), 1./rate_tr)
        Sampler_val.update_samples(validation_points, constant(rate_val), 1. / rate_val)

        # optimizer_p_w.zero_grad()
        # optimizer_n_w.zero_grad()

        losses_tr = elec_model.train_step(optimizer, Sampler_tr)
        losses_val = elec_model.evaluate(Sampler_val)

        if j % 1000 == 0:
            print(j, "E\t\t", losses_tr, "\t\t", losses_val)
            print("\t\tWeights\t\tN\t\t", elec_model.weights, "\t\tP\t\t", elec_model.weights)
            print("\t\tTime per iter: ", (time.time() - t) * 1000, "ms")
            history["electrolyte"]["loss_tr"].append(np.sum([l * w for l, w in zip(losses_tr, elec_model.weights)]))
            history["electrolyte"]["losses_tr"].append(losses_tr)
            history["electrolyte"]["loss_val"].append(np.sum([l * w for l, w in zip(losses_val, elec_model.weights)]))
            history["electrolyte"]["losses_val"].append(losses_val)
            history["electrolyte"]["iteration"].append(j)

    # PINN.pos_model.save_model("../models/TL_pos_HardIV.pt")
    # PINN.neg_model.save_model("../models/TL_neg_HardIV.pt")
    #
    # with open('../models/TL_HardIV.pkl', 'wb') as fp:
    #     pickle.dump(history, fp)

    # Plot
    plt.figure()
    plt.grid()
    plt.semilogy(history["electrolyte"]["iteration"], history["electrolyte"]["loss_tr"], "-g", label="Training")
    plt.semilogy(history["electrolyte"]["iteration"], history["electrolyte"]["loss_val"], "--g", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE Loss")
    plt.legend()
    plt.show()

    # Evaluation
    # param = pybamm.ParameterValues("ORegan2022")
    param = pybamm.ParameterValues("Chen2020")
    param['Electrolyte diffusivity [m2.s-1]'] = lambda c, T:  1.7694e-10  # 4.862e-10 #  8.794e-11 * (c*1000)**2 - 3.972e-10 * (c*1000)**2 + 4.862e-10#
    # param['Electrolyte diffusivity [m2.s-1]'] = lambda c, T:  8.794e-11 * (c/1000)**2 - 3.972e-10 * (c/1000) + 4.862e-10
    PBM_model = pybamm.lithium_ion.SPMe()
    experiment = pybamm.Experiment(["Discharge at " + str(test_beta) + "C for 100000 seconds or until 2.5 V"])
    sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
    sol = sim.solve(initial_soc=1)

    ce = sol["Electrolyte concentration [mol.m-3]"]

    # pos_SPM_r0 = sol['X-averaged positive particle concentration'].entries[0, :]
    # pos_SPM_r1 = sol['X-averaged positive particle concentration'].entries[-1, :]

    t = sol["Time [s]"].entries
    x = sol["x [m]"].entries[:, 0]

    ce_t0 = ce(t=t[0], x=x)
    ce_tend = ce(t=t[-1], x=x)

    t_end = t[-1]

    t = np.linspace(0, t_end, num=len(ce_tend)) / elec_model.tc

    elec_model.update_tc(1.)

    bcs_sample_x = torch.linspace(0., 1., 1000)
    bcs_sample_t = torch.zeros_like(bcs_sample_x)
    bcs_sample_I = torch.ones_like(bcs_sample_t) * test_beta
    bcs_sample = torch.stack([bcs_sample_t, bcs_sample_x, bcs_sample_I]).t()
    ce_PINN_t0 = elec_model(bcs_sample)

    bcs_sample_t = torch.ones_like(bcs_sample_x)
    bcs_sample = torch.stack([bcs_sample_t, bcs_sample_x, bcs_sample_I]).t()
    ce_PINN_tend = elec_model(bcs_sample)

    plt.figure()
    plt.grid()
    # plt.plot(bcs_sample_t.detach().numpy() * PINN_elec.tc / 3600., ce_PINN_t0.detach().numpy(), "k", label="PINN")
    # plt.plot(t, ce_t0, "r", label="Pybamm")
    plt.plot(bcs_sample_x.detach().numpy() * elec_model.tc / 3600., (ce_PINN_tend - ce_PINN_tend[0]).detach().numpy(),
             "r", label="PINN")
    plt.plot(x / parameters["L"], (ce_tend - ce_tend[0]) / parameters["ce0"], "k", label="Pybamm")
    plt.legend()
    plt.xlabel("r [-]")
    plt.ylabel("x [-]")
    plt.show()

    plt.figure()
    plt.grid()
    # plt.plot(bcs_sample_t.detach().numpy() * PINN_elec.tc / 3600., ce_PINN_t0.detach().numpy(), "k", label="PINN")
    # plt.plot(t, ce_t0, "r", label="Pybamm")
    plt.plot(bcs_sample_x.detach().numpy() * elec_model.tc / 3600., ce_PINN_tend.detach().numpy(),
             "r", label="PINN")
    plt.plot(x / parameters["L"], ce_tend / parameters["ce0"], "k", label="Pybamm")
    plt.legend()
    plt.xlabel("r [-]")
    plt.ylabel("x [-]")
    plt.show()

    num = 1000.

    t_test = (torch.arange(0, num, dtype=torch.float, requires_grad=True) / num)[::10]
    r_rand = (torch.arange(0, num, dtype=torch.float, requires_grad=True) / num)[::10]

    t_new, r_new = torch.meshgrid(t_test, r_rand)
    pos = np.zeros_like(t_new.detach().numpy())

    pos_dcdt = np.zeros_like(t_new.detach().numpy())
    pos_dcdr = np.zeros_like(t_new.detach().numpy())

    for i, (t, r) in enumerate(zip(t_new, r_new)):
        cur = torch.ones_like(t)
        residuals = elec_model.compute_residuals(torch.stack((t, r, cur)).t())
        pos[i, :] = np.abs(residuals.detach().numpy())
        gradients = elec_model.compute_gradients(torch.stack((t, r, cur)).t())
        pos_dcdt[i, :] = np.abs(gradients[:, 0].detach().numpy())
        pos_dcdr[i, :] = np.abs(gradients[:, 1].detach().numpy())


    def plot_area(points, style, cmap_label):
        plt.subplots(figsize=(7, 3), tight_layout=True)
        plot = plt.pcolormesh(t_new.detach().numpy(), r_new.detach().numpy(), points, cmap=style, shading='gouraud')
        cbar = plt.colorbar(plot)
        plt.contour(t_new.detach().numpy(), r_new.detach().numpy(), points, 10, colors='gray')
        plt.ylabel('$x$')
        plt.xlabel('$t$ [h]')
        cbar.set_label(cmap_label)
        # plt.savefig('/content/drive/MyDrive/Datos/con_PINN_pos.png')
        plt.show()


    plot_area(pos, 'RdBu_r', 'res')
    plot_area(pos_dcdt, 'Oranges', 'dcdt')
    plot_area(pos_dcdr, 'Oranges', 'dcdr')