from scr.SPMe import Solid_Phase, Electrolyte
from scr.pinn import FFNN
from scr.sampling import Sampler
from scr.utils import load_params

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

    weights = {"PDE": 100., "IV": 1., "BC_Left": 10., "BC_Right": 1.}

    # C_rates_tr = [0.3, 0.5, 0.6, 0.7, 1.]
    # C_rates_val = [0.4, 0.8]

    C_rates_tr = [1.]
    C_rates_val = [1.]

    model = FFNN(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    PINN_elec = Electrolyte(model, parameters, criterion=RMSELoss, weights=weights)

    Sampler_tr = Sampler(training_points)
    Sampler_val = Sampler(validation_points)

    history = {"loss_tr": [], "losses_tr": [], "loss_val": [], "losses_val": [], "iteration": []}

    print("Iter \t\t PDE \t IV \t BC Left \t BC Right \t\t\t PDE \t IV \t BC Left \t BC Right")
    for j in range(3000 + 1):

        PINN_elec.update_crate(np.random.choice(C_rates_tr))
        loss_tr, losses_tr = PINN_elec.train_step(optimizer, Sampler_tr)

        PINN_elec.update_crate(np.random.choice(C_rates_val))
        loss_val, losses_val = PINN_elec.evaluate(Sampler_val)

        if j % 1000 == 0:
            print(j, "\t\t", losses_tr, "\t\t", losses_val)
            history["loss_tr"].append(loss_tr)
            history["losses_tr"].append(losses_tr)
            history["loss_val"].append(loss_val)
            history["losses_val"].append(losses_val)
            history["iteration"].append(j)

    # PINN_neg.save_model("models/electrolyte.pt")
    #
    # with open('models/electrolyte.pkl', 'wb') as fp:
    #     pickle.dump(history, fp)

    # Plot
    plt.figure()
    plt.grid("on")
    plt.semilogy(history["iteration"], history["loss_tr"], label="Training")
    plt.semilogy(history["iteration"], history["loss_val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE Loss")
    plt.legend()
    plt.show()

    # Evaluation
    # param = pybamm.ParameterValues("ORegan2022")
    param = pybamm.ParameterValues("Chen2020")
    PBM_model = pybamm.lithium_ion.SPMe()
    experiment = pybamm.Experiment(["Discharge at 1C for 10000 seconds or until 2.5 V"])
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

    t = np.linspace(0, t_end, num=len(ce_tend))/3600.

    PINN_elec.update_crate(1.)

    bcs_sample_x = torch.linspace(0., 1., 1000)
    bcs_sample_t = torch.zeros_like(bcs_sample_x)
    bcs_sample_I = torch.ones_like(bcs_sample_t) * 1.
    bcs_sample = torch.stack([bcs_sample_t, bcs_sample_x, bcs_sample_I]).t()
    ce_PINN_t0 = PINN_elec(bcs_sample)

    bcs_sample_t = torch.ones_like(bcs_sample_x)
    bcs_sample = torch.stack([bcs_sample_t, bcs_sample_x, bcs_sample_I]).t()
    ce_PINN_tend = PINN_elec(bcs_sample)

    plt.figure()
    plt.grid("on")
    # plt.plot(bcs_sample_t.detach().numpy() * PINN_elec.tc / 3600., ce_PINN_t0.detach().numpy(), "k", label="PINN")
    # plt.plot(t, ce_t0, "r", label="Pybamm")
    plt.plot(bcs_sample_x.detach().numpy() * PINN_elec.tc / 3600., ce_PINN_tend.detach().numpy(), "k", label="PINN")
    plt.plot(t, ce_tend/parameters["ce0"], "r", label="Pybamm")
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
        residuals = PINN_elec.compute_residuals(torch.stack((t, r, cur)).t(), cur[0])
        pos[i, :] = np.abs(residuals.detach().numpy())
        gradients = PINN_elec.compute_gradients(torch.stack((t, r, cur)).t())
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
