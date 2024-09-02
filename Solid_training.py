from scr.SPMe import Solid_Phase
from scr.dependencies.pinn import FFNN, DeepONet
from scr.dependencies.sampling import Sampler, Sampler_DONet
from scr.dependencies.utils import load_params
from scr.dependencies.profiles import zheng_current, constant

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
                       "IV": {"type": "IV", "N": 100},
                       "BC_Center": {"type": "BC", "N": 100, "BC_pos": 0.},
                       "BC_Surf": {"type": "BC", "N": 100, "BC_pos": 1.}}
    # validation_points = {"PDE": 20, "IV": 10, "BC_Center": 10, "BC_Surf": 10}
    validation_points = {"PDE": {"type": "PDE", "N": 30},
                         "IV": {"type": "IV", "N": 15},
                         "BC_Center": {"type": "BC", "N": 15, "BC_pos": 0.},
                         "BC_Surf": {"type": "BC", "N": 15, "BC_pos": 1.}}

    pos_weights = {"PDE": 1e4, "IV": 10., "BC_Center": 1., "BC_Surf": 1e4}
    # pos_weights = {"PDE": 1e4, "IV": 10., "BC_Center": 1., "BC_Surf": 10.}
    neg_weights = {"PDE": 1e4, "IV": 10., "BC_Center": 1., "BC_Surf": 1e4}

    betas_tr = [0.2, 0.3, 0.5, 0.7, 0.75, 0.8, 0.85, 0.95, 1.]
    betas_val = [0.4, 0.6]
    test_beta = 0.4

    # model = FFNN(layers=[32, 32, 32], input_dim=3, output_dim=1, dropout=0.)
    model = DeepONet(branch_layers=[64, 64, 64, 64], trunk_layers=[64, 64, 64, 64], dim_branch=360, dim_trunk=3,
                     dim_int=100, dim_out=1, dropout=0.)
    # model = DeepONet(branch_layers=[32, 32, 32], trunk_layers=[32, 32, 32], dim_branch=360, dim_trunk=3,
    #                  dim_int=100, dim_out=1, dropout=0.)
    # model = FFNN_old(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    PINN = Solid_Phase(model, parameters, criterion=RMSELoss, electrode="pos", weights=pos_weights)

    # Sampler_tr = Sampler(training_points)
    # Sampler_val = Sampler(validation_points)
    # Sampler_tr = Sampler_DONet(training_points, zheng_current(1.), mode="uniform")
    # Sampler_val = Sampler_DONet(validation_points, zheng_current(1.), mode="uniform")
    Sampler_tr = Sampler_DONet(training_points, zheng_current(1.), mode="uniform")
    Sampler_val = Sampler_DONet(validation_points, zheng_current(1.), mode="uniform")

    history = {"loss_tr": [], "losses_tr": [], "loss_val": [], "losses_val": [], "iteration": []}

    print("Iter \t\t PDE \t IV \t BC Centre \t BC Surf \t\t\t PDE \t IV \t BC Centre \t BC Surf")
    for j in range(50000 + 1):

        # Sampler_tr.update_current_func(zheng_current(np.random.choice(betas_tr)))
        Sampler_tr.update_current_func(zheng_current(np.random.choice(betas_tr)))
        loss_tr, losses_tr = PINN.train_step(optimizer, Sampler_tr)

        # Sampler_val.update_current_func(zheng_current(np.random.choice(betas_val)))
        Sampler_val.update_current_func(zheng_current(np.random.choice(betas_val)))
        loss_val, losses_val = PINN.evaluate(Sampler_val)

        if j % 1000 == 0:
            print(j, "\t\t", losses_tr, "\t\t", losses_val)
            history["loss_tr"].append(loss_tr)
            history["losses_tr"].append(losses_tr)
            history["loss_val"].append(loss_val)
            history["losses_val"].append(losses_val)
            history["iteration"].append(j)

    # j_prev = j
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01, max_iter=50)
    #
    # for j in range(j_prev, j_prev + 200 + 1):
    #
    #     Sampler_tr.update_current_func(zheng_current(np.random.choice(betas_tr)))
    #     loss_tr, losses_tr = PINN.train_step(optimizer, Sampler_tr)
    #
    #     Sampler_val.update_current_func(zheng_current(np.random.choice(betas_val)))
    #     loss_val, losses_val = PINN.evaluate(Sampler_val)
    #
    #     if j % 10 == 0:
    #         print(j, "\t\t", losses_tr, "\t\t", losses_val)
    #         history["loss_tr"].append(loss_tr)
    #         history["losses_tr"].append(losses_tr)
    #         history["loss_val"].append(loss_val)
    #         history["losses_val"].append(losses_val)
    #         history["iteration"].append(j)

    # PINN_neg.save_model("models/negative_current.pt")
    #
    # with open('models/negative_current.pkl', 'wb') as fp:
    #     pickle.dump(history, fp)

    # Plot
    plt.figure()
    plt.grid("on")
    plt.semilogy(history["iteration"], history["loss_tr"], label="Training")
    plt.semilogy(history["iteration"], history["loss_val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

    # Evaluation
    # param = pybamm.ParameterValues("ORegan2022")
    param = pybamm.ParameterValues("Chen2020")

    t_eval = np.arange(0, 3600)
    # cur_fun = zheng_current(test_beta)
    cur_fun = zheng_current(test_beta)

    current_interpolant = pybamm.Interpolant(t_eval, cur_fun(t_eval) * parameters["I_typ"], pybamm.t)
    param["Current function [A]"] = current_interpolant

    PBM_model = pybamm.lithium_ion.SPM()
    sim = pybamm.Simulation(PBM_model, parameter_values=param)
    sol = sim.solve(initial_soc=1., t_eval=t_eval)

    # pos_SPM_r0 = sol['X-averaged positive particle concentration'].entries[0, :]
    # pos_SPM_r1 = sol['X-averaged positive particle concentration'].entries[-1, :]

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

    t_end = t[-1]
    t = np.linspace(0, t_end, num=len(pos_SPM_r1))/3600.

    # cur_fun = zheng_current(test_beta)
    cur_fun = zheng_current(test_beta)

    bcs_sample_t = torch.linspace(0., 1., 1000)
    N = torch.tensor(cur_fun(np.arange(0., 3600, 10)))
    bcs_sample_r = torch.ones_like(bcs_sample_t)
    bcs_sample_I = torch.tensor(cur_fun(bcs_sample_t.numpy()*3600.))
    bcs_sample = torch.stack([bcs_sample_t, bcs_sample_r, bcs_sample_I]).t()
    c_bcs = PINN((bcs_sample, N))

    plt.figure()
    plt.grid(True)
    plt.plot(bcs_sample_t.detach().numpy()*PINN.tc/3600., c_bcs.detach().numpy(), "k", label="PINN")
    plt.plot(t, pos_SPM_r1, "r", label="Pybamm")
    plt.legend()
    plt.xlabel("t [h]")
    plt.ylabel("x [-]")
    plt.show()

    bcs_sample_r = torch.zeros_like(bcs_sample_t)
    bcs_sample_r0 = torch.stack([bcs_sample_t, bcs_sample_r, bcs_sample_I]).t()
    c_bcs_r0 = PINN((bcs_sample_r0, N))

    plt.figure()
    plt.grid(True)
    plt.plot(bcs_sample_t.detach().numpy()*PINN.tc/3600., c_bcs_r0.detach().numpy(), "k", label="PINN")
    plt.plot(t, pos_SPM_r0, "r", label="Pybamm")
    plt.legend()
    plt.xlabel("t [h]")
    plt.ylabel("x [-]")
    plt.show()



    num = 1000.

    t_test = (torch.arange(0, num, dtype=torch.float, requires_grad=True) / num)[::10]
    r_rand = (torch.arange(0, num, dtype=torch.float, requires_grad=True) / num)[::10]

    N = torch.tensor(cur_fun(np.arange(0., 3600, 10)))

    t_new, r_new = torch.meshgrid(t_test, r_rand)
    pos = np.zeros_like(t_new.detach().numpy())

    pos_dcdt = np.zeros_like(t_new.detach().numpy())
    pos_dcdr = np.zeros_like(t_new.detach().numpy())

    for i, (t, r) in enumerate(zip(t_new, r_new)):
        cur = torch.tensor(cur_fun(t.detach().numpy() * 3600.))
        sample = (torch.stack((t, r, cur)).t(), N)
        residuals = PINN.compute_residuals(sample)
        pos[i, :] = np.abs(residuals.detach().numpy())
        gradients = PINN.compute_gradients(sample)
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
