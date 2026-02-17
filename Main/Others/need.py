from scr.SPMe import Solid_Phase
from scr.utils.pinn import DeepONet_TL
from scr.utils.parameters import load_params


import pybamm
import torch
import os

# np.set_printoptions(precision=3)
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print("Device: ", device, "\n")
# print(torch.get_num_threads())
torch.set_num_threads(1)


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat-y)**2))

if __name__ == "__main__":

    dr_p = 1.
    dr_n = 0.8
    dr_Dp = 1.
    dr_Dn = 1.

    iterations = 2000

    crate = -1.

    file = "2025-03-07_16-00"
    iter = ""
    if iter != "":
        fold = os.path.join("../models", file, iter)
    else:
        fold = os.path.join("../models", file)

    # Run PyBaMM simulation
    param = pybamm.ParameterValues("Chen2020")
    param["Positive electrode active material volume fraction"] *= dr_p
    param["Negative electrode active material volume fraction"] *= dr_n
    param["Positive electrode diffusivity [m2.s-1]"] *= dr_Dp
    param["Negative electrode diffusivity [m2.s-1]"] *= dr_Dn

    PBM_model = pybamm.lithium_ion.SPM()

    experiment = pybamm.Experiment(["Charge at 1C for 100000 seconds or until 4.2 V"])
    sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
    solution = sim.solve(initial_soc=0.)

    t = solution["Time [s]"].entries
    V = solution["Terminal voltage [V]"].entries

    # PybaMM concentrations
    c_s_n = solution["Negative particle concentration"]
    c_s_p = solution["Positive particle concentration"]
    r_n = solution["r_n [m]"].entries[:, 0, 0]
    r_p = solution["r_p [m]"].entries[:, 0, 0]
    t_pbm = solution["Time [s]"].entries
    x = solution["x [m]"].entries[:, 0]

    c_p_pbm = c_s_p(r=r_p[-1], t=t, x=x[-1])
    c_n_pbm = c_s_n(r=r_n[-1], t=t, x=x[0])


    parameters = load_params()
    # Pybamm takes stechiometric coefficients differently, so we need to adjust the parameters
    # parameters["SOL_p"][0] = c_p_pbm[0]
    # parameters["SOL_n"][0] = c_n_pbm[0]
    eps_p_0 = parameters["eps_p"]
    eps_n_0 = parameters["eps_n"]
    D_p_0 = parameters["D_p"]
    D_n_0 = parameters["D_n"]

    training_points = {"PDE": {"type": "PDE", "N": 1000},
                       "IV": {"type": "IV", "N": 100},
                       "BC_Center": {"type": "BC", "N": 100, "BC_pos": 0.},
                       "BC_Surf": {"type": "BC", "N": 100, "BC_pos": 1.}}
    validation_points = {"PDE": {"type": "PDE", "N": 30},
                         "IV": {"type": "IV", "N": 15},
                         "BC_Center": {"type": "BC", "N": 15, "BC_pos": 0.},
                         "BC_Surf": {"type": "BC", "N": 15, "BC_pos": 1.}}

    model_p = DeepONet_TL(branch_layers=[64, 64, 64, 64], trunk_layers=[64, 64, 64, 64], fine_layers=[32, 32],
                          dim_branch=360, dim_trunk=3, dim_int=128, dim_out=1, dropout=0.).to(device)
    pos_model = Solid_Phase(model_p, parameters, [1., 1., 1.], criterion=RMSELoss, electrode="pos").to(device)

    model_n = DeepONet_TL(branch_layers=[64, 64, 64, 64], trunk_layers=[64, 64, 64, 64], fine_layers=[32, 32],
                          dim_branch=360, dim_trunk=3, dim_int=128, dim_out=1, dropout=0.).to(device)
    neg_model = Solid_Phase(model_n, parameters, [1., 1., 1.], criterion=RMSELoss, electrode="neg").to(device)

    param_size_p = 0
    param_nelem_p = 0
    for param in pos_model.parameters():
        param_size_p += param.nelement() * param.element_size()
        param_nelem_p += param.nelement()

    size_all_mb_p = (param_size_p) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb_p))
    print('model parameters: {:.4f}'.format(param_nelem_p))

    param_size_n = 0
    param_nelem_n = 0
    for param in neg_model.parameters():
        param_size_n += param.nelement() * param.element_size()
        param_nelem_n += param.nelement()

    size_all_mb_n = (param_size_n) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb_n))
    print('model parameters: {:.4f}'.format(param_nelem_n))

    param_size_n = 0
    param_nelem_n = 0
    for param in model_n.fine.parameters():
        param_size_n += param.nelement() * param.element_size()
        param_nelem_n += param.nelement()

    size_all_mb_n = (param_size_n) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb_n))
    print('model parameters: {:.4f}'.format(param_nelem_n))