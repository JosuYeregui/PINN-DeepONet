import torch
from scr.Elec_split import Elec_Neg_Split, Elec_Sep_Split, Elec_Pos_Split
from scr.utils.optimizers import NTK_Adaptive, Adam_Custom
import matplotlib.pyplot as plt
import pybamm


n_model = Elec_Neg_Split([1., 1000.])
s_model = Elec_Sep_Split([1.])
p_model = Elec_Pos_Split([1., 1000.])
#optimizer = NTK_Adaptive(elec_model.parameters(), elec_model.weights, adam_param = {'lr': 0.0005, 'betas': (0.9, 0.999)})
optimizer_n = Adam_Custom(n_model.parameters(), n_model.weights, adam_param = {'lr': 0.0005, 'betas': (0.9, 0.999)})
optimizer_s = Adam_Custom(s_model.parameters(), s_model.weights, adam_param = {'lr': 0.0005, 'betas': (0.9, 0.999)})
optimizer_p = Adam_Custom(p_model.parameters(), p_model.weights, adam_param = {'lr': 0.0005, 'betas': (0.9, 0.999)})

for epoch in range(3000):
    data_t = torch.rand((1000, 1), requires_grad=True)
    data_x = torch.rand((1000, 1), requires_grad=True)
    data_I = torch.ones_like(data_t, requires_grad=True)
    data = torch.cat([data_t, data_x, data_I], dim=1)
    # append a data point at x = 1 at the end

    data_bc0 = data.clone()
    data_bc0[:, 1] = 0.0

    data_bc1 = data.clone()
    data_bc1[:, 1] = 1.0

    n_model.eval()
    s_model.eval()
    p_model.eval()

    neg_coef = n_model.predict(data_bc1).detach()
    sep_coef = s_model.predict(data_bc1, neg_coef).detach()

    n_model.train()
    s_model.train()
    p_model.train()

    optimizer_n.zero_grad()
    loss_n = n_model.compute_loss(data, data_bc0)
    optimizer_n.step(loss_n)

    optimizer_s.zero_grad()
    loss_s = s_model.compute_loss(data, neg_coef)
    optimizer_s.step(loss_s)

    optimizer_p.zero_grad()
    loss_p = p_model.compute_loss(data, data_bc1, sep_coef)
    optimizer_p.step(loss_p)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss_n: {loss_n.detach().numpy()}, Loss_s: {loss_s.detach().numpy()}, Loss_p: {loss_p.detach().numpy()}")

param = pybamm.ParameterValues("Chen2020")
param['Electrolyte diffusivity [m2.s-1]'] = lambda c, T:  1.7694e-10  # 4.862e-10 #  8.794e-11 * (c*1000)**2 - 3.972e-10 * (c*1000)**2 + 4.862e-10#
# param['Electrolyte diffusivity [m2.s-1]'] = lambda c, T:  8.794e-11 * (c/1000)**2 - 3.972e-10 * (c/1000) + 4.862e-10
PBM_model = pybamm.lithium_ion.SPMe()
experiment = pybamm.Experiment(["Discharge at 1C for 100000 seconds or until 2.5 V"])
sim = pybamm.Simulation(PBM_model, experiment=experiment, parameter_values=param)
sol = sim.solve(initial_soc=1)

ce = sol["Electrolyte concentration [mol.m-3]"]

t = sol["Time [s]"].entries
pos = sol["x [m]"].entries[:, 0]

ce_t0 = ce(t=t[0], x=pos) /1000
ce_tend = ce(t=t[-1], x=pos) /1000

data_t = torch.ones((1000, 1))
data_x = torch.linspace(0, 1, 1000).unsqueeze(1)
data_I = torch.ones_like(data_t)
data = torch.cat([data_t, data_x, data_I], dim=1)
# append a data point at x = 1 at the end

data_bc1 = data.clone()
data_bc1[:, 1] = 1.0

n_model.eval()
s_model.eval()
p_model.eval()
with torch.no_grad():
    neg_coef = n_model.predict(data_bc1)
    sep_coef = s_model.predict(data_bc1, neg_coef)

    neg_c = n_model.predict(data)
    sep_c = s_model.predict(data, neg_coef)
    pos_c = p_model.predict(data, sep_coef)

L = n_model.param["L"] + s_model.param["L"] + p_model.param["L"]
x_neg_norm = data[:, 1] * n_model.param["L"] / L
x_sep_norm = (data[:, 1] * s_model.param["L"] + n_model.param["L"]) / L
x_pos_norm = (data[:, 1] * p_model.param["L"] + n_model.param["L"] + s_model.param["L"]) / L
# x_neg_norm = data[:, 1]
# x_sep_norm = data[:, 1]
# x_pos_norm = data[:, 1]
# Join resulting x positions and concentrations
x = torch.cat([x_neg_norm, x_sep_norm, x_pos_norm], dim=0)
c = torch.cat([neg_c, sep_c, pos_c], dim=0)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(x, c , label='PINN', color='black')
plt.plot(pos/L, ce_tend, label='Pybamm', color='red')
plt.xlabel('x (normalized)')
plt.ylabel('Concentration (normalized)')
plt.title('Electrolyte Concentration Profiles')
plt.legend()
plt.grid()
plt.show()