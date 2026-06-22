import torch
from scr.Elec_split import Electrolyte_Split
from scr.utils.optimizers import NTK_Adaptive, Adam_Custom
import matplotlib.pyplot as plt
import pybamm


#elec_model = Electrolyte_Split([1., 0.01, 1., 1000., 1000., 100, 100])
elec_model = Electrolyte_Split([1., 1, 1., 1., 1., 1, 1])
#elec_model = Electrolyte_Split([1., 0.01, 1., 1000., 1000., 100., 100.])
#optimizer = NTK_Adaptive(elec_model.parameters(), elec_model.weights, adam_param = {'lr': 0.0005, 'betas': (0.9, 0.999)})
# optimizer = Adam_Custom(elec_model.parameters(), elec_model.weights, adam_param = {'lr': 0.0005, 'betas': (0.9, 0.999)})
optimizer = NTK_Adaptive(elec_model.parameters(), elec_model.weights, adam_param = {'lr': 0.0005, 'betas': (0.9, 0.999)})

for epoch in range(1000):
    data_t = torch.rand((1000, 1), requires_grad=True)
    data_x = torch.rand((1000, 1), requires_grad=True)
    data_I = torch.ones_like(data_t, requires_grad=True)
    data = torch.cat([data_t, data_x, data_I], dim=1)
    # append a data point at x = 1 at the end

    data_bc0 = data.clone()
    data_bc0[:, 1] = 0.0

    data_bc1 = data.clone()
    data_bc1[:, 1] = 1.0

    elec_model.train()
    optimizer.zero_grad()
    losses = elec_model.compute_loss(data, data_bc0, data_bc1)
    optimizer.step(losses)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {losses}, Weights: {elec_model.weights}")

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

elec_model.eval()
with torch.no_grad():
    neg_c, sep_c, pos_c = elec_model.predict(data, data_bc1)

L = elec_model.params_n["L"] + elec_model.params_s["L"] + elec_model.params_p["L"]
x_neg_norm = data[:, 1] * elec_model.params_n["L"] / L
x_sep_norm = (data[:, 1] * elec_model.params_s["L"] + elec_model.params_n["L"]) / L
x_pos_norm = (data[:, 1] * elec_model.params_p["L"] + elec_model.params_n["L"] + elec_model.params_s["L"]) / L
# x_neg_norm = data[:, 1]
# x_sep_norm = data[:, 1]
# x_pos_norm = data[:, 1]
# Join resulting x positions and concentrations
x = torch.cat([x_neg_norm, x_sep_norm, x_pos_norm], dim=0)
c = torch.cat([neg_c, sep_c, pos_c], dim=0)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(x, c, label='PINN', color='black')
plt.plot(pos/L, ce_tend, label='Pybamm', color='red')
plt.xlabel('x (normalized)')
plt.ylabel('Concentration (normalized)')
plt.title('Electrolyte Concentration Profiles')
plt.legend()
plt.grid()
plt.show()