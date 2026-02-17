import pickle as pkl
from scr.utils.parameters import load_params

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap
import scienceplots
import os

import numpy as np

with open("data_full.pkl", "rb") as f:
    data = pkl.load(f)

dr_p = np.array(data["dr_p"])
dr_n = np.array(data["dr_n"])
eps_n = np.array(data["eps_n"])
eps_p = np.array(data["eps_p"])

params = load_params()
epc_p_target = params["eps_p"] * dr_p
epc_n_target = params["eps_n"] * dr_n

error = np.sqrt((np.abs(eps_p - epc_p_target)/epc_p_target * 100.) ** 2 + (np.abs(eps_n - epc_n_target)/epc_n_target * 100.) ** 2)
error = np.sqrt((np.abs(eps_p - epc_p_target)) ** 2 + (np.abs(eps_n - epc_n_target)) ** 2)
# reshape error so it can be plotted
error = error.reshape((20, 20)).T

eps_p = eps_p.reshape((20, 20))
eps_n = eps_n.reshape((20, 20))
epc_p_target = epc_p_target.reshape((20, 20))
epc_n_target = epc_n_target.reshape((20, 20))

print((eps_p[-1, -1] - epc_p_target[-1, -1])/epc_p_target[-1, -1], (eps_n[-1, -1] - epc_n_target[-1, -1])/epc_n_target[-1, -1])
print(params["eps_p"], params["eps_n"])

contour = False
cmap_label = "Relative error [\%]"
cmap_label = "Absolute error [-]"

dr_p = np.linspace(0.8, 1., 20)
dr_n = np.linspace(0.8, 1., 20)
levels = MaxNLocator(nbins=200).tick_values(error.min(), error.max())

with plt.style.context(['science', 'ieee']):
    plt.subplots()  # , tight_layout=True)
    plot = plt.pcolormesh(100*(1 - dr_p[1:]), 100*( 1 -dr_n[1:]), error[1:, 1:], cmap='viridis')
    cbar = plt.colorbar(plot)
    if contour:
        plt.contour(dr_p[1:], dr_n[1:], error[1:, 1:], levels=levels, cmap="PiYG_r")
    plt.ylabel('$LAM_n$ [\%]')
    plt.xlabel('$LAM_p$ [\%]')
    cbar.set_label(cmap_label)
    # plt.savefig('/content/drive/MyDrive/Datos/con_PINN_pos.png')
    # plt.savefig(os.path.join("Paper_results", cmap_label+".svg"), format="svg")
    plt.savefig(os.path.join("Paper_results", "barrido.pdf"), format="pdf")
    plt.show()