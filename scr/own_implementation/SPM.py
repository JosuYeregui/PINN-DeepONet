from pinn import PINN
import torch
from torch import nn
import numpy as np

class Solid_Phase(PINN):

    def __int__(self, model, parameters, criterion=nn.MSELoss()):
        super().__init__(model, criterion)

        self.params = parameters

    def pde(self, r, t, c):

        dcdt = torch.autograd.grad(c, t, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        dcdr = torch.autograd.grad(c, r, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]
        dr2Ndr = torch.autograd.grad(dcdr * torch.pow(r, 2), r, grad_outputs=torch.ones_like(dcdr),
                                     create_graph=True)[0]

        return dcdt * torch.pow(r, 2) - self.params["D_p"] / np.power(self.params["R_p"], 2) * dr2Ndr

    def bc_centre(self, r, c):

        dcdr = torch.autograd.grad(c, r, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        return dcdr

    def bc_surf(self, r, c, I):

        dcdr = torch.autograd.grad(c, r, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        return dcdr - torch.pow(self.params["R_p"], 2) * I / (
                3 * self.params["Positive electrode active material volume fraction"] *
                self.params["D_p"] * self.params["L_p"] * self.params["F"] *
                self.params["A"] * self.params["c_p_max"])

    def iv(self, c0, SOC):

        return c0 - self.params["SOL_pos"][0] + ((self.params["SOL_pos"][1] - self.params["SOL_pos"][0]) * SOC)
