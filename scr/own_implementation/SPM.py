from pinn import PINN
from sampling import sample
import torch
from torch import nn
import numpy as np


class Solid_Phase(PINN):

    def __init__(self, model, parameters, criterion=nn.MSELoss(), r_scale=1., t_scale=1.):
        super().__init__(model, criterion)

        self.params = parameters

        self.r_scale = r_scale
        self.t_scale = t_scale

    def compute_loss(self):

        loss = 0.

        pde_sample = sample(50, 2)
        c_pde = self.model(pde_sample)

        loss += self.criterion(self._pde(pde_sample[:, 1], pde_sample[:, 0], c_pde))

        iv_sample_r = sample(50, 1)
        iv_sample = torch.concat([torch.zeros_like(iv_sample_r), iv_sample_r], dim=0)
        c_iv = self.model(iv_sample)

        loss += self.criterion(self._iv(c_iv, self.params["SOC_0"]))

        bcc_sample_t = sample(50, 1)
        bcc_sample_r = torch.zeros_like(bcc_sample_t)
        bcc_sample = torch.concat([bcc_sample_t, bcc_sample_r], dim=0)
        c_bcc = self.model(bcc_sample)

        loss += self.criterion(self._bc_centre(bcc_sample_r, c_bcc))

        bcs_sample_t = sample(50, 1)
        bcs_sample_r = torch.ones_like(bcs_sample_t)
        bcs_sample = torch.concat([bcs_sample_t, bcs_sample_r], dim=0)
        c_bcs = self.model(bcs_sample)

        loss += self.criterion(self._bc_surf(bcc_sample_r, c_bcs, self.I))

        return loss

    def _pde(self, r, t, c):

        dcdt = torch.autograd.grad(c, t, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        dcdr = torch.autograd.grad(c, r, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]
        dr2Ndr = torch.autograd.grad(dcdr * torch.pow(r, 2), r, grad_outputs=torch.ones_like(dcdr),
                                     create_graph=True)[0]

        return dcdt * torch.pow(r, 2) - self.params["D_p"] / np.power(self.params["R_p"], 2) * dr2Ndr

    def _bc_centre(self, r, c):

        dcdr = torch.autograd.grad(c, r, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        return dcdr

    def _bc_surf(self, r, c, I):

        dcdr = torch.autograd.grad(c, r, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        return dcdr - torch.pow(self.params["R_p"], 2) * I / (
                3 * self.params["Positive electrode active material volume fraction"] *
                self.params["D_p"] * self.params["L_p"] * self.params["F"] *
                self.params["A"] * self.params["c_p_max"])

    def _iv(self, c0, SOC):

        return c0 - self.params["SOL_pos"][0] + ((self.params["SOL_pos"][1] - self.params["SOL_pos"][0]) * SOC)
