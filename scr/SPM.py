from scr.pinn import PINN
from scr.sampling import sample

import torch
from torch import nn
import numpy as np


class Solid_Phase(PINN):

    def __init__(self, model, parameters, criterion=nn.MSELoss(), r_scale=1., t_scale=1.,
                 electrode="pos", weights=None):
        super().__init__(model, criterion)

        self.params = parameters

        self.r_scale = r_scale
        self.t_scale = t_scale

        self.weights = {"PDE": 1., "IV": 1., "BC_Center": 1., "BC_Surf": 1.}
        if weights is not None:
            self.weights = weights

        if electrode == "pos":
            self.electrode = -1.
            self.el_name = "p"
        elif electrode == "neg":
            self.electrode = 1.
            self.el_name = "n"
        else:
            raise ValueError("Selected electrode type is not compatible")

    def compute_loss(self, points):

        loss = []

        pde_sample = sample(points["PDE"], 2)
        c_pde = self(pde_sample)

        loss.append(self.weights["PDE"] * self.criterion(self._pde(pde_sample, c_pde),
                                                         torch.zeros_like(c_pde)))

        iv_sample_r = sample(points["IV"], 1)
        iv_sample = torch.concat([torch.zeros_like(iv_sample_r), iv_sample_r], dim=1)
        c_iv = self(iv_sample)

        loss.append(self.weights["IV"] * self.criterion(self._iv(c_iv, self.params["SOC_0"]),
                                                         torch.zeros_like(c_iv)))

        bcc_sample_t = sample(points["BC_Center"], 1)
        bcc_sample_r = torch.zeros_like(bcc_sample_t, requires_grad=True)
        bcc_sample = torch.concat([bcc_sample_t, bcc_sample_r], dim=1)
        c_bcc = self(bcc_sample)

        loss.append(self.weights["BC_Center"] * self.criterion(self._bc_centre(bcc_sample, c_bcc),
                                                               torch.zeros_like(c_bcc)))

        bcs_sample_t = sample(points["BC_Surf"], 1)
        bcs_sample_r = torch.ones_like(bcs_sample_t, requires_grad=True)
        bcs_sample = torch.concat([bcs_sample_t, bcs_sample_r], dim=1)
        c_bcs = self(bcs_sample)

        loss.append(self.weights["BC_Surf"] * self.criterion(self._bc_surf(bcs_sample, c_bcs, -self.params["I_typ"]),
                                                             torch.zeros_like(c_bcs)))

        hist = np.array([l.detach().numpy() for l in loss])

        loss = torch.sum(torch.stack(loss))

        return loss, hist

    def _pde(self, x, c):

        dcdx = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        dr2Ndr = - torch.autograd.grad(dcdx[:, 1] * torch.pow(x[:, 1], 2), x, grad_outputs=torch.ones_like(dcdx[:, 1]),
                                     create_graph=True)[0]

        return dcdx[:, 0] * torch.pow(x[:, 1], 2) / 3600. + self.params["D_"+self.el_name] / np.power(self.params["R_"+self.el_name], 2) * dr2Ndr[:, 1]

    def _bc_centre(self, x, c):

        dcdr = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        return dcdr[:, 1]

    def _bc_surf(self, x, c, I):

        dcdr = - torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        return - dcdr[:, 1] - self.electrode * np.power(self.params["R_"+self.el_name], 2) * I / (
                3 * self.params["eps_"+self.el_name] * self.params["D_"+self.el_name] * self.params["L_"+self.el_name] * self.params["F"] *
                self.params["A"] * self.params["c_"+self.el_name+"_max"])

    def _iv(self, c0, SOC):
        return c0 - (self.params["SOL_"+self.el_name][0] + ((self.params["SOL_"+self.el_name][1] - self.params["SOL_"+self.el_name][0]) * SOC))
