from scr.pinn import PINN

import torch
from torch import nn
import numpy as np


class Solid_Phase(PINN):

    def __init__(self, model, parameters, criterion=nn.MSELoss(), c_rate=1.,
                 electrode="pos", weights=None):
        super().__init__(model, criterion)

        self.params = parameters

        self.C_rate = c_rate
        self.tc = (1./c_rate)*3600.

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

    def update_crate(self, c_rate):
        self.C_rate = c_rate
        self.tc = (1. / c_rate) * 3600.

    def compute_loss(self, sampler):

        loss = []

        pde_sample = sampler.points["PDE"]
        c_pde = self(pde_sample)

        loss.append(self.weights["PDE"] * self.criterion(self._pde(pde_sample, c_pde),
                                                         torch.zeros_like(c_pde)))

        iv_sample = sampler.points["IV"]
        c_iv = self(iv_sample)

        loss.append(self.weights["IV"] * self.criterion(self._iv(c_iv, self.params["SOC_0"]),
                                                        torch.zeros_like(c_iv)))

        bcc_sample = sampler.points["BC_Center"]
        c_bcc = self(bcc_sample)

        loss.append(self.weights["BC_Center"] * self.criterion(self._bc_centre(bcc_sample, c_bcc),
                                                               torch.zeros_like(c_bcc)))

        bcs_sample = sampler.points["BC_Surf"]
        c_bcs = self(bcs_sample)

        loss.append(self.weights["BC_Surf"] *
                    self.criterion(self._bc_surf(bcs_sample, c_bcs, -self.C_rate * self.params["I_typ"]),
                                   torch.zeros_like(c_bcs)))

        hist = np.array([l_hist.detach().numpy() for l_hist in loss])

        loss = torch.sum(torch.stack(loss))

        return loss, hist

    def compute_residuals(self, samples):

        self.model.eval()
        #
        c_pde = self(samples)
        residuals = self._pde(samples, c_pde)

        return residuals

    def compute_gradients(self, samples):

        self.model.eval()
        #
        c = self(samples)
        dcdx = torch.autograd.grad(c, samples, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        return dcdx

    def _pde(self, x, c):

        dcdx = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        dr2Ndr = - torch.autograd.grad(dcdx[:, 1] * torch.pow(x[:, 1], 2), x, grad_outputs=torch.ones_like(dcdx[:, 1]),
                                       create_graph=True)[0]

        return (dcdx[:, 0] * torch.pow(x[:, 1], 2) / self.tc + self.params["D_"+self.el_name] /
                np.power(self.params["R_"+self.el_name], 2) * dr2Ndr[:, 1])

    def _bc_centre(self, x, c):

        dcdr = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        return dcdr[:, 1]

    def _bc_surf(self, x, c, i_app):

        dcdr = - torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                     create_graph=True)[0]

        def regularization(t):
            return 1.
            # return 0.5 * (1 + torch.tanh((t - 10. / 3600.) / (10. / 3600.)))

        return (- dcdr[:, 1] - regularization(x[:, 0]) * self.electrode * np.power(self.params["R_"+self.el_name], 2) *
                i_app / (3 * self.params["eps_"+self.el_name] * self.params["D_"+self.el_name] *
                         self.params["L_"+self.el_name] * self.params["F"] * self.params["A"] *
                         self.params["c_"+self.el_name+"_max"]))

    def _iv(self, c0, soc):
        return c0 - (self.params["SOL_"+self.el_name][0] + ((self.params["SOL_"+self.el_name][1] -
                                                             self.params["SOL_"+self.el_name][0]) * soc))
