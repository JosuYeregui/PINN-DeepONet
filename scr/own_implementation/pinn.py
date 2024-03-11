import torch
from torch import nn
import numpy as np


class PINN(nn.Module):

    def __int__(self, model, criterion=nn.MSELoss()):
        self.model = model
        self.model.apply(self.init_weights)

        self.criterion = criterion

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.model(x)

    def update_model(self, model):
        self.model = model


class Pos_Electrode(PINN):

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

    def bc_centre(self, r, t, c):

        dcdt = torch.autograd.grad(c, t, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        return dcdt

    def bc_surf(self, r, t, c):
        # TODO

        dcdt = torch.autograd.grad(c, t, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        return dcdt

    def iv(self, c0, SOC):

        return c0 - self.params["SOL_pos"][0] + ((self.params["SOL_pos"][1] - self.params["SOL_pos"][0]) * SOC)


