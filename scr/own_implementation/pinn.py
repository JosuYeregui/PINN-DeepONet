import torch
from torch import nn
from sampling import sample
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

    def train(self):
        loss = 0

        for func in self.loss_factors["pde"]:
            loss += self.criterion()

