import torch
from torch import nn
from sampling import sample
import numpy as np


class PINN(nn.Module):

    def __init__(self, model, optimizer, criterion=nn.MSELoss()):
        super().__init__()
        self.model = model
        self.model.apply(self.init_weights)

        self.criterion = criterion

        self.optimizer = optimizer

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.model(x).flatten()

    def update_model(self, model):
        self.model = model

    def save_model(self, PATH):
        torch.save(self.model.state_dict(), PATH)

    def load_model(self, PATH):
        self.model.load_state_dict(torch.load(PATH))

    def compute_loss(self):
        raise NotImplementedError

    def train_step(self):

        self.model.train()

        # Zero your gradients for every batch!
        self.optimizer.zero_grad()

        # Compute the loss and its gradients
        loss, losses = self.compute_loss()
        loss.backward()

        # Adjust learning weights
        self.optimizer.step()

        return losses


class FFNN(nn.Module):
    def __init__(self,input_size,output_size):
        super(FFNN, self).__init__()
        self.tanh = nn.Tanh()
        self.l1 = nn.Linear(input_size, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 32)
        self.l4 = nn.Linear(32, output_size)

    def forward(self, x):
        output = self.tanh(self.l1(x))
        output = self.tanh(self.l2(output))
        output = self.tanh(self.l3(output))
        output = self.l4(output)
        return output

