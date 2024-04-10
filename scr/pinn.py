import torch
from torch import nn


class PINN(nn.Module):

    def __init__(self, model, criterion=nn.MSELoss()):
        super().__init__()
        self.model = model
        self.model.apply(self.init_weights)

        self.criterion = criterion

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.model(x).flatten()

    def update_model(self, model):
        self.model = model

    def save_model(self, PATH):
        torch.save(self.model, PATH)

    def load_model(self, PATH):
        self.model = torch.load(PATH)
        self.model.eval()

    def compute_loss(self, points):
        raise NotImplementedError

    def train_step(self, optimizer, sampler):

        self.model.train()

        def get_loss():

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Compute the loss and its gradients
            loss, losses = self.compute_loss(sampler)
            loss.backward()
            return loss, losses

        def closure():
            loss, _ = get_loss()
            return loss

        # Adjust learning weights
        if isinstance(optimizer, torch.optim.LBFGS):
            optimizer.step(closure)
            self.model.eval()
            loss_tr, losses_tr = get_loss()
        else:
            loss_tr, losses_tr = get_loss()
            optimizer.step()

        loss_tr = loss_tr.detach().numpy()

        return loss_tr, losses_tr

    def evaluate(self, sampler):

        self.model.eval()
        # Compute the loss and its gradients
        loss_val, losses_val = self.compute_loss(sampler)
        loss_val = loss_val.detach().numpy()

        return loss_val, losses_val


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
        # output = self.tanh(self.l3(output))
        output = self.l4(output)
        return output

