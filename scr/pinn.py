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
        # x = sampler.points[mode]
        # if isinstance(self.model, DeepONet):
        #     return self.model(x, sampler.N).flatter()
        # else:
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


class FFNN_old(nn.Module):
    def __init__(self,input_size, output_size, hidden=32):
        super(FFNN_old, self).__init__()
        self.tanh = nn.Tanh()
        self.l1 = nn.Linear(input_size, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, hidden)
        # self.l4 = nn.Linear(hidden, hidden)
        # self.l5 = nn.Linear(hidden, hidden)
        # self.l6 = nn.Linear(hidden, hidden)
        self.lout = nn.Linear(hidden, output_size)

    def forward(self, x):
        output = self.tanh(self.l1(x))
        output = self.tanh(self.l2(output))
        output = self.tanh(self.l3(output))
        output = self.lout(output)
        return output


class FFNN(nn.Module):
    def __init__(self, layers, input_dim, output_dim, activation=nn.Tanh, dropout=0.2):
        super(FFNN, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, layers[0]))

        # Hidden layers
        for i in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[i-1], layers[i]))

        # Output layer
        self.layers.append(nn.Linear(layers[-1], output_dim))

        self.activation = activation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(self.dropout(layer(x)))
        x = self.layers[-1](x)
        return x


class DeepONet(nn.Module):
    def __init__(self, branch_layers, trunk_layers, dim_branch, dim_trunk, dim_int, dim_out,
                 activation=nn.Tanh, dropout=0.2):
        super(DeepONet, self).__init__()
        self.layers = nn.ModuleList()

        self.branch = FFNN(branch_layers, dim_branch, dim_int, activation=activation, dropout=dropout)
        self.trunk = FFNN(trunk_layers, dim_trunk, dim_int, activation=activation, dropout=dropout)
        self.b = nn.Parameter(torch.zeros(1, dim_out))  # Initialize with zeros

    def forward(self, x):

        out_B = self.branch.activation(self.branch(x[1].T))
        out_T = self.trunk.activation(self.trunk(x[0]))

        out_nn = out_B * out_T
        u_pred = torch.sum(out_nn, dim=-1) + self.b
        return u_pred

