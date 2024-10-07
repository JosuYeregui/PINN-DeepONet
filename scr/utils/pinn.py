import torch
from torch import nn
import numpy as np


class PINN(nn.Module):
    """
    Base wrapper for the PINN models used in the project, based on Pytorch. Contains the methods necessary to
    initialize the NN architectures, execute training/evaluation steps, update the model and save/load the
    trained weights.
    """

    def __init__(self, model, criterion=nn.MSELoss()):
        super().__init__()
        # Load NN and apply initialization
        self.model = model
        self.model.apply(self._init_weights)

        self.criterion = criterion

    def forward(self, x):
        """
        Performs the forward pass to a given input data.
        :param x: Input tensor (or tuple in case of DeepONet), containing the spatial and condition information
        :return: Model response to input data
        """
        return self.model(x).flatten()

    def save_model(self, PATH):
        """
        Saves the pytorch model as .tp file with
        :param PATH: Local path to save the model
        """
        torch.save(self.state_dict(), PATH)

    def load_model(self, PATH):
        """
        Loads the pytorch model stored as .tp file if possible
        :param PATH: Local path where model is stored
        """
        try:
            self.load_state_dict(torch.load(PATH, weights_only=True))
            self.model.eval()
        except:
            print("Could not load the model from " + PATH + "!")

    def compute_loss(self, points):
        raise NotImplementedError

    def train_step(self, optimizer, sampler):
        """
        Performs one training step to the NN and returns the step loss.
        :param optimizer: PyTorch optimizer
        :param sampler: Sampler object defined in scr/sampling.py
        :return: Returns the overall loss and component loss after running the forward pass
        """

        # Set the model in training mode
        self.model.train()
        # Zero gradients for every batch
        optimizer.zero_grad()

        losses = self.compute_loss(sampler)
        optimizer.step(losses)

        # Detach the loss values from the computational graph
        losses_tr = np.array([l_hist.cpu().detach().numpy() for l_hist in losses])

        return losses_tr

    def train_step_with_loss(self, optimizer, loss):
        """
        Performs one training step with a given loss to the NN and returns the step loss.
        :param optimizer: PyTorch optimizer
        :param loss: Sampler object defined in scr/sampling.py
        :return: Returns the overall loss and component loss after running the forward pass
        """

        # Set the model in training mode
        self.model.train()
        # Zero gradients for every batch
        optimizer.zero_grad()

        def closure():
            # Wrapper to avoid issues with the optimizer.step method used with the LBFGS optimizer
            return loss

        # Adjust learning weights
        if isinstance(optimizer, torch.optim.LBFGS):
            # LBFGS requires to perform the step inserting the loss function as argument
            optimizer.step(closure)
            self.model.eval()
        else:
            # Otherwise the step is performed normally
            optimizer.step()

    def evalueate_old(self, sampler):
        """
        Performs an evaluation pass to return loss.
        :param sampler: Sampler object defined in scr/sampling.py
        :return: Returns the overall loss and component loss after running the forward pass
        """

        # Set the model in evaluation mode
        self.model.eval()
        # Compute the loss
        loss_val, losses_val = self.compute_loss(sampler)
        loss_val = loss_val.detach().numpy()
        losses_val = np.array([l_hist.detach().numpy() for l_hist in losses_val])

        return loss_val, losses_val

    def evaluate(self, sampler):
        """
        Performs an evaluation pass to return loss.
        :param sampler: Sampler object defined in scr/sampling.py
        :return: Returns the overall loss and component loss after running the forward pass
        """

        # Set the model in evaluation mode
        self.model.eval()
        # Compute the loss
        losses_val = self.compute_loss(sampler)
        losses_val = np.array([l_hist.cpu().detach().numpy() for l_hist in losses_val])

        return losses_val

    @staticmethod
    def _init_weights(m):
        """
        Initializes the weights of the NN (see Xavier Glorot initialization)
        :param m: Element of the model, applies initialization if it is a nn.Linear type layer.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class FFNN(nn.Module):
    """
    Implements a Feed-Forward Neural Network (FFNN) with a specified number of layers,
    input dimension, output dimension, activation function, and dropout rate.
    """
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
        """
        Performs a forward pass through the FFNN.
        :param x: Input data tensor of shape (batch_size, input_dim)
        :return: Response tensor of shape (batch_size, output_dim)
        """
        for layer in self.layers[:-1]:
            x = self.activation(self.dropout(layer(x)))
        x = self.layers[-1](x)
        return x


class DeepONet(nn.Module):
    """
    Implements a DeepONet architecture consisting of two sub-networks, one for encoding the input function
    at a fixed number of sensors (branch net), and another for encoding the locations for the
    output functions (trunk net), which should be able to learn operators accurately and efficiently from a
    relatively small dataset
    """
    def __init__(self, branch_layers, trunk_layers, dim_branch, dim_trunk, dim_int, dim_out,
                 activation=nn.Tanh, dropout=0.2):
        super(DeepONet, self).__init__()

        # The sub-networks are defined as standard FFNN
        self.branch = FFNN(branch_layers, dim_branch, dim_int, activation=activation, dropout=dropout)
        self.trunk = FFNN(trunk_layers, dim_trunk, dim_int, activation=activation, dropout=dropout)
        self.b = nn.Parameter(torch.zeros(1, dim_out))  # Initialize with zeros

    def forward(self, x):
        """
        Performs a forward pass through the DeepONet architecture.
        :param x: Input data tensor of shape (batch_size, input_dim)
        :return: Response tensor of shape (batch_size, output_dim)
        """
        # Forward passes the Branch net and Trunk net
        out_B = self.branch.activation(self.branch(x[1].T))
        out_T = self.trunk.activation(self.trunk(x[0]))

        # Aggregate the results of both sub-networks and add the bias
        out_nn = out_B * out_T
        u_pred = torch.sum(out_nn, dim=-1) + self.b
        return u_pred

class NN_TL_Diffusion(nn.Module):
    """
    Implements a DeepONet architecture consisting of two sub-networks, one for encoding the input function
    at a fixed number of sensors (branch net), and another for encoding the locations for the
    output functions (trunk net), which should be able to learn operators accurately and efficiently from a
    relatively small dataset
    """


    def __init__(self, general_layers, fine_layers, dim_in,  dim_int, dim_out,
                 activation=nn.Tanh, dropout=0.2):
        super(NN_TL_Diffusion, self).__init__()

        # The sub-networks are defined as standard FFNN
        self.generalize = FFNN(general_layers, dim_in, dim_int, activation=activation, dropout=dropout)
        self.fine = FFNN(fine_layers, dim_int, dim_out, activation=activation, dropout=dropout)


    def forward(self, x):
        """
        Performs a forward pass through the DeepONet architecture.
        :param x: Input data tensor of shape (batch_size, input_dim)
        :return: Response tensor of shape (batch_size, output_dim)
        """
        # Forward passes the Branch net and Trunk net
        out_G_preact = self.generalize(x)
        out_G = self.generalize.activation(out_G_preact)
        out_F = self.fine(out_G)
        return out_F

    def freeze_general_model(self):
        self.generalize.requires_grad_(False)

    def unfreeze_general_model(self):
        self.generalize.requires_grad_(True)
