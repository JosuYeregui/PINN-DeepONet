import pylab as pl

import torch
from torch import nn
import numpy as np

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

class Electrolyte_Split(nn.Module):

    def __init__(self, weights, criterion=RMSELoss):
        super().__init__()

        # self.params_p = {"I_typ": 5., "L": 7.56e-05, "F": 96485.33212, "t_plus": 0.2594, "ce0": 1000.,
        #                  "D": 1.7694e-10, "brug": 1.5, "por": 0.335, "tc": 3600., "A": 0.10270000000000001,
        #                  "L_tot": 0.0001728}
        # self.params_n = {"I_typ": 5., "L": 8.52e-05, "F": 96485.33212, "t_plus": 0.2594, "ce0": 1000.,
        #                  "D": 1.7694e-10, "brug": 1.5, "por": 0.25, "tc": 3600., "A": 0.10270000000000001,
        #                  "L_tot": 0.0001728}
        # self.params_s = {"I_typ": 5., "L": 1.2e-05, "F": 96485.33212, "t_plus": 0.2594, "ce0": 1000.,
        #                  "D": 1.7694e-10, "brug": 1.5, "por": 0.47, "tc": 3600., "A": 0.10270000000000001,
        #                  "L_tot": 0.0001728}

        self.params_p = {"I_typ": 5., "L": 7.56e-05, "F": 96485.33212, "t_plus": 0.2594, "ce0": 1000.,
                         "D": 1.7694e-10, "brug": 1.5, "por": 0.335, "tc": 3600., "A": 0.10270000000000001,
                         "L_tot": 0.0001728}
        self.params_n = {"I_typ": 5., "L": 8.52e-05, "F": 96485.33212, "t_plus": 0.2594, "ce0": 1000.,
                         "D": 1.7694e-10, "brug": 1.5, "por": 0.25, "tc": 3600., "A": 0.10270000000000001,
                         "L_tot": 0.0001728}
        self.params_s = {"I_typ": 5., "L": 1.2e-05, "F": 96485.33212, "t_plus": 0.2594, "ce0": 1000.,
                         "D": 1.7694e-10, "brug": 1.5, "por": 0.47, "tc": 3600., "A": 0.10270000000000001,
                         "L_tot": 0.0001728}

        self.neg_model = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.sep_model = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.pos_model = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.weights = weights
        self.criterion = criterion

    def predict(self, data, data_bc1):
        """
        Performs the forward pass to a given input data.
        :param data: Input tensor (or tuple in case of DeepONet), containing the spatial and condition information
        :return: Model response to input data
        """
        neg_c = self._output_transform_neg(data)
        neg_coef = self._output_transform_neg(data_bc1)
        sep_c = self._output_transform_sep(data, neg_coef)
        sep_coef = self._output_transform_sep(data_bc1, neg_coef)
        pos_c = self._output_transform_pos(data, sep_coef)

        return neg_c, sep_c, pos_c

    def compute_loss(self, data, data_bc0, data_bc1):

        loss = []

        neg_c, sep_c, pos_c = self.predict(data, data_bc1)
        for c, param, e in zip([neg_c, sep_c, pos_c],
                            [self.params_n, self.params_s, self.params_p],
                            ["negative", "separator", "positive"]):
            loss.append(self.criterion(self._pde(data, c, param, electrode=e), torch.zeros_like(c)))
        #loss.append(loss_pde)

        #neg_c_bc0 = self._output_transform_neg(data_bc0)
        neg_c_bc0, sep_c_bc0, pos_c_bc0 = self.predict(data_bc0, data_bc1)
        loss.append(self.criterion(self._bc(data_bc0, neg_c_bc0), torch.zeros_like(neg_c_bc0)))

        neg_c_bc1, sep_c_bc1, pos_c_bc1 = self.predict(data_bc1, data_bc1)
        loss.append(self.criterion(self._bc(data_bc1, pos_c_bc1), torch.zeros_like(pos_c_bc1)))

        loss.append(self.criterion(neg_c_bc1, sep_c_bc0))
        loss.append(self.criterion(sep_c_bc1, pos_c_bc0))

        return torch.stack(loss)

    def _pde(self, x, c, param, electrode="positive"):
        i_app = x[:, 2] * param["I_typ"] / param["A"]

        dcdx = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        left = dcdx[:, 0] * param["ce0"] / 3600. * param["por"]
        # left = torch.ones_like(c)

        De_eff_val = param["D"] * param["por"] ** param["brug"]
        De_eff = torch.ones_like(c) * De_eff_val
        right = torch.autograd.grad(De_eff * dcdx[:, 1], x, grad_outputs=torch.ones_like(dcdx[:, 1]),
                                    create_graph=True)[0][:, 1]

        right *= param["ce0"] / param["L"] ** 2

        if electrode == "positive":
            right -= (i_app/ (param["F"] * param["L"]) * (1 - param["t_plus"]))
        elif electrode == "negative":
            right += (i_app / (param["F"] * param["L"]) * (1 - param["t_plus"]))
        return (left - right)


    def _bc(self, x, c):

        dcdx = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]
        return dcdx[:, 1]
        # return self._compute_flux(x, c, i_app)

    def _output_transform_neg(self, x):
        """
        Performs the forward pass to a given input data.
        :param x: Input tensor (or tuple in case of DeepONet), containing the spatial and condition information
        :return: Model response to input data
        """
        u = self.neg_model(x).flatten()
        u = x[:, 0] * u + 1.
        return u

    def _output_transform_sep(self, x, coef):
        """
        Enforces y(t=0, xs) = 1 and y(t, xs=0) = coef using a general blending method.

        :param x: Input tensor with t = x[:, 0] and xs = x[:, 1]
        :param coef: Value of the boundary condition at xs=0
        :return: Model response with hard constraints applied
        """
        nn_output = self.sep_model(x).flatten()
        t = x[:, 0]  # Keep dimensions for broadcasting
        xs = x[:, 1]

        # To prevent division by zero at (0, 0)
        epsilon = 1e-8
        denominator = t + xs + epsilon

        # Form 1 satisfies y(t=0)=1
        y1 = t * nn_output + 1.0

        # Form 2 satisfies y(xs=0)=coef
        y2 = xs * nn_output + coef

        # Blend the two forms
        u = (xs / denominator) * y1 + (t / denominator) * y2

        u = x[:, 0] * nn_output + 1.

        return u

    def _output_transform_pos(self, x, coef):
        """
        Enforces y(t=0, xs) = 1 and y(t, xs=0) = coef using a general blending method.

        :param x: Input tensor with t = x[:, 0] and xs = x[:, 1]
        :param coef: Value of the boundary condition at xs=0
        :return: Model response with hard constraints applied
        """
        nn_output = self.pos_model(x).flatten()
        t = x[:, 0]  # Keep dimensions for broadcasting
        xs = x[:, 1]

        # To prevent division by zero at (0, 0)
        epsilon = 1e-8
        denominator = t + xs + epsilon

        # Form 1 satisfies y(t=0)=1
        y1 = t * nn_output + 1.0

        # Form 2 satisfies y(xs=0)=coef
        y2 = xs * nn_output + coef

        # Blend the two forms
        u = (xs / denominator) * y1 + (t / denominator) * y2

        u = x[:, 0] * nn_output + 1.

        return u


class Elec_Neg_Split(nn.Module):

    def __init__(self, weights, criterion=RMSELoss):
        super().__init__()

        self.param = {"I_typ": 5., "L": 8.52e-05, "F": 96485.33212, "t_plus": 0.2594, "ce0": 1000.,
                    "D": 1.7694e-10, "brug": 1.5, "por": 0.25, "tc": 3600., "A": 0.10270000000000001, "L_tot": 0.0001728}

        self.neg_model = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.weights = weights
        self.criterion = criterion

    def predict(self, data):
        neg_c = self._output_transform_neg(data)

        return neg_c

    def compute_loss(self, data, data_bc0):

        loss = []

        neg_c = self.predict(data)

        loss.append(self.criterion(self._pde(data, neg_c, self.param), torch.zeros_like(neg_c)))

        neg_c_bc0 = self._output_transform_neg(data_bc0)
        loss.append(self.criterion(self._bc(data_bc0, neg_c_bc0), torch.zeros_like(neg_c_bc0)))

        return torch.stack(loss)

    def _pde(self, x, c, param):
        i_app = x[:, 2] * param["I_typ"] / param["A"]

        dcdx = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        left = dcdx[:, 0] * param["ce0"] / 3600. * param["por"]
        # left = torch.ones_like(c)

        De_eff_val = param["D"] * param["por"] ** param["brug"]
        De_eff = torch.ones_like(c) * De_eff_val
        right = torch.autograd.grad(De_eff * dcdx[:, 1], x, grad_outputs=torch.ones_like(dcdx[:, 1]),
                                    create_graph=True)[0][:, 1]

        right *= param["ce0"] / param["L"] ** 2

        right += (i_app / (param["F"] * param["L"]) * (1 - param["t_plus"]))
        return (left - right) * x[:, 0]


    def _bc(self, x, c):

        dcdx = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]
        return dcdx[:, 1]
        # return self._compute_flux(x, c, i_app)

    def _output_transform_neg(self, x):
        """
        Performs the forward pass to a given input data.
        :param x: Input tensor (or tuple in case of DeepONet), containing the spatial and condition information
        :return: Model response to input data
        """
        u = self.neg_model(x).flatten()
        u = x[:, 0] * u + 1.
        return u


class Elec_Sep_Split(nn.Module):

    def __init__(self, weights, criterion=RMSELoss):
        super().__init__()

        self.param = {"I_typ": 5., "L": 1.2e-05, "F": 96485.33212, "t_plus": 0.2594, "ce0": 1000.,
                         "D": 1.7694e-10, "brug": 1.5, "por": 0.47, "tc": 3600., "A": 0.10270000000000001,
                         "L_tot": 0.0001728}

        self.sep_model = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.weights = weights
        self.criterion = criterion

    def predict(self, data, neg_coef):
        sep_c = self._output_transform_sep(data, neg_coef)

        return sep_c

    def compute_loss(self, data, neg_coef):

        loss = []

        sep_c = self.predict(data, neg_coef)
        loss.append(self.criterion(self._pde(data, sep_c, self.param), torch.zeros_like(sep_c)))

        return torch.stack(loss)

    def _pde(self, x, c, param):

        dcdx = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        left = dcdx[:, 0] * param["ce0"] / 3600. * param["por"]
        # left = torch.ones_like(c)

        De_eff_val = param["D"] * param["por"] ** param["brug"]
        De_eff = torch.ones_like(c) * De_eff_val
        right = torch.autograd.grad(De_eff * dcdx[:, 1], x, grad_outputs=torch.ones_like(dcdx[:, 1]),
                                    create_graph=True)[0][:, 1]

        right *= param["ce0"] / param["L"] ** 2

        return (left - right)

    def _output_transform_sep(self, x, coef):
        """
        Enforces y(t=0, xs) = 1 and y(t, xs=0) = coef using a general blending method.

        :param x: Input tensor with t = x[:, 0] and xs = x[:, 1]
        :param coef: Value of the boundary condition at xs=0
        :return: Model response with hard constraints applied
        """
        nn_output = self.sep_model(x).flatten()
        t = x[:, 0]  # Keep dimensions for broadcasting
        xs = x[:, 1]

        # To prevent division by zero at (0, 0)
        epsilon = 1e-8
        denominator = t + xs + epsilon

        # Form 1 satisfies y(t=0)=1
        y1 = t * nn_output + 1.0

        # Form 2 satisfies y(xs=0)=coef
        y2 = xs * nn_output + coef

        # Blend the two forms
        u = (xs / denominator) * y1 + (t / denominator) * y2

        return u


class Elec_Pos_Split(nn.Module):

    def __init__(self, weights, criterion=RMSELoss):
        super().__init__()

        self.param = {"I_typ": 5., "L": 7.56e-05, "F": 96485.33212, "t_plus": 0.2594, "ce0": 1000.,
                         "D": 1.7694e-10, "brug": 1.5, "por": 0.335, "tc": 3600., "A": 0.10270000000000001,
                         "L_tot": 0.0001728}

        self.pos_model = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.weights = weights
        self.criterion = criterion

    def predict(self, data, sep_coef):
        pos_c = self._output_transform_pos(data, sep_coef)

        return pos_c

    def compute_loss(self, data, data_bc1, sep_coef):

        loss = []

        pos_c = self.predict(data, sep_coef)

        loss.append(self.criterion(self._pde(data, pos_c, self.param), torch.zeros_like(pos_c)))

        pos_c_bc1 = self._output_transform_pos(data_bc1, sep_coef)
        loss.append(self.criterion(self._bc(data_bc1, pos_c_bc1), torch.zeros_like(pos_c_bc1)))

        return torch.stack(loss)

    def _pde(self, x, c, param):
        i_app = x[:, 2] * param["I_typ"] / param["A"]

        dcdx = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        left = dcdx[:, 0] * param["ce0"] / 3600. * param["por"]
        # left = torch.ones_like(c)

        De_eff_val = param["D"] * param["por"] ** param["brug"]
        De_eff = torch.ones_like(c) * De_eff_val
        right = torch.autograd.grad(De_eff * dcdx[:, 1], x, grad_outputs=torch.ones_like(dcdx[:, 1]),
                                    create_graph=True)[0][:, 1]

        right *= param["ce0"] / param["L"] ** 2

        right -= (i_app / (param["F"] * param["L"]) * (1 - param["t_plus"]))
        return (left - right)

    def _bc(self, x, c):

        dcdx = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]
        return dcdx[:, 1]
        # return self._compute_flux(x, c, i_app)

    def _output_transform_pos(self, x, coef):
        """
        Enforces y(t=0, xs) = 1 and y(t, xs=0) = coef using a general blending method.

        :param x: Input tensor with t = x[:, 0] and xs = x[:, 1]
        :param coef: Value of the boundary condition at xs=0
        :return: Model response with hard constraints applied
        """
        nn_output = self.pos_model(x).flatten()
        t = x[:, 0]  # Keep dimensions for broadcasting
        xs = x[:, 1]

        # To prevent division by zero at (0, 0)
        epsilon = 1e-8
        denominator = t + xs + epsilon

        # Form 1 satisfies y(t=0)=1
        y1 = t * nn_output + 1.0

        # Form 2 satisfies y(xs=0)=coef
        y2 = xs * nn_output + coef

        # Blend the two forms
        u = (xs / denominator) * y1 + (t / denominator) * y2

        return u