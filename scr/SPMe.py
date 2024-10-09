import pylab as pl

from scr.utils.pinn import PINN

import torch
from torch import nn
import numpy as np


class Cell():
    def __init__(self, pos_model, neg_model, electrolyte=None):
        self.pos_model = pos_model
        self.neg_model = neg_model
        self.electrolyte = electrolyte

    def compute_V(self, samples):
        if isinstance(samples, tuple):
            I = samples[0][:, 2] * self.pos_model.params["I_typ"]
        else:
            I = samples[:, 2]*self.pos_model.params["I_typ"]

        c_pos = self.pos_model(samples)
        U_0_p, eta_p = self._get_solid_vcomps(c_pos, self.pos_model.params, I, elec="p")

        c_neg = self.neg_model(samples)
        U_0_n, eta_n = self._get_solid_vcomps(c_neg, self.neg_model.params, -I, elec="n")

        return U_0_p - U_0_n + eta_p - eta_n

    @staticmethod
    def _get_solid_vcomps(c, parameters, I, elec="p"):
        OCV = parameters["E_"+elec](c)

        RT_F = parameters["R"] * parameters["T"] / parameters["F"]
        j = - I / (parameters["L_"+elec] * parameters["as_"+elec] * parameters["A"])

        j0 = parameters["m_ref_"+elec] * 31.62 * parameters["c_"+elec+"_max"] * torch.sqrt(c) * torch.sqrt(1 - c)

        eta = 2 * RT_F * torch.arcsinh(j / (2 * j0))

        return OCV, eta



class Solid_Phase(PINN):
    """
    PINN for Solid phase of the battery for both negative and positive electrodes.
    """
    def __init__(self, model, parameters, weigths, criterion=nn.MSELoss(), c_rate=1.,
                 electrode="pos"):
        super().__init__(model, criterion)

        self.params = parameters

        # For scaling purposes a characteristic time is specified
        self.tc = (1./c_rate)*3600.

        # The loss function has weighted terms, received as input or not scaled
        # self.weights = {"PDE": 1., "IV": 1., "BC_Center": 1., "BC_Surf": 1.}
        # if weights is not None and not adjustable_weights:
        #     self.weights = weights
        #
        # self.adjustable_weights = adjustable_weights
        #
        # self.adj_w = nn.Parameter(data=torch.Tensor([1, 1, 1]), requires_grad=adjustable_weights)
        self.weigths = weigths

        # There are certain differences between the positive and negative domain solid equations,
        # mainly in flux direction
        if electrode == "pos":
            self.electrode = 1.
            self.el_name = "p"
        elif electrode == "neg":
            self.electrode = -1.
            self.el_name = "n"
        else:
            raise ValueError("Selected electrode type is not compatible")

    def update_tc(self, c_rate):
        """
        Updates the current characteristic time
        :param c_rate: New C-rate to base the time on
        """
        self.tc = (1. / c_rate) * 3600.

    def compute_loss(self, sampler):
        """
        Computes the loss function based on the solid-phase equations of a SPMe.
        See https://docs.pybamm.org/en/stable/source/examples/notebooks/models/SPMe.html
        :param sampler: Sampler object to obtain evaluation points on different domains
        :return: Total (in computational graph) and spliced loss value s
        """
        loss = []

        # PDE loss
        pde_sample = sampler.sample("PDE")
        if isinstance(pde_sample, tuple):  # In case of a DeepONet a tuple with (x, N) is returned
            x_pde = pde_sample[0]
        else:
            x_pde = pde_sample
        c_pde = self(pde_sample)

        loss.append(self.criterion(self._pde(x_pde, c_pde),
                                                         torch.zeros_like(c_pde)))

        # # Initial Value loss
        # iv_sample = sampler.sample("IV")
        # c_iv = self(iv_sample)
        #
        # loss.append(self.weights["IV"] * self.criterion(self._iv(c_iv, self.params["SOC_0"]),
        #                                                 torch.zeros_like(c_iv)))

        # Boundary Condition (Centre) loss
        bcc_sample = sampler.sample("BC_Center")
        if isinstance(bcc_sample, tuple):  # In case of a DeepONet a tuple with (x, N) is returned
            x_bcc = bcc_sample[0]
        else:
            x_bcc = bcc_sample
        c_bcc = self(bcc_sample)

        loss.append(self.criterion(self._bc_centre(x_bcc, c_bcc),
                                                               torch.zeros_like(c_bcc)))

        # Boundary Condition (Surface) loss
        bcs_sample = sampler.sample("BC_Surf")
        if isinstance(bcs_sample, tuple):  # In case of a DeepONet a tuple with (x, N) is returned
            x_bcs = bcs_sample[0]
        else:
            x_bcs = bcs_sample
        c_bcs = self(bcs_sample)

        loss.append(
                    self.criterion(self._bc_surf(x_bcs, c_bcs),
                                   torch.zeros_like(c_bcs)))

        return torch.stack(loss)
        # return torch.sum(torch.stack(loss)), loss

    def compute_residuals(self, samples):
        """
        Auxiliary function to compute the PDE residuals for evaluation purposes.
        :param samples: Evaluation points for which to compute the residuals. Takes tensor of shape (n_points, 3)
        :return: Residual values of the PDE left-right equations at sample evaluated points
        """

        self.model.eval()
        # Model forward pass and PDE evaluation
        c_pde = self(samples)
        if isinstance(samples, tuple):
            x = samples[0]
        else:
            x = samples
        residuals = self._pde(x, c_pde)

        return residuals

    def compute_gradients(self, samples):
        """
        Auxiliary function to compute the gradients of concentrations wrt the input independent variables.
        :param samples: Evaluation points for which to compute the residuals. Takes tensor of shape (n_points, 3)
        :return: Gradient values of the model response wrt input points. Outputs a tensor of shape (n_points, 3)
        """

        self.model.eval()
        # Model forward pass and automatic differentiation
        c = self(samples)
        if isinstance(samples, tuple):
            x = samples[0]
        else:
            x = samples
        dcdx = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        return dcdx

    def _pde(self, x, c):
        """
        Computes the residual of the main PDE equation.
        $dc/dt = 1/r^2 * d/dr(D * r^2 * dc/dr)$
        :param x: Independent variables of the PDE.
        :param c: Concentration at given radius point r and time t.
        :return: Residual of the PDE.
        """
        dcdx = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        dr2Ndr = - torch.autograd.grad(dcdx[:, 1] * torch.pow(x[:, 1], 2), x, grad_outputs=torch.ones_like(dcdx[:, 1]),
                                       create_graph=True)[0]

        return (dcdx[:, 0] * torch.pow(x[:, 1], 2) / self.tc + self.params["D_"+self.el_name] /
                np.power(self.params["R_"+self.el_name], 2) * dr2Ndr[:, 1])

    def _bc_centre(self, x, c):
        """
        Boundary condition at the centre of the solid particle.
        $N = 0$
        :param x: Independent variables of the PDE.
        :param c: Concentration at the center of the solid particle.
        :return: Returns the residual for the center BC equation.
        """
        dcdr = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        return dcdr[:, 1]

    def _bc_surf(self, x, c):
        """
        Surface condition at the centre of the solid particle.
        $N = -j$
        :param x: Independent variables of the PDE.
        :param c: Concentration at the surface of the solid particle.
        :return: Returns the residual for the surface BC equation.
        """
        i_app = - x[:, 2] * self.params["I_typ"] / self.params["A"]

        dcdr = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        # A regularization is added for the initial t points of the equations where the derivatives are more agressive
        def regularization(t):
            return 1. #  0.5 * (1 + torch.tanh((t - self.tc*0.01/self.tc) / (self.tc*0.01/self.tc)))

        N = (dcdr[:, 1] * self.params["D_"+self.el_name] * self.params["c_"+self.el_name+"_max"] /
             self.params["R_"+self.el_name])

        # The flux j is an algebraic equation based on the applied current
        j = i_app * self.electrode / (self.params["as_"+self.el_name] *
                                      self.params["L_"+self.el_name] * self.params["F"])

        return N + j * regularization(x[:, 0])

    def _iv(self, c0, soc):
        """
        Initial value condition term of the PDE.
        $c_ini = c0$
        :param c0: Concentration at t=0
        :param soc: State-of-charge of the battery at t=0
        :return: Returns the residual for the IV condition.
        """
        return c0 - (self.params["SOL_"+self.el_name][0] + ((self.params["SOL_"+self.el_name][1] -
                                                             self.params["SOL_"+self.el_name][0]) * soc))

    def forward(self, x):
        """
        Performs the forward pass to a given input data.
        :param x: Input tensor (or tuple in case of DeepONet), containing the spatial and condition information
        :return: Model response to input data
        """
        u = self.model(x)
        if isinstance(x, tuple):
            x = x[0]   # In case of DeepONet the input is a tuple with (x, N)
        u = x[:, 0] * u.flatten() + (self.params["SOL_"+self.el_name][0] + ((self.params["SOL_"+self.el_name][1] -
                                                             self.params["SOL_"+self.el_name][0]) *
                                                                  self.params["SOC_0"]))
        return u


class Electrolyte(PINN):

    def __init__(self, model, parameters, weights, criterion=nn.MSELoss(), c_rate=1.):
        super().__init__(model, criterion)

        self.params = parameters

        # For scaling purposes a characteristic time is specified
        self.tc = (1. / c_rate) * 3600.

        self.weights = weights

    def update_tc(self, c_rate):
        """
        Updates the current characteristic time
        :param c_rate: New C-rate to base the time on
        """
        self.tc = (1. / c_rate) * 3600.

    def compute_loss(self, sampler):

        loss = []

        pde_sample = sampler.sample("PDE")
        c_pde = self(pde_sample)

        loss.append(self.criterion(self._pde(pde_sample, c_pde),
                                                         torch.zeros_like(c_pde)))

        # iv_sample = sampler.sample("IV", torch.Tensor([1., 1., self.C_rate]), self.C_rate)
        # c_iv = self(iv_sample)
        #
        # loss.append(self.criterion(self._iv(c_iv), torch.zeros_like(c_iv)))

        bcc_sample = sampler.sample("BC_Left")
        c_bcc = self(bcc_sample)

        loss.append(self.criterion(self._bc(bcc_sample, c_bcc),
                                                             torch.zeros_like(c_bcc)))

        bcs_sample = sampler.sample("BC_Right")
        c_bcs = self(bcs_sample)

        loss.append(self.criterion(self._bc(bcs_sample, c_bcs),
                                   torch.zeros_like(c_bcs)))

        return torch.stack(loss)

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

    def _pde_old(self, x, c, i_app):
        dcdx = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        N = self._compute_flux(x, c, i_app)

        dNdx = torch.autograd.grad(N, x, grad_outputs=torch.ones_like(N),
                                   create_graph=True)[0]

        L_n = self.params["L_n"] / self.params["L"]
        L_s = self.params["L_s"] / self.params["L"]

        idx_Ln = x[:, 1] < L_n
        idx_Lp = x[:, 1] > L_n + L_s
        idx_Ls = (L_n <= x[:, 1]) & (x[:, 1] <= L_n + L_s)

        left = dcdx[:, 0] * self.params["ce0"] / self.tc  # * self.params["por_n"]
        left[idx_Ln] *= self.params["por_n"]
        left[idx_Ls] *= self.params["por_s"]
        left[idx_Lp] *= self.params["por_p"]

        right = -dNdx[:, 1] / self.params["L"]
        right[idx_Ln] += i_app / (self.params["F"] * self.params["L_n"])
        right[idx_Lp] -= i_app / (self.params["F"] * self.params["L_p"])

        return left - right

    def _pde(self, x, c):
        i_app = x[:, 2] * self.params["I_typ"] / self.params["A"]

        dcdx = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        L_n = self.params["L_n"] / self.params["L"]
        L_s = self.params["L_s"] / self.params["L"]

        idx_Ln = x[:, 1] < L_n
        idx_Lp = x[:, 1] > L_n + L_s
        idx_Ls = (L_n <= x[:, 1]) & (x[:, 1] <= L_n + L_s)

        left = dcdx[:, 0] * self.params["ce0"] / self.tc  # * self.params["por_n"]
        left[idx_Ln] *= self.params["por_n"]
        left[idx_Ls] *= self.params["por_s"]
        left[idx_Lp] *= self.params["por_p"]

        De_eff = torch.ones_like(c) * self.params["D_e_const"]
        # De_eff = self.params["D_e"](c)
        De_eff[idx_Ln] *= self.params["por_n"] ** self.params["brug"]
        De_eff[idx_Ls] *= self.params["por_s"] ** self.params["brug"]
        De_eff[idx_Lp] *= self.params["por_p"] ** self.params["brug"]
        right = torch.autograd.grad(De_eff * dcdx[:, 1], x, grad_outputs=torch.ones_like(dcdx[:, 1]),
                                    create_graph=True)[0][:, 1]

        right *= self.params["ce0"] / self.params["L"] ** 2

        right[idx_Ln] += i_app[idx_Ln] / (self.params["F"] * self.params["L_n"]) * (1 - self.params["t_plus"])
        right[idx_Lp] -= i_app[idx_Lp] / (self.params["F"] * self.params["L_p"]) * (1 - self.params["t_plus"])

        return left - right

    def _bc(self, x, c):

        dcdx = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]
        return dcdx[:, 1]
        # return self._compute_flux(x, c, i_app)

    def _iv(self, c0):
        return c0 - 1.

    def _compute_flux(self, x, c, i_app):

        dcdx = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                   create_graph=True)[0]

        L_n = self.params["L_n"] / self.params["L"]
        L_p = self.params["L_p"] / self.params["L"]
        L_s = self.params["L_s"] / self.params["L"]

        idx_Ln = x[:, 1] < L_n
        idx_Lp = x[:, 1] > L_n + L_s
        idx_Ls = (L_n <= x[:, 1]) & (x[:, 1] <= L_n + L_s)

        term_1 = - dcdx[:, 1] * self.params["D_e"](c) * self.params["ce0"] / self.params["L"]  # * self.params["por_n"]
        term_1[idx_Ln] *= self.params["por_n"]
        term_1[idx_Ls] *= self.params["por_s"]
        term_1[idx_Lp] *= self.params["por_p"]

        term_2 = (self.params["t_plus"] * i_app / self.params["F"]
                  * torch.ones_like(term_1))
        term_2[idx_Ln] *= x[idx_Ln, 1] / L_n
        term_2[idx_Lp] *= (1 - x[idx_Lp, 1]) / L_p

        return term_1 + term_2

    def forward(self, x):
        """
        Performs the forward pass to a given input data.
        :param x: Input tensor (or tuple in case of DeepONet), containing the spatial and condition information
        :return: Model response to input data
        """
        u = self.model(x)
        u = x[:, 0] * u.flatten() + 1.
        return u
