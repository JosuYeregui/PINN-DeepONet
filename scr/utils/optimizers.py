import torch
from torch.optim import Optimizer

import numpy as np


class NTK_Adaptive(Optimizer):
    """
    Implements an adaptive optimization algorithm that adjusts the weights of multiple loss components based on their
    gradients. The weights are updated every 10 iterations using a moving average approach, where the new weight is a
    combination of the previous weight and the ratio of the sum of all gradients to the individual gradient.
    :param params: Iterable of parameters to optimize
    :param weights: Initial weights for each loss component
    :param adam_param: Dictionary of parameters for the Adam optimizer (default: {'lr': 0.0005, 'betas': (0.9, 0.999)})
    :param alpha: Smoothing factor for the moving average of weights (default: 0.9)
    :param device: Device to perform computations on (default: "cpu")
    """

    def __init__(self, params, weights, adam_param = {'lr': 0.0005, 'betas': (0.9, 0.999)}, alpha=0.9, device="cpu"):
        self.params = list(params)
        self.weights = weights
        self.adam = torch.optim.Adam(self.params, **adam_param)
        self.alpha = alpha
        self.iter = 0
        self.device = device

        super().__init__(self.params, defaults={})

    def step(self, losses):
        """
        Performs a single optimization step, updating the weights of the loss components every 10 iterations based on their gradients.
        :param losses: List of loss components to be combined and optimized
        """
        self.iter += 1
        def closure():
            loss = torch.sum(losses * torch.tensor(self.weights).to(self.device))
            return loss.backward()

        # THis function is called every 10 iterations. Change the frequency at will.

        if self.iter % 10 == 0:

            grads = []
            for loss in losses:
                m_grad = []
                self.zero_grad()
                loss.backward(retain_graph=True)
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            m_grad.append(torch.zeros(p.size()).reshape(-1))
                        else:
                            m_grad.append(torch.abs(p.grad).reshape(-1))
                grads.append(torch.sqrt(torch.sum(torch.cat(m_grad) ** 2)).item())

            # New weights = alpha * old weights + (1 - alpha) * (sum of all gradients / individual gradient)
            for i, g in enumerate(grads):
                w_delta = np.sum(grads) / g
                self.weights[i] = self.alpha * self.weights[i] + (1 - self.alpha) * w_delta

        self.zero_grad()
        self.adam.step(closure)

class Adam_Custom(Optimizer):

    def __init__(self, params, weights, adam_param={'lr': 0.0005, 'betas': (0.9, 0.999)}):
        self.params = list(params)
        self.weights = weights
        self.adam = torch.optim.Adam(self.params, **adam_param)
        self.iter = 0

        super().__init__(self.params, defaults={})

    def step(self, losses):
        self.iter += 1

        def closure():
            loss = torch.sum(losses * torch.tensor(self.weights))
            return loss.backward()

        self.adam.step(closure)


class Adam_LBFGS(Optimizer):
# TODO
    def __init__(
            self,
            params,
            switch_epoch=10000,
            adam_param={'lr': 0.0005, 'betas': (0.9, 0.999)},
            lbfgs_param={'lr': 1, 'max_iter': 20}
    ):
        self.params = list(params)
        self.switch_epoch = switch_epoch
        self.adam = torch.optim.Adam(self.params, **adam_param)
        self.lbfgs = torch.optim.LBFGS(self.params, **lbfgs_param)

        super().__init__(self.params, defaults={})

        self.state['current_step'] = 0

    def step(self, closure=None):
        self.state['current_step'] += 1

        if self.state['current_step'] < self.switch_epoch:
            self.adam.step(closure)
        else:
            self.lbfgs.step(closure)
            if self.state['current_step'] == self.switch_epoch:
                print(f"Switch to LBFGS at epoch {self.switch_epoch}")