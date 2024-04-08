from pinn import PINN
import torch

class Diff_example(PINN):

    def pde(self, t, x, y):
        dydt = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y),
                                   create_graph=True)[0]

        dydx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                   create_graph=True)[0]
        d2ydx2 = torch.autograd.grad(dydx, x, grad_outputs=torch.ones_like(y),
                                   create_graph=True)[0]

        return dydt - d2ydx2