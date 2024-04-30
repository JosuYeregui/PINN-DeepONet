import numpy as np
import skopt
import torch


class Sampler:

    def __init__(self, init_point_data, mode="pseudo"):

        self.points = dict()
        self.mode = mode

        self.update_samples(init_point_data)

    def sample(self, cond, scale, sc_none):
        return self.points[cond] * scale

    def get_points(self, cond):
        return self.points[cond]

    def update_samples(self, point_data):

        for cond in point_data:
            if point_data[cond]["type"] == "IV":
                points = self.update_iv(point_data[cond]["N"])
            elif point_data[cond]["type"] == "BC":
                points = self.update_bc(point_data[cond]["N"], point_data[cond]["BC_pos"])
            elif point_data[cond]["type"] == "PDE":
                points = self.update_pde(point_data[cond]["N"])
            else:
                raise NotImplementedError("The sampling type does not exist")

            points = torch.concat([points, torch.ones((points.size()[0], 1))], dim=1)
            self.points[cond] = points

    def update_iv(self, n):
        x = self.sample_modes(n, 1)
        t = torch.zeros_like(x)
        return torch.concat([t, x], dim=1)

    def update_bc(self, n, bc_pos):
        t = self.sample_modes(n, 1)
        x = torch.ones_like(t) * bc_pos
        return torch.concat([t, x], dim=1)

    def update_pde(self, n):
        return self.sample_modes(n, 2)

    def sample_modes(self, n, n_dim):

        if self.mode == "uniform":
            return torch.tensor(self._quasirandom(n, n_dim).astype(np.float32), requires_grad=True)
        elif self.mode == "pseudo":
            return torch.tensor(self._pseudorandom(n, n_dim).astype(np.float32), requires_grad=True)
        elif self.mode == "quasi":
            return torch.tensor(self._quasirandom(n, n_dim).astype(np.float32), requires_grad=True)
        raise ValueError("Sampler method not defined!")

    @staticmethod
    def _pseudorandom(n_samples, dimension):
        """Pseudo-random sampling"""
        return np.random.random(size=(n_samples, dimension))

    @staticmethod
    def _quasirandom(n_samples, dimension):
        """Quasi-random sampling by Hammersley's method"""
        if dimension == 1:
            sampler = skopt.sampler.Hammersly(min_skip=1, max_skip=1)
            skip = 0
        else:
            sampler = skopt.sampler.Hammersly()
            skip = 1
        space = [(0.0, 1.0)] * dimension
        return np.asarray(sampler.generate(space, n_samples + skip)[skip:])


class Sampler_DONet(Sampler):

    def __init__(self, init_point_data, mode="pseudo"):
        super(Sampler_DONet, self).__init__(init_point_data, mode)

        self.N = torch.ones((3600, 1))

    def sample(self, cond, scale_point, scale_N, **kwargs):

        return (self.points[cond] * scale_point, self.N * scale_N)



