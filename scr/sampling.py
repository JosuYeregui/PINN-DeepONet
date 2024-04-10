import numpy as np
import skopt
import torch


class Sampler:

    def __init__(self, mode="pseudo"):

        self.points = None
        self.mode = mode

    def sample(self):

        if self.points is not None:

            return self.points

        else:
            return self.sample()

    def update_samples(self):

        return 1

    def sample(self, n, n_dim, scale=1., mode="pseudo"):

        if mode == "pseudo":
            return torch.tensor(self._pseudorandom(n, n_dim).astype(np.float32) * scale, requires_grad=True)
        elif mode == "quasi":
            return torch.tensor(self._quasirandom(n, n_dim).astype(np.float32) * scale, requires_grad=True)
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

