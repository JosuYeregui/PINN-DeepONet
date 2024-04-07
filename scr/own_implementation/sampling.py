import numpy as np
import skopt


def sample(n, n_dim, scale=1., mode="pseudo"):

    if mode == "pseudo":
        return pseudorandom(n, n_dim) * scale
    elif mode == "quasi":
        return quasirandom(n, n_dim) * scale
    raise ValueError("Sampler method not defined!")


def pseudorandom(n_samples, dimension):
    """Pseudo-random sampling"""
    return np.random.random(size=(n_samples, dimension)).astype(np.real(32))


def quasirandom(n_samples, dimension):
    """Quasi-random sampling by Hammersley's method"""
    sampler = skopt.sampler.Hammersly()
    skip = 1
    space = [(0.0, 1.0)] * dimension
    return np.asarray(sampler.generate(space, n_samples + skip)[skip:], dtype=np.real(32))
