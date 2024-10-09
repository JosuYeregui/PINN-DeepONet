import numpy as np
import skopt
import torch


class Sampler:
    """
    Sampler to feed the NN following the PINN needs
    """
    def __init__(self, init_point_data, current_func, t_max=1., mode="pseudo", device="cpu"):

        self.points = dict()
        self.points_tch = dict()
        self.mode = mode
        self.current_func = current_func
        self.device=device

        self.update_samples(init_point_data, current_func, t_max)

    def sample(self, cond):
        return torch.tensor(self.points[cond], requires_grad=True).to(self.device)

    def get_points(self, cond):
        return self.points_tch[cond]

    def update_current_func(self, current_func):

        self.current_func = current_func

        for cond in self.points:
            self.points[cond][:, 2] = self.current_func(self.points[cond][:, 0] * 3600.)

    def update_t(self, c_rate):

        for cond in self.points:
            if cond != "IV":
                self.points[cond][:, 0] *= 1. / (np.max(self.points[cond][:, 0]) * c_rate)

    def update_samples(self, point_data, current_func, t_max):

        for cond in point_data:
            if point_data[cond]["type"] == "IV":
                points = self.update_iv(point_data[cond]["N"])
            elif point_data[cond]["type"] == "BC":
                points = self.update_bc(point_data[cond]["N"], point_data[cond]["BC_pos"])
            elif point_data[cond]["type"] == "PDE":
                points = self.update_pde(point_data[cond]["N"])
            else:
                raise NotImplementedError("The sampling type does not exist")

            points[:, 0] *= t_max
            self.current_func = current_func

            cur = self.current_func(points[:, 0] * 3600.)
            points = np.concatenate([points, cur.reshape((-1, 1))], axis=1)

            self.points[cond] = points
            self.points_tch[cond] = self._cast_torch(points)

    def update_iv(self, n):
        x = self.sample_modes(n, 1)
        t = np.zeros_like(x)
        return np.concatenate([t, x], axis=1)

    def update_bc(self, n, bc_pos):
        t = self.sample_modes(n, 1)
        x = np.ones_like(t) * bc_pos
        return np.concatenate([t, x], axis=1)

    def update_pde(self, n):
        return self.sample_modes(n, 2)

    def sample_modes(self, n, n_dim):

        if self.mode == "uniform":
            return self._uniform(n, n_dim).astype(np.float32)
        elif self.mode == "pseudo":
            return self._pseudorandom(n, n_dim).astype(np.float32)
        elif self.mode == "quasi":
            return self._quasirandom(n, n_dim).astype(np.float32)
        raise ValueError("Sampler method not defined!")

    @staticmethod
    def _uniform(n_samples, dimension):
        """Uniform sampling"""
        if dimension == 1:
            return np.linspace(0, 1, n_samples).reshape(-1, 1)
        elif dimension == 2:
            num_points_per_dim = int(np.sqrt(n_samples))

            # Calculate the spacing between points
            step_size = 1.0 / num_points_per_dim

            # Generate points
            points = []
            for i in range(num_points_per_dim):
                for j in range(num_points_per_dim):
                    x = (i + 0.5) * step_size  # Center of the grid cell
                    y = (j + 0.5) * step_size  # Center of the grid cell
                    points.append((x, y))

            return np.array(points)
        else:
            raise ValueError("Dimension number not supported!")

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

    def _cast_torch(self, array):
        return torch.tensor(array, requires_grad=True).to(self.device)


class Sampler_DONet(Sampler):
    """
    Sampler compatible with the DeepONet architecture. The data needed for the network differs as the input to the
    DeepONet requires a separated input of the sensors (input current) to the branch net.
    """
    def __init__(self, init_point_data, current_func, t_max=1., mode="pseudo", device="cpu"):
        super(Sampler_DONet, self).__init__(init_point_data, current_func, t_max, mode, device)

        self.t = np.arange(0, 3600, 10, dtype=np.float32)
        self.N = current_func(self.t)
        self.N_tch = self._cast_torch(self.N)

    def sample(self, cond, **kwargs):

        return (self.points_tch[cond], self.N_tch)

    def update_current_func(self, current_func):
        self.current_func = current_func

        for cond in self.points:
            self.points[cond][:, 2] = self.current_func(self.points[cond][:, 0] * 3600)
            self.points_tch[cond] = self._cast_torch(self.points[cond])

        self.N = current_func(self.t)
        self.N_tch = self._cast_torch(self.N)

    def update_N(self, current_func):
        self.N = current_func(self.t)
        self.N_tch = self._cast_torch(self.N)