
from scr.utils.profiles import constant, grf
from matplotlib import pyplot as plt
import numpy as np
from time import time


if __name__ == "__main__":
    start = time()
    cur_func = grf(1, 1000, 10, 0.5, 0.1)
    t = np.linspace(0, 1, 1000)

    res = cur_func()

    print("Time: ", time() - start)

    plt.plot(t, res)
    plt.show()

    start = time()
    cur_func = constant(1)
    t = np.linspace(0, 1, 1000)

    res = cur_func(t)

    print("Time: ", time() - start)

    # # Parameters for the Gaussian random field
    # n_points = 100  # Number of points in the field
    # mean = 1  # Mean of the Gaussian distribution
    # std_dev = 0.001  # Standard deviation
    # length_scale = 0.05  # Controls the smoothness of the field
    #
    # # Generate spatial points
    # x = np.linspace(0, 1, n_points)
    #
    #
    # # Generate covariance matrix based on a Gaussian kernel
    # def gaussian_kernel(x1, x2, length_scale):
    #     return np.exp(-0.5 * ((x1 - x2) ** 2) / (length_scale ** 2))
    #
    #
    # # Create covariance matrix
    # covariance_matrix = np.array([[gaussian_kernel(xi, xj, length_scale) for xj in x] for xi in x])
    #
    # # Generate samples from the multivariate normal distribution
    # current_profile = np.random.multivariate_normal(mean * np.ones(n_points), covariance_matrix)
    #
    # # Plot the current profile
    # plt.plot(x, current_profile, label='Current Profile')
    # plt.title('1D Gaussian Random Field')
    # plt.xlabel('Position')
    # plt.ylabel('Current')
    # plt.grid()
    # plt.legend()
    # plt.show()