import matplotlib.pyplot as plt
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq
from matplotlib import _cntr as cntr

class Random2DGaussian():
    def __init__(self, minx=0, maxx=10, miny=0, maxy=10):
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        self.mean = (np.random.random_sample() * (self.maxx - self.minx) + self.minx,
                     np.random.random_sample() * (self.maxy - self.miny) + self.miny)
        self.cov_matrix = self.create_cov_matrix()

    def create_cov_matrix(self):
        eigenvalx = (np.random.random_sample() * (self.maxx - self.minx) / 5) ** 2
        eigenvaly = (np.random.random_sample() * (self.maxy - self.miny) / 5) ** 2
        D = np.array([[eigenvalx, 0], [0, eigenvaly]])
        # angle = np.random.random_sample() * 2 * np.pi # angle in radians
        # R = np.array([[np.cos(angle), np.sin(-angle)], [np.sin(angle), np.cos(angle)]])
        angle = np.random.random_sample()*360
        theta = np.radians(angle)
        cos, sin = np.cos(theta), np.sin(theta)
        R = np.array([[cos,-sin],[sin,cos]])
        cov_matrix = np.dot(np.dot(np.transpose(R), D), R)
        return cov_matrix

    def get_sample(self, n):
        return np.random.multivariate_normal(self.mean, self.cov_matrix, n)


def sample_gauss_2d(C, N):
    if C == 1:
        G = Random2DGaussian()
        X = G.get_sample(N)
        Y = np.array([0] * N)
        return X, Y
    else:
        G = Random2DGaussian()
        X = G.get_sample(N)
        Y = np.array([0] * N)
        for i in range(1, C):
            G = Random2DGaussian()
            X = np.vstack((X, G.get_sample(N)))
            Y = np.vstack((Y, [i] * N))
        return X, Y.flatten()


def sample_gmm_2d(K, C, N):
    if K == 0 or C < 1:
        raise Exception("Number of classes needs to be 1 or more, and number of bivariant distributions needs to be greater than zero")
    for i in range(0, K):
        G = Random2DGaussian()
        rand_class = np.random.randint(low=0, high=C)
        if i == 0:
            X = G.get_sample(N)
            Y = [rand_class] * N
            continue
        X = np.vstack((X, G.get_sample(N)))
        Y = np.vstack((Y, [rand_class] * N))

    return X, Y.flatten()


# def graph_data(X, Y_, Y):
#     correctly_classified = X[np.where(Y == Y_)]
#     incorrectly_classified = X[np.where(Y != Y_)]
#
#     colors = np.array([0.3] * len(X))
#     colors[np.where(Y_ == 1)] = 1
#
#     color_correct = colors[np.where(Y == Y_)]
#     color_incorrect = colors[np.where(Y != Y_)]
#
#     plt.scatter(correctly_classified[:, 0], correctly_classified[:, 1],
#                 color=zip(color_correct, color_correct, color_correct), marker='o', edgecolors=(0, 0, 0))
#     plt.scatter(incorrectly_classified[:, 0], incorrectly_classified[:, 1],
#                 color=zip(color_incorrect, color_incorrect, color_incorrect), marker='s', edgecolors=(0, 0, 0))


def graph_data(X, Y_, Y, special=[]):
    correct = np.where(Y == Y_)
    incorrect = np.where(Y != Y_)
    correctly_classified = X[correct]
    incorrectly_classified = X[incorrect]

    sizes = np.array([50] * len(X))
    if len(special) != 0:
        sizes[special] *= 3

    correct_sizes = sizes[correct]
    incorrect_sizes = sizes[incorrect]

    colors = np.array([0.3] * len(X))
    colors[np.where(Y_ == 1)] = 1

    color_correct = colors[np.where(Y == Y_)]
    color_incorrect = colors[np.where(Y != Y_)]

    plt.scatter(correctly_classified[:, 0], correctly_classified[:, 1],
                color=zip(color_correct, color_correct, color_correct), marker='o', edgecolors=(0, 0, 0),
                s=correct_sizes)
    plt.scatter(incorrectly_classified[:, 0], incorrectly_classified[:, 1],
                color=zip(color_incorrect, color_incorrect, color_incorrect), marker='s', edgecolors=(0, 0, 0),
                s=incorrect_sizes)


# def graph_surface(fun, rect, offset=0):
#     xmin, ymin = rect[0]
#     xmax, ymax = rect[1]
#     x_range = np.linspace(xmin, xmax, num=1000)
#     y_range = np.linspace(ymin, ymax, num=1000)
#     x, y = np.meshgrid(x_range, y_range)
#     grid = np.stack((x.flatten(), y.flatten())).transpose()
#     dec = fun(grid)
#
#     delta = np.abs(dec - offset).max()
#     c = cntr.Cntr(x, y, dec.reshape(x.shape))
#     res = c.trace(offset)
#     nseg = len(res) // 2
#     segments, codes = res[:nseg], res[nseg:]
#     plt.plot(segments[0][:, 0], segments[0][:, 1], color=(0, 0, 0))
#     plt.pcolormesh(x, y, dec.reshape(x.shape), vmin=offset - delta, vmax=offset + delta)


def graph_surface(fun, rect, offset=0):
    xmin, ymin = rect[0]
    xmax, ymax = rect[1]
    x_range = np.linspace(xmin, xmax, num=1000)
    y_range = np.linspace(ymin, ymax, num=1000)
    x, y = np.meshgrid(x_range, y_range)

    grid = np.stack((x.flatten(), y.flatten())).transpose()  # isto ko i ono sto radis sa unravelom
    dec = fun(grid)  # vrijednost funkcije
    delta = np.abs(dec - offset).max()  # delta blabla...
    plt.contour(x, y, dec.reshape(x.shape), levels=[offset])  # reshapeam ko ti jelte
    plt.pcolormesh(x, y, dec.reshape(x.shape), vmin=offset - delta, vmax=offset + delta)

def labels_to_one_hot(Y,C):
    Yoh_ = np.zeros((Y.shape[0],C))
    Yoh_[range(Y.shape[0]),Y] = 1
    return Yoh_
