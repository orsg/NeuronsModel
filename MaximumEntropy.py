import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import networkx as nx
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from itertools import product

class MaxEntropyOptimizer(object):

    MAX_ITERATIONS = 1000
    STOP_CRITERION = 0.001
    STEP = 0.01

    def __init__(self, neurons_data):
        self.data = neurons_data
        self.lambdas = None
        self.model_marginals = None
        self.emp_marginals = None
        self.sample_space = np.array(list(product([0, 1], repeat=self.data.shape[1])))
        self.delta = 0

    def optimize(self, max_iterations=MAX_ITERATIONS):
        self.lambdas = np.ones((max_iterations+1, self.data.shape[1], self.data.shape[1]))
        self.delta = np.zeros((max_iterations + 1, self.data.shape[1], self.data.shape[1]))
        self.compute_emp_constraints()
        for self.iteration in tqdm(range(max_iterations)):
            self.compute_model_constraints()
            # update with the gradient direction
            self.delta[self.iteration] = (self.model_marginals - self.emp_marginals)
            self.lambdas[self.iteration + 1] = self.lambdas[self.iteration] + self.STEP * self.delta[self.iteration]
            if np.abs(self.delta[self.iteration]).max() < self.STOP_CRITERION:
                break
            # TODO: regularization?
            # TODO: dynamic steps
            # TODO: stop criterion
        return self.lambdas, self.iteration, self.delta, self.emp_marginals

    def compute_model_constraints(self):
        probs = np.apply_along_axis(lambda x: np.exp(-1 * np.triu(np.outer(x, x) * self.lambdas[self.iteration]).sum()),
                                    1,
                                    self.sample_space)
        probs = probs / probs.sum()
        self.model_marginals = np.apply_along_axis(lambda x: np.triu(np.outer(x, x)),
                                                   1,
                                                   (self.sample_space.T*np.sqrt(probs)).T).sum(axis=0)

    def compute_emp_constraints(self):
        # use the diagonal of the matrix as the independent constraints
        self.emp_marginals = np.apply_along_axis(lambda x: np.triu(np.outer(x, x)),
                                                 1,
                                                 self.data.values).mean(axis=0)
