"""Element assembly shared by the static solvers."""

import numpy as np


class DenseAssembly:
    def __init__(self, size):
        self.stiffness = np.zeros((size, size))

    def add(self, dofs, stiffness):
        self.stiffness[np.ix_(dofs, dofs)] += stiffness

    def finish(self):
        return self.stiffness
