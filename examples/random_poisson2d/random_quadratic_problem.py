from dolfin import *
from problem import QuadraticProblem
import numpy as np

class RandomQuadraticProblem(QuadraticProblem):

    def __init__(self, n, m, alpha=1e-3, beta=1e-3,
                yd=Constant(0.0), 
                g=Constant(0.0),
                lb=Constant(-np.inf), 
                ub=Constant(np.inf)):

        super(RandomQuadraticProblem, self).__init__(n=n,m=m, 
                            alpha=alpha, 
                            beta=beta, 
                            yd=yd, g=g, 
                            lb=lb, ub=ub)

        self._sample = [1.0, 1.0, 0.0]
    
    def smooth_objective(self, u_vec):
        raise NotImplementedError()

    @property
    def sample(self):

        return self._sample

    @sample.setter
    def sample(self, sample):

        self._sample[:] = sample

    @property
    def N_mat(self):
        """Multiplies N_mat by sample[0]."""

        return self._sample[0]*self._N_mat

    @property
    def yd_vec(self):
        """Multiplies yd_vec by sample[1]."""

        return self._sample[1]*self._yd_vec

    @property
    def g_vec(self):
        """Multiplies g_vec by sample[2]."""

        return self._sample[2]*self._g_vec
