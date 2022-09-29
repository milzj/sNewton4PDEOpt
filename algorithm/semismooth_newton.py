import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.linalg import norm
import numpy as np

import warnings

from globalize import NewtonStep
from globalize import MonotonicityTest


class SemismoothNewton(object):
    """Semismooth Newton method for linear quadratic optimal control
    problems.

    - The Newton method is applied to a normal mapping (see eq. (3.3)
    in Pieper (2015)). The normal mapping is a potentially nonsmooth
    operator equation.

    - Terminates if `norm(normal mapping) <= gtol` or the maximum number
    of iterations is reached.

    - As globalization either `NewtonStep` (no globalization) or the
    `MonotonicityTest` can be used.

    - The method does not check whether the initial value `problem.u_vec`
    is a solution.

    - The initial control used for the semismooth Newton method is computed
    using phase-one. It computes the solution to the unconstrained problem with
    beta = 0 (cf. sect. 5 in Stadler (2009)).

    - Linear systems are solved using `npsolve` of scipy.

    - Currently, no print level can be specified.


    Attributes:
    -----------
        problem : QuadraticProblem
            control problem
        gtol : float, optional
            termination tolerance
        max_iter : int, optional
            maximum number of iterations
        globalization : string, optional
            globalization (either NewtonStep or MonotonicityTest)


    References:
    -----------
    K. Pieper, Finite element discretization and efficient numerical solution of
    elliptic and parabolic sparse control problems, Dissertation, TUM, Munich, 2015

    G. Stadler, Elliptic optimal control problems with L1-control cost and applications
    for the placement of control devices, Comput. Optim. Appl., 44 (2009), pp. 159–181
    """

    def __init__(self, problem, gtol = 1e-8, max_iter = 30, globalization = "NewtonStep"):
        """Initializes the semismooth Newton method.

        The function's input parameter are class attributes.
        """

        self.problem = problem
        self.gtol = gtol
        self.max_iter = max_iter
        self.globalization = get_globalization(globalization)

    def solve(self):
        """Solves optimal control problem.

        The solution is stored in `problem.u_vec` (as a numpy
        vector) and in `problem.u` (as a fenics function).
        """

        print("\n ==================")
        print("\n Semismooth Newton.")
        print("\n ==================")

        # Problem data
        alpha = self.problem.alpha

        A_mat = self.problem.A_mat
        G_mat = self.problem.G_mat
        N_mat = self.problem.N_mat
        G_matT = self.problem.G_matT
        M_mat = self.problem.M_mat
        P_mat = self.problem.P_mat

        ny, nu = G_mat.shape

        g_vec = self.problem.g_vec
        yd_vec = self.problem.yd_vec
        u_vec = self.problem.u_vec

        # Phase-one
        print("\n Phase-One \n")
        alpha_I = alpha*sp.identity(nu, format='csr')

        C = sp.bmat([	[A_mat,None,-G_mat],\
                [N_mat,A_mat,None],\
                [None, -P_mat, alpha_I]],format='csr')

        d = np.hstack([g_vec, yd_vec, np.zeros(nu)])

        yzw = spsolve(C, d)
        y_vec,z_vec, v_vec = np.split(yzw, [ny, 2*ny])

        norm_normal_map_old = self.problem.norm_normal_map_vec(v_vec)
        print("norm of normal map={}".format(norm_normal_map_old))

        print("\n Main loop \n")
        i = 0
        while i < self.max_iter and norm_normal_map_old > self.gtol:
            i += 1

            # Assemble and solve (full) Newton system
            p_vec, D_mat = self.problem.grad_prox(v_vec)

            DG =  G_mat @ D_mat
            C = sp.bmat([	[A_mat,None,-DG],\
                    [N_mat,A_mat,None],
                    [None, -P_mat, alpha_I]],format='csr')

            d = -np.hstack([ A_mat @ y_vec  - G_mat @ p_vec - g_vec,\
                     A_mat @ z_vec  + N_mat @ y_vec - yd_vec,\
                     -P_mat @ z_vec + alpha * v_vec])

            yzv = spsolve(C, d)
            dy_vec,dz_vec,dv_vec = np.split(yzv, [ny, 2*ny])

            norm_residual_map = lambda sigma: self.problem.norm_normal_map_vec(v_vec+sigma*dv_vec)
            sigma, norm_normal_map_new = self.globalization.globalize(norm_residual_map, norm_normal_map_old, self.gtol)

            y_vec += sigma*dy_vec
            z_vec += sigma*dz_vec

            v_vec += sigma*dv_vec
            norm_normal_map_old = norm_normal_map_new

            norm_dv_vec = self.problem.norm_vec(dv_vec)

            # Print data
            print("iter={}".format(i))
            print("Step length={}".format(sigma))
            print("norm of step={}".format(norm_dv_vec))
            print("norm of normal map={}".format(norm_normal_map_old))
            print("\n")

        if i >= self.max_iter:
            warnings.warn("Maximum number of iterations = {} reached".format(i))
        elif norm_normal_map_old > self.gtol:
            raise ValueError("Convergence failure.")

        # Compute and save control
        u_vec = self.problem.prox(v_vec)
        self.problem.u_vec = u_vec



def get_globalization(globalization):

    if globalization == "NewtonStep":

        globalize = NewtonStep()

    elif globalization == "MonotonicityTest":

        globalize = MonotonicityTest()

    else:
        raise ValueError("Unknown globalization={}".format(globalization))

    return globalize
