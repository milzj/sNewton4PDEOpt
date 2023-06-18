from dolfin import *

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from prox import proj_box, grad_proj_box, prox_box_l1, grad_prox_box_l1

from discretization import Discretization

class QuadraticProblem(Discretization):
    """Defines discretized linear elliptic control problem."""

    def __init__(self, n=2**3, m=0,
            alpha=1e-3,
            beta=0.0,
            yd=Constant(0.0),
            g=Constant(0.0),
            kappa=Constant(1.0),
            lb=Constant(-np.inf),
            ub=Constant(np.inf)):
        """

        Parameters:
        ----------
            n : int
                number of cells in horizontal direction
            m : int
                number of cells in vertical direction, m = 0 if domain is unit interval
            alpha : float
                positve regularization parameter
            beta : float
                nonnegative sparsity parameter
            yd, g, kappa, lb, ub : Constant, Expression, Function
                desired state, source term, 
                diffusion coefficient,
                lower and upper bounds
        """


        self.__check_arguments(n, m, alpha, beta, yd, g, kappa, lb, ub)
        super().__init__(n, m, yd, g, kappa, lb, ub)

        self.alpha = alpha
        self.beta = beta

    def __check_arguments(self, n, m, alpha, beta, yd, g, kappa, lb, ub):

        if n <= 0 or not isinstance(n, int):
            raise ValueError("n = {} must be a positive integer.".format(n))
        if m < 0 or not isinstance(m, int):
            raise ValueError("m = {} must be a nonnegative integer.".format(m))
        if alpha <= 0.0:
            raise ValueError("alpha = {} must be a positive number".format(alpha))
        if beta < 0.0:
            raise ValueError("beta = {} must be a nonnegative number".format(beta))
        if isinstance(alpha, int):
            raise TypeError("alpha = {} must a float".format(alpha))
        if isinstance(beta, int):
            raise TypeError("beta = {} must a float".format(beta))

        inputs = [yd, g, kappa, lb, ub]
        classinfo = (Constant, Expression, Function)

        for input in inputs:
            if not isinstance(input, classinfo):
                raise TypeError("yd, g, kappa, lb, ub should be instances of" +
                        "Constant, Expression or Function.")



    def criticality_measure(self, u_vec):
        """Evaluates a criticality measure at u_vec.

        Computes the norm of `u_vec-prox(-1/alpha * grad(u_vec))`
        which is a criticality measure for the control problem, that is,
        the function returns zero if and only if the input `u_vec`
        is the solution to the strongly convex control problem.

        Parameters:
        -----------
            u_vec : ndarray
                design variable as numpy vector, e.g.,
                approximate solution.

        Returns:
        --------
            
            criticality_measure : float
                criticality measure evaluated at u_vec.
        """

        alpha = self.alpha
        gamma = 1.0/alpha

        gradient = self.gradient_vec(u_vec)
        w_vec = u_vec-self.prox(u_vec-gamma*gradient)

        return self.norm_vec(w_vec)

    def smooth_objective(self, u_vec):
        """Evaluates the smooth part of the composite objective function.

        Evaluates the tracking-type cost plus the control cost.
        `errornorm` is used to evaluate the tracking-type part.

        Parameters:
        -----------
            u_vec : ndarray
                design variable as numpy vector, e.g.,
                approximate solution.
        """

        alpha = self.alpha

        self.state_vec(u_vec)

        J = .5*errornorm(self.yd, self.y, degree_rise = 0)**2
        J += .5*alpha*self.norm_vec(u_vec)**2

        return J


    def prox(self, v_vec):
        """Computes the proximity operator.

        Parameter:
        ----------
            v_vec : ndarray
                vector whose proximity operator is desired.

        Returns:
        --------
            prox_v_vec : float
                proximity operator of v_vec.

        References:
        -----------

        p. 818 in

        E. Casas, R. Herzog, and G. Wachsmuth, Optimality conditions
        and error analysis of semilinear elliptic control problems with
        L1 cost functional, SIAM J. Optim., 22 (2012),
        pp. 795--820, https://doi.org/10.1137/110834366.
        """

        return prox_box_l1(v_vec, self.lb_vec, self.ub_vec, self.beta/self.alpha)

    def grad_prox(self, v_vec):
        """Computes proximal operator and its generalized derivative.

        Similar to `prox` but also computes a generalized derivative of the
        prox operator.

        Parameters:
        ----------
            v_vec : ndarray
                vector whose proximity operator is desired.

        Returns:
        --------
            (proximal_operator, subgradient) : (ndarray, scipy.sparse.dia.dia_matrix)
                proximal operator and one of its generalized subgradients.

        """

        return grad_prox_box_l1(v_vec, self.lb_vec, self.ub_vec, self.beta/self.alpha)

    def norm_vec(self, v_vec):
        """Computes input's vector norm defined by stiffness matrix."""

        return np.sqrt(np.dot(v_vec, self.M_mat.dot(v_vec)))

    def error_norm(self, v, w, degree_rise = 0):
        """Calls fenics' errornorm function."""

        return errornorm(v, w, degree_rise = degree_rise)

    def state_vec(self, u_vec):
        """Computes solution to PDE given the input.

        The PDE solution is stored in `y_vec` as a vector
        and in `y` as a function.
        """

        y_vec = spsolve(self.A_mat, self.G_mat @ u_vec + self.g_vec)
        self.y_vec = y_vec

    def gradient_vec(self, u_vec):
        """Compute gradient of smooth objective function.

        Solves the state and adjoint equations at once.

        Returns:
        -------
            grad_vec : ndarray
                gradient
        """
        C = sp.bmat([	[self.A_mat, None],\
                [self.N_mat, self.A_mat]],format='csr')

        d = np.hstack([self.G_mat @ u_vec + self.g_vec, self.yd_vec])
        yz = spsolve(C, d)
        y_vec, z_vec = np.split(yz, 2)

        return - self.P_mat @ z_vec + self.alpha * u_vec

    def normal_map_vec(self, v_vec):
        """Compute normal map.

        Solves the state and adjoint equations at once.

        Returns:
        --------
            nm_vec : ndarray
                evaluation of normal mapping.
        """
        alpha = self.alpha
        u_vec = self.prox(v_vec)

        C = sp.bmat([	[self.A_mat, None],\
                [self.N_mat, self.A_mat]],format='csr')

        d = np.hstack([self.G_mat @ u_vec + self.g_vec, self.yd_vec])
        yz = spsolve(C, d)
        y_vec, z_vec = np.split(yz, 2)

        return  -self.P_mat @ z_vec + alpha * v_vec

    def norm_normal_map_vec(self, v_vec):
        """Computes the norm of the normal map."""

        return self.norm_vec(self.normal_map_vec(v_vec))


