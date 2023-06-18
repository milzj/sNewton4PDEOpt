from dolfin import *
import scipy.sparse as sp
import numpy as np

from assemble_system import assemble_system_csr, assemble_csr, assemble_rhs


class Discretization(object):
    """Discretizes a linear-quadratic optimal control problem.

    Discretizes the Poisson equation on either 
    the unit interval or the unit square with
    Dirichlet boundary to obtain the sparse linear system

        A_mat y_vec = G_mat u_vec + g_vec.

    Moreover it discretizes the input functions yd, lb and ub 
    used to define a linear quadratic control problem, and computes 
    the state and control stiffness matrix.

    The state space H0^1([0,1]^d) is approximated using
    piecewise linear functions (CG1) und the control
    space (L2([0,1]^d)is approximated using piecewise constant
    functions (DG0). 

    The input functions yd, lb, and ub are approximated by DG0 functions.
    """

    def __init__(self, n, m, yd, g, kappa, lb, ub):

        self.__mesh(n,m)
        self.__function_spaces()
        self.__boundary_conditions()
        self.__trial_test_functions()


        self.__stiffness_matrix(kappa)
        self.__mass_matrix_state()
        self.__mass_matrix_control()

        self.__linear_operator()

        self.__projection_matrix()

        self.__functions(yd, g, lb, ub)

        self.__vecs()


    def __mesh(self, n, m):
        """Create mesh

        If m == 0, then a UnitIntervalMesh with n cells is created.
        """

        if m == 0:
            mesh = UnitIntervalMesh(n)
        elif m > 0:
            mesh = UnitSquareMesh(n,m)

        self.mesh = mesh

    def __function_spaces(self):
        """Define function spaces."""

        mesh = self.mesh

        V = FunctionSpace(mesh, "CG", 1)
        U = FunctionSpace(mesh, "DG", 0)

        self.V = V
        self.U = U

    def __functions(self, yd, g, lb, ub):

        self._y = Function(self.V)
        self._u = Function(self.U)

        self._yd = Function(self.U)
        self.yd = yd

        self._g = Function(self.U)
        self.g = g

        self._lb = Function(self.U)
        self.lb = lb
        self._ub = Function(self.U)
        self.ub = ub

    def __vecs(self):

        ny, nu = self._G_mat.shape

        self._u_vec = np.zeros(nu)
        self._y_vec = np.zeros(ny)

    def __boundary_conditions(self):
        "Defines Dirichlet boundary conditions."

        self.bcs = DirichletBC(self.V, Constant(0.0), lambda x, on_boundary: on_boundary)

    def __trial_test_functions(self):
        """Trial and test functions for control and state space."""

        V = self.V
        U = self.U

        self.__test_fun_V = TestFunction(V)
        self.__trial_fun_V = TrialFunction(V)
        self.__trial_fun_U = TrialFunction(U)
        self.__test_fun_U = TestFunction(U)


    def __stiffness_matrix(self, kappa):
        """Computes the stiffness matrix.

        Parameters:
        ----------
            kappa : Constant, Function
                diffusion coefficient				
        """

        trial_fun_V = self.__trial_fun_V
        test_fun_V = self.__test_fun_V

        a = kappa*dot(grad(trial_fun_V), grad(test_fun_V))*dx
        L = Constant(1.0)*test_fun_V*dx

        A_mat, b_vec = assemble_system_csr(a, L, bcs = self.bcs)
        self._A_mat = A_mat

        A_matT, b_vec = assemble_system_csr(adjoint(a), L, bcs = self.bcs)
        self._A_matT = A_matT

        self.__L = L
        self.__b_vec = b_vec

    def __mass_matrix_state(self):
        """Computes the state mass matrix N_mat."""

        test_fun_V = self.__test_fun_V
        trial_fun_V = self.__trial_fun_V
        L = self.__L

        N = test_fun_V*trial_fun_V*dx

        N_mat, b_vec = assemble_system_csr(N, L, bcs = self.bcs)
        self._N_mat = N_mat

    def __mass_matrix_control(self):
        """Computes the control mass matrix M_mat."""

        trial_fun_U = self.__trial_fun_U
        test_fun_U = self.__test_fun_U

        M = trial_fun_U*test_fun_U*dx

        M_mat = assemble_csr(M)
        self._M_mat = M_mat

    def __linear_operator(self):
        """Computes the control mass matrix G_mat."""

        trial_fun_U = self.__trial_fun_U
        test_fun_V = self.__test_fun_V
        b_vec = self.__b_vec

        G = trial_fun_U*test_fun_V*dx

        G_mat_ = assemble_csr(G)
        b_vec[:] = 1.0
        self.bcs.apply(b_vec)
        self.b_vec = b_vec
        G_mat = sp.diags(b_vec) @ G_mat_
        self._G_mat = G_mat

        self._G_matT = sp.csr_matrix.transpose(G_mat)

    def __projection_matrix(self):
        """Computes the projection matrix P_mat = inv(M_mat) @ G_mat.T.

        The projection matrix `P_mat` can be used to compute
        the gradient of the smooth objective function given the
        solution to the adjoint equation.
        """

        M_mat = self._M_mat
        M_mat_inv = sp.csr_matrix((1.0/M_mat.data, M_mat.indices, M_mat.indptr))
        self._P_mat = M_mat_inv @ self.G_matT

    @property
    def A_mat(self):
        return self._A_mat

    @property
    def N_mat(self):
        return self._N_mat

    @property
    def M_mat(self):
        return self._M_mat

    @property
    def P_mat(self):
        return self._P_mat

    @property
    def G_mat(self):
        return self._G_mat

    @property
    def G_matT(self):
        return self._G_matT

    @property
    def yd(self):
        return self._yd

    @yd.setter
    def yd(self, yd):
        test_fun_V = self.__test_fun_V
        self._yd.interpolate(yd)
        self._yd_vec = assemble_rhs(self._yd*test_fun_V*dx, bcs = self.bcs)

    @property
    def yd_vec(self):
        return self._yd_vec

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, g):
        """
        Note:
        -----
        Using self._g_vec = assemble_rhs(g*self.v*dx, bcs = self.bcs)
        might yield a different _g_vec than computed below. 
        """
        self._g.interpolate(g)
        trial_fun_V = self.__trial_fun_V
        test_fun_V = self.__test_fun_V
        a = dot(grad(trial_fun_V), grad(test_fun_V))*dx
        L = self._g*test_fun_V*dx
        A_mat, b_vec = assemble_system_csr(a, L, bcs = self.bcs)
        self._g_vec = b_vec

    @property
    def g_vec(self):
        return self._g_vec

    @property
    def u_vec(self):
        return self._u_vec

    @u_vec.setter
    def u_vec(self, u_vec):
        self._u_vec[:] = u_vec
        self._u.vector()[:] = u_vec

    @property
    def u(self):
        return self._u

    @property
    def y_vec(self):
        return self._y_vec

    @y_vec.setter
    def y_vec(self, y_vec):

        self._y_vec[:] = y_vec
        self._y.vector()[:] = y_vec

    @property
    def y(self):
        return self._y

    # Lower bounds
    @property
    def lb_vec(self):
        return self._lb_vec

    @property
    def lb(self):
        return self._lb

    @lb.setter
    def lb(self, lb):
        self._lb.interpolate(lb)
        self._lb_vec = self._lb.vector()[:]

    # Upper bounds
    @property
    def ub_vec(self):
        return self._ub_vec

    @property
    def ub(self):
        return self._ub

    @ub.setter
    def ub(self, ub):
        self._ub.interpolate(ub)
        self._ub_vec = self._ub.vector()[:]


