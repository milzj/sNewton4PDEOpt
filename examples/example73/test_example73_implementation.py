from example73 import Solution, Parameter, State, LaplaceAdjoint, DesiredState, Adjoint
from dolfin import *
import numpy as np

import problem

def test_state_adjoint(alpha=1e-4, beta=1.5e-4, n=128, atol = 1e-5):
    """Solves elliptic PDE with right-hand side equal to the optimal solution and
    checks whether the optimal state is approximately equal
    to the PDE solution. A related test is applied to the adjoint
    question.
    """
    degree_rise = 0

    mesh = UnitSquareMesh(n, n)
    element = FiniteElement("DG", mesh.ufl_cell(), 0)
    U = FunctionSpace(mesh, element)

    params = Parameter(alpha, beta)

    # solution
    solution = Solution(params.lb, params.ub,\
                element = element,\
                domain = mesh)

    u = project(solution, U)

    # state
    element = FiniteElement("CG", mesh.ufl_cell(), 1)
    state = State(params, element = element, domain = mesh)

    V = FunctionSpace(mesh, "CG", 1)

    y = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(y), grad(v))*dx
    L = solution*v*dx

    y = Function(V)
    bcs = DirichletBC(V, 0.0, "on_boundary")
    solve(a == L, y, bcs=bcs)

    assert errornorm(state, y, degree_rise = degree_rise, mesh = mesh) < atol

    # adjoint
    p = Function(V)
    element = FiniteElement("DG", mesh.ufl_cell(), 0)
    laplace_adjoint = LaplaceAdjoint(params, element = element, domain = mesh)
    L = -laplace_adjoint*v*dx
    solve(a == L, p, bcs = bcs)

    adjoint = Adjoint(params, element = element, domain = mesh)

    assert errornorm(adjoint, p, degree_rise = degree_rise, mesh = mesh) < atol


def test_optimality(alpha = 1e-4, beta = 1.5e-4, n = 256, atol = 1e-14):
    """Computes the proximal operator of 1/alpha * (optimal adjoint state)
    and checks whether it is close to the optimal solution.
    """
    degree_rise = 0

    mesh = UnitSquareMesh(n, n)
    element = FiniteElement("DG", mesh.ufl_cell(), 0)
    U = FunctionSpace(mesh, element)

    params = Parameter(alpha, beta)

    solution = Solution(params.lb, params.ub, element = element, domain = mesh)

    lb = Constant(params.lb)
    ub = Constant(params.ub)
    g = Constant(0.0)

    quadratic = problem.QuadraticProblem(n=n,m=n, alpha = alpha, beta = beta, lb = lb, ub = ub, g = g)

    adjoint = Adjoint(params, element = element, domain = mesh)

    p = project(adjoint, U)
    p_vec = p.vector()[:]
    z_vec = quadratic.prox(1/alpha*p_vec)
    z = Function(U)
    z.vector()[:] = z_vec

    assert errornorm(solution, z, degree_rise = degree_rise) < atol
