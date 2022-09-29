import pytest

from dolfin import *
from problem import QuadraticProblem
from algorithm import SemismoothNewton

import numpy as np

def problem(n=32):
    """Simple test problem."""

    alpha = 1e-3
    beta = 1e-3

    lb = Constant(-30.0)
    ub = Constant(30.0)

    yd = Expression("exp(2*x[0])*sin(2*pi*x[0])*sin(2*pi*x[1])/6.", degree = 1)
    g = Constant(1.0)

    return QuadraticProblem(n=n, m=n, yd=yd, g=g, alpha=alpha, beta=beta, lb=lb, ub=ub)


def test_nonsmooth(n=128):

    gtol = 1e-12

    quadratic = problem(n=n)

    newton = SemismoothNewton(quadratic, gtol)
    newton.solve()

    cmeasure = quadratic.criticality_measure(quadratic.u_vec)

    assert cmeasure < 10.0*gtol


def test_zero_solution(n=128):
    """Tests if zero is solution if beta equals inf norm of gradient."""

    gtol = 1e-15

    quadratic = problem(n=n)

    grad = quadratic.gradient_vec(0.0*quadratic.u_vec)

    beta = np.linalg.norm(grad, ord=np.inf)
    quadratic.beta = beta

    measure = quadratic.criticality_measure(0.0*quadratic.u_vec)
    assert measure  < gtol

    newton = SemismoothNewton(quadratic, gtol=gtol)
    newton.solve()

    assert norm(quadratic.u, "L2") < gtol

    alpha = quadratic.alpha
    norm_normal_map = quadratic.norm_normal_map_vec(-1/alpha*grad)

    assert norm_normal_map < gtol

