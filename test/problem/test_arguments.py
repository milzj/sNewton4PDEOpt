import pytest

from dolfin import *
from problem import QuadraticProblem


def test_check_arguments():

    n = 32

    lb = Constant(-10.0)
    ub = Expression('x[0] <= 0.25 ? 0 : -5.0+20.0*x[0]', degree=1)

    yd = Expression("sin(4*pi*x[0])*cos(8*pi*x[1])*exp(2.0*x[0])", degree = 1)
    g = Expression("10.0*cos(8*pi*x[0])*cos(8*pi*x[1])", degree = 1)

    alpha = 0.1
    beta = 0.0

    quadratic = QuadraticProblem(n=n,m=n, yd=yd, g=g, alpha=alpha, beta=beta, lb=lb, ub=ub)

    with pytest.raises(ValueError):
        QuadraticProblem(n=-1,m=n, yd=0.0, g=g, alpha=alpha, beta=beta, lb=lb, ub=ub)

    with pytest.raises(ValueError):
        QuadraticProblem(n=n,m=-10, yd=0.0, g=g, alpha=alpha, beta=beta, lb=lb, ub=ub)

    with pytest.raises(ValueError):
        QuadraticProblem(n=n,m=n, yd=0.0, g=g, alpha=-10.0, beta=beta, lb=lb, ub=ub)

    with pytest.raises(ValueError):
        QuadraticProblem(n=n,m=n, yd=0.0, g=g, alpha=alpha, beta=-10., lb=lb, ub=ub)


    with pytest.raises(TypeError):
        QuadraticProblem(n=n,m=n, yd=yd, g=g, alpha=1, beta=beta, lb=lb, ub=ub)

    with pytest.raises(TypeError):
        QuadraticProblem(n=n,m=n, yd=0.0, g=g, alpha=alpha, beta=beta, lb=lb, ub=ub)


    with pytest.raises(TypeError):
        QuadraticProblem(n=n,m=n, yd=0.0, g=g, alpha=alpha, beta=0.0, lb=lb, ub=0.0)
