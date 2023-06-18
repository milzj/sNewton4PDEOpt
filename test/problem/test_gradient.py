import pytest

from dolfin import *
from problem import QuadraticProblem

import numpy as np

def convergence_rates(E_values, eps_values, show=True):
    """
    Source https://github.com/dolfin-adjoint/pyadjoint/blob/master/pyadjoint/verification.py

    """
    from numpy import log
    r = []
    for i in range(1, len(eps_values)):
        r.append(log(E_values[i] / E_values[i - 1])
        / log(eps_values[i] / eps_values[i - 1]))
    if show:
        print("Computed convergence rates: {}".format(r))
    return r

def taylor_test(fun, u_vec, h_vec, dJdm=0.0, Hm=0.0):
    """
    Adapted from https://github.com/dolfin-adjoint/pyadjoint/blob/master/pyadjoint/verification.py
    """

    Jm = fun(u_vec)
    def perturbe(eps):
        return u_vec + eps * h_vec
    
    residuals = []
    epsilons = [0.01 / 2 ** i for i in range(6)]
    for eps in epsilons:
        Jp = fun(perturbe(eps))
        res = abs(Jp - Jm - eps * dJdm - 0.5 * eps ** 2 * Hm)
        residuals.append(res)
    
    if min(residuals) < 1E-15:
        logging.warning("The taylor remainder is close to machine precision.")
    print("Computed residuals: {}".format(residuals))
    return np.median(convergence_rates(residuals, epsilons))

def test_gradient():
    """Preforms a Taylor test."""

    n = 64

    alpha = 1e-4
    beta = 0.003
    rtol = 1e-1

    lb = Constant(-30.0)
    ub = Constant(30.0)

    yd = Expression("exp(2*x[0])*sin(2*pi*x[0])*sin(2*pi*x[1])/6.", degree = 1)
    g = Expression("10.0*cos(8*pi*x[0])*cos(8*pi*x[1])", degree = 1)

    quadratic =  QuadraticProblem(n=n, m=n, yd=yd, g=g, alpha=alpha, beta=beta, lb=lb, ub=ub)

    u_vec = np.random.randn(quadratic.U.dim())
    h_vec = np.random.randn(quadratic.U.dim())


    fun = lambda u_vec: quadratic.smooth_objective(u_vec)

    assert np.isclose(taylor_test(fun, u_vec, h_vec), 1.0, rtol=rtol)

    grad = quadratic.gradient_vec(u_vec)
    deriv = quadratic.M_mat @ grad
    dJdm = deriv.dot(h_vec)

    assert np.isclose(taylor_test(fun, u_vec, h_vec, dJdm=dJdm), 2.0, rtol=rtol)
