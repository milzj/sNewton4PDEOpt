import pytest

from dolfin import *
import numpy as np
from problem  import QuadraticProblem
import scipy.sparse as sp

@pytest.mark.parametrize("n", [32, 64, 128, 256])
@pytest.mark.parametrize("m", [15, 64, 128, 128])

def test_projection(n, m):
    """Checks if projections computed using projection matrix and dolfin's project 
    are close."""

    tol = 1e-14
    degree_rise = 0

    quadratic = QuadraticProblem(n=n,m=n)

    Psp = quadratic.P_mat

    V = quadratic.V
    U = quadratic.U
    bcs = quadratic.bcs

    z = Function(V)
    z_vec = np.random.randn(V.dim())
    z.vector()[:] = z_vec

    u = Function(U)
    p_vec = Psp.dot(z_vec)
    u.vector()[:] = p_vec

    # Making sure z in H_0^1
    bcs.apply(z.vector())
    p = project(z, U)

    err = errornorm(p, u, degree_rise = degree_rise)
    assert err < tol

    u_vec = np.random.randn(U.dim())
    quadratic.state_vec(u_vec)

    p_vec = Psp @ quadratic.y_vec
    u.vector()[:] = p_vec

    z.vector()[:] = quadratic.y_vec
    p = project(z, U)


    err = errornorm(p, u, degree_rise = degree_rise)
    assert err < tol

