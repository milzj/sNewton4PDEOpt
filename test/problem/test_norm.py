import pytest

from dolfin import *
from problem import QuadraticProblem
from algorithm import SemismoothNewton

import numpy as np

@pytest.mark.parametrize("n", [32, 64, 128])

def test_norm(n):

    rtol = 1e-14
    quadratic = QuadraticProblem(n=n, m=n)

    U = quadratic.U
    u = Function(U)

    u.vector()[:] = np.random.randn(U.dim())

    u_vec = u.vector()[:]

    u_vec_norm = quadratic.norm_vec(u_vec)
    u_norm = norm(u, norm_type="L2")

    assert np.isclose(u_vec_norm, u_norm, rtol=rtol)

