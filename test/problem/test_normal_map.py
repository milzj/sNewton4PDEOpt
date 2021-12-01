import pytest

from dolfin import *
from problem import QuadraticProblem
from algorithm import SemismoothNewton

import numpy as np

def problem(n=32):

	alpha = 1e-7
	beta = 0.003

	lb = Constant(-30.0)
	ub = Constant(30.0)

	yd = Expression("exp(2*x[0])*sin(2*pi*x[0])*sin(2*pi*x[1])/6.", degree = 1)
	g = Constant(1.0)

	return QuadraticProblem(n=n, m=n, yd=yd, g=g, alpha=alpha, beta=beta, lb=lb, ub=ub)


def test_normal_map(n=64):
	"""Checks if normal map equals gradient evaluated as the zero control."""

	gtol = 1e-15

	quadratic = problem(n=n)

	u_zero = 0.0*quadratic.u_vec

	grad = quadratic.gradient_vec(u_zero)
	normal_map = quadratic.normal_map_vec(u_zero)

	assert np.linalg.norm(grad-normal_map, ord=np.inf) < gtol
