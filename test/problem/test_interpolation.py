import pytest

import dolfin
from problem import QuadraticProblem
import numpy as np


@pytest.mark.parametrize("lbs", [-np.inf, -10.0])
def test_interpolation(lbs):
	"""Tests if yd, g, lb, ub are approximated as expected."""
	n = 32
	tol = 1e-14


	lb = dolfin.Constant(lbs)
	ub_expr = dolfin.Expression('x[0] <= 0.25 ? 0 : -5.0+20.0*x[0]', degree=1)

	yd_expr = dolfin.Expression("sin(4*pi*x[0])*cos(8*pi*x[1])*exp(2.0*x[0])", degree = 1)
	g_expr = dolfin.Expression("10.0*cos(8*pi*x[0])*cos(8*pi*x[1])", degree = 1)

	alpha = 0.1
	beta = 0.1

	quadratic = QuadraticProblem(n=n, m=n, yd=yd_expr, g=g_expr, alpha=alpha, beta=beta, lb=lb, ub=ub_expr)

	mesh = dolfin.UnitSquareMesh(n,n)
	U = dolfin.FunctionSpace(mesh, "DG", 0)
	V = dolfin.FunctionSpace(mesh, "CG", 1)

	ub = dolfin.Function(U)
	ub.interpolate(ub_expr)
	ub_vec = ub.vector()[:]

	lb_vec = lbs*np.ones(2*n**2)

	assert dolfin.errornorm(quadratic.ub, ub, degree_rise = 0) < tol
	assert np.all(quadratic.lb_vec == lb_vec)
	assert np.all(quadratic.ub_vec == ub_vec)

	yd = dolfin.Function(U)
	yd.interpolate(yd_expr)

	assert dolfin.errornorm(quadratic.yd, yd, degree_rise = 0) < tol

	g = dolfin.Function(U)
	g.interpolate(g_expr)

	assert dolfin.errornorm(quadratic.g, g, degree_rise = 0) < tol
