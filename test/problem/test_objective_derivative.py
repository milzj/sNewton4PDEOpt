import pytest

from problem import QuadraticProblem
from algorithm import SemismoothNewton
from dolfin import *
import numpy as np

try:
	from dolfin_adjoint import *
except ImportError:
	import sys
	print("dolfin_adjoint unavailable, skipping test.")
	sys.exit(0)

set_log_level(LogLevel.ERROR)

def reduced_functional(n, yd, alpha, g):

	tape = Tape()
	set_working_tape(tape)

	mesh = UnitSquareMesh(n, n)
	V = FunctionSpace(mesh, "CG", 1)
	U = FunctionSpace(mesh, "DG", 0)

	bcs = DirichletBC(V, Constant(0.0), lambda x, on_boundary: on_boundary)
	y = Function(V)
	v = TestFunction(V)
	u = Function(U)

	_g = Function(U)
	_g.interpolate(g)

	_yd = Function(U)
	_yd.interpolate(yd)

	F = (inner(grad(y), grad(v)) - u * v - _g*v) * dx

	solve(F == 0, y, bcs)

	J = 0.5 * (y - _yd)**2 * dx + alpha / 2 * u ** 2* dx

	control = Control(u)
	rf = ReducedFunctional(assemble(J), control)

	return rf


@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("alpha", [1e-6, 1e-3, 1e-1])

def test_objective_derivative(n, alpha):
	"""Tests if objective values and derivatives are close."""
	degree_rise = 0
	beta = 0.0

	yd = Expression("sin(x[0])*sin(x[1])", degree = 1)
	g = Expression("exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree = 1)

	quadratic = QuadraticProblem(n=n, m=n, yd=yd, g=g, alpha=alpha)

	rf = reduced_functional(n, yd, alpha, g)

	U = quadratic.U
	u = Function(U)
	u.vector()[:] =  np.random.randn(U.dim())
	u_vec = u.vector()[:]

	val = quadratic.smooth_objective(u_vec)
	grad_vec = quadratic.gradient_vec(u_vec)
	grad_vec = quadratic.M_mat @ grad_vec

	derivative = Function(U)
	derivative.vector()[:] = grad_vec


	rf_val = rf(u)
	rf_derivative = rf.derivative()

	assert np.isclose(val, rf_val, rtol=1e-12)

	assert errornorm(derivative, rf_derivative, degree_rise = degree_rise) < 1e-12
