import pytest

from dolfin import *
from problem import QuadraticProblem


def reference_solution(mesh, kappa=Constant(1.0), u=Constant(1.0), g=Constant(1.0)):

	V = FunctionSpace(mesh, 'CG', 1)
	bc = DirichletBC(V, Constant(0), 'on_boundary')
	y, v = TrialFunction(V), TestFunction(V)
	a = inner(kappa*grad(y), grad(v))*dx
	L = u*v*dx + g*v*dx
	yh = Function(V)
	solve(a == L, yh, bcs=bc)

	return yh


@pytest.mark.parametrize("n", [32, 64, 128])

def test_pde_solution(n):
	"""Test if PDE solution computed with scipy equals PDE solution computed with dolfin."""
	degree_rise = 0
	gtol = 1e-12

	u_expr = Expression("sin(4*pi*x[0])*cos(8*pi*x[1])*exp(2.0*x[0])", degree = 1)
	g_expr = Expression("10.0*cos(8*pi*x[0])*cos(8*pi*x[1])", degree = 1)
	kappa = Expression("x[1]*x[1]+0.05", degree = 1)


	mesh = UnitSquareMesh(MPI.comm_self, n, n)
	U = FunctionSpace(mesh, 'DG', 0)
	u = Function(U)
	u.interpolate(u_expr)

	u_vec = u.vector()[:]

	quadratic = QuadraticProblem(n=n, m=n, g=g_expr, kappa=kappa)
	quadratic.state_vec(u_vec)
	y = quadratic.y

	g = Function(U)
	g.interpolate(g_expr)

	y_ref = reference_solution(mesh, g=g, kappa=kappa, u=u)

	error = errornorm(y, y_ref, degree_rise = degree_rise)
	assert error < 10.0*gtol

