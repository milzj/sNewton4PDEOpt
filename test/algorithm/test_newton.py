import pytest

from problem import QuadraticProblem
from algorithm import SemismoothNewton
from dolfin import *

try:
	from dolfin_adjoint import *
except ImportError:
	import sys
	print("dolfin_adjoint unavailable, skipping test.")
	sys.exit(0)

try:
	from moola import *
except ImportError:
	import sys
	print("moola unavailable, skipping test.")
	sys.exit(0)

set_log_level(LogLevel.ERROR)

def reference_solution(n, yd, alpha, g, gtol):

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

	problem = MoolaOptimizationProblem(rf)
	u_moola = DolfinPrimalVector(u)

	solver = NewtonCG(problem, u_moola, options={'gtol': gtol,
                	                                   'maxiter': 20,
                        	                           'display': 2,
                                	                   'ncg_hesstol': 0})
	sol = solver.solve()

	return sol, problem


@pytest.mark.parametrize("n", [32, 64])
@pytest.mark.parametrize("alpha", [1e-3, 1e-2])

def test_newton(n, alpha):
	"""Tests if solution computed by semismooth Newton method
	is close to the solution computed with moola.NewtonCG."""

	gtol = 1e-9

	degree_rise = 0
	beta = 0.0

	yd = Expression("sin(x[0])*sin(x[1])", degree = 1)
	g = Expression("exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree = 1)

	quadratic = QuadraticProblem(n=n, m=n, yd=yd, g=g, alpha=alpha, beta=beta)

	newton = SemismoothNewton(quadratic, gtol)
	newton.solve()

	u_newton = quadratic.u
	obj_u_newton = quadratic.smooth_objective(quadratic.u_vec)

	sol, problem = reference_solution(n, yd, alpha, g, gtol)

	u_ref = sol["control"].data
	u_ref_vec = u_ref.vector().get_local()

	# checks if reference solution is approximate solution
	grad_vec = quadratic.gradient_vec(u_ref_vec)
	assert quadratic.norm_vec(grad_vec) < 1e3*gtol

	# checks if solution solves reference problem
	u_newton_moola = DolfinPrimalVector(u_newton)
	gradient = problem.obj.derivative(u_newton_moola).primal()
	grad_norm = gradient.norm()
	assert grad_norm < 1e3*gtol

	# checks if solutions are close
	error = errornorm(u_ref, u_newton, degree_rise = degree_rise)
	assert error < 1e3*gtol

