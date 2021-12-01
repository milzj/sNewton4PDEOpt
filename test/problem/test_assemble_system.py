from dolfin import *
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy.sparse.linalg import spsolve
import numpy as np

from problem import assemble_system_csr, assemble_csr, assemble_rhs

def issymmetric(A, atol=1e-14):

	B = A - A.T

	return splinalg.norm(B) < atol

def test_issymmetric(n=10):

	A = sp.random(n, n)
	A = .5*(A+A.T)

	assert issymmetric(A)

def _solve(a, L, bcs, V):

	y = Function(V)
	solve( a == L, y, bcs = bcs)

	return y

def _spsolve(A_mat, b_vec, V):

	y_vec = spsolve(A_mat, b_vec)

	y = Function(V)
	y.vector()[:] = y_vec

	return y

def pde(n=128):

	mesh = UnitSquareMesh(n, n)
	V = FunctionSpace(mesh, "CG", 1)

	bcs = DirichletBC(V, 0.0, "on_boundary")

	u = TrialFunction(V)
	v = TestFunction(V)

	kappa = Expression("x[1]*x[1]+0.05", degree = 1)
	f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)",degree=1)

	L = f*v*dx
	a = kappa*inner(grad(u), grad(v))*dx

	return a, L, bcs, V

def test_assemble_system_csr(n=128, atol=1e-11):
	"""Check if fenics.solve and scipy.linalg.spsolve
	produce same solution and if assembled bilinear
	form is symmetric.
	"""

	a, L, bcs, V = pde(n=n)

	y = _solve(a, L, bcs, V)

	A_mat, b_vec = assemble_system_csr(a, L, bcs = bcs)

	assert issymmetric(A_mat)

	_y = _spsolve(A_mat, b_vec, V)

	err = errornorm(y, _y, degree_rise = 0)

	assert err < atol

def test_assemble_csr(n=128, atol=1e-11):

	a, L, bcs, V = pde(n=n)

	y = _solve(a, L, bcs, V)

	A_mat = assemble_csr(a, bcs = bcs)
	b_vec = assemble_rhs(L, bcs = bcs)

	_y = _spsolve(A_mat, b_vec, V)

	err = errornorm(y, _y, degree_rise = 0)

	assert err < atol


