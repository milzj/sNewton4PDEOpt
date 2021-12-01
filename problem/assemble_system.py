from dolfin import *
import scipy.sparse as sp

def assemble_system_csr(A_form, b_form, bcs = None):
	"""Assemble bilinear form and rhs with boundary conditions to CSR sparse matrix.

	Returns symmetric stiffness matrix by calling `assemble_system`.

	Parameters:
	----------
		A_form : ufl.Form
			bilinear form
		b_form : ufl.Form
			linear form
		bcs : DirichletBC
			boundary conditions

	Returns:
	--------
		A_mat : scipy.sparse.csr_matrix
			assembled bilinear form with boundary conditions applied.
		b : dolfin.cpp.la.PETScVector
			right-hand side with boundary conditions applied.
	"""

	A = PETScMatrix()
	b = PETScVector()

	assemble_system(A_form, b_form, bcs=bcs, A_tensor = A, b_tensor = b)

	mat = as_backend_type(A).mat()
	A_mat = sp.csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)

	return A_mat, b

def assemble_system_csc(A_form, b_form, bcs = None):
	"""Assemble bilinear form and rhs with boundary conditions to CSC sparse matrix.

	The method calls assemble_system_csr and then converts the CSR matrix to a CSC matrix.

	The right-hand side b is converted a a numpy.array.

	TODO: 	Avoid convertion of matrix format.
		See https://github.com/scipy/scipy/blob/v1.6.2/scipy/sparse/linalg/dsolve/linsolve.py#L318

	Parameters:
	----------
		A_form : ufl.Form
			bilinear form
		b_form : ufl.Form
			linear form
		bcs : DirichletBC
			boundary conditions

	Returns:
	--------
		A_mat : scipy.sparse.csc_matrix
			assembled bilinear form with boundary conditions applied 
			as symmetric matrix.
		b : dolfin.cpp.la.PETScVector
			right-hand side with boundary conditions applied.
	"""

	A_mat, b = assemble_system_csr(A_form, b_form, bcs = bcs)
	A_mat = sp.csc_matrix(A_mat)

	return A_mat, b

def assemble_csr(A_form, bcs = None):
	"""Assemble bilinear form with boundary conditions to CSR sparse matrix.

	Parameters:
	----------
		A_form : ufl.Form
			bilinear form
		bcs : DirichletBC
			boundary conditions

	Returns:
	--------
		A_mat : scipy.sparse.csc_matrix
			assembled bilinear form with boundary conditions applied 
			as possibly nonsymmetric matrix.

	References:
	-----------

	C. Clason (2019): https://github.com/clason/tvwavecontrol/blob/master/tvwavecontrol.py#L21

	"""
	A = assemble(A_form)

	if bcs:
		bcs.apply(A)

	mat = as_backend_type(A).mat()

	return sp.csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)


def assemble_rhs(b_form, bcs = None):
	"""Assemble linear form with boundary conditions.

	Parameters:
	----------
		b_form : ufl.Form
			linear form
		bcs : DirichletBC
			boundary conditions

	Returns:
	--------
		b_vec : ndarray
			right-hand side with boundary conditions applied.

	"""

	b = assemble(b_form)

	if bcs:
		bcs.apply(b)

	b_vec = b[:]

	return b_vec
