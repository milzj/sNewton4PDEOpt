from .prox_l1 import prox_l1
from .proj_box import proj_box

import scipy.sparse as sp

def grad_prox_box_l1(v, lb, ub, lam):
	"""Compute proximal operator for box constaints and l1-norm and its generalized derivative.

	Computes proximal operator and an element of the subdifferential.
	The proximal operator is computed using a composition formula.

	TODO: Should we use (abs(v) >= lam) instead of (abs(v) > lam)?

	Parameters:
	-----------
		v : ndarray
			input array
		lb, ub : ndarray or float
			lower and upper bound
		lam : float
			parameter

	Returns:
	--------
		(proximal_operator, subgradient) : (ndarray, scipy.sparse.dia.dia_matrix)
			proximal operator and one of its generalized subgradients.

	References:
	-----------

	Examples 3.2.8, 3.2.9 and 4.2.17 in

	A. Milzarek, Numerical methods and second order theory for nonsmooth problems,
	Dissertation, TUM, Munich, http://mediatum.ub.tum.de/?id=1289514
	"""

	w = prox_l1(v, lam)

	delta = (w > lb) * (w < ub) * (abs(v) > lam)

	return ( proj_box(w, lb, ub), sp.diags(delta.astype(float)) )
