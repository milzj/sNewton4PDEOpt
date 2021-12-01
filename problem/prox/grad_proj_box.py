from prox_l1 import prox_l1
from proj_box import proj_box

import scipy.sparse as sp

def grad_proj_box(v, lb, ub):
	"""Computes projection onto box and its generalized derivative.

	Computes projection of `v` onto `[lb, ub]` and an element of the subdifferential.

	Parameters:
	----------
		v : np.ndarray
			input array
		lb : np.ndarray, float
			lower bound
		ub : np.ndarray, float
			upper bound

	Returns:
	--------
		(projection, subgradient) : (ndarray, scipy.sparse.dia.dia_matrix)
			tuple of projection and subgradient

	References:
	-----------

	Examples 3.2.8, 3.2.9 and 4.2.17 in

	A. Milzarek, Numerical methods and second order theory for nonsmooth problems,
	Dissertation, TUM, Munich, http://mediatum.ub.tum.de/?id=1289514

	p. 93 in

	S. Garreis, Optimal Control under Uncertainty: Theory and Numerical Solution
	with Low-Rank Tensors, TUM, Munich, http://mediatum.ub.tum.de/node?id=1452538
	"""

	w = proj_box(v, lb, ub)
	delta = (w > lb) * (w < ub)

	return ( w, sp.diags(delta.astype(float)) )
