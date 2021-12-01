from .globalization import Globalization

import warnings


class MonotonicityTest(Globalization):
	"""Restricted residual monotonicity test

	The restricted residual monotonicity test computes the largest
	step size `sigma` in {1, 1/2, 1/4, ..., sigma_min} such that

		residual_map(sigma) <= (1 + sigma/100 + 2e-4)*residual_map_old.

	If such a parameter `sigma` does not exist, then `sigma = sigma_min/2`
	is chosen and a warning is printed.

	Instead of using 1/4 as in eq. (3.32) in Deulfhard (2011), we use 1/100.

	The term 2e-4 tries to take into account for potentially inaccurate evaluations
	of the residual mapping, assumed to be accurate up to four significant figures.

	The default value of `sigma_min` is taken from Mannel and Rund (2020).

	References:
	-----------

	P. Deuflhard, Newton Methods for Nonlinear Problems, Springer Ser. Comput.
	Math. 35, Springer, Berlin, 2011, https://doi.org/10.1007/978-3-642-23899-4

	F. Mannel and A. Rund, A hybrid semismooth quasi-Newton method for
	nonsmooth optimal control with PDEs, Optim. Eng., (2020),
	https://doi.org/10.1007/s11081-020-09523-w
	"""


	def __init__(self):

		self.sigma_min = .5**10


	def globalize(self, residual_map, residual_map_old, gtol):

		sigma = 1.0
		sigma_min = self.sigma_min

		residual_map_new = residual_map(sigma)

		while residual_map_new > (1.0-sigma/100.0+2e-4)*residual_map_old and sigma >= sigma_min:
			sigma /= 2.0
			residual_map_new = residual_map(sigma)

		if sigma <= sigma_min:
			warnings.warn("Step size sigma = {} is smaller than minimum step size = {}.".format(sigma, sigma_min))

		return sigma, residual_map_new
