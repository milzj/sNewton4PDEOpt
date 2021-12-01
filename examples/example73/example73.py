"""Implements Example 7.3 from Wachsmuth and Wachsmuth (2011).

References:
-----------

G. Wachsmuth and D. Wachsmuth, Convergence and regularization results for optimal
control problems with sparsity functional, ESAIM Control. Optim. Calc. Var., 17 (2011),
pp. 858–886, https://doi.org/10.1051/cocv/2010027.

"""

from dolfin import *
import numpy as np

class Parameter(object):

	def __init__(self,alpha, beta):


		self.__check_arguments(alpha, beta)

		self.alpha = alpha
		self.beta = beta

		lb = -1.
		ub = 54.0/7.0

		self.lb = lb
		self.ub = ub

		self.a = 23328.0*beta - 5832.0*alpha - 5832.0*alpha*ub
		self.b = -9720.0*beta + 2268.0*alpha + 2592.0*alpha*ub
		self.c = 1296.0*beta  - 288.0*alpha  - 378.0*alpha*ub
		self.d = -55.0*beta   + 12.0*alpha   + 18.0*alpha*ub
		self.e = -432.0*beta  + 648.0*alpha
		self.f = 108.0*beta   - 216.0*alpha


	def __check_arguments(self, alpha, beta):

		if not isinstance(alpha, float):
			raise TypeError("alpha should be float.")
		if not isinstance(beta, float):
			raise TypeError("beta should be float.")
		if not (0 < alpha and alpha < beta):
			raise ValueError("Need: 0 < alpha < beta.")


class Solution(UserExpression):
	"""Implements the optimal control.

	__init__ requires calling super as noted by D. Kamensky.

	References:
	-----------

	D. Kamensky (2019): https://fenicsproject.discourse.group/t/meshes-with-subdomains-broken-tutorial/435

	"""
	def __init__(self, lb, ub, **kwargs):

		super(Solution, self).__init__(**kwargs)

		self.lb = lb
		self.ub = ub

	def eval(self, value, x):
		r2 = (.5 - x[0])**2 + (.5 - x[1])**2
		r = sqrt(r2)

		if r < 1.0/18.0:
			v = self.ub
		if 1.0/18.0 <= r < 1.0/9.0:
			v = -18.0*self.ub*(r - 1.0/9.0)
		if 1.0/9.0 <= r < 1./6.0:
			v = 0.0
		if 1./6.0 <= r < 2.0/9.0:
#			v = -18.0*(x[0] - 1.0/6.0)
			v = -18.0*(r - 1.0/6.0)
		if 2.0/9.0 <= r < 5.0/18.0:
			v = self.lb
		if 5.0/18.0 <= r < 1.0/3.0:
			v = -6.0 + 18.0 * r
		if 1.0/3.0 <= r:
			v = 0.0

		value[0] = v

	def value_shape(self):

		return (1,)

class Adjoint(UserExpression):
	"""Implements the optimal adjoint state."""
	def __init__(self, params, **kwargs):

		super(Adjoint, self).__init__(**kwargs)

		self.params = params
		self.solution = Solution(params.lb, params.ub, **kwargs)

	def eval(self, value, x):
		r2 = (.5 - x[0])**2 + (.5 - x[1])**2
		r = sqrt(r2)

		alpha = self.params.alpha
		beta = self.params.beta
		ub = self.params.ub
		solution = self.solution

		v = 0.0

		if r < 1.0/18.0:
			v = -162.0*alpha*ub*r**2 + beta + 3.0*alpha*ub/2
		if 1.0/18.0 <= r < 1.0/9.0:
			val = [0.0]
			solution.eval(val, x)
			v = beta + alpha*val[0]
		if 1.0/9.0 <= r < 1./6.0:
			v = self.params.a*r**3 + self.params.b*r**2 + self.params.c*r + self.params.d
		if 1./6.0 <= r < 2.0/9.0:
			val = [0.0]
			solution.eval(val, x)
			v = -beta + alpha*val[0]
		if 2.0/9.0 <= r < 5.0/18.0:
			v = 324.0*alpha*(r-0.25)**2 - beta - 5*alpha/4
		if 5.0/18.0 <= r < 1.0/3.0:
			val = [0.0]
			solution.eval(val, x)
			v = -beta + alpha*val[0]
		if 1.0/3.0 <= r < 1.0/2.0:
			v = self.params.e*(r-1/3)**3 \
				+ self.params.f*(r-1/3)**2 \
				+ 18.0*alpha*(r-1/3) \
				 -beta

		value[0] = v

	def value_shape(self):

		return (1,)


class LaplaceAdjoint(UserExpression):
	"""Implements the Laplacian of optimal adjoint state."""
	def __init__(self, params, **kwargs):

		super(LaplaceAdjoint, self).__init__(**kwargs)

		self.params = params

	def eval(self, value, x):
		r2 = (.5 - x[0])**2 + (.5 - x[1])**2
		r = sqrt(r2)

		v = 0.0

		if r < 1.0/18.0:
			v = -648.0*self.params.alpha*self.params.ub
		if 1.0/18.0 <= r < 1.0/9.0:
#			v = self.params.alpha*self.params.ub - 18.0/r
			v = -self.params.alpha*self.params.ub*18.0/r
		if 1.0/9.0 <= r < 1./6.0:
			v = 9.0*self.params.a*r + 4.0*self.params.b + self.params.c/r
		if 1./6.0 <= r < 2.0/9.0:
#			v = self.params.alpha - 18.0/r
			v = -self.params.alpha*18.0/r
		if 2.0/9.0 <= r < 5.0/18.0:
			v = 324.0*self.params.alpha*(4.0-1/(2.0*r))
		if 5.0/18.0 <= r < 1.0/3.0:
			v = 18.0*self.params.alpha/r
		if 1.0/3.0 <= r < 1.0/2.0:
			v = self.params.e*(9.0*r-4.0+1/(3.0*r)) \
				+ self.params.f*(4.0-2.0/(3.0*r)) \
				+ 18.0*self.params.alpha/r

		value[0] = v

	def value_shape(self):

		return (1,)


class State(UserExpression):
	"""Implements the optimal state."""
	def __init__(self, params, **kwargs):

		super(State, self).__init__(**kwargs)

		self.params = params
		self.values = []

	def eval(self, value, x):

		r2 = (.5 - x[0])**2 + (.5 - x[1])**2
		r = sqrt(r2)
		v = 0.0

		if r < 1.0/18.0:
			v += self.params.ub*r**2/4.0 - 1.0/168.0
			r = 1.0/18.0
		if 1.0/18.0 <= r < 1.0/9.0:
			v += self.params.ub*r**2/2.0 - 2.0*self.params.ub*r**3\
					-np.log(9.0*r)/252.0-5.0/189.0
			r = 1.0/9.0
		if 1.0/9.0 <= r < 1./6.0:
			v += np.log(6.0*r)/36.0
			r = 1.0/6.0
		if 1.0/6.0 <= r < 2.0/9.0:
			v += 3.0*r**2/4.0 - 2.0*r**3 + np.log(9.0*r/2.0)/72.0 - 11.0/729.0
			r = 2.0/9.0
		if 2.0/9.0 <= r < 5.0/18.0:
			v += -r**2/4.0 + 91.0*np.log(18.0*r/5.0)/1944.0 + 25.0/1296.0
			r = 5.0/18.0
		if 5.0/18.0 <= r < 1.0/3.0:
			v += -3.0*r**2/2.0 + 2.0*r**3 + np.log(3.0*r)/9.0 + 5.0/54.0

#		value[0] = v
		value[0] = -v

	def value_shape(self):

		return (1,)


class DesiredState(UserExpression):
	"""Implements the desired state."""

	def __init__(self, params, **kwargs):

		super(DesiredState, self).__init__(**kwargs)

		self.state = State(params)
		self.laplace_adjoint = LaplaceAdjoint(params)

	def eval(self, value, x):

		value1 = [0.0]
		self.state.eval(value1, x)

		value2 = [0.0]
		self.laplace_adjoint.eval(value2, x)

		value[0] = value1[0] - value2[0]

	def value_shape(self):

		return (1,)


