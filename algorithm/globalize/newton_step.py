from .globalization import Globalization

class NewtonStep(Globalization):

	def globalize(self, residual_map, residual_map_old, gtol):

		sigma = 1.0

		return sigma, residual_map(sigma)

