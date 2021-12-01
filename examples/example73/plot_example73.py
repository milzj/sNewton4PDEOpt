"""Creates 2D plots of solution, optimal (adjoint) state and desired state."""
from dolfin import *

import matplotlib.pyplot as plt

import sys
import numpy as np

from example73 import Solution, Parameter, State, LaplaceAdjoint, DesiredState, Adjoint

if __name__ == "__main__":

	outdir = "output/"
	import os
	if not os.path.exists(outdir):
	    os.makedirs(outdir)

	N = 2*64

	alpha = 1e-4
	beta = 1.5e-4

	degree = 0
	mesh = UnitSquareMesh(N, N)
	element = FiniteElement("DG", mesh.ufl_cell(), degree)
	U = FunctionSpace(mesh, element)

	params = Parameter(alpha, beta)

	# solution
	solution = Solution(params.lb, params.ub,\
				element = element,\
				domain = mesh)

	u = project(solution, U)

	plot(u)
	plt.savefig(outdir + "/solution_n=" + str(N) +".pdf", bbox_inches="tight")
	plt.close()


	# state
	state = State(params, element = element, domain = mesh)

	V = FunctionSpace(mesh, "CG", 1)
	y = project(state, V)

	plot(y)
	plt.savefig(outdir + "/state_n=" + str(N) +".pdf", bbox_inches="tight")
	plt.close()

	# adjoint
	adjoint = Adjoint(params, element = element, domain = mesh)

	V = FunctionSpace(mesh, "CG", 1)
	z = project(adjoint, V)

	plot(z)
	plt.savefig(outdir + "/adjoint_n=" + str(N) +".pdf", bbox_inches="tight")
	plt.close()

	# desired sate
	desired_state = DesiredState(params, \
				element = element, domain = mesh)

	yd = project(desired_state, U)

	plot(yd)
	plt.savefig(outdir + "/yd_n=" + str(N) +".pdf", bbox_inches="tight")
	plt.close()




