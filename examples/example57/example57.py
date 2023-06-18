"""Implements Example 1 from Stadler (2009).

References:
-----------
G. Stadler, Elliptic optimal control problems with L 1 -control cost and applications
for the placement of control devices, Comput. Optim. Appl., 44 (2009), pp. 159–181

K. Kunisch and D. Walter. On fast convergence rates for generalized conditional gradient
methods with backtracking stepsize. arXiv:2109.15217
"""

from dolfin import *
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt

from algorithm import SemismoothNewton
from problem import QuadraticProblem

gtol = 1e-12

n = 64

beta = 0.001
alpha = 1e-5

lb = Constant(-30.0)
ub = Constant(30.0)

yd = Expression("exp(2*x[0])*sin(2*pi*x[0])*sin(2*pi*x[1])/6.", degree = 1)
g = Constant(0.0)

quadratic = QuadraticProblem(n=n,m=n, yd=yd, g=g, alpha=alpha, beta=beta, lb=lb, ub=ub)

newton = SemismoothNewton(quadratic, gtol, globalization = "MonotonicityTest")
newton.solve()

# Check optimality
measure = quadratic.criticality_measure(quadratic.u_vec)
print("Criticality measure={}".format(measure))

file = File("solution.pvd")
file << quadratic.u

quadratic.state_vec(quadratic.u_vec)
file = File("state.pvd")
file << quadratic.y


c = plot(quadratic.u)
plt.colorbar(c)
plt.savefig("solution.pdf")
plt.savefig("solution.png")
plt.close()

plot(quadratic.y)
plt.savefig("state.pdf")
plt.savefig("state.png")
plt.close()

C = sp.bmat([	[quadratic.A_mat, None],\
        [quadratic.N_mat, quadratic.A_mat]],format='csr')

d = np.hstack([quadratic.G_mat @ quadratic.u_vec + quadratic.g_vec, quadratic.yd_vec])
yz = spsolve(C, d)
y_vec, z_vec = np.split(yz, 2)

V = quadratic.V
z = Function(V)
z.vector().set_local(z_vec)

c = plot(z)
plt.colorbar(c)
plt.savefig("adjoint.pdf")
plt.savefig("adjoint.png")
plt.close()



# Postprocessing 1

solution = np.zeros(quadratic.U.dim())
solution[:] = quadratic.u_vec
val = quadratic.smooth_objective(solution)

quadratic.alpha = 0.0
gradient = quadratic.gradient_vec(solution)

u_post = np.zeros(quadratic.U.dim())

idx = gradient > beta
u_post[idx] = -30.0

idx = gradient < -beta
u_post[idx] = 30.0

quadratic.u_vec = u_post

c = plot(quadratic.u)
plt.colorbar(c)
plt.savefig("solution_postporcessed.pdf")
plt.savefig("solution_postprocessed.png")
plt.close()

# Compute objective function values

val = quadratic.smooth_objective(solution)
u = quadratic.u
L1norm = assemble(abs(u)*dx(None, {'quadrature_degre': 5}))
obj_val = val + beta*L1norm
F1 = "F(solution)+beta*norm(solution,L1)={}".format(obj_val)
print(F1)

val = quadratic.smooth_objective(u_post)
_u_post = Function(quadratic.U)
_u_post.vector().set_local(u_post)
L1norm = assemble(abs(_u_post)*dx(None, {'quadrature_degre': 5}))
obj_val = val + beta*L1norm
F2 = "F(solution_post)+beta*norm(solution_post,L1)={}".format(obj_val)
print(F2)


np.savetxt("objective_values.txt", ["n={}".format(n), F1, F2], fmt="%s")
