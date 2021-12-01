from dolfin import *
import numpy as np

from algorithm import SemismoothNewton
from problem import QuadraticProblem

import matplotlib.pyplot as plt

gtol = 1e-12

n = 2**7

alpha = 1e-4
beta = 0.0
ratio = 0.1

lb = Constant(-30.0)
ub = Constant(30.0)

yd = Expression("exp(2*x[0])*sin(2*pi*x[0])*sin(2*pi*x[1])/6.", degree = 1)
g = Constant(0.0)

quadratic = QuadraticProblem(n=n,m=n, yd=yd, g=g, alpha=alpha, beta=beta, lb=lb, ub=ub)

# Update beta
u_vec = quadratic.u_vec
z = quadratic.gradient_vec(0.0*u_vec)

z_norm_inf = np.linalg.norm(z, ord=np.inf)

beta = ratio * z_norm_inf
quadratic.beta = beta
print("beta={}".format(beta))

newton = SemismoothNewton(quadratic, gtol)
newton.solve()

# Check optimality
measure = quadratic.criticality_measure(quadratic.u_vec)
print("Criticality measure={}".format(measure))

file = File("solution.pvd")
file << quadratic.u

quadratic.state_vec(quadratic.u_vec)
file = File("state.pvd")
file << quadratic.y


plot(quadratic.u)
plt.savefig("solution.pdf")
plt.close()

plot(quadratic.y)
plt.savefig("state.pdf")
plt.close()
