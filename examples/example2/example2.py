"""Implements Example 2 from Stadler (2009).

References:
-----------
G. Stadler, Elliptic optimal control problems with L 1 -control cost and applications
for the placement of control devices, Comput. Optim. Appl., 44 (2009), pp. 159–181
"""

from dolfin import *
import numpy as np

import matplotlib.pyplot as plt

from algorithm import SemismoothNewton
from problem import QuadraticProblem

gtol = 1e-12

n = 128

alpha = 1e-6
beta = 0.001

lb = Constant(-np.inf)
ub = Constant(np.inf)

yd = Expression("exp(2*x[0])*sin(2*pi*x[0])*sin(2*pi*x[1])/6.", degree = 1)
g = Constant(0.0)

quadratic = QuadraticProblem(n=n,m=n, yd=yd, g=g, alpha=alpha, beta=beta, lb=lb, ub=ub)

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
