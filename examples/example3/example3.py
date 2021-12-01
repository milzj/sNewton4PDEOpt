"""Implements Example 3 from Stadler (2009).

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

alpha = 0.0002
beta = 0.002

lb = Constant(-10.0)
ub = Expression('x[0] <= 0.25 ? 0 : -5.0+20.0*x[0]', degree=1)

yd = Expression("sin(4*pi*x[0])*cos(8*pi*x[1])*exp(2.0*x[0])", degree = 1)
g = Expression("10.0*cos(8*pi*x[0])*cos(8*pi*x[1])", degree = 1)

quadratic = QuadraticProblem(n=n, m=n, yd=yd, g=g, alpha=alpha, beta=beta, lb=lb, ub=ub)

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
