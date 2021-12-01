from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

from algorithm import SemismoothNewton
from problem import QuadraticProblem

gtol = 1e-12

n = 2**5

alpha = 1e-3
beta = 1e-3

lb = Constant(-10.)
ub = Constant(1.)

g = Constant(0.0)

yd = Expression("cos(75.0*x[0]/pi)*sin(5.0*x[1])", degree = 1)
kappa = Constant(1.0/100.)

quadratic = QuadraticProblem(n=n,m=n,yd=yd, g=g, alpha=alpha, beta=beta, lb=lb, ub=ub, kappa=kappa)
newton = SemismoothNewton(quadratic, gtol, max_iter = 50, globalization = "MonotonicityTest")
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
plt.savefig("control.pdf")
plt.close()

plot(quadratic.y)
plt.savefig("state.pdf")
plt.close()

quadratic = QuadraticProblem(n=n,m=n,yd=yd, g=g, alpha=alpha, beta=beta, lb=lb, ub=ub, kappa=kappa)
newton = SemismoothNewton(quadratic, gtol, max_iter = 50, globalization = "NewtonStep")
newton.solve()

# Check optimality
measure = quadratic.criticality_measure(quadratic.u_vec)
print("Criticality measure={}".format(measure))

file = File("solution_newtonstep.pvd")
file << quadratic.u

quadratic.state_vec(quadratic.u_vec)
file = File("state_newtonstep.pvd")
file << quadratic.y

plot(quadratic.u)
plt.savefig("control_newtonstep.pdf")
plt.close()

plot(quadratic.y)
plt.savefig("state_newtonstep.pdf")
plt.close()

