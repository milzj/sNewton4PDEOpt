"""Solves a sample average approximation of

min (1/2) E[norm(y(u,xi)-yd)**2] + (alpha/2) norm(u)**2 + beta norm(u,L1),

where y(u,xi) solves the weak form of Poisson equation with random inputs

    - kappa(xi) Laplacian(y) = u + g.

Here g and kappa are a real-valued random variables. Since kappa is 
real-valued, the expectation function's gradient
can be computed with two PDE solutions (see random_poisson2d.tex) by
modifying problem data N_mat, yd_vec, and g_vec.
"""

from dolfin import *
import numpy as np
from random_quadratic_problem import RandomQuadraticProblem
from sampler import Sampler
from algorithm import SemismoothNewton

import matplotlib.pyplot as plt

n = 2**7
# Number of samples
N = 1000

alpha = 1e-4
beta = 0.0

yd = Expression("exp(2.0*x[0])*sin(2.0*pi*x[0])*sin(2.0*pi*x[1])/6.0", degree = 0)
g = Constant(1.0)

lb = Constant(-10.0)
ub = Constant(10.0)

gtol = 1e-12

quadratic = RandomQuadraticProblem(n, n,
                yd=yd, g=g,
                alpha=alpha,
                beta=beta,
                lb=lb, ub=ub)

sampler = Sampler()

# Update beta
u_vec = 0.0*quadratic.u_vec
grad_vecs = []

for i in range(0, 10):
    sample = sampler.sample(1)
    quadratic.sample = sample
    g_vec = quadratic.gradient_vec(u_vec)
    grad_vecs.append(g_vec)

grad_vec = np.mean(grad_vecs, axis=0)
beta = .1*np.linalg.norm(grad_vec, ord=np.inf)

print("beta={}".format(beta))
quadratic.beta = beta

# Generate a sample of size N
sample = sampler.sample(N)
quadratic.sample = sample

# Solve equivalent reformulation of sample average approximation problem
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
