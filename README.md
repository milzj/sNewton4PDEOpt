# sNewton4PDEOpt


The Python module implements a semismooth Newton method for solving
finite-element discretizations of the strongly convex, linear elliptic PDE-constrained optimization
problem

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Balign*%7D%0A%09%5Cmin_%7Bu%20%5Cin%20U_%7B%5Ctext%7Bad%7D%7D%7D%5C%2C%20(1%2F2)%20%0A%09%5C%7CS(u)-y_d%5C%7C_%7BL%5E2(D)%7D%5E2%0A%20%20%20%20%2B%20(%5Calpha%2F2)%5C%7Cu%5C%7C_%7BL%5E2(D)%7D%5E2%20%0A%20%20%20%20%2B%20%5Cbeta%20%5C%7Cu%5C%7C_%7BL%5E1(D)%7D%2C%20%0A%5Cend%7Balign*%7D">

where <img src="https://render.githubusercontent.com/render/math?math=S(u)%20%5Cin%20H_0%5E1(D)">
solves the linear elliptic PDE:

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Balign*%7D%0A%09%5Ctext%7Bfind%7D%20%5Cquad%20y%20%5Cin%20H_0%5E1(D)%20%5Cquad%20%0A%09%5Ctext%7Bwith%7D%20%5Cquad%20%0A%09(%5Ckappa%20%5Cnabla%20y%2C%5Cnabla%20v)_%7BL%5E2(D)%5Ed%7D%0A%09%3D%20(u%2Bg%2Cv)_%7BL%5E2(D)%7D%0A%09%5Cquad%20%5Ctext%7Bfor%20all%7D%20%5Cquad%20v%20%5Cin%20H_0%5E1(D).%0A%5Cend%7Balign*%7D">

The feasible set is given by
<img src="https://render.githubusercontent.com/render/math?math=U_%7B%5Ctext%7Bad%7D%7D%20%3D%20%5C%7Bu%20%5Cin%20L%5E2(D)%3A%20%5Clb%20%5Cleq%20u%20%5Cleq%20%5Cub%20%5C%7D%24">.

The control space is discretized using piecewise constant functions and the
state space is discretized using piecewise continuous functions. [FEniCS](https://fenicsproject.org/)
is used to preform the discretization. Sparse linear systems are solved using
`npsolve` of [scipy](https://www.scipy.org/).

The parameter `alpha` must be positive and `beta` be nonnegative. The domain `D`
is `(0,1)^d` with `d` being one or two. The diffusion coefficient `kappa` maps
from the domain to the positive reals. The lower and upper bounds `lb` and `ub`, the desired state `yd`, 
the diffusion coefficient `kappa`, and the 
source term `g` must be instances of either 
[`Constant`](https://fenicsproject.org/olddocs/dolfin/2017.2.0/python/programmers-reference/functions/constant/Constant.html), 
[`Function`](https://fenicsproject.org/olddocs/dolfin/1.5.0/python/programmers-reference/functions/function/Function.html), or
[`Expression`](https://fenicsproject.org/olddocs/dolfin/1.5.0/python/programmers-reference/functions/expression/Expression.html).

The semismooth Newton method is applied a normal map,
a reformulation of the first-order optimality conditions as a nonsmooth operator equation
(see eq. (3.3) in Ref. [3]). 

The module implements a globalization via a restricted residual monotonicity test
(see [MonotonicityTest](./algorithm/globalize/monotonicity_test.py)), while 
[NewtonStep](./algorithm/globalize/newton_step.py) chooses the step size equal to one. 
The implementation of the restricted residual monotoncitity test is based on
eq. (3.32) in Ref. [1].

## Installation

The code can be downloaded using `git clone`.

## Examples

- [example1](./examples/example1), [example2](./examples/example2), 
[example3](./examples/example3), and [example4](./examples/example4) implement Examples 1--4 in Ref. [4].
- [poisson1d](./examples/poisson1d) and [possion2d](./examples/poisson2d) use an L1-heuristic to determine `beta`
(compare with p. 199 in Ref. [2] and Lemma 3.1 in Ref. [4]).
- [example73](./examples/example73) implements Example 7.3 in Ref. [5] and it is used for code verification.
- [random_poisson2d](./examples/random_poisson2d) demonstrates how to use the module to solve sample average approximations of a simple risk-neutral problem.
- [cdc](./examples/cdc) provides an example where globalization using [MonotonicityTest](./algorithm/globalize/monotonicity_test.py) requires fewer iterations than [NewtonStep](./algorithm/globalize/newton_step.py).

## Dependencies

- [numpy](https://numpy.org/)
- [scipy](https://www.scipy.org/)
- [FEniCS](https://fenicsproject.org/)

Some tests use the following packages:

- [dolfin-adjoint](http://www.dolfin-adjoint.org/)
- [moola](https://github.com/funsim/moola)

The code was tested using 
python version 3.8.11, scipy version 1.7.0, 
numpy version 1.21.1, fenics version 2019.1.0, 
moola version 0.1.6, and matplotlib version 3.4.2.

## References

* [1] P. Deuflhard, [Newton Methods for Nonlinear Problems](https://doi.org/10.1007/978-3-642-23899-4), Springer Ser. Comput.
Math. 35, Springer, Berlin, 2011
* [2] N. Parikh and S. Boyd, [Proximal Algorithms](https://doi.org/10.1561/2400000003), Found. Trends Mach. Learning, 1
(2014), pp. 127--239
* [3] K. Pieper, [Finite element discretization and efficient numerical solution of elliptic and parabolic sparse control problems](http://nbn-resolving.de/urn/resolver.pl?urn:nbn:de:bvb:91-diss-20150420-1241413-1-4), Dissertation, Technische Universität
München, München, 2015
* [4] G. Stadler, [Elliptic optimal control problems with L1-control cost and applications
for the placement of control devices](https://doi.org/10.1007/s10589-007-9150-9), Comput. Optim. Appl., 44 (2009), pp. 159--181
* [5] G. Wachsmuth and D. Wachsmuth, [Convergence and regularization results for 
optimal control problems with sparsity functional](https://doi.org/10.1051/cocv/2010027), ESAIM Control. Optim. Calc.
Var., 17 (2011), pp. 858--886

## Author

The code has been implemented by Johannes Milz.

