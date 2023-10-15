**Work in progress**

### Variance reduction 

The following codebase implements a simple technique based 
on Ito's formula (c.f. [Ito lemma Wikipedia](https://en.wikipedia.org/wiki/It%C3%B4%27s_lemma)) for reducing the variance 
of a Monte Carlo estimator for partial differential equations. 

### Motivation
The Feynman-Kac formula establishes a link between partial differential equations
and stochastic processes, namely under certain conditions 
the solution of a pde can be written as a conditional expectation value (c.f.
[Feynman-Kac formula Wikipedia](https://en.wikipedia.org/wiki/Feynman%E2%80%93Kac_formula)). 
To numerically approximate the expectation value one needs to sample a Brownian motion. 
This estimator can have a high variance and with that a large
number of samples are necessary to reduce the approximation error.
The following codebase implements a variance reduction technique.
Instead of just sampling the Monte Carlo estimator 
one learns with a neural network the derivatives of
the solution to the pde through minimization of the variance of estimator.

Currently, two example pde's are implemented:

1. 1D Black Scholes model
2. Heston model

For more details of the implemented technique see Algorithm 4 
of Vidales et al. "Unbiased deep solvers for linear parametric PDEs
" (arXiv:1810.05094).

### Getting started

1. Setup the environment (see the environment file `env.yml`).
2. Under `configs/config.yaml` the settings for two pde's are defined (Heston & Black Scholes model).
3. To run the code, just call `main.py`.
4. In case you want to add an additional pde or extent to higher dimension, have a look at `pde.py` and the `class Equation`.


