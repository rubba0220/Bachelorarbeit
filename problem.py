import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController

import nifty8.re as jft

jax.config.update("jax_enable_x64", True)

seed = 42
key = random.PRNGKey(seed)

parameter1 = jft.UniformPrior(0.0, 1.0, name="p1", shape=(1,))
parameter2 = jft.UniformPrior(1.1, 2.0, name="p2", shape=(1,))
parameter3 = jft.UniformPrior(2.1, 3.0, name="p3", shape=(1,))
# ...


class ForwardModel(jft.Model):
    def __init__(self):
        # This is where you initialise all input parameters
        # This tells nifty what it has to sample
        self.parameter1 = parameter1
        self.parameter2 = parameter2
        self.parameter3 = parameter3

        super().__init__(
            init=self.parameter1.init | self.parameter2.init | self.parameter3.init
        )

    def __call__(self, x):
        # This gets some abstract input x that allows you to retrieve your
        # parameters using the previously initialised priors (functions).
        p1 = self.parameter1(x)
        p2 = self.parameter2(x)
        p3 = self.parameter3(x)

        def complicated_function(a, b, c):

            vector_field = lambda t, y, args: -y
            term = ODETerm(vector_field)
            solver = Dopri5()
            saveat = SaveAt(ts=[a[0], b[0], c[0]])
            stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

            sol = diffeqsolve(term, solver, t0=0, t1=3, dt0=0.1, y0=1, saveat=saveat,
                            stepsize_controller=stepsize_controller)

            return sol.ys

        return complicated_function(p1, p2, p3)

# This initialises your forward-model which computes something data-like
fwd = ForwardModel()

# Now we create synthetic data and add some noise to it:

# Nifty likes functional programming, so it expects a function that applies the noise-covariance
# For non-correlated noise, this is a diagonal matrix, so multiplication of every
# datapoint by the square of the corresponding noise value.
# uniform noise of 5% of data value:
noise_cov = lambda x: 0.05 * x
noise_cov_inv = lambda x: 1. / 0.05 * x

key, subkey = random.split(key)
# position holds a set of abstract values defining your position in probability space
pos_truth = jft.random_like(subkey, fwd.domain)

# This is you synthetic "true" data
fwd_truth = fwd(pos_truth)

# And this adds some random noise
key, subkey = random.split(key)
noise_truth = (
    (noise_cov(jft.ones_like(fwd.target))) ** 0.5 # sqrt to get from cov->std
) * jft.random_like(key, fwd.target) # random means white noise
data = fwd_truth + noise_truth

# This defines the Likelihood in Bayes' law. "Amend" glues your 
# forward model to the input
lh = jft.Gaussian(data, noise_cov_inv).amend(fwd)



# Now lets run the main inference scheme:
n_vi_iterations = 6
delta = 1e-4
n_samples = 10

key, k_i, k_o = random.split(key, 3)
# NOTE, changing the number of samples always triggers a resampling even if
# `resamples=False`, as more samples have to be drawn that did not exist before.
samples, state = jft.optimize_kl(
    lh,
    jft.Vector(lh.init(k_i)),
    n_total_iterations=n_vi_iterations,
    n_samples=lambda i: n_samples // 2 if i < 2 else n_samples,
    # Source for the stochasticity for sampling
    key=k_o,
    # Arguments for the conjugate gradient method used to drawing samples from
    # an implicit covariance matrix
    draw_linear_kwargs=dict(
        cg_name="SL",
        cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=100),
    ),
    # Arguements for the minimizer in the nonlinear updating of the samples
    nonlinearly_update_kwargs=dict(
        minimize_kwargs=dict(
            name="SN",
            xtol=delta,
            cg_kwargs=dict(name=None),
            maxiter=5,
        )
    ),
    # Arguments for the minimizer of the KL-divergence cost potential
    kl_kwargs=dict(
        minimize_kwargs=dict(
            name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=35
        )
    ),
    sample_mode="nonlinear_resample",
    odir="./results_test",
    resume=False,
)

# Now the samples-object contains all the abstract parameters that were inferred
# Reading out the physical input parameter values goes e.g. like this:
res = jft.mean_and_std(tuple(parameter1(s) for s in samples))
# ...
# Please save your results in some way:
# TODO

print("Inferred values:", "MEAN:", res[0], "STD:", res[1])
print("with the true value being:", parameter1(pos_truth))

plt.show()
