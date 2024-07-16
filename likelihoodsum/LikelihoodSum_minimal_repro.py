import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random
from dataclasses import field

import nifty8.re as jft
jax.config.update("jax_enable_x64", True)

seed = 42
key = random.PRNGKey(seed)

parameter1 = jft.LogNormalPrior(3.0, 1.0, shape=(1,), name = 'p1') #name = 'p1', dann ist type(lh1.domain)=dict --> domains of ... must support core arithmetic operations; ohne name = 'p1' ist type(lh1.domain)=ShapeDtypeStruct

class ForwardModel1(jft.Model):
    def __init__(self):
        self.parameter1 = parameter1

        super().__init__(init=self.parameter1.init )

    def __call__(self, x):
        p1 = self.parameter1(x)

        def complicated_function(a):
            return a

        return complicated_function(p1)

class NoValue:
    pass

class LikelihoodSum(jft.Likelihood):
    left_likelihood: jft.Likelihood = field(metadata=dict(static=False))
    right_likelihood: jft.Likelihood = field(metadata=dict(static=False))
    def __init__(
        self,
        left,
        right,
        /,
        domain=NoValue,
        init=NoValue,
        _left_key="lh_left",
        _right_key="lh_right",
    ):
        if not (isinstance(left, jft.Likelihood) and isinstance(right, jft.Likelihood)):
            te = (
                "object which to add to this instance is of invalid type"
                f" {type(right)!r}"
            )
            raise TypeError(te)
        self._lkey, self._rkey = _left_key, _right_key
        joined_tangents_shape = {
            self._lkey: left._lsm_tan_shp,
            self._rkey: right._lsm_tan_shp,
        }
        if isinstance(left._lsm_tan_shp,
                      jft.Vector) or isinstance(right._lsm_tan_shp, jft.Vector):
            joined_tangents_shape = jft.Vector(joined_tangents_shape)
        if (
            domain is NoValue and left.domain is not NoValue and
            right.domain is not NoValue
        ):
            lvec = isinstance(left.domain, jft.Vector)
            rvec = isinstance(right.domain, jft.Vector)
            ldomain = left.domain.tree if lvec else left.domain
            rdomain = right.domain.tree if rvec else right.domain
            domain = ldomain | rdomain #ldomain|rdomain funktioniert zwar für dict, aber nicht für ShapeDtypeStruct, ldomain, dann probleme bei backward-pass vgl Ticket ,NoValue takes no arguments
            domain = jft.Vector(domain) if lvec or rvec else domain
            isswd = hasattr(domain, "shape") and hasattr(domain, "dtype")
            if not isswd and not jft.has_arithmetics(domain):
                ve = (
                    "domains of the Likelihood-summands must support core"
                    " arithmetic operations"
                    "\nmaybe you forgot to wrap your inputs to the liklihoods"
                    " in `Vector`s"
                )
                raise ValueError(ve)
        self.left_likelihood = left
        self.right_likelihood = right
        super().__init__(
            domain=domain, init=init, lsm_tangents_shape=joined_tangents_shape
        )

fwd1 = ForwardModel1()

noise_cov = lambda x: 0.05 * x
noise_cov_inv = lambda x: 1. / 0.05 * x

key, subkey = random.split(key)
pos_truth = jft.random_like(subkey, fwd1.domain)
fwd1_truth = fwd1(pos_truth)

key, subkey = random.split(key)
noise1_truth = ((noise_cov(jft.ones_like(fwd1.target))) ** 0.5) * jft.random_like(key, fwd1.target)
data = fwd1_truth + noise1_truth

lh1 = jft.Gaussian(data, noise_cov_inv).amend(fwd1)
lh2 = jft.Gaussian(data, noise_cov_inv).amend(fwd1)
print('''''''')
print(lh1.domain)
print(isinstance(lh1.domain, jft.Vector))
print(hasattr(lh1.domain, "shape") and hasattr(lh1.domain, "dtype"))
print(jft.has_arithmetics(lh1.domain))
print('''''''')

lh = LikelihoodSum(lh1, lh2)
#lh = lh1 + lh2





n_vi_iterations = 6
delta = 1e-4
n_samples = 10

key, k_i, k_o = random.split(key, 3)
samples, state = jft.optimize_kl(
    lh,
    jft.Vector(lh.init(k_i)),
    n_total_iterations=n_vi_iterations,
    n_samples=lambda i: n_samples // 2 if i < 2 else n_samples,
    key=k_o,
    draw_linear_kwargs=dict(
        cg_name="SL",
        cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=100),
    ),
    nonlinearly_update_kwargs=dict(
        minimize_kwargs=dict(
            name="SN",
            xtol=delta,
            cg_kwargs=dict(name=None),
            maxiter=5,
        )
    ),
    kl_kwargs=dict(
        minimize_kwargs=dict(
            name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=35
        )
    ),
    sample_mode="nonlinear_resample",
    odir="./results_test",
    resume=False,
)


res = jft.mean_and_std(tuple(parameter1(s) for s in samples))
print("Inferred values:", "MEAN:", res[0], "STD:", res[1])
print("with the true value being:", parameter1(pos_truth))