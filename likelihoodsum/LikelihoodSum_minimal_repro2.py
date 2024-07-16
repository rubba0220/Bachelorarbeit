import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random

import nifty8.re as jft

jax.config.update("jax_enable_x64", True)

seed = 42
key = random.PRNGKey(seed)

dims = (128, 128)

cf_zm = dict(offset_mean=0.0, offset_std=(1e-3, 1e-4))
cf_fl = dict(
    fluctuations=(1e-1, 5e-3),
    loglogavgslope=(-1.0, 1e-2),
    flexibility=(1e0, 5e-1),
    asperity=(5e-1, 5e-2),
)
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(**cf_zm)
cfm.add_fluctuations(
    dims, distances=1.0 / dims[0], **cf_fl, prefix="ax1", non_parametric_kind="power"
)
correlated_field = cfm.finalize()

scaling = jft.LogNormalPrior(3.0, 1.0, name="scaling", shape=(1,))


class Signal(jft.Model):
    def __init__(self, correlated_field, scaling):
        self.cf = correlated_field
        self.scaling = scaling
        # Init methods of the Correlated Field model and any prior model in
        # NIFTy.re are aware that their input is standard normal a priori.
        # The `domain` of a model does not know this. Thus, tracking the `init`
        # methods should be preferred over tracking the `domain`.
        super().__init__(init=self.cf.init | self.scaling.init)

    def __call__(self, x):
        # NOTE, think of `Model` as being just a plain function that takes some
        # input and performs all the necessary computation for your model.
        # Note, `scaling` here is completely degenarate with `offset_std` in the
        # likelihood but the priors for them are very different.
        return self.scaling(x) * jnp.exp(self.cf(x))


signal = Signal(correlated_field, scaling)

signal_response = signal
noise_cov = lambda x: 0.1**2 * x
noise_cov_inv = lambda x: 0.1**-2 * x

# Create synthetic data
key, subkey = random.split(key)
pos_truth = jft.random_like(subkey, signal_response.domain)
signal_response_truth = signal_response(pos_truth)
key, subkey = random.split(key)
noise_truth = (
    (noise_cov(jft.ones_like(signal_response.target))) ** 0.5
) * jft.random_like(key, signal_response.target)
data = signal_response_truth + noise_truth

lh = jft.Gaussian(data, noise_cov_inv).amend(signal_response) + jft.Gaussian(data, noise_cov_inv).amend(signal_response)
