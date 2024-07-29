from jax import random
import nifty8.re as jft
import numpy as np

import jax

seed = 42
key = random.PRNGKey(seed)
shape = (10, 10)

cf_zm = {"offset_mean": 0., "offset_std": (1e-3, 1e-4)}
cf_fl = {
    "fluctuations": (1e-1, 5e-3),
    "loglogavgslope": (-3., 1e-2),
    "flexibility": (1e+0, 5e-1),
    "asperity": (5e-1, 5e-2),
}
cfm = jft.CorrelatedFieldMaker("jcf_")
cfm.set_amplitude_total_offset(**cf_zm)
cfm.add_fluctuations(
    shape,
    distances=1. / shape[0],
    **cf_fl,
    prefix="",
    non_parametric_kind="power",
)

jcf = cfm.finalize()
key, subkey = random.split(key)
pos = jft.random_like(subkey, jcf.domain)
noise_level = 0.2
datar = jcf(pos) + np.random.normal(0, 1, shape)*noise_level

dom_key = 'a'
# m = jft.Model(
#     lambda x: {dom_key: jcf(x)},
#     domain=jcf.domain)
m = jft.Model(lambda x: jft.Vector({dom_key: jcf(x)}), domain=jcf.domain) #Fix von dort


R = jft.Model(lambda x: x[dom_key], domain=m.target)
like1 = jft.Gaussian(datar, lambda x: 1/noise_level**2 * x).amend(R)
like2 = jft.Gaussian(datar, lambda x: 2/noise_level**2 * x).amend(R)

ll = (like1 + like2).amend(m)

pos = jft.random_like(key, jcf.domain)

n_iterations = 2
n_samples = 2
delta = 1e-3
absdelta = 1e-4

samples, _ = jft.optimize_kl(
    ll,
    jft.Vector(pos),
    n_total_iterations=n_iterations,
    n_samples=n_samples,
    # Source for the stochasticity for sampling
    key=key,
    draw_linear_kwargs=dict(cg_name="SL",
                            cg_kwargs=dict(absdelta=absdelta / 10., maxiter=100)),
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
            name="M", absdelta=absdelta, cg_kwargs=dict(name="MCG"), maxiter=35
        )
    ),
    sample_mode="nonlinear_resample",
    resume=False)
