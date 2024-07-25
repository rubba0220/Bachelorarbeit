import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random
from jax import jit
from jax.scipy.interpolate import RegularGridInterpolator
import pandas as pd
import numpy as np
import nifty8.re as jft

jax.config.update("jax_enable_x64", True)

df = pd.read_csv("v2_58.txt")
z = jnp.array(df["z"].values)
sorted_indices = jnp.argsort(z)
z = z[sorted_indices]
v2 = jnp.array(df["v2"].values)
v2 = v2[sorted_indices]
poly = np.loadtxt(f'real data/poly_58.txt')

seed = 42
key = random.PRNGKey(seed)

dims = (1801, )

cf_zm = dict(offset_mean=jnp.log(450.), offset_std=(jnp.log(200), jnp.log(100)))
cf_fl = dict(
    fluctuations=(1e-1, 5e-3),
    loglogavgslope=(-6.0, 2.),
    flexibility=(1e0, 5e-1),
    asperity=(5e-1, 5e-2),
)
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(**cf_zm)
cfm.add_fluctuations(dims, distances=1.0, **cf_fl, prefix="ax1", non_parametric_kind="power")
correlated_field = cfm.finalize()

class Signal(jft.Model):
    def __init__(self, correlated_field):
        self.cf = correlated_field
        
        super().__init__(init=self.cf.init)
    @jit
    def __call__(self, x):
        grid = jnp.linspace(0, 1800, dims[0])
        sig = RegularGridInterpolator((grid,), jnp.exp(self.cf(x)))
        return sig(z)


signal = Signal(correlated_field)

signal_response = signal
noise_cov = lambda x: 1200000 * x
noise_cov_inv = lambda x:1200000 * x

# Create synthetic data
# key, subkey = random.split(key)
# pos_truth = jft.random_like(subkey, signal_response.domain)
# signal_response_truth = signal_response(pos_truth)
# key, subkey = random.split(key)
# noise_truth = ((noise_cov(jft.ones_like(signal_response.target))) ** 0.5) * jft.random_like(key, signal_response.target)
# data = signal_response_truth + noise_truth

data = v2

lh = jft.Gaussian(data, noise_cov_inv).amend(signal_response)

n_vi_iterations = 6
delta = 1e-4
n_samples = 4

key, k_i, k_o = random.split(key, 3)
samples, state = jft.optimize_kl(
    lh,
    jft.Vector(lh.init(k_i)),
    n_total_iterations=n_vi_iterations,
    n_samples=lambda i: n_samples // 2 if i < 2 else n_samples,
    key=k_o,
    # Names of parameters that should not be sampled but still optimized
    # can be specified as point_estimates (effectively we are doing MAP for
    # these degrees of freedom).
    # point_estimates=("cfax1flexibility", "cfax1asperity"),
    # Arguments for the conjugate gradient method used to drawing samples from
    # an implicit covariance matrix
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
    odir="results_intro",
    resume=False,
)

namps = cfm.get_normalized_amplitudes()
post_sr_mean = jft.mean(tuple(signal(s) for s in samples))
post_a_mean = jft.mean(tuple(cfm.amplitude(s)[1:] for s in samples))
grid = correlated_field.target_grids[0]
to_plot = [("Data", data, 'scatter'), ("Reconstruction", post_sr_mean, 'plot')]

fig, axs = plt.subplots(2, 1, figsize=(20, 20), sharex=True)
for ax, v in zip(axs.flat, to_plot):
    title, field, tp = v
    ax.set_title(title)
    ax.grid()
    if tp == 'scatter':
        ax.scatter(z, jnp.sqrt(field), marker='.')
    elif tp == 'plot':
        ax.plot(z, jnp.sqrt(field))
    ax.plot(z, jnp.sqrt(poly[0]*z+poly[1]))
for ax in axs.flat[len(to_plot) :]:
    ax.set_axis_off()
fig.tight_layout()
fig.subplots_adjust(hspace=0)
plt.show()
