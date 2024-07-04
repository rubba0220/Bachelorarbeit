#jax bisher nur für CPU intslliert (pip install -U "jax[cpu]")
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import jit, random
from functools import partial
from matplotlib import pyplot as plt
from scipy import constants as const
import nifty8.re as jft
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController, DirectAdjoint
import pandas as pd

import time
jax.config.update("jax_enable_x64", True)

# Plot-Formatierung
plt.rcParams['font.size'] = 24.0
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.0

n = 120
dz = 1.

f = lambda params, z, u: jnp.array([u[1], -u[0]*params[0] + z*params[1]])
z0 = 0.
u0 = jnp.array([1.,0.])

@partial(jit, static_argnames=['f', 'n']) 
def eigenerSolverV2(params, z0, u0, f, n, dz):
                                                        
    # Runge-Kutta 4. Ordnung
    # @partial(jit, static_argnames=['f']) #nötig ??
    def rk4_step(params, z0, u0, dz, f):
        k1 = dz * f(params, z0, u0)
        k2 = dz * f(params, z0 + dz / 2, u0 + k1 / 2)
        k3 = dz * f(params, z0 + dz / 2, u0 + k2 / 2)
        k4 = dz * f(params, z0 + dz, u0 + k3)
        u1 = u0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return u1

    def rk4_step_scan(u, i):
        return rk4_step(params, z0+i*dz, u, dz, f), \
            rk4_step(params, z0+i*dz, u, dz, f)

    _, uz = lax.scan(rk4_step_scan, u0, jnp.linspace(0, n*dz, n))

    return uz

''' Test des Algorithmus zur MGVI '''

roh_1 = jft.UniformPrior(0.0001, 0.001, name="roh_1", shape=(1,))
sigma_1 = jft.UniformPrior(0.00001, 0.0001, name="sigma_1", shape=(1,))

class ForwardModel(jft.Model):
    def __init__(self):

        self.roh_1 = roh_1
        self.sigma_1 = sigma_1

        super().__init__(
            init =   self.roh_1.init | self.sigma_1.init)

    def __call__(self, x):
        r1 = self.roh_1(x)
        s1 = self.sigma_1(x)

        def complicated_function(roh_1, sigma_1):
            params = [roh_1[0], sigma_1[0]]

            uz = eigenerSolverV2(params, z0, u0, f, n, dz)
            val = uz[:,0]

            return val

        return complicated_function(r1, s1)

# This initialises your forward-model which computes something data-like

fwd = ForwardModel()

seed = 4
key = random.PRNGKey(seed)
noise_cov = lambda x: 0.1 * x
noise_cov_inv = lambda x: 1. / 0.1 * x
key, subkey = random.split(key)
pos_truth = jft.random_like(subkey, fwd.domain)
fwd_truth = fwd(pos_truth)

key, subkey = random.split(key)
noise_truth = (
    (noise_cov(jft.ones_like(fwd.target))) ** 0.5 # sqrt to get from cov->std
) * jft.random_like(subkey, fwd.target) # random means white noise
data = fwd_truth + noise_truth

#Visualisierung
fig, ax = plt.subplots(figsize=(20,10))
ax.set_xlabel('z/pc')
#ax.set_yscale('log')
ax.set_ylabel('$\\nu / \\nu_0 $')
ax.scatter([0.+i*dz for i in range(n)], [data], marker='o')
ax.grid()
fig.tight_layout()
plt.show()

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
results = {}

results["rohs1"] = tuple(roh_1(s).tolist()[0] for s in samples)
results["roh1"] = jft.mean_and_std(results["rohs1"])
results["sigmas1"] = tuple(sigma_1(s).tolist()[0] for s in samples)
results["sigma1"] = jft.mean_and_std(results["sigmas1"])

truthr = [roh_1(pos_truth)[0]]
meanr = [results['roh1'][0]]
stdr = [results['roh1'][1]]

truths = [sigma_1(pos_truth)[0]]
means = [results['sigma1'][0]]
stds = [results['sigma1'][1]]

data_roh = {
    "True Value roh": truthr,

    "Inferred Value roh": meanr,

    "Standard Deviation roh": stdr,

    "Samples roh": [results[f'rohs1']],# + [results['rohsdm']],

    "Abweichung roh": list((jnp.array(truthr) - jnp.array(meanr))/jnp.array(stdr))
}

data_sigma = {
    "True Value sigma": truths,

    "Inferred Value sigma": means,

    "Standard Deviation sigma": stds,

    "Samples sigma": [results[f'sigmas1']],# + [results['sigmasdm']],

    "Abweichung sigma": list((jnp.array(truths) - jnp.array(means))/jnp.array(stds))
}

dfr = pd.DataFrame(data_roh)
dfr.to_csv(f'data_roh_simp_3.csv', mode='a', header=False, index=False)
dfs = pd.DataFrame(data_sigma)
dfs.to_csv(f'data_sigma_simp_3.csv', mode='a', header=False, index=False)


# for i in range(10):
#     print(i)
#     test(i)

#seed = 3; jft.random_like(key, fwd.target) -> error
#seed = 3; jft.random_like(subkey, fwd.target) -> kein error
#seed = 4; jft.random_like(subkey, fwd.domain) -> error