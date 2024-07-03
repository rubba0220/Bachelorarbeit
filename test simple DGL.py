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
dz = 10.

f = lambda params, u: jnp.array([u[1], - params**2 * u[0]])
z0 = 0.
u0 = jnp.array([1.,0.])

@partial(jit, static_argnames=['f', 'n']) 
def eigenerSolverV2(params, u0, f, n, dz):
                                                        
    # Runge-Kutta 4. Ordnung
    # @partial(jit, static_argnames=['f']) #nötig ??
    def rk4_step(params, u0, dz, f):
        k1 = dz * f(params, u0)
        k2 = dz * f(params, u0 + k1 / 2)
        k3 = dz * f(params, u0 + k2 / 2)
        k4 = dz * f(params, u0 + k3)
        u1 = u0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return u1

    def rk4_step_scan(u, i):
        return rk4_step(params, u, dz, f), \
            rk4_step(params, u, dz, f)

    _, uz = lax.scan(rk4_step_scan, u0, jnp.linspace(0, n*dz, n))

    return uz

''' Test des Algorithmus zur MGVI '''

roh_1 = jft.UniformPrior(0.0001, 0.005, name="roh_1", shape=(1,))

class ForwardModel(jft.Model):
    def __init__(self):

        self.roh_1 = roh_1

        super().__init__(
            init =   self.roh_1.init)

    def __call__(self, x):
        r1 = self.roh_1(x)

        def complicated_function(roh_1):
            params = roh_1[0]

            uz = eigenerSolverV2(params, u0, f, n, dz)
            val = uz[:,0]

            return val

        return complicated_function(r1)

# This initialises your forward-model which computes something data-like
fwd = ForwardModel()

def test_mgvi(s, it):

    seed = s
    key = random.PRNGKey(seed)
    noise_cov = lambda x: 0.001 * x
    noise_cov_inv = lambda x: 1. / 0.001 * x
    key, subkey = random.split(key)
    pos_truth = jft.random_like(subkey, fwd.domain)
    fwd_truth = fwd(pos_truth)
    key, subkey = random.split(key)
    noise_truth = (
        (noise_cov(jft.ones_like(fwd.target))) ** 0.5 # sqrt to get from cov->std
    ) * jft.random_like(key, fwd.target) # random means white noise
    data = fwd_truth + noise_truth

    #Visualisierung
    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_xlabel('z/pc')
    #ax.set_yscale('log')
    ax.set_ylabel('$\\nu / \\nu_0 $')
    ax.scatter([0.+i*dz for i in range(n)], [data], marker='o')
    ax.grid()
    fig.tight_layout()

    lh = jft.Gaussian(data, noise_cov_inv).amend(fwd)

    for i in range(it):
        
        t0 = time.time()

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

        truthr = [roh_1(pos_truth)[0]]
        meanr = [results['roh1'][0]]
        stdr = [results['roh1'][1]]


        data_roh = {
            "True Value roh": truthr,

            "Inferred Value roh": meanr,

            "Standard Deviation roh": stdr,

            "Samples roh": [results[f'rohs1']],# + [results['rohsdm']],

            "Abweichung roh": list((jnp.array(truthr) - jnp.array(meanr))/jnp.array(stdr))
        }

        dfr = pd.DataFrame(data_roh)
        dfr.to_csv(f'data_roh_start_cond_{seed}.csv', mode='a', header=False, index=False)
        
        t1 = time.time()
        print('Time:', t1-t0, 's')

seed = 78
key = random.PRNGKey(seed)
key, subkey = random.split(key)
seeds = random.randint(subkey, (15,), 0, 1000)
for seed in seeds:
    test_mgvi(seed, 1)



