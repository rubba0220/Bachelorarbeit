#MGVI_real.py
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random
from matplotlib import pyplot as plt
import nifty8.re as jft
import pandas as pd
import util
import importlib
importlib.reload(util)

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

''' Algorithmus zur MGVI '''
rhos = jnp.array([0.021, 0.016, 0.012, 0.0009, 0.0006, 0.0031, 0.0015, 0.0020, 0.0022, 0.007, 0.0135, 0.006, 0.002, 0.0035, 0.0001])
sigmas = jnp.array([4., 7., 9., 40., 20., 7.5, 10.5, 14., 18., 18.5, 18.5, 20., 20., 37., 100.])
erhos = jnp.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * rhos
esigmas = jnp.array([1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 5., 5., 5., 10.])

rho_s = jft.LogNormalPrior(rhos, erhos, name="rho_s", shape=(15,))
sigma_s = jft.LogNormalPrior(sigmas, esigmas, name="sigma_s", shape=(15,))
rho_dm = jft.UniformPrior(0., 0.2, name="rho_dm", shape=(1,))

def mgvi(am_min, am_max, z2, z1, n):
    poly = np.loadtxt(f'real data/poly_58.txt')
    bins = np.loadtxt(f'real data/bins_{am_min:.0f}{am_max:.0f}.txt')
    i2 = np.where(bins<=z2)[0][-1]
    i1 = np.where(bins>=z1)[0][0]
    z2 = bins[i2]
    z1 = bins[i1]
    bins = bins[i2:i1+1]
    data = np.loadtxt(f'real data/n_{am_min:.0f}{am_max:.0f}.txt', dtype='int')
    data = np.flip(data)[i2:i1] + data[i2:i1]
    norm = np.sum(data)
    n_bins = int(len(bins)-1)

    class ForwardModel(jft.Model):
        def __init__(self):
            self.rho_s = rho_s
            self.sigma_s = sigma_s
            self.rho_dm = rho_dm

            super().__init__(init =  self.rho_s.init| self.sigma_s.init | self.rho_dm.init)

        @jit
        def __call__(self, x):
            rs = self.rho_s(x)
            ss = self.sigma_s(x)
            rdm = self.rho_dm(x)

            def complicated_function(rho_s, sigma_s, rho_dm):
                params = jnp.column_stack((rho_s, sigma_s))
                rho_dm = rho_dm[0]

                uz, zs = util.diffraxDopri5(rho_dm, params, z1, n)
                vdfo_norm_calc, z = util.vdfo_norm(z2, z1, zs, uz, n, poly)
                integral, z_borders = util.binning(vdfo_norm_calc, z, z2, z1, n, n_bins)
                surface_density_calc = util.surface_density(params, uz, z1, n)

                return integral * norm/jnp.sum(integral), surface_density_calc
            dfo, sd = complicated_function(rs, ss, rdm)
            return jft.Vector({'dfo': dfo, 'sd': sd})

    fwd = ForwardModel()
    R_dfo = jft.Model(lambda x: x['dfo'], domain=fwd.target)
    R_sd = jft.Model(lambda x: x['sd'], domain=fwd.target)

    lh_dfo = jft.Poissonian(data).amend(R_dfo)
    lh_sd = jft.Gaussian(49.4, lambda x: 1/4.6**2 * x).amend(R_sd)
    lh = (lh_dfo + lh_sd).amend(fwd)

    n_vi_iterations = 6
    delta = 1e-4
    n_samples = 10
    seed = 42
    key = random.PRNGKey(seed)

    key, k_i, k_o = random.split(key, 3)
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

    results = {}

    for k in range(15):
        exec(f'results["rhos{k+1}"] = tuple(rho_s(s)[{k}].tolist() for s in samples)')
        exec(f'results["sigmas{k+1}"] = tuple(sigma_s(s)[{k}].tolist() for s in samples)')
        exec(f'results["rho{k+1}"] = jft.mean_and_std(results["rhos{k+1}"])')
        exec(f'results["sigma{k+1}"] = jft.mean_and_std(results["sigmas{k+1}"])')
    results["rhosdm"] = tuple(rho_dm(s).tolist()[0] for s in samples)
    results["rhodm"] = jft.mean_and_std(results["rhosdm"])
    results["surfds"] = tuple((fwd(s))['sd'] for s in samples)
    results["surfd"] = jft.mean_and_std(results["surfds"])

    meanr = [results[f'rho{k+1}'][0] for k in range(15)] + [results['rhodm'][0]]
    stdr = [results[f'rho{k+1}'][1] for k in range(15)] + [results['rhodm'][1]]


    data_rho = {
        "Inferred Value rho": meanr,

        "Standard Deviation rho": stdr,

        "Samples rho": [results[f'rhos{k+1}'] for k in range(15)] + [results['rhosdm']]
    }

    means = [results[f'sigma{k+1}'][0] for k in range(15)]
    stds = [results[f'sigma{k+1}'][1] for k in range(15)]


    data_sigma = {
        "Inferred Value sigma": means,

        "Standard Deviation sigma": stds,

        "Samples sigma": [results[f'sigmas{k+1}'] for k in range(15)]
    }
    meansd = [results['surfd'][0]]
    stdsd = [results['surfd'][1]]

    data_sd = {
        "Inferred Value sd": meansd,

        "Standard Deviation sd": stdsd,

        "Samples sd": [results['surfds']],
    }
    dfr = pd.DataFrame(data_rho)
    dfs = pd.DataFrame(data_sigma)
    dfsd = pd.DataFrame(data_sd)
    dfr.to_csv(f'real data2/rho_{am_min:.0f}{am_max:.0f}.csv', mode='a', header=False, index=False)
    dfs.to_csv(f'real data2/sigma_{am_min:.0f}{am_max:.0f}.csv', mode='a', header=False, index=False)
    dfsd.to_csv(f'real data2/sd_{am_min:.0f}{am_max:.0f}.csv', mode='a', header=False, index=False)

t0 = time.time()
mgvi(5,6, 0., 1600., 1601)
t1 = time.time()
print(t1-t0)
# t0 = time.time()
# mgvi(6,7, 0., 1600., 1601)
# t1 = time.time()
# print(t1-t0)
#mgvi(7,8, 100., 650., 651)
#mgvi(5,8, 240., 680., 681)

