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
rhos = jnp.array([  0.021, 0.016, 0.012, 
                    0.0009, 0.0006, 0.0031, 
                    0.0015, 0.0020, 0.0022, 
                    0.007, 0.0135, 0.006, 
                    0.002, 0.0035, 0.0001])

sigmas = jnp.array([4., 7., 9., 
                    40., 20., 7.5, 
                    10.5, 14., 18., 
                    18.5, 18.5, 20., 
                    20., 37., 100.])

erhos = jnp.array([ 0.5, 0.5, 0.5,
                    0.5, 0.2, 0.2,
                    0.2, 0.2, 0.2,
                    0.2, 0.2, 0.2,
                    0.2, 0.2, 0.2]) * rhos

esigmas = jnp.array([   1., 1., 1.,
                        1., 2., 2.,
                        2., 2., 2.,
                        2., 2., 5.,
                        5., 5., 10.])

rho_s = jft.LogNormalPrior(rhos, erhos, name="rho_s", shape=(15,))
sigma_s = jft.LogNormalPrior(sigmas, esigmas, name="sigma_s", shape=(15,))
rho_dm = jft.UniformPrior(0., 0.2, name="rho_dm", shape=(1,))

def mgvi(am_min, am_max, z2, z1, n):
    poly = np.loadtxt(f'real data/poly_58.txt')
    bins = np.loadtxt(f'real data/bins_{am_min:.0f}{am_max:.0f}.txt')
    i2 = np.where(bins<=z2)[0][-1]
    i1 = np.where(bins>=z1)[0][0]
    bins = bins[i1:i2+1]
    n = np.loadtxt(f'real data/n_{am_min:.0f}{am_max:.0f}.txt', dtype='int')
    data = np.flip(n)[i1:i2] + n[i1:i2]
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
                params = jnp.array([
                        [rho_s[0], sigma_s[0]], [rho_s[1], sigma_s[1]], [rho_s[2], sigma_s[2]],
                        [rho_s[3], sigma_s[3]], [rho_s[4], sigma_s[4]], [rho_s[5], sigma_s[5]],
                        [rho_s[6], sigma_s[6]], [rho_s[7], sigma_s[7]], [rho_s[8], sigma_s[8]],
                        [rho_s[9], sigma_s[9]], [rho_s[10], sigma_s[10]], [rho_s[11], sigma_s[11]],
                        [rho_s[12], sigma_s[12]], [rho_s[13], sigma_s[13]], [rho_s[14], sigma_s[14]]])
                
                rho_dm = rho_dm[0]
                uz, zs = util.diffraxDopri5(rho_dm, params, z1, n)
                vdfo_norm_calc, z = util.vdfo_norm(z2, z1, zs, uz, n, poly)
                integral, z_borders = util.binning(vdfo_norm_calc, z, z2, z1, n, n_bins)

                return integral * norm/jnp.sum(integral)

            return complicated_function(rs, ss, rdm)


    fwd = ForwardModel()

    seed = 42
    key = random.PRNGKey(seed)
    lh = jft.Poissonian(data).amend(fwd)

    n_vi_iterations = 6
    delta = 1e-4
    n_samples = 10

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

    dfr = pd.DataFrame(data_rho)
    dfs = pd.DataFrame(data_sigma)
    dfr.to_csv(f'real data/rho_{am_min:.0f}{am_max:.0f}.csv', mode='a', header=False, index=False)
    dfs.to_csv(f'real data/sigma_{am_min:.0f}{am_max:.0f}.csv', mode='a', header=False, index=False)


mgvi(5,6, 300, 1600., 1601)
mgvi(6,7, 100, 1000., 1001)
mgvi(7,8, 100, 650., 651)
mgvi(5,8, 240, 680., 681)

