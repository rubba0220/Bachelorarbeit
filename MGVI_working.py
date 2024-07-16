#MGVI_test.py
import jax
import jax.numpy as jnp
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

''' Massenmodell '''
rhos = jnp.array([0.021, 0.016, 0.012, 0.0009, 0.0006, 0.0031, 0.0015, 0.0020, 0.0022, 0.007, 0.0135, 0.006, 0.002, 0.0035, 0.0001])
sigmas = jnp.array([4., 7., 9., 40., 20., 7.5, 10.5, 14., 18., 18.5, 18.5, 20., 20., 37., 100.])
erhos = jnp.array([ 0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * rhos
esigmas = jnp.array([1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 5., 5., 5., 10.])

rho_s = jft.LogNormalPrior(rhos, erhos, name="rho_s", shape=(15,))
sigma_s = jft.LogNormalPrior(sigmas, esigmas, name="sigma_s", shape=(15,))
rho_dm = jft.UniformPrior(0., 0.2, name="rho_dm", shape=(1,))

''' Domain '''
z2 = 0.
z1 = 3000.
n = 3001
poly = (0,0)
norm = 5000
n_bins = 20

''' Forward Models '''

def complicated_function(rho_s, sigma_s, rho_dm):
    params = jnp.column_stack((rho_s, sigma_s))
    rho_dm = rho_dm[0]

    uz, zs = util.diffraxDopri5(rho_dm, params, z1, n)
    vdfo_norm_calc, z = util.vdfo_norm(z2, z1, zs, uz, n, poly, mock=True)
    integral, z_borders = util.binning(vdfo_norm_calc, z, z2, z1, n, n_bins)
    surface_density_calc = util.surface_density(params, uz, z1, n)

    return integral * norm/jnp.sum(integral), surface_density_calc

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
        dfo, sd = complicated_function(rs, ss, rdm)
        return jft.Vector({'dfo': dfo, 'sd': sd})

# This initialises your forward-model which computes something data-like
fwd = ForwardModel()

R_dfo = jft.Model(lambda x: x['dfo'], domain=fwd.target)

R_sd = jft.Model(lambda x: x['sd'], domain=fwd.target)

#jit macht es hier langsamer ??? Schon wieder ...

''' Testing MGVI '''
def test_mgvi(s):
    seed = s
    key = random.PRNGKey(seed)

    key, subkey = random.split(key)
    pos_truth = jft.random_like(subkey, fwd.domain)
    dfo_truth = fwd(pos_truth)['dfo']
    sd_truth = fwd(pos_truth)['sd']

    dfo_truth = jnp.round(dfo_truth, 0)
    dfo_truth = dfo_truth.astype(int)

    noise_cov = lambda x: 3 * x
    noise_cov_inv = lambda x: 1. / 3 * x

    key, subkey = random.split(key)
    noise_truth = ((noise_cov(jft.ones_like((fwd.target)['sd']))) ** 0.5) * jft.random_like(key, (fwd.target)['sd'])
    sd_truth = sd_truth + noise_truth
    
    print(sd_truth, noise_truth)

    # #Visualisierung
    # i_s = int((z2-0.)/(z1-0.) * (n-1))
    # l = int((n-i_s-1)/n_bins)
    # z = jnp.linspace(0., z1, n)[i_s:]
    # z_borders = z[0::l]
    # fig, ax = plt.subplots(figsize=(20,10))
    # ax.set_xlabel('z/pc')
    # ax.set_ylabel('$\\nu / \\nu_0 $')
    # ax.scatter(z_borders[:-1], dfo_truth, marker='o')
    # ax.grid()
    # fig.tight_layout()

    lh_dfo = jft.Poissonian(dfo_truth).amend(R_dfo)
    lh_sd = jft.Gaussian(sd_truth, noise_cov_inv).amend(R_sd)

    lh = (lh_dfo + lh_sd).amend(fwd)


    ''# Now lets run the main inference scheme:
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

    for k in range(15):
        exec(f'results["rhos{k+1}"] = tuple(rho_s(s)[{k}].tolist() for s in samples)')
        exec(f'results["sigmas{k+1}"] = tuple(sigma_s(s)[{k}].tolist() for s in samples)')
        exec(f'results["rho{k+1}"] = jft.mean_and_std(results["rhos{k+1}"])')
        exec(f'results["sigma{k+1}"] = jft.mean_and_std(results["sigmas{k+1}"])')
    results["rhosdm"] = tuple(rho_dm(s).tolist()[0] for s in samples)
    results["rhodm"] = jft.mean_and_std(results["rhosdm"])
    results["sds"] = tuple(R_sd(s) for s in samples)
    results["sd"] = jft.mean_and_std(results["sds"])

    truthr = [*rho_s(pos_truth), rho_dm(pos_truth)[0]]
    meanr = [results[f'rho{k+1}'][0] for k in range(15)] + [results['rhodm'][0]]
    stdr = [results[f'rho{k+1}'][1] for k in range(15)] + [results['rhodm'][1]]


    data_rho = {
        "True Value rho": truthr,

        "Inferred Value rho": meanr,

        "Standard Deviation rho": stdr,

        "Samples rho": [results[f'rhos{k+1}'] for k in range(15)] + [results['rhosdm']],

        "Abweichung rho": list((jnp.array(truthr) - jnp.array(meanr))/jnp.array(stdr))
    }

    truths = [*sigma_s(pos_truth)]
    means = [results[f'sigma{k+1}'][0] for k in range(15)]
    stds = [results[f'sigma{k+1}'][1] for k in range(15)]


    data_sigma = {
        "True Value sigma": truths,

        "Inferred Value sigma": means,

        "Standard Deviation sigma": stds,

        "Samples sigma": [results[f'sigmas{k+1}'] for k in range(15)],

        "Abweichung sigma": list((jnp.array(truths) - jnp.array(means))/jnp.array(stds))
    }

    truthsd = [sd_truth]
    meansd = [results['sd'][0]]
    stdsd = [results['sd'][1]]

    data_sd = {
        "True Value sd": truthsd,

        "Inferred Value sd": meansd,

        "Standard Deviation sd": stdsd,

        "Samples sd": results['sds'],

        "Abweichung sd": list((jnp.array(truthsd) - jnp.array(meansd))/jnp.array(stdsd))
    }


    dfr = pd.DataFrame(data_rho)
    dfs = pd.DataFrame(data_sigma)
    dfsd = pd.DataFrame(data_sd)
    dfr.to_csv(f'data_rho.csv', mode='a', header=False, index=False)
    dfs.to_csv(f'data_sigma.csv', mode='a', header=False, index=False)
    dfsd.to_csv(f'data_sd.csv', mode='a', header=False, index=False)



seed = 4
key = random.PRNGKey(seed)

key, subkey = random.split(key)
seeds = random.randint(subkey, (25,), 1, 1000000)

def has_duplicates(arr):
    seen = set()
    for element in arr:
        element = int(element)
        if element in seen:
            return True
        seen.add(element)
    return False

if has_duplicates(seeds):
    print("Das Array enthält doppelte Elemente.")

else:
    for s in seeds:
        t0 = time.time()
        test_mgvi(s)
        t1 = time.time()
        print('Time:', t1-t0, 's')




