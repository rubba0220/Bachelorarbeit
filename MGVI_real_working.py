#MGVI_real.py
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random
from jax.scipy.interpolate import RegularGridInterpolator
import nifty8.re as jft
import pandas as pd
import matplotlib.pyplot as plt
import util_working as util
import importlib
import time
import sys
jnp.set_printoptions(threshold=sys.maxsize)
importlib.reload(util)

jax.config.update("jax_enable_x64", True)

''' Massenmodell '''
rhos = jnp.array([0.021, 0.016, 0.012, 0.0009, 0.0006, 0.0031, 0.0015, 0.0020, 0.0022, 0.007, 0.0135, 0.006, 0.002, 0.0035, 0.0001])
sigmas = jnp.array([4., 7., 9., 40., 20., 7.5, 10.5, 14., 18., 18.5, 18.5, 20., 20., 37., 100.])
erhos = jnp.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * rhos
esigmas = jnp.array([1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 5., 5., 5., 10.])

rho_s = jft.LogNormalPrior(rhos, erhos, name="rho_s", shape=(15,))
sigma_s = jft.LogNormalPrior(sigmas, esigmas, name="sigma_s", shape=(15,))
rho_dm = jft.UniformPrior(0., 0.2, name="rho_dm", shape=(1,))

''' Domain '''
am_min = 5
am_max = 6
z2 = 270.
z1 = 1800.
z3 = 5000.
interval = 'both'

poly = np.loadtxt(f'real data/poly_57.txt')

bins = np.loadtxt(f'real data/bins_{am_min:.0f}{am_max:.0f}.txt')
df_v2 = pd.read_csv(f'real data/v2_57.txt')
z_v2 = jnp.array(df_v2["z"].values)
v2 = jnp.array(df_v2["v2"].values)
sorted_indices = jnp.argsort(z_v2)
z_v2 = z_v2[sorted_indices]
v2 = jnp.array(df_v2["v2"].values)
v2 = v2[sorted_indices]

i2 = np.where(bins<=z2)[0][-1]
i1 = np.where(bins>=z1)[0][0]
z2 = bins[i2]
z1 = bins[i1]
n = int(z1)+1
n3 = int(z3-z1)+1

bins = bins[i2:i1+1]
n_bins = int(len(bins)-1)

dims = (n, )
# cf_zm = dict(offset_mean=900., offset_std=(700., 700.))
# cf_fl = dict(   fluctuations=(1000., 1000.), 
#                 loglogavgslope=(-20., 5.), #dickes ???
#                 flexibility=(1e-3, 1e-16),
#                 asperity=(1e-3, 1e-16),)

cf_zm = dict(offset_mean=7, offset_std=(4, 4))
cf_fl = dict(   fluctuations=(7, 7),
                loglogavgslope=(-30., 5.),
                flexibility=(1e-3, 1e-16),
                asperity=(1e-3, 1e-16),)

cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(**cf_zm)
cfm.add_fluctuations(dims, distances=1.0, **cf_fl, prefix="ax1", non_parametric_kind="power")
correlated_field = cfm.finalize()

data = np.loadtxt(f'real data/n_{am_min:.0f}{am_max:.0f}.txt', dtype='int')
if interval == 'pos':
    data = data[i2:i1] 
elif interval == 'neg':
    data = np.flip(data)[i2:i1]
elif interval == 'both':
    data = np.flip(data)[i2:i1] + data[i2:i1]

''' Forward Model '''
norm = jnp.sum(data)
string = f'z2:{z2} z1:{z1} norm:sum vel:poly data:{interval}'

class ForwardModel(jft.Model):
    def __init__(self):
        self.rho_s = rho_s
        self.sigma_s = sigma_s
        self.rho_dm = rho_dm
        self.correlated_field = correlated_field

        super().__init__(init =  self.rho_s.init| self.sigma_s.init | self.rho_dm.init | self.correlated_field.init)

    @jit
    def __call__(self, x):
        rs = self.rho_s(x)
        ss = self.sigma_s(x)
        rdm = self.rho_dm(x)
        cf = jnp.exp(self.correlated_field(x)) 
        # cf = self.correlated_field(x)

        def complicated_function(rho_s, sigma_s, rho_dm, cf):
            params = jnp.column_stack((rho_s, sigma_s))
            rho_dm = rho_dm[0]

            uz, zs = util.diffraxDopri5(rho_dm, params, z1, n)
            uz_, zs_ = util.Solver(rho_dm, params, z1, z3, uz[-1], n3)
            vdfo_norm_calc, z, sig2 = util.vdfo_norm(z2, z1, zs, uz, n, poly, cf, z_v2)
            integral, z_borders = util.binning(vdfo_norm_calc, z, z2, z1, n, n_bins)
            surface_density_calc = util.surface_density(params, jnp.append(uz, uz_, axis=0), jnp.append(zs, zs_))

            return integral * norm/jnp.sum(integral), surface_density_calc, sig2
        dfo, sd, sig2 = complicated_function(rs, ss, rdm, cf)
        return jft.Vector({'dfo': dfo, 'sd': sd, 'sig2': sig2})

fwd = ForwardModel()
R_dfo = jft.Model(lambda x: x['dfo'], domain=fwd.target)
R_sd = jft.Model(lambda x: x['sd'], domain=fwd.target)
R_sig2 = jft.Model(lambda x: x['sig2'], domain=fwd.target)

lh_dfo = jft.Poissonian(data).amend(R_dfo)
lh_sd = jft.Gaussian(49.4, lambda x: 1/4.6**2 * x).amend(R_sd)
lh_sig2 = jft.Gaussian(v2, lambda x: 1/1100**2 * x).amend(R_sig2)	#7 #sinnvoller wählen !!!!

lh = (lh_dfo + lh_sd + lh_sig2).amend(fwd)

#lh_dfo + lh_sd + lh_sig2

''' Optimization '''
n_vi_iterations = 6
delta = 1e-4
n_samples = 10

seed = 42
key = random.PRNGKey(seed)
# key, subkey = random.split(key)
# pos_truth = jft.random_like(subkey, fwd.domain)
# dfo_truth = fwd(pos_truth)['dfo']
# sd_truth = fwd(pos_truth)['sd']
# sig2_truth = fwd(pos_truth)['sig2']
# print(len(dfo_truth), sd_truth, len(sig2_truth))
key, k_i, k_o = random.split(key, 3)

t0 = time.time()
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

''' Results '''
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

''' Save Results '''
meanr = [results[f'rho{k+1}'][0] for k in range(15)] 
stdr = [results[f'rho{k+1}'][1] for k in range(15)] 

data_rho = {
    "Run": [f'rho_{k+1} ' + string for k in range(15)],
    "Inferred Value rho": meanr,
    "Standard Deviation rho": stdr,
    "Samples rho": [results[f'rhos{k+1}'] for k in range(15)]
}

meanrd = [results['rhodm'][0]]
stdrd = [results['rhodm'][1]]

data_rd = { 
    "Run": [f'rho_dm ' + string],
    "Inferred Value rho": meanrd,
    "Standard Deviation rho": stdrd,
    "Samples rho": [results['rhosdm']]
}

means = [results[f'sigma{k+1}'][0] for k in range(15)]
stds = [results[f'sigma{k+1}'][1] for k in range(15)]

data_sigma = {
    "Run": [f'sigma_{k+1} ' + string for k in range(15)],
    "Inferred Value sigma": means,
    "Standard Deviation sigma": stds,
    "Samples sigma": [results[f'sigmas{k+1}'] for k in range(15)]
}

meansd = [results['surfd'][0]]
stdsd = [results['surfd'][1]]

data_sd = {
    "Run": ['surfdens ' + string],
    "Inferred Value sd": meansd,
    "Standard Deviation sd": stdsd,
    "Samples sd": [results['surfds']],
}

t1 = time.time()
print('Time: ', t1-t0)

dfr = pd.DataFrame(data_rho)
dfr.set_index('Run', inplace=True)
dfrd = pd.DataFrame(data_rd)
dfrd.set_index('Run', inplace=True)
dfs = pd.DataFrame(data_sigma)
dfs.set_index('Run', inplace=True)
dfsd = pd.DataFrame(data_sd)
dfsd.set_index('Run', inplace=True)

dfr.to_csv(f'real data test/rho_{am_min:.0f}{am_max:.0f}_v2.csv', mode='a', header=False)
dfrd.to_csv(f'real data test/rd_{am_min:.0f}{am_max:.0f}_v2.csv', mode='a', header=False)
dfs.to_csv(f'real data test/sigma_{am_min:.0f}{am_max:.0f}_v2.csv', mode='a', header=False)
dfsd.to_csv(f'real data test/sd_{am_min:.0f}{am_max:.0f}_v2.csv', mode='a', header=False)



namps = cfm.get_normalized_amplitudes()
post_sr_mean = jft.mean(tuple(fwd(s)['sig2'] for s in samples))
corrfield = jft.mean_and_std(tuple(jnp.exp(correlated_field(s)) for s in samples))
# corrfield = jft.mean_and_std(tuple(correlated_field(s) for s in samples))
post_a_mean = jft.mean(tuple(cfm.amplitude(s)[1:] for s in samples))
grid = correlated_field.target_grids[0]
to_plot = [("Data", v2, 'scatter'), ("Reconstruction", post_sr_mean, 'plot')]

data_cf = {
    "Run": ['corrfield ' + string],
    "Inferred Value cf": [corrfield[0]],
    "Standard Deviation cf": [corrfield[1]],
}

dfcf = pd.DataFrame(data_cf)
dfcf.set_index('Run', inplace=True)
dfcf.to_csv(f'real data test/cf_{am_min:.0f}{am_max:.0f}_v2.csv', mode='a', header=False)

fig, axs = plt.subplots(2, 1, figsize=(20, 20), sharex=True)
for ax, v in zip(axs.flat, to_plot):
    title, field, tp = v
    ax.set_title(title)
    ax.grid()
    if tp == 'scatter':
        ax.scatter(z_v2, field, marker='.')
    elif tp == 'plot':
        ax.plot(z_v2, field)
    ax.plot(z_v2, poly[0]*z_v2+poly[1])
fig.tight_layout()
fig.subplots_adjust(hspace=0)
plt.show()

fig = plt.figure(figsize=(20, 10))
plt.plot(jnp.linspace(0., z1, n), jnp.sqrt(corrfield[0]))
plt.grid()
fig.tight_layout()
plt.show()
