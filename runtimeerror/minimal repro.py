#MGVI_real.py
import jax
import jax.numpy as jnp
from jax import jit, random

import nifty8.re as jft
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import importlib

current_dir = os.path.abspath('')
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import util_working as util

jnp.set_printoptions(threshold=sys.maxsize)
importlib.reload(util)

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)

''' Massenmodell '''
rhos = jnp.array([0.021, 0.016, 0.012, 0.0009, 0.0006, 0.0031, 0.0015, 0.0020, 0.0022, 0.007, 0.0135, 0.006, 0.002, 0.0035, 0.0001])
sigmas = jnp.array([4., 7., 9., 40., 20., 7.5, 10.5, 14., 18., 18.5, 18.5, 20., 20., 37., 100.])
erhos = jnp.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * rhos
esigmas = jnp.array([1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 5., 5., 5., 10.])

rho_s = jft.LogNormalPrior(rhos, erhos, name="rho_s", shape=(15,))
sigma_s = jft.LogNormalPrior(sigmas, esigmas, name="sigma_s", shape=(15,))
rho_dm = jft.UniformPrior(0., 0.2, name="rho_dm", shape=(1,))

''' Domain '''
am_min = 6.000
am_max = 6.724
z2 = 100.
z1 = 1000.
z3 = 5000.

bins = np.loadtxt(f'../real data new intervals/bins_{am_min*1000:.0f}{am_max*1000:.0f}.txt')

i2 = np.where(bins<=z2)[0][-1]
i1 = np.where(bins>=z1)[0][0]
z2 = bins[i2]
z1 = bins[i1]
n = int(z1)+1
n3 = int(z3-z1)+1

bins = bins[i2:i1+1]
n_bins = int(len(bins)-1)

''' Run '''
seed = 371

poly = np.loadtxt(f'../real data new intervals/poly_{am_min*1000:.0f}{am_max*1000:.0f}.txt')
inference = True

''' Correlated Field '''
dims = (n, )

                       
cf_zm = dict(offset_mean=0., offset_std=(0.3, 0.3))
cf_fl = dict(   fluctuations=(0.5, 0.3), 
                loglogavgslope=(-3., 0.5),
                flexibility=(1e-3, 1e-16),
                asperity=(1e-3, 1e-16),)

cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(**cf_zm)
cfm.add_fluctuations(dims, distances=1.0, **cf_fl, prefix="ax1", non_parametric_kind="power")
correlated_field = cfm.finalize()


m_poly = jft.LogNormalPrior(poly[0], 0.2*poly[0], name="m_steig", shape=(1,))
b_poly = jft.LogNormalPrior(poly[1], 0.2*poly[1], name="b_steig", shape=(1,))


''' Data '''
data = np.loadtxt(f'../real data new intervals/n_{am_min*1000:.0f}{am_max*1000:.0f}.txt', dtype='int')

data = np.flip(data)[i2:i1]

''' Forward Model '''
norm = jnp.sum(data)
def complicated_function(rho_s, sigma_s, rho_dm, cf):
            params = jnp.column_stack((rho_s, sigma_s))
            rho_dm = rho_dm[0]

            uz, zs = util.diffraxDopri5(rho_dm, params, z1, n)
            uz_, zs_ = util.Solver(rho_dm, params, z1, z3, uz[-1], n3)
            vdfo_norm_calc, z, sig2 = util.vdfo_norm(z2, z1, zs, uz, n, poly, cf, jnp.array([1,2,3,4,5]))
            integral, z_borders = util.binning(vdfo_norm_calc, z, z2, z1, n, n_bins)

            return integral * norm/jnp.sum(integral)

class ForwardModel(jft.Model):
    def __init__(self):
        self.rho_s = rho_s
        self.sigma_s = sigma_s
        self.rho_dm = rho_dm
        self.correlated_field = correlated_field
        self.m_poly = m_poly
        self.b_poly = b_poly

        super().__init__(init =  self.rho_s.init| self.sigma_s.init | self.rho_dm.init | self.correlated_field.init | self.m_poly.init | self.b_poly.init)
        
    @jit
    def __call__(self, x):
        rs = self.rho_s(x)
        ss = self.sigma_s(x)
        rdm = self.rho_dm(x)

    
        m = self.m_poly(x)
        b = self.b_poly(x)
        rough_func = (b + m*jnp.linspace(0, z1, n))
            
        cf = rough_func * jnp.exp(self.correlated_field(x))

        dfo = complicated_function(rs, ss, rdm, cf)
        return jft.Vector({'dfo': dfo})

fwd = ForwardModel()
R_dfo = jft.Model(lambda x: x['dfo'], domain=fwd.target)
lh_dfo = jft.Poissonian(data).amend(R_dfo)

lh = (lh_dfo).amend(fwd)

''' Optimization '''
n_vi_iterations = 6
delta = 1e-4
n_samples = 10

key = random.PRNGKey(seed)
key, k_i, k_o = random.split(key, 3)

if inference:
    samples, state = jft.optimize_kl(
        lh,
        jft.Vector(lh.init(k_i)),
        n_total_iterations=n_vi_iterations,
        n_samples=lambda i: n_samples // 2 if i < 2 else n_samples,
        # Source for the stochasticity for sampling
        key=k_o,
        # Arguments for the conjugate gradient method used to drawing samples from an implicit covariance matrix
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
        callback=lambda x,y: print('\n', jft.mean(tuple(fwd(s)['dfo'] for s in x)),'\n'),
        sample_mode="nonlinear_resample",
        odir="./runtimeerror/results_test",
        resume=False,
    )

    print(fwd(samples[-1])['dfo'])

else:
      print()
    #  for i in range(100):
    #     key, subkey = random.split(key)
    #     sample = jft.random_like(subkey, fwd.domain)
    #     print(fwd(sample)['dfo'])







































# ''' Results '''
# results = {}

# for k in range(15):
#     exec(f'results["rhos{k+1}"] = tuple(rho_s(s)[{k}].tolist() for s in samples)')
#     exec(f'results["sigmas{k+1}"] = tuple(sigma_s(s)[{k}].tolist() for s in samples)')
#     exec(f'results["rho{k+1}"] = jft.mean_and_std(results["rhos{k+1}"])')
#     exec(f'results["sigma{k+1}"] = jft.mean_and_std(results["sigmas{k+1}"])')
# results["rhosdm"] = tuple(rho_dm(s).tolist()[0] for s in samples)
# results["rhodm"] = jft.mean_and_std(results["rhosdm"])
# if run == 'rough_func':
#     print('slope: ', jft.mean_and_std(tuple(m_poly(s) for s in samples)))
#     print('offset: ', jft.mean_and_std(tuple(b_poly(s) for s in samples)))

# if run == 'rough_func':
#     if label == 'fit':
#         Sigma_sq = jft.mean_and_std(tuple((b_poly(s) + m_poly(s)*jnp.linspace(0, z1, n)) * jnp.exp(correlated_field(s)) for s in samples))
# corrfield = jft.mean_and_std(tuple(correlated_field(s) for s in samples))

# meanr = [results[f'rho{k+1}'][0] for k in range(15)] 
# stdr = [results[f'rho{k+1}'][1] for k in range(15)] 

# data_rho = {
#     "Run": [name + f'rho_{k+1} ' + string for k in range(15)],
#     "Inferred Value rho": meanr,
#     "Standard Deviation rho": stdr,
#     "Samples rho": [results[f'rhos{k+1}'] for k in range(15)]
# }

# meanrd = [results['rhodm'][0]]
# stdrd = [results['rhodm'][1]]

# data_rd = { 
#     "Run": [name + f'rho_dm ' + string],
#     "Inferred Value rho": meanrd,
#     "Standard Deviation rho": stdrd,
#     "Samples rho": [results['rhosdm']]
# }

# means = [results[f'sigma{k+1}'][0] for k in range(15)]
# stds = [results[f'sigma{k+1}'][1] for k in range(15)]

# data_sigma = {
#     "Run": [name + f'sigma_{k+1} ' + string for k in range(15)],
#     "Inferred Value sigma": means,
#     "Standard Deviation sigma": stds,
#     "Samples sigma": [results[f'sigmas{k+1}'] for k in range(15)]
# }

# data_cf = {
#     "Run": [name + 'corrfield ' + string],
#     "Inferred Value cf": [corrfield[0]],
#     "Standard Deviation cf": [corrfield[1]],
#     "Inferred Value Sigma_sq": [Sigma_sq[0]],
#     "Standard Deviation Sigma_sq": [Sigma_sq[1]],
# }

# t1 = time.time()
# print('Time: ', t1-t0)

# dfr = pd.DataFrame(data_rho)
# dfr.set_index('Run', inplace=True)
# dfrd = pd.DataFrame(data_rd)
# dfrd.set_index('Run', inplace=True)
# dfs = pd.DataFrame(data_sigma)
# dfs.set_index('Run', inplace=True)
# dfcf = pd.DataFrame(data_cf)
# dfcf.set_index('Run', inplace=True)

# ''' Plot Results cf'''
# to_plot = [("Reconstruction", Sigma_sq, 'plot'), ("Correlated Field", corrfield, 'plot2')]

# fig, axs = plt.subplots(3, 1, figsize=(20, 20))
# grid = jnp.linspace(0, z1, n)
# for ax, v in zip(axs.flat, to_plot):
#     title, field, tp = v
#     ax.set_title(title)
#     ax.grid()
#     if tp == 'plot':
#         ax.plot(grid, field[0])
#         ax.plot(grid, field[0]+field[1], alpha=0.5)
#         ax.plot(grid, field[0]-field[1], alpha=0.5)
#         ax.plot(grid, poly[0]*grid+poly[1])
#         ax.sharex(axs[0])
#     elif tp == 'plot2':
#         ax.plot(grid, field[0])
#         ax.plot(grid, field[0]+field[1], alpha=0.5)
#         ax.plot(grid, field[0]-field[1], alpha=0.5)
#         ax.sharex(axs[0])
# fig.tight_layout()
# #fig.savefig(f'Plots/corrfield_{am_min:.0f}{am_max:.0f}_{run}{label}_{z2:.0f}_{z1:.0f}.png')
# plt.show()
