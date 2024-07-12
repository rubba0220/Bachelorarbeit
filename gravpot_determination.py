import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import nifty8.re as jft
import pandas as pd
import time
import util
import importlib
importlib.reload(util)

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

n = 121
z1 = 1200.
z2 = 200.
poly = (0,0)
n_bins = 10



rho_dm = 0.025
params = jnp.array([[0.021, 4.], [0.016, 7.], [0.012, 9.], \
                    [0.0009, 40.], [0.0006, 20.], [0.0031, 7.5], \
                    [0.0015, 10.5], [0.0020, 14.], [0.0022, 18.], \
                    [0.007, 18.5], [0.0135, 18.5], [0.006, 20.], \
                    [0.002, 20.], [0.0035, 37.], [0.0001, 100.]]) 

uz, zs = util.diffraxDopri5(rho_dm, params, z1, n)
vdfo_norm_calc, z = util.vdfo_norm(z2, z1, zs, uz, n, poly)
integral, z_borders = util.binning(vdfo_norm_calc, z, z2, z1, n, n_bins)



data = pd.read_csv('data new new/data_fail.csv', header=None)

rho_dm_truth = data.iloc[15,0]
rho_dm_inferred = data.iloc[15,1]
rho_dm_std = data.iloc[15,2]
# rho_dm_samples = data.iloc[15,3]
# rho_dm_samples = tuple(map(float, rho_dm_samples.strip("()").split(", ")))

rho_truth = data.iloc[:15,0]
rho_inferred = data.iloc[:15,1]
rho_std = data.iloc[:15,2]

sigma_truth = data.iloc[16:,0]
sigma_inferred = data.iloc[16:,1]
sigma_std = data.iloc[16:,2]

params_truth = jnp.transpose(jnp.vstack((jnp.array(rho_truth.to_numpy()), jnp.array(sigma_truth.to_numpy()))))
params_inferred = jnp.transpose(jnp.vstack((jnp.array(rho_inferred.to_numpy()), jnp.array(sigma_inferred.to_numpy()))))

uz_truth, zs_truth = util.diffraxDopri5(rho_dm_truth, params_truth, z1, n)
vdfo_norm_calc_truth, z_truth = util.vdfo_norm(z2, z1, zs_truth, uz_truth, n, poly)
uz_inferred, zs_inferred = util.diffraxDopri5(rho_dm_inferred, params_inferred, z1, n)
vdfo_norm_calc_inferred, z_inferred = util.vdfo_norm(z2, z1, zs_inferred, uz_inferred, n, poly)

uz_partial_truth, zs_partial_truth = util.diffraxDopri5(rho_dm_truth, params, z1, n)
vdfo_norm_calc_partial_truth, z_partial_truth = util.vdfo_norm(z2, z1, zs_partial_truth, uz_partial_truth, n, poly)
uz_partial_inferred, zs_partial_inferred = util.diffraxDopri5(rho_dm_inferred, params, z1, n)
vdfo_norm_calc_partial_inferred, z_partial_inferred = util.vdfo_norm(z2, z1, zs_partial_inferred, uz_partial_inferred, n, poly)



rhos_dm = jnp.unique(jnp.concatenate((jnp.round(jnp.linspace(0., 0.2, 11), decimals = 3), jnp.round(jnp.linspace(0.005, 0.03, 6), decimals=3))))

erhos = jnp.array([ 0.5, 0.5, 0.5,
                    0.5, 0.2, 0.2,
                    0.2, 0.2, 0.2,
                    0.2, 0.2, 0.2,
                    0.2, 0.2, 0.2]) * jnp.array([   0.021, 0.016, 0.012, 
                                                    0.0009, 0.0006, 0.0031, 
                                                    0.0015, 0.0020, 0.0022, 
                                                    0.007, 0.0135, 0.006, 
                                                    0.002, 0.0035, 0.0001])

esigmas = jnp.array([   1., 1., 1.,
                        1., 2., 2.,
                        2., 2., 2.,
                        2., 2., 5.,
                        5., 5., 10.])

params_pp = params + jnp.transpose(jnp.vstack((erhos, esigmas)))
params_pm = params + jnp.transpose(jnp.vstack((erhos, -esigmas)))
params_mp = params + jnp.transpose(jnp.vstack((-erhos, esigmas)))
params_mm = params + jnp.transpose(jnp.vstack((-erhos, -esigmas)))





''' Visualisierung fail'''
fig, ax = plt.subplots(2, 1, figsize=(20,20), sharex=True, gridspec_kw={'height_ratios': [1,1]})
plt.title('Run with failed reconstruction of $\\rho_{dm}$')
ax[0].set_xlabel('z/pc')
ax[0].set_ylabel('$\Phi / (km/s)^2$')
ax[0].scatter(zs_truth, [u[0] for u in uz_truth], marker='o', color='red', label='Truth')
ax[0].scatter(zs_inferred, [u[0] for u in uz_inferred], marker='x', color='black', label='Inferred')
ax[0].scatter(zs_partial_truth, [u[0] for u in uz_partial_truth], marker='o', color='blue', label='Partially Truth')
ax[0].scatter(zs_partial_inferred, [u[0] for u in uz_partial_inferred], marker='x', color='green', label='Partially Inferred')
ax[0].grid()
ax[1].set_xlabel('z/pc')
ax[1].set_ylabel('$\\nu / \\nu_0 $')
ax[1].scatter(z_truth, vdfo_norm_calc_truth, marker='o', color='red', label='Truth')
ax[1].scatter(z_inferred, vdfo_norm_calc_inferred, marker='x', color='black', label='Inferred')
ax[1].scatter(z_partial_truth, vdfo_norm_calc_partial_truth, marker='o', color='blue', label='Partially Truth')
ax[1].scatter(z_partial_inferred, vdfo_norm_calc_partial_inferred, marker='x', color='green', label='Partially Inferred')
ax[1].grid()
ax[1].legend()
fig.tight_layout()
fig.subplots_adjust(hspace=0.0)



''' Visualisierung rho_dm'''
fig, ax = plt.subplots(2, figsize=(20,20), sharex=True, gridspec_kw={'height_ratios': [1,1]})
plt.title('Influence of $\\rho_{dm}$')
ax[0].set_xlabel('z/pc')
ax[0].set_ylabel('$\Phi / (km/s)^2$')
ax[1].set_xlabel('z/pc')
ax[1].set_ylabel('$\\nu / \\nu_0 $')

for rho_dm in rhos_dm:
    uz, zs = util.diffraxDopri5(rho_dm, params, z1, n)
    vdfo_norm_calc, z = util.vdfo_norm(z2, z1, zs, uz, n, poly)
    ax[0].scatter(zs, [u[0] for u in uz], label=f'$\\rho_dm = {rho_dm:.3f}$')
    ax[1].scatter(z, vdfo_norm_calc, label=f'$\\rho_dm = {rho_dm:.3f}$')

ax[0].grid()
ax[1].grid()
ax[1].legend(prop={'size': 23})
fig.tight_layout()
fig.subplots_adjust(hspace=0.0)

''' Visualisierung rho_dm'''
fig, ax = plt.subplots(2, figsize=(20,20), sharex=True, gridspec_kw={'height_ratios': [1,1]})
plt.title('Influence of $\\rho_{dm}$')
ax[0].set_xlabel('z/pc')
ax[0].set_ylabel('$\Phi / (km/s)^2$')
ax[1].set_xlabel('z/pc')
ax[1].set_ylabel('$\\nu / \\nu_0 $')

for rho_dm in rhos_dm:
    uz, zs = util.diffraxDopri5(rho_dm, params, z1, n)
    vdfo_norm_calc, z = util.vdfo_norm(z2, z1, zs, uz, n, poly)
    ax[0].scatter(zs, [u[0] for u in uz], label=f'$\\rho_dm = {rho_dm:.3f}$')
    ax[1].scatter(z, vdfo_norm_calc/jnp.sum(vdfo_norm_calc), label=f'$\\rho_dm = {rho_dm:.3f}$')

ax[0].grid()
ax[1].grid()
ax[1].legend(prop={'size': 23})
fig.tight_layout()
fig.subplots_adjust(hspace=0.0)



''' Visualisierung other params'''
fig, ax = plt.subplots(2, figsize=(20,20), sharex=True, gridspec_kw={'height_ratios': [1,1]})
plt.title('Influence of other parameters')
ax[0].set_xlabel('z/pc')
ax[0].set_ylabel('$\Phi / (km/s)^2$')
ax[1].set_xlabel('z/pc')
ax[1].set_ylabel('$\\nu / \\nu_0 $')
uz_pp, zs_pp = util.diffraxDopri5(rho_dm, params_pp, z1, n)
ax[0].scatter(zs_pp, [u[0] for u in uz_pp], label='pp')
vdfo_norm_calc_pp, z_pp = util.vdfo_norm(z2, z1, zs_pp, uz_pp, n, poly)
ax[1].scatter(z_pp, vdfo_norm_calc_pp, label='pp')
uz_pm, zs_pm = util.diffraxDopri5(rho_dm, params_pm, z1, n)
ax[0].scatter(zs_pm, [u[0] for u in uz_pm], label='pm')
vdfo_norm_calc_pm, z_pm = util.vdfo_norm(z2, z1, zs_pm, uz_pm, n, poly)
ax[1].scatter(z_pm, vdfo_norm_calc_pm, label='pm')
uz_mp, zs_mp = util.diffraxDopri5(rho_dm, params_mp, z1, n)
ax[0].scatter(zs_mp, [u[0] for u in uz_mp], label='mp')
vdfo_norm_calc_mp, z_mp = util.vdfo_norm(z2, z1, zs_mp, uz_mp, n, poly)
ax[1].scatter(z_mp, vdfo_norm_calc_mp, label='mp')
uz_mm, zs_mm = util.diffraxDopri5(rho_dm, params_mm, z1, n)
ax[0].scatter(zs_mm, [u[0] for u in uz_mm], label='mm')
vdfo_norm_calc_mm, z_mm = util.vdfo_norm(z2, z1, zs_mm, uz_mm, n, poly)
ax[1].scatter(z_mm, vdfo_norm_calc_mm, label='mm')
ax[0].grid()
ax[1].grid()
ax[1].legend()
fig.tight_layout()
fig.subplots_adjust(hspace=0.0)



''' Visualisierung binning'''
fig, ax = plt.subplots(2, 1, figsize=(20,20), sharex=True, gridspec_kw={'height_ratios': [1,1]})
plt.title('Binning')
ax[0].set_xlabel('z/pc')
ax[0].set_ylabel('$\\nu / \\nu_0 $ in bins')
for i in range(len(z_borders)-1):
    xmin = z_borders[i]
    xmax = z_borders[i+1]
    ax[0].hlines(integral[i], xmin, xmax, color='black')
ax[0].scatter(z_borders[:-1], integral, marker='o', color='red')
ax[0].grid()
ax[0].legend()
ax[1].set_xlabel('z/pc')
ax[1].set_ylabel('$\\nu / \\nu_0 $')
ax[1].scatter(z, vdfo_norm_calc, marker='o', color='blue')
ax[1].grid()
ax[1].legend()
fig.tight_layout()
fig.subplots_adjust(hspace=0.0)
                                   

