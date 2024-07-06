#jax bisher nur für CPU intslliert (pip install -U "jax[cpu]")
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import jit, random
from functools import partial
from matplotlib import pyplot as plt
from scipy import constants as const
import nifty8.re as jft
import diffrax as dif
import pandas as pd

import time
jax.config.update("jax_enable_x64", True)

t0 = time.time()

# Plot-Formatierung
plt.rcParams['font.size'] = 24.0
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.0

G = const.G / (3.0857E+16)**3 * 1.989E+30 * (3.0857E+13)**2
                            #Umrechnung in pc^3/M_sun/s^2 (grav pot in (km/s)^2)
                            #Umrechnung, sodass z in parsec

n = 121
i_s = int(200/1200 * (n-1))
i_n = int(100/1200 * (n-1))

#Formulierung des Anfangswertproblems (z taucht in den Formeln auf, um an anderen DGLs zu testen)
f = lambda rho_dm, params, z, u: jnp.array([u[1], \
            4*jnp.pi*G * (jnp.sum(params[:,0]*jnp.exp(-u[0]/params[:,1]**2)) + rho_dm)])
z0 = 0.
z1 = 1200.
u0 = jnp.array([0.,0.]) #freie Nullpunktswahl/Symmetrie

#numerische Lösung (mittels Dopri5/rk4)
@partial(jit, static_argnames=['f', 'n']) 
def diffraxDopri5(rho_dm, params, z0, z1, u0, f, n):

    vector_field = lambda z, y, args: f(args[0], args[1], z, y) #wrapper für reihenfolge
    term = dif.ODETerm(vector_field)
    solver = dif.Dopri5()
    saveat = dif.SaveAt(ts=jnp.linspace(0, z1, n))
    stepsize_controller = dif.PIDController(rtol=1e-3, atol=1e-6)
    adjoint = dif.DirectAdjoint()

    sol = dif.diffeqsolve(  term, solver, 
                            t0=z0, t1=z1, dt0=None, y0=u0, args=(rho_dm, params), 
                            saveat=saveat,
                            adjoint=adjoint,
                            stepsize_controller=stepsize_controller) #throw=False, max_steps=None

    zs = sol.ts
    uz = sol.ys

    return uz, zs

@partial(jit, static_argnames=['f', 'n'])
def eigenerSolverV2(rho_dm, params, z0, z1, u0, f, n):
    dz = (z1-z0)/(n-1)

    # Runge-Kutta 4. Ordnung
    def rk4_step(rho_dm, params, z0, dz, u0, f):
        k1 = dz * f(rho_dm, params, z0, u0)
        k2 = dz * f(rho_dm, params, z0 + dz / 2, u0 + k1 / 2)
        k3 = dz * f(rho_dm, params, z0 + dz / 2, u0 + k2 / 2)
        k4 = dz * f(rho_dm, params, z0 + dz, u0 + k3)
        u1 = u0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return u1

    def rk4_step_scan(u, x):
        return rk4_step(rho_dm, params, x, dz, u, f), \
            rk4_step(rho_dm, params, x, dz, u, f)

    zs = jnp.linspace(z0, z1, n)
    _, uz = lax.scan(rk4_step_scan, u0, zs[:-1])
    uz = jnp.concatenate([jnp.array([u0]), uz], axis=0)

    return uz, zs

#Berechnung des tracer density drop off
#neu:lax.scan()
@partial(jit, static_argnames=['n', 'i_s', 'i_n'])
def vdfo_norm(z0, z1, i_s, i_n, uz, n):
    dz = (z1-z0)/(n-1)

    #mock velocity dispersion function
    def sigma(z):
        return 20. + 17.*z/1000. #z in pc, sigma in km/s
    
    z = jnp.linspace(z0+i_s*dz, z1, n-i_s)
    sigma_sq_norm = (sigma(z)/sigma(z0+i_n*dz))**(2)
    z_ns = jnp.linspace(z0+i_n*dz, z0+(i_s-1)*dz, i_s-i_n)
    sigmass = sigma(z_ns)

    exp_int = jnp.exp(-jnp.sum(\
                sigmass**(-2) \
                * jnp.array(uz)[i_n:i_s,1] * dz))

    def exp_int_step(exp_int, i):
        return exp_int * jnp.exp(-sigma(z0+i*dz)**(-2) * jnp.array(uz)[i,1] * dz), \
                exp_int * jnp.exp(-sigma(z0+i*dz)**(-2) * jnp.array(uz)[i,1] * dz)

    _, exp_int_list = lax.scan(exp_int_step, exp_int, jnp.arange(i_s, n-1, 1))

    exp_int_list = jnp.concatenate([jnp.array([exp_int]), exp_int_list], axis=0)

    vdfo_norm_calc = jnp.multiply(sigma_sq_norm**(-1), jnp.array(exp_int_list))

    return vdfo_norm_calc, z

rho_dm = 0.025

rhos_dm = jnp.unique(jnp.concatenate((jnp.round(jnp.linspace(0., 0.2, 11), decimals = 3), jnp.round(jnp.linspace(0.005, 0.03, 6), decimals=3))))
params = jnp.array([[0.021, 4.], [0.016, 7.], [0.012, 9.], \
                    [0.0009, 40.], [0.0006, 20.], [0.0031, 7.5], \
                    [0.0015, 10.5], [0.0020, 14.], [0.0022, 18.], \
                    [0.007, 18.5], [0.0135, 18.5], [0.006, 20.], \
                    [0.002, 20.], [0.0035, 37.], [0.0001, 100.]]) 

data = pd.read_csv('data_fail.csv', header=None)
rho_dm_truth = data.iloc[15,0]
rho_dm_inferred = data.iloc[15,1]
rho_dm_std = data.iloc[15,2]
rho_dm_samples = data.iloc[15,3]
rho_dm_samples = tuple(map(float, rho_dm_samples.strip("()").split(", ")))

rho_truth = data.iloc[:15,0]
rho_inferred = data.iloc[:15,1]
rho_std = data.iloc[:15,2]

sigma_truth = data.iloc[16:,0]
sigma_inferred = data.iloc[16:,1]
sigma_std = data.iloc[16:,2]

params_truth = jnp.transpose(jnp.vstack((jnp.array(rho_truth.to_numpy()), jnp.array(sigma_truth.to_numpy()))))
params_inferred = jnp.transpose(jnp.vstack((jnp.array(rho_inferred.to_numpy()), jnp.array(sigma_inferred.to_numpy()))))

uz_truth, zs_truth = diffraxDopri5(rho_dm_truth, params_truth, z0, z1, u0, f, n)
vdfo_norm_calc_truth, z_truth = vdfo_norm(z0, z1, i_s, i_n, uz_truth, n)
uz_inferred, zs_inferred = diffraxDopri5(rho_dm_inferred, params_inferred, z0, z1, u0, f, n)
vdfo_norm_calc_inferred, z_inferred = vdfo_norm(z0, z1, i_s, i_n, uz_inferred, n)

uz_partial_truth, zs_partial_truth = diffraxDopri5(rho_dm_truth, params, z0, z1, u0, f, n)
vdfo_norm_calc_partial_truth, z_partial_truth = vdfo_norm(z0, z1, i_s, i_n, uz_partial_truth, n)
uz_partial_inferred, zs_partial_inferred = diffraxDopri5(rho_dm_inferred, params, z0, z1, u0, f, n)
vdfo_norm_calc_partial_inferred, z_partial_inferred = vdfo_norm(z0, z1, i_s, i_n, uz_partial_inferred, n)

''' Visualisierung fail'''
fig, ax = plt.subplots(2, 1, figsize=(20,20), sharex=True, gridspec_kw={'height_ratios': [1,1]})
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

''' Visualisierung'''
fig, ax = plt.subplots(2, figsize=(20,20), sharex=True, gridspec_kw={'height_ratios': [1,1]})
ax[0].set_xlabel('z/pc')
ax[0].set_ylabel('$\Phi / (km/s)^2$')
ax[1].set_xlabel('z/pc')
ax[1].set_ylabel('$\\nu / \\nu_0 $')

for rho_dm in rhos_dm:
    uz, zs = diffraxDopri5(rho_dm, params, z0, z1, u0, f, n)
    vdfo_norm_calc, z = vdfo_norm(z0, z1, i_s, i_n, uz, n)
    ax[0].scatter(zs, [u[0] for u in uz], label=f'$\\rho_dm = {rho_dm:.3f}$')
    ax[1].scatter(z, vdfo_norm_calc, label=f'$\\rho_dm = {rho_dm:.3f}$')

ax[0].grid()
ax[1].grid()
ax[1].legend()
fig.tight_layout()
fig.subplots_adjust(hspace=0.0)

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

''' Visualisierung'''
fig, ax = plt.subplots(2, figsize=(20,20), sharex=True, gridspec_kw={'height_ratios': [1,1]})
ax[0].set_xlabel('z/pc')
ax[0].set_ylabel('$\Phi / (km/s)^2$')
ax[1].set_xlabel('z/pc')
ax[1].set_ylabel('$\\nu / \\nu_0 $')
uz_pp, zs_pp = diffraxDopri5(rho_dm, params_pp, z0, z1, u0, f, n)
ax[0].scatter(zs_pp, [u[0] for u in uz_pp], label='pp')
vdfo_norm_calc_pp, z_pp = vdfo_norm(z0, z1, i_s, i_n, uz_pp, n)
ax[1].scatter(z_pp, vdfo_norm_calc_pp, label='pp')
uz_pm, zs_pm = diffraxDopri5(rho_dm, params_pm, z0, z1, u0, f, n)
ax[0].scatter(zs_pm, [u[0] for u in uz_pm], label='pm')
vdfo_norm_calc_pm, z_pm = vdfo_norm(z0, z1, i_s, i_n, uz_pm, n)
ax[1].scatter(z_pm, vdfo_norm_calc_pm, label='pm')
uz_mp, zs_mp = diffraxDopri5(rho_dm, params_mp, z0, z1, u0, f, n)
ax[0].scatter(zs_mp, [u[0] for u in uz_mp], label='mp')
vdfo_norm_calc_mp, z_mp = vdfo_norm(z0, z1, i_s, i_n, uz_mp, n)
ax[1].scatter(z_mp, vdfo_norm_calc_mp, label='mp')
uz_mm, zs_mm = diffraxDopri5(rho_dm, params_mm, z0, z1, u0, f, n)
ax[0].scatter(zs_mm, [u[0] for u in uz_mm], label='mm')
vdfo_norm_calc_mm, z_mm = vdfo_norm(z0, z1, i_s, i_n, uz_mm, n)
ax[1].scatter(z_mm, vdfo_norm_calc_mm, label='mm')
ax[0].grid()
ax[1].grid()
ax[1].legend()
fig.tight_layout()
fig.subplots_adjust(hspace=0.0)


                                   

