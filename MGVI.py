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

n = 1201
i_s = int(200/1200 * (n-1))
i_n = int(200/1200 * (n-1))

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

    _, exp_int_list = lax.scan(exp_int_step, exp_int, jnp.arange(i_s, n-1, 1)) #letzter Punkt wird nicht genutzt (Riemannsumme links)

    exp_int_list = jnp.concatenate([jnp.array([exp_int]), exp_int_list], axis=0)

    vdfo_norm_calc = jnp.multiply(sigma_sq_norm**(-1), jnp.array(exp_int_list))

    return vdfo_norm_calc, z

''' Test des Algorithmus zur MGVI '''
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

            uz, zs = diffraxDopri5(rho_dm, params, z0, z1, u0, f, n)

            vdfo_norm_calc, z = vdfo_norm(z0, z1, i_s, i_n, uz, n)

            return vdfo_norm_calc

        return complicated_function(rs, ss, rdm)

# This initialises your forward-model which computes something data-like
fwd = ForwardModel()
def test_mgvi(s):
    seed = s
    key = random.PRNGKey(seed)

    noise_cov = lambda x: 0.001 * x
    noise_cov_inv = lambda x: 1. / 0.001 * x

    key, subkey = random.split(key)
    pos_truth = jft.random_like(subkey, fwd.domain)
    fwd_truth = fwd(pos_truth)

    key, subkey = random.split(key)
    noise_truth = (
        (noise_cov(jft.ones_like(fwd.target))) ** 0.5) * jft.random_like(subkey, fwd.target)
    data = fwd_truth + noise_truth

    # #Visualisierung
    # dz = (z1-z0)/(n-1)
    # z = jnp.linspace(z0+i_s*dz, z1, n-i_s)
    # fig, ax = plt.subplots(figsize=(20,10))
    # ax.set_xlabel('z/pc')
    # ax.set_ylabel('$\\nu / \\nu_0 $')
    # ax.scatter(z, data, marker='o')
    # ax.grid()
    # fig.tight_layout()

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

    for k in range(15):
        exec(f'results["rhos{k+1}"] = tuple(rho_s(s)[{k}].tolist() for s in samples)')
        exec(f'results["sigmas{k+1}"] = tuple(sigma_s(s)[{k}].tolist() for s in samples)')
        exec(f'results["rho{k+1}"] = jft.mean_and_std(results["rhos{k+1}"])')
        exec(f'results["sigma{k+1}"] = jft.mean_and_std(results["sigmas{k+1}"])')
    results["rhosdm"] = tuple(rho_dm(s).tolist()[0] for s in samples)
    results["rhodm"] = jft.mean_and_std(results["rhosdm"])

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

    dfr = pd.DataFrame(data_rho)
    dfs = pd.DataFrame(data_sigma)
    dfr.to_csv(f'data_rho_neuenorm.csv', mode='a', header=False, index=False)
    dfs.to_csv(f'data_sigma_neuenorm.csv', mode='a', header=False, index=False)

seed = 4
key = random.PRNGKey(seed)

key, subkey = random.split(key)
seeds = random.randint(subkey, (10,), 1, 1000000)

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

# print(seeds[59])

# #Ausreißer bei run 60, also pos 59 --> seed 390196

# test_mgvi(390196)




