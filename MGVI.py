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

n = 1200
dz = 1.

i1 = int(200/1200 * n)
i2 = n
i3 = int(100/1200 * n)

#Formulierung des Anfangswertproblems (z taucht in den Formeln auf, um an anderen DGLs zu testen)
f = lambda roh_dm, params, z, u: jnp.array([u[1], \
            4*jnp.pi*G * (jnp.sum(params[:,0]*jnp.exp(-u[0]/params[:,1]**2)) + roh_dm)])
z0 = 0.
u0 = jnp.array([0.,0.]) #freie Nullpunktswahl/Symmetrie

#numerische Lösung (mittels Dopri5/rk4)
@partial(jit, static_argnames=['f', 'n']) 
def diffraxDopri5(roh_dm, params, z0, u0, f, n, dz):

    vector_field = lambda z, y, args: f(args[0], args[1], z, y) #wrapper für reihenfolge
    term = ODETerm(vector_field)
    solver = Dopri5()
    saveat = SaveAt(ts=jnp.linspace(0, n*dz, n))
    stepsize_controller = PIDController(rtol=1e-3, atol=1e-6)
    adjoint = DirectAdjoint()

    sol = diffeqsolve(term, solver, t0=z0, t1=z0+n*dz, dt0=dz, y0=u0,
                        args=(roh_dm, params),
                    saveat=saveat,
                    stepsize_controller=stepsize_controller,
                    adjoint = adjoint, throw=False)
                    #max_steps=65536)

    #zs = sol.ts
    uz = sol.ys

    return uz

@partial(jit, static_argnames=['f', 'n']) 
def eigenerSolverV2(roh_dm, params, z0, u0, f, n, dz):
                                                        
    # Runge-Kutta 4. Ordnung
    # @partial(jit, static_argnames=['f']) #nötig ??
    def rk4_step(roh_dm, params, z0, u0, dz, f):
        k1 = dz * f(roh_dm, params, z0, u0)
        k2 = dz * f(roh_dm, params, z0 + dz / 2, u0 + k1 / 2)
        k3 = dz * f(roh_dm, params, z0 + dz / 2, u0 + k2 / 2)
        k4 = dz * f(roh_dm, params, z0 + dz, u0 + k3)
        u1 = u0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return u1

    def rk4_step_scan(u, i):
        return rk4_step(roh_dm, params, z0+i*dz, u, dz, f), \
            rk4_step(roh_dm, params, z0+i*dz, u, dz, f)

    _, uz = lax.scan(rk4_step_scan, u0, jnp.linspace(0, n*dz, n))

    return uz

#Berechnung des tracer density drop off
#neu:lax.scan()
@partial(jit, static_argnames=['i1', 'i2', 'i3'])
def vdfo_norm(i1, i2, i3, z0, dz, uz):
    
    #mock velocity dispersion function
    def sigma(z):
        return 20. + 17.*z/1000. #z in pc, sigma in km/s
    
    zs = jnp.linspace(z0+i1*dz, z0+(i2-1)*dz, i2-i1)
    sigma_sq_norm = (sigma(zs)/sigma(z0+i3*dz))**(2)
    zss = jnp.linspace(z0+i3*dz, z0+(i1-1)*dz, i1-i3)
    sigmass = sigma(zss)

    exp_int = jnp.exp(-jnp.sum(\
                sigmass**(-2) \
                * jnp.array(uz)[i3:i1,1] * dz))

    def exp_int_step(exp_int, i):
        return exp_int * jnp.exp(-sigma(z0+i*dz)**(-2) * jnp.array(uz)[i,1] * dz), \
                exp_int * jnp.exp(-sigma(z0+i*dz)**(-2) * jnp.array(uz)[i,1] * dz)

    _, exp_int_list = lax.scan(exp_int_step, exp_int, jnp.arange(i1, i2, 1))

    vdfo_norm_calc = jnp.multiply(sigma_sq_norm**(-1), jnp.array(exp_int_list))

    return vdfo_norm_calc

''' Test des Algorithmus zur MGVI '''
rohs = jnp.array([  0.021, 0.016, 0.012, 
                    0.0009, 0.0006, 0.0031, 
                    0.0015, 0.0020, 0.0022, 
                    0.007, 0.0135, 0.006, 
                    0.002, 0.0035, 0.0001])

sigmas = jnp.array([4., 7., 9., 
                    40., 20., 7.5, 
                    10.5, 14., 18., 
                    18.5, 18.5, 20., 
                    20., 37., 100.])

erohs = jnp.array([ 0.5, 0.5, 0.5,
                    0.5, 0.2, 0.2,
                    0.2, 0.2, 0.2,
                    0.2, 0.2, 0.2,
                    0.2, 0.2, 0.2]) * rohs

esigmas = jnp.array([   1., 1., 1.,
                        1., 2., 2.,
                        2., 2., 2.,
                        2., 2., 5.,
                        5., 5., 10.])

for k in range(1):
    exec(f'roh_{k+1} = jft.LogNormalPrior(rohs[{k}], erohs[{k}], name="roh_{k+1}".format({k}), shape=(1,))')
    exec(f'sigma_{k+1} = jft.LogNormalPrior(sigmas[{k}], esigmas[{k}], name="sigma_{k+1}".format({k}), shape=(1,))')

roh_dm = jft.UniformPrior(0., 0.2, name="roh_dm", shape=(1,))

class ForwardModel(jft.Model):
    def __init__(self):
        for k in range(1):
            exec(f'self.roh_{k+1} = roh_{k+1}')
            exec(f'self.sigma_{k+1} = sigma_{k+1}')
        self.roh_dm = roh_dm

        super().__init__(
            init =   self.roh_1.init | self.sigma_1.init | self.roh_dm.init)

    def __call__(self, x):
        r1 = self.roh_1(x)
        s1 = self.sigma_1(x)
        rdm = self.roh_dm(x)

        def complicated_function(roh_1, sigma_1, roh_dm):
            params = jnp.array([
                    [roh_1[0], sigma_1[0]]])
            roh_dm = roh_dm[0]

            uz = diffraxDopri5(roh_dm, params, z0, u0, f, n, dz)

            vdfo_norm_calc = vdfo_norm(i1, i2, i3, z0, dz, uz)

            return vdfo_norm_calc

        return complicated_function(r1, s1, rdm)

# This initialises your forward-model which computes something data-like
fwd = ForwardModel()
def test_mgvi(s, ns = 6):
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

    # #Visualisierung
    # fig, ax = plt.subplots(figsize=(20,10))
    # ax.set_xlabel('z/pc')
    # #ax.set_yscale('log')
    # ax.set_ylabel('$\\nu / \\nu_0 $')
    # ax.scatter([0.+i*dz for i in range(i1,i2)], [data], marker='o')
    # ax.grid()
    # fig.tight_layout()

    lh = jft.Gaussian(data, noise_cov_inv).amend(fwd)


    # Now lets run the main inference scheme:
    n_vi_iterations = ns
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

    for k in range(1):
        exec(f'results["rohs{k+1}"] = tuple(roh_{k+1}(s).tolist()[0] for s in samples)')
        exec(f'results["sigmas{k+1}"] = tuple(sigma_{k+1}(s).tolist()[0] for s in samples)')
        exec(f'results["roh{k+1}"] = jft.mean_and_std(results["rohs{k+1}"])')
        exec(f'results["sigma{k+1}"] = jft.mean_and_std(results["sigmas{k+1}"])')
    results["rohsdm"] = tuple(roh_dm(s).tolist()[0] for s in samples)
    results["rohdm"] = jft.mean_and_std(results["rohsdm"])
    # Please save your results in some way:
    # TODO

    truthr = [roh_1(pos_truth)[0], roh_dm(pos_truth)[0]]
    meanr = [results[f'roh{k+1}'][0] for k in range(1)] + [results['rohdm'][0]]
    stdr = [results[f'roh{k+1}'][1] for k in range(1)] + [results['rohdm'][1]]


    data_roh = {
        "True Value roh": truthr,

        "Inferred Value roh": meanr,

        "Standard Deviation roh": stdr,

        "Samples roh": [results[f'rohs{k+1}'] for k in range(1)] + [results['rohsdm']],

        "Abweichung roh": list((jnp.array(truthr) - jnp.array(meanr))/jnp.array(stdr))
    }

    truths = [sigma_1(pos_truth)[0]]
    means = [results[f'sigma{k+1}'][0] for k in range(1)]
    stds = [results[f'sigma{k+1}'][1] for k in range(1)]


    data_sigma = {
        "True Value sigma": truths,

        "Inferred Value sigma": means,

        "Standard Deviation sigma": stds,

        "Samples sigma": [results[f'sigmas{k+1}'] for k in range(1)],

        "Abweichung sigma": list((jnp.array(truths) - jnp.array(means))/jnp.array(stds))
    }

    dfr = pd.DataFrame(data_roh)
    dfs = pd.DataFrame(data_sigma)
    dfr.to_csv(f'data_roh_1.csv', mode='a', header=False, index=False)
    dfs.to_csv(f'data_sigma_1.csv', mode='a', header=False, index=False)

    t1 = time.time()
    print('Time:', t1-t0, 's')

seed = 33
key = random.PRNGKey(seed)

key, subkey = random.split(key)
seeds = random.randint(subkey, (20,), 1, 1000000)

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
        test_mgvi(s, 6)


