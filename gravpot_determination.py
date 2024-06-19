#jax bisher nur für CPU intslliert (pip install -U "jax[cpu]")
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import jit
from functools import partial
from matplotlib import pyplot as plt
from scipy import constants as const
from jax import random
import nifty8.re as jft

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

G = const.G / (3.0857E+16)**3 * 1.989E+30 * (3.0857E+13)**2 #impurity ist ok (G ist konstant über die Auswertung)
                            #Umrechnung in pc^3/M_sun/s^2 (grav pot in (km/s)^2)
                            #Umrechnung, sodass z in parsec

# Runge-Kutta 4. Ordnung
@partial(jit, static_argnames=['f'])
def rk4_step(roh_dm, params, z0, u0, dz, f):
    k1 = dz * f(roh_dm, params, z0, u0)
    k2 = dz * f(roh_dm, params, z0 + dz / 2, u0 + k1 / 2)
    k3 = dz * f(roh_dm, params, z0 + dz / 2, u0 + k2 / 2)
    k4 = dz * f(roh_dm, params, z0 + dz, u0 + k3)
    u1 = u0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return u1

#Formulierung des Anfangswertproblems (z taucht in den Formeln auf, um an anderen DGLs zu testen)
f = lambda roh_dm, params, z, u: jnp.array([u[1], \
            4*jnp.pi*G * (jnp.sum(params[:,0]*jnp.exp(-u[0]/params[:,1]**2)) \
                          + roh_dm)])
z0 = 0.
u0 = jnp.array([0.,0.]) #freie Nullpunktswahl/Symmetrie
roh_dm = 0.025 #M_sun/pc^3
params = jnp.array([[0.021, 4.], [0.016, 7.], [0.012, 9.], \
                    [0.0009, 40.], [0.0006, 20.], [0.0031, 7.5], \
                    [0.0015, 10.5], [0.0020, 14.], [0.0022, 18.], \
                    [0.007, 18.5], [0.0135, 18.5], [0.006, 20.], \
                    [0.002, 20.], [0.0035, 37.], [0.0001, 100.]]) 
                                                        #mu(0) ??? in M_sun/pc^3, sigma(0) in km/s

#numerische Lösung (Optimierung?)
uz = []
dz = 1
n = 1200

#neu mit lax.scan
def rk4_step_scan(u, i):
    return rk4_step(roh_dm, params, z0+i*dz, u, dz, f), \
        rk4_step(roh_dm, params, z0+i*dz, u, dz, f)

_, uz = lax.scan(rk4_step_scan, u0, jnp.arange(0, n*dz, dz))



# #neu mit diffrax
# from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController

# vector_field = lambda z, y, args: f(args[0], args[1], z, y) #wrapper für reihenfolge
# term = ODETerm(vector_field)
# solver = Dopri5()
# saveat = SaveAt(ts=jnp.linspace(0, 1200, n))
# stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

# sol = diffeqsolve(term, solver, t0=0, t1=1200, dt0=dz, y0=u0,
#                     args=(roh_dm, params),
#                   saveat=saveat,
#                   stepsize_controller=stepsize_controller)

# zs = sol.ts
# uz = sol.ys

# #Visualisierung
# fig, ax1 = plt.subplots(figsize=(20,10))
# plt.title('Test')
# ax1.set_xlabel('z/pc')
# ax1.set_ylabel('$\Phi / (km/s)^2$')
# ax1.scatter([z0+i*dz for i in range(n)], [u[0] for u in uz], marker='o')
# ax1.grid()
# fig.tight_layout()

# #mock data: sekante in Figur 5
# def vdfo_norm(z):
#     return 10**(0.57 + -1.63*z/1000) #z in pc

def sigma(z):
    return 20 + 17*z/1000 #z in pc, sigma in km/s

# fig, ax = plt.subplots(1,2, figsize=(20,10))
# ax[0].set_xlabel('z/pc')
# #ax[0].set_yscale('log')
# ax[0].set_ylabel('$\\nu / \\nu(z_0) $')
# ax[0].scatter([100+i*10 for i in range(100)], [vdfo_norm(100+i*10) for i in range(100)], marker='o')
# ax[0].grid()
# ax[1].set_xlabel('z/pc')
# ax[1].set_ylabel('$\sigma_z / (km/s)$')
# ax[1].scatter([100+i*10 for i in range(100)], [sigma(100+i*10) for i in range(100)], marker='o')
# ax[1].grid()
# fig.tight_layout()

#Berechnung des tracer density drop off
i1 = int(200/1200 * n)
i2 = n
i3 = int(100/1200 * n)

#neu:lax.scan()
sigma_sq_norm = jnp.array([(sigma(z0+i*dz)/sigma(z0+i3*dz))**(2) for i in range(i1,i2)])
exp_int = jnp.exp(-jnp.sum(\
            sigma(jnp.array([z0+i*dz for i in range(i3,i1)]))**(-2) \
            * jnp.array(uz)[i3:i1,1] * dz))

def exp_int_step(exp_int, i):
    return exp_int * jnp.exp(-sigma(z0+i*dz)**(-2) * jnp.array(uz)[i,1] * dz), \
            exp_int * jnp.exp(-sigma(z0+i*dz)**(-2) * jnp.array(uz)[i,1] * dz)

carry, exp_int_list = lax.scan(exp_int_step, exp_int, jnp.arange(i1, i2, 1))

vdfo_norm_calc = jnp.multiply(sigma_sq_norm**(-1), jnp.array(exp_int_list))

# #Visualisierung
# fig, ax = plt.subplots(figsize=(20,10))
# ax.set_xlabel('z/pc')
# #ax.set_yscale('log')
# ax.set_ylabel('$\\nu / \\nu_0 $')
# ax.scatter([z0+i*dz for i in range(i1,i2)], [vdfo_norm_calc], marker='o')
# ax.grid()
# fig.tight_layout()

seed = 42
key = random.PRNGKey(seed)

roh_1 = jft.UniformPrior(0.04, 0.05, name="roh_1", shape=(1,))
sigma_1 = jft.UniformPrior(3., 6., name="sigma_1", shape=(1,))
roh_2 = jft.UniformPrior(0.04, 0.05, name="roh_2", shape=(1,))
sigma_2 = jft.UniformPrior(5., 9., name="sigma_2", shape=(1,))
roh_dm = jft.UniformPrior(0.02, 0.04, name="roh_dm", shape=(1,))

class ForwardModel(jft.Model):
    def __init__(self):
        self.roh_1 = roh_1
        self.sigma_1 = sigma_1
        self.roh_2 = roh_2
        self.sigma_2 = sigma_2
        self.roh_dm = roh_dm

        super().__init__(
            init=self.roh_1.init | self.sigma_1.init | self.roh_2.init | self.sigma_2.init | self.roh_dm.init
        )

    def __call__(self, x):
        p1 = self.roh_1(x)
        p2 = self.sigma_1(x)
        p3 = self.roh_2(x)
        p4 = self.sigma_2(x)
        p5 = self.roh_dm(x)

        def complicated_function(roh_1, sigma_1, roh_2, sigma_2, roh_dm):
            params = jnp.array([[roh_1[0], sigma_1[0]], [roh_2[0], sigma_2[0]]])
            roh_dm = roh_dm[0]

            ''' wichtige Frage !!!'''

            # Runge-Kutta 4. Ordnung
            @partial(jit, static_argnames=['f'])
            def rk4_step(roh_dm, params, z0, u0, dz, f):
                k1 = dz * f(roh_dm, params, z0, u0)
                k2 = dz * f(roh_dm, params, z0 + dz / 2, u0 + k1 / 2)
                k3 = dz * f(roh_dm, params, z0 + dz / 2, u0 + k2 / 2)
                k4 = dz * f(roh_dm, params, z0 + dz, u0 + k3)
                u1 = u0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                return u1

            #Formulierung des Anfangswertproblems (z taucht in den Formeln auf, um an anderen DGLs zu testen)
            f = lambda roh_dm, params, z, u: jnp.array([u[1], \
                        4*jnp.pi*G * (jnp.sum(params[:,0]*jnp.exp(-u[0]/params[:,1]**2)) \
                          + roh_dm)])

            z0 = 0.
            u0 = jnp.array([0.,0.]) #freie Nullpunktswahl/Symmetrie

            #numerische Lösung (Optimierung?)
            dz = 1
            n = 1200

            #neu mit lax.scan
            def rk4_step_scan(u, i):
                return rk4_step(roh_dm, params, z0+i*dz, u, dz, f), \
                    rk4_step(roh_dm, params, z0+i*dz, u, dz, f)

            _, uz = lax.scan(rk4_step_scan, u0, jnp.arange(0, n*dz, dz))

            #Berechnung des tracer density drop off
            i1 = int(200/1200 * n)
            i2 = n
            i3 = int(100/1200 * n)

            def sigma(z):
                return 20 + 17*z/1000 #z in pc, sigma in km/s

            #neu:lax.scan()
            sigma_sq_norm = jnp.array([(sigma(z0+i*dz)/sigma(z0+i3*dz))**(2) for i in range(i1,i2)])
            exp_int = jnp.exp(-jnp.sum(\
                        sigma(jnp.array([z0+i*dz for i in range(i3,i1)]))**(-2) \
                        * jnp.array(uz)[i3:i1,1] * dz))

            def exp_int_step(exp_int, i):
                return exp_int * jnp.exp(-sigma(z0+i*dz)**(-2) * jnp.array(uz)[i,1] * dz), \
                        exp_int * jnp.exp(-sigma(z0+i*dz)**(-2) * jnp.array(uz)[i,1] * dz)

            carry, exp_int_list = lax.scan(exp_int_step, exp_int, jnp.arange(i1, i2, 1))

            vdfo_norm_calc = jnp.multiply(sigma_sq_norm**(-1), jnp.array(exp_int_list))

            return vdfo_norm_calc

        return complicated_function(p1, p2, p3, p4, p5)

# This initialises your forward-model which computes something data-like
fwd = ForwardModel()

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
ax.scatter([z0+i*dz for i in range(i1,i2)], [data], marker='o')
ax.grid()
fig.tight_layout()

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
roh1 = jft.mean_and_std(tuple(roh_1(s) for s in samples))
sigma1 = jft.mean_and_std(tuple(sigma_1(s) for s in samples))
roh2 = jft.mean_and_std(tuple(roh_2(s) for s in samples))
sigma2 = jft.mean_and_std(tuple(sigma_2(s) for s in samples))
rohdm = jft.mean_and_std(tuple(roh_dm(s) for s in samples))

# Please save your results in some way:
# TODO

print("Inferred values:", "MEAN:", roh1[0], "STD:", roh1[1])
print("with the true value being:", roh_1(pos_truth))
print('')
print("Inferred values:", "MEAN:", sigma1[0], "STD:", sigma1[1])
print("with the true value being:", sigma_1(pos_truth))
print('')
print("Inferred values:", "MEAN:", roh2[0], "STD:", roh2[1])
print("with the true value being:", roh_2(pos_truth))
print('')
print("Inferred values:", "MEAN:", sigma2[0], "STD:", sigma2[1])
print("with the true value being:", sigma_2(pos_truth))
print('')
print("Inferred values:", "MEAN:", rohdm[0], "STD:", rohdm[1])
print("with the true value being:", roh_dm(pos_truth))

plt.show()