#jax bisher nur für CPU intslliert (pip install -U "jax[cpu]")
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import jit, random
from functools import partial
from matplotlib import pyplot as plt
from scipy import constants as const
import nifty8.re as jft
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController

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

n = 1200
dz = 1

i1 = int(200/1200 * n)
i2 = n
i3 = int(100/1200 * n)



''' Test des Algorithmus zur MGVI '''
seed = 42
key = random.PRNGKey(seed)

roh_1 = jft.LogNormalPrior(0.09, 0.03, name="roh_1", shape=(1,))
sigma_1 = jft.LogNormalPrior(12., 10., name="sigma_1", shape=(1,))
roh_dm = jft.LogNormalPrior(0.02, 0.04, name="roh_dm", shape=(1,))

class ForwardModel(jft.Model):
    def __init__(self):
        self.roh_1 = roh_1
        self.sigma_1 = sigma_1
        self.roh_dm = roh_dm

        super().__init__(
            init=self.roh_1.init | self.sigma_1.init | self.roh_dm.init
        )

    def __call__(self, x):
        p1 = self.roh_1(x)
        p2 = self.sigma_1(x)
        p3 = self.roh_dm(x)

        def complicated_function(roh_1, sigma_1, roh_dm):
            params = jnp.array([[roh_1[0], sigma_1[0]], ])
            roh_dm = roh_dm[0]

            #Formulierung des Anfangswertproblems (z taucht in den Formeln auf, um an anderen DGLs zu testen)
            f = lambda roh_dm, params, z, u: jnp.array([u[1], \
                        4*jnp.pi*G * (jnp.sum(params[:,0]*jnp.exp(-u[0]/params[:,1]**2)) + roh_dm)])
            z0 = 0.
            u0 = jnp.array([0.,0.]) #freie Nullpunktswahl/Symmetrie

            #numerische Lösung (mittels Dopri5)
            @partial(jit, static_argnames=['f', 'n']) 
            def diffraxDopri5(roh_dm, params, z0, u0, f, n, dz):

                vector_field = lambda z, y, args: f(args[0], args[1], z, y) #wrapper für reihenfolge
                term = ODETerm(vector_field)
                solver = Dopri5()
                saveat = SaveAt(ts=jnp.linspace(0, n*dz, n))
                stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

                sol = diffeqsolve(term, solver, t0=z0, t1=z0+n*dz, dt0=dz, y0=u0,
                                    args=(roh_dm, params),
                                saveat=saveat,
                                stepsize_controller=stepsize_controller)

                zs = sol.ts
                uz = sol.ys

                return uz

            uz = diffraxDopri5(roh_dm, params, z0, u0, f, n, dz)

            #mock velocity dispersion function
            def sigma(z):
                return 20 + 17*z/1000 #z in pc, sigma in km/s

            #Berechnung des tracer density drop off
            #neu:lax.scan()
            sigma_sq_norm = jnp.array([(sigma(z0+i*dz)/sigma(z0+i3*dz))**(2) for i in range(i1,i2)])
            exp_int = jnp.exp(-jnp.sum(\
                        sigma(jnp.array([z0+i*dz for i in range(i3,i1)]))**(-2) \
                        * jnp.array(uz)[i3:i1,1] * dz))

            def exp_int_step(exp_int, i):
                return exp_int * jnp.exp(-sigma(z0+i*dz)**(-2) * jnp.array(uz)[i,1] * dz), \
                        exp_int * jnp.exp(-sigma(z0+i*dz)**(-2) * jnp.array(uz)[i,1] * dz)

            _, exp_int_list = lax.scan(exp_int_step, exp_int, jnp.arange(i1, i2, 1))

            vdfo_norm_calc = jnp.multiply(sigma_sq_norm**(-1), jnp.array(exp_int_list))

            return vdfo_norm_calc

        return complicated_function(p1, p2, p3)

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
ax.scatter([0.+i*dz for i in range(i1,i2)], [data], marker='o')
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
rohdm = jft.mean_and_std(tuple(roh_dm(s) for s in samples))

# Please save your results in some way:
# TODO

print("Inferred values:", "MEAN:", roh1[0], "STD:", roh1[1])
print("with the true value being:", roh_1(pos_truth))
print('')
print("Inferred values:", "MEAN:", sigma1[0], "STD:", sigma1[1])
print("with the true value being:", sigma_1(pos_truth))
print('')
print("Inferred values:", "MEAN:", rohdm[0], "STD:", rohdm[1])
print("with the true value being:", roh_dm(pos_truth))

plt.show()