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

for k in range(15):
    exec(f'roh_{k+1} = jft.NormalPrior(rohs[{k}], erohs[{k}], name="roh_{k+1}".format({k}), shape=(1,))')
    exec(f'sigma_{k+1} = jft.NormalPrior(sigmas[{k}], esigmas[{k}], name="sigma_{k+1}".format({k}), shape=(1,))')

roh_dm = jft.UniformPrior(0, 0.2, name="roh_dm", shape=(1,))

class ForwardModel(jft.Model):
    def __init__(self):
        for k in range(15):
            exec(f'self.roh_{k+1} = roh_{k+1}')
            exec(f'self.sigma_{k+1} = sigma_{k+1}')
        self.roh_dm = roh_dm

        super().__init__(
            init=   self.roh_1.init | self.sigma_1.init | \
                    self.roh_2.init | self.sigma_2.init | \
                    self.roh_3.init | self.sigma_3.init | \
                    self.roh_4.init | self.sigma_4.init | \
                    self.roh_5.init | self.sigma_5.init | \
                    self.roh_6.init | self.sigma_6.init | \
                    self.roh_7.init | self.sigma_7.init | \
                    self.roh_8.init | self.sigma_8.init | \
                    self.roh_9.init | self.sigma_9.init | \
                    self.roh_10.init | self.sigma_10.init | \
                    self.roh_11.init | self.sigma_11.init | \
                    self.roh_12.init | self.sigma_12.init | \
                    self.roh_13.init | self.sigma_13.init | \
                    self.roh_14.init | self.sigma_14.init | \
                    self.roh_15.init | self.sigma_15.init | \
                    self.roh_dm.init
        )

    def __call__(self, x):
        p1 = self.roh_1(x)
        p2 = self.sigma_1(x)
        p3 = self.roh_2(x)
        p4 = self.sigma_2(x)
        p5 = self.roh_3(x)
        p6 = self.sigma_3(x)
        p7 = self.roh_4(x)
        p8 = self.sigma_4(x)
        p9 = self.roh_5(x)
        p10 = self.sigma_5(x)
        p11 = self.roh_6(x)
        p12 = self.sigma_6(x)
        p13 = self.roh_7(x)
        p14 = self.sigma_7(x)
        p15 = self.roh_8(x)
        p16 = self.sigma_8(x)
        p17 = self.roh_9(x)
        p18 = self.sigma_9(x)
        p19 = self.roh_10(x)
        p20 = self.sigma_10(x)
        p21 = self.roh_11(x)
        p22 = self.sigma_11(x)
        p23 = self.roh_12(x)
        p24 = self.sigma_12(x)
        p25 = self.roh_13(x)
        p26 = self.sigma_13(x)
        p27 = self.roh_14(x)
        p28 = self.sigma_14(x)
        p29 = self.roh_15(x)
        p30 = self.sigma_15(x)
        p31 = self.roh_dm(x)

        def complicated_function(roh_1, sigma_1, roh_2, sigma_2, 
                                 roh_3, sigma_3, roh_4, sigma_4, 
                                 roh_5, sigma_5, roh_6, sigma_6, 
                                 roh_7, sigma_7, roh_8, sigma_8, 
                                 roh_9, sigma_9, roh_10, sigma_10, 
                                 roh_11, sigma_11, roh_12, sigma_12, 
                                 roh_13, sigma_13, roh_14, sigma_14, 
                                 roh_15, sigma_15, roh_dm):
            params = jnp.array([
                    [roh_1[0], sigma_1[0]], [roh_2[0], sigma_2[0]], 
                    [roh_3[0], sigma_3[0]], [roh_4[0], sigma_4[0]], 
                    [roh_5[0], sigma_5[0]], [roh_6[0], sigma_6[0]], 
                    [roh_7[0], sigma_7[0]], [roh_8[0], sigma_8[0]], 
                    [roh_9[0], sigma_9[0]], [roh_10[0], sigma_10[0]], 
                    [roh_11[0], sigma_11[0]], [roh_12[0], sigma_12[0]], 
                    [roh_13[0], sigma_13[0]], [roh_14[0], sigma_14[0]], 
                    [roh_15[0], sigma_15[0]]])
            roh_dm = roh_dm[0]

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
                stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

                sol = diffeqsolve(term, solver, t0=z0, t1=z0+n*dz, dt0=dz, y0=u0,
                                    args=(roh_dm, params),
                                saveat=saveat,
                                stepsize_controller=stepsize_controller)

                zs = sol.ts
                uz = sol.ys

                return uz
            
            @partial(jit, static_argnames=['f', 'n']) 
            def eigenerSolverV2(roh_dm, params, z0, u0, f, n, dz):
                                                                    
                # Runge-Kutta 4. Ordnung
                @partial(jit, static_argnames=['f']) #nötig ??
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

            uz = eigenerSolverV2(roh_dm, params, z0, u0, f, n, dz)

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

        return complicated_function(p1, p2, p3, p4, p5, 
                                    p6, p7, p8, p9, p10, 
                                    p11, p12, p13, p14, p15, 
                                    p16, p17, p18, p19, p20,
                                    p21, p22, p23, p24, p25,
                                    p26, p27, p28, p29, p30, p31)

# This initialises your forward-model which computes something data-like
fwd = ForwardModel()
seed = 42
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
for k in range(15):
        exec(f'roh{k+1} = jft.mean_and_std(tuple(roh_{k+1}(s) for s in samples))')
        exec(f'sigma{k+1} = jft.mean_and_std(tuple(sigma_{k+1}(s) for s in samples))')
rohdm = jft.mean_and_std(tuple(roh_dm(s) for s in samples))

# Please save your results in some way:
# TODO

for k in range(15):
    if eval(f'(roh{k+1}[0]-roh_{k+1}(pos_truth))/roh{k+1}[1] > 1.5'):
        exec(f'print("Inferred values:", "MEAN:", roh{k+1}[0], "STD:", roh{k+1}[1])')
        exec(f'print("with the true value being:", roh_{k+1}(pos_truth))')
        exec(f'print("Abweichung: ", (roh{k+1}[0]-roh_{k+1}(pos_truth))/roh{k+1}[1], "STDs")')
    if eval(f'(sigma{k+1}[0]-sigma_{k+1}(pos_truth))/sigma{k+1}[1] > 1.5'):
        exec(f'print("Inferred values:", "MEAN:", sigma{k+1}[0], "STD:", sigma{k+1}[1])')
        exec(f'print("with the true value being:", sigma_{k+1}(pos_truth))')
        exec(f'print("Abweichung: ", (sigma{k+1}[0]-sigma_{k+1}(pos_truth))/sigma{k+1}[1], "STDs")')

print("")

if (rohdm[0]-roh_dm(pos_truth))/rohdm[1] > 1.5:
    print("Inferred values:", "MEAN:", rohdm[0], "STD:", rohdm[1])
    print("with the true value being:", roh_dm(pos_truth))
    print("Abweichung: ", (rohdm[0]-roh_dm(pos_truth))/rohdm[1], "STDs")


print(jnp.mean(jnp.array([roh1[1]/roh1[0], roh2[1]/roh2[0], roh3[1]/roh3[0], roh4[1]/roh4[0], roh5[1]/roh5[0], roh6[1]/roh6[0], roh7[1]/roh7[0], roh8[1]/roh8[0], roh9[1]/roh9[0], roh10[1]/roh10[0], roh11[1]/roh11[0], roh12[1]/roh12[0], roh13[1]/roh13[0], roh14[1]/roh14[0], roh15[1]/roh15[0], rohdm[1]/rohdm[0]])))
print(jnp.mean(jnp.array([(roh1[0]-roh_1(pos_truth))/roh1[1], (roh2[0]-roh_2(pos_truth))/roh2[1], (roh3[0]-roh_3(pos_truth))/roh3[1], (roh4[0]-roh_4(pos_truth))/roh4[1], (roh5[0]-roh_5(pos_truth))/roh5[1], (roh6[0]-roh_6(pos_truth))/roh6[1], (roh7[0]-roh_7(pos_truth))/roh7[1], (roh8[0]-roh_8(pos_truth))/roh8[1], (roh9[0]-roh_9(pos_truth))/roh9[1], (roh10[0]-roh_10(pos_truth))/roh10[1], (roh11[0]-roh_11(pos_truth))/roh11[1], (roh12[0]-roh_12(pos_truth))/roh12[1], (roh13[0]-roh_13(pos_truth))/roh13[1], (roh14[0]-roh_14(pos_truth))/roh14[1], (roh15[0]-roh_15(pos_truth))/roh15[1], (rohdm[0]-roh_dm(pos_truth))/rohdm[1]])))

plt.show()