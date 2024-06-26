#jax bisher nur für CPU intslliert (pip install -U "jax[cpu]")
import jax.numpy as jnp
import jax.lax as lax
from jax import jit, vmap
from functools import partial
from matplotlib import pyplot as plt
from scipy import constants as const
import time

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
            4*jnp.pi*G * (jnp.sum(params[:,0]*jnp.exp(-u[0]/params[:,1]**2)) + roh_dm)])
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
dz = 0.1
n = 12000

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

#Visualisierung
fig, ax1 = plt.subplots(figsize=(20,10))
plt.title('Test')
ax1.set_xlabel('z/pc')
ax1.set_ylabel('$\Phi / (km/s)^2$')
ax1.scatter([z0+i*dz for i in range(n)], [u[0] for u in uz], marker='o')
ax1.grid()
fig.tight_layout()

#mock data: sekante in Figur 5
def vdfo_norm(z):
    return 10**(0.57 + -1.63*z/1000) #z in pc

def sigma(z):
    return 20 + 17*z/1000 #z in pc, sigma in km/s

fig, ax = plt.subplots(1,2, figsize=(20,10))
ax[0].set_xlabel('z/pc')
#ax[0].set_yscale('log')
ax[0].set_ylabel('$\\nu / \\nu(z_0) $')
ax[0].scatter([100+i*10 for i in range(100)], [vdfo_norm(100+i*10) for i in range(100)], marker='o')
ax[0].grid()
ax[1].set_xlabel('z/pc')
ax[1].set_ylabel('$\sigma_z / (km/s)$')
ax[1].scatter([100+i*10 for i in range(100)], [sigma(100+i*10) for i in range(100)], marker='o')
ax[1].grid()
fig.tight_layout()

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

#Visualisierung
fig, ax = plt.subplots(figsize=(20,10))
ax.set_xlabel('z/pc')
#ax.set_yscale('log')
ax.set_ylabel('$\\nu / \\nu_0 $')
ax.scatter([z0+i*dz for i in range(i1,i2)], [vdfo_norm_calc], marker='o')
ax.grid()
fig.tight_layout()
