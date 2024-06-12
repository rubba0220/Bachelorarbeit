#jax bisher nur für CPU intslliert (pip install -U "jax[cpu]")
import jax.numpy as jnp
import jax.lax as lax
from jax import jit
from functools import partial
from matplotlib import pyplot as plt
from scipy import constants as const

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

#Formulierung des Anfangswertproblems
f = lambda roh_dm, params, z, u: jnp.array([u[1], \
            4*jnp.pi*G * (jnp.sum(params[:,0]*jnp.exp(-u[0]/params[:,1])) + roh_dm)])
z0 = 0.
u0 = jnp.array([0.,0.]) #freie Nullpunktswahl/Symmetrie
roh_dm = 0.025 #M_sun/pc^3
params = jnp.array([[0.021, 4.], [0.016, 7.], [0.012, 9.], \
                    [0.0009, 40.], [0.0006, 20.], [0.0031, 7.5], \
                    [0.0015, 10.5], [0.0020, 14.], [0.0022, 18.], \
                    [0.007, 18.5], [0.0135, 20.], [0.006, 20.], \
                    [0.002, 37.], [0.0035, 37.], [0.0001, 100.]]) 
                                                        #mu(0) in M_sun/pc^3, sigma(0)^2 in km^2/s^2

#numerische Lösung (Optimierung?)
uz = []
dz = 1
n = 1000
for i in range(n):
    u0 = rk4_step(roh_dm, params, z0+i*dz, u0, dz, f)
    uz.append(u0)

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
    return 17 + 85*z/1000 #z in pc, sigma in km/s

fig, ax = plt.subplots(1,2, figsize=(20,10))
ax[0].set_xlabel('z/pc')
ax[0].set_yscale('log')
ax[0].set_ylabel('$\\nu / \\nu_0 $')
ax[0].scatter([100+i*dz for i in range(n)], [vdfo_norm(100+i*dz) for i in range(n)], marker='o')
ax[0].grid()
ax[1].set_xlabel('z/pc')
ax[1].set_ylabel('$\sigma_z / (km/s)$')
ax[1].scatter([100+i*dz for i in range(n)], [sigma(100+i*dz) for i in range(n)], marker='o')
ax[1].grid()
fig.tight_layout()

#Berechnung des tracer density drop off
#Hier muss definitiv optimiert werden
from scipy.integrate import trapezoid as int

# vdfo_norm_calc = jnp.vstack([(sigma_sq[i]/sigma_sq[0])**(-1) * \
#                 jnp.exp(-int(1/sigma_sq[:i] * jnp.array(uz)[:i,1], dx=dz)) for i in range(n)])
# print(vdfo_norm_calc)