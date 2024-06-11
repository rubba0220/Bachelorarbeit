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

# Runge-Kutta 4. Ordnung
@partial(jit, static_argnames=['f'])
def rk4_step(params, z0, u0, dz, f):
    k1 = dz * f(params, z0, u0)
    k2 = dz * f(params, z0 + dz / 2, u0 + k1 / 2)
    k3 = dz * f(params, z0 + dz / 2, u0 + k2 / 2)
    k4 = dz * f(params, z0 + dz, u0 + k3)
    u1 = u0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return u1

#Formulierung des Anfangswertproblems
f = lambda params, z, u: jnp.array([u[1], 4*jnp.pi*params[0] * \
    (params[1]*jnp.exp(-u[0]/params[2]) + params[3])])
z0 = 0.
u0 = jnp.array([0.,0.]) #Warum auf 0 ???
params = jnp.array([1., 10., 6., 1.])

#numerische Lösung (Optimierung?)
uz = []
dz = 0.01
n = 1000
for i in range(n):
    u0 = rk4_step(params, z0+i*dz, u0, dz, f)
    uz.append(u0)

#Visualisierung
fig, ax1 = plt.subplots(figsize=(20,10))
plt.title('Test')
ax1.set_xlabel('z')
ax1.set_ylabel('$\Phi$')
ax1.scatter([z0+i*dz for i in range(n)], [u[0] for u in uz], marker='o', label='u')
ax1.grid()
fig.tight_layout()
plt.legend()


