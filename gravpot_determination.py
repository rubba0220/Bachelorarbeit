#jax bisher nur für CPU intslliert (pip install -U "jax[cpu]")
import jax.numpy as jnp
import jax.lax as lax
from jax import jit, vmap
from functools import partial
from matplotlib import pyplot as plt
from scipy import constants as const
import timeit

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

# lax.scan(step)
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

# neu mit diffrax
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController, DirectAdjoint

# Dopri5
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
                    stepsize_controller=stepsize_controller,
                    adjoint=DirectAdjoint())

    zs = sol.ts
    uz = sol.ys

    return uz

n = 1200
dz = 1

rd = 0.02
rds = jnp.linspace(0, 0.2, 10)
rs = 0.01
rss = jnp.linspace(0, 0.1, 10)
s = 12.
ss = jnp.linspace(2, 40, 10)

fig, ax = plt.subplots(figsize=(20,10))
plt.title('Rho_dm in einfachem Massenmodell')
ax.set_xlabel('z / kpc')
ax.set_ylabel('$\Phi$ / (km/s)$^2$')
for rdi in rds:
    p = jnp.array([[rs, s]])
    uz = diffraxDopri5(rdi, p, z0, u0, f, n, dz)
    plt.plot(jnp.linspace(0, n*dz, n), uz[:,0], label = f'rd = {rdi:.3f} M_sun/pc^3')
ax.grid()
ax.legend()
fig.tight_layout()

fig, ax = plt.subplots(figsize=(20,10))
plt.title('Rho_s in einfachem Massenmodell')
ax.set_xlabel('z / kpc')
ax.set_ylabel('$\Phi$ / (km/s)$^2$')
for rsi in rss:
    p = jnp.array([[rsi, s]])
    uz = diffraxDopri5(rd, p, z0, u0, f, n, dz)
    plt.plot(jnp.linspace(0, n*dz, n), uz[:,0], label = f'rd = {rsi:.3f} M_sun/pc^3')
ax.grid()
ax.legend()
fig.tight_layout()

fig, ax = plt.subplots(figsize=(20,10))
plt.title('Sigma in einfachem Massenmodell')
ax.set_xlabel('z / kpc')
ax.set_ylabel('$\Phi$ / (km/s)$^2$')
for si in ss:
    p = jnp.array([[rs, si]])
    uz = diffraxDopri5(rd, p, z0, u0, f, n, dz)
    plt.plot(jnp.linspace(0, n*dz, n), uz[:,0], label = f's = {si:.3f} km/s')
ax.grid()
ax.legend()
fig.tight_layout()

fig, ax = plt.subplots(figsize=(20,10))
plt.title('Sigma/Rho_s in einfachem Massenmodell')
ax.set_xlabel('z / kpc')
ax.set_ylabel('$\Phi$ / (km/s)$^2$')
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
i = 0
for rsi in rss[3:6]:
    c = colors[i]
    for si in ss[6-i*2:10-i*2]:
        p = jnp.array([[rsi, si]])
        uz = diffraxDopri5(rd, p, z0, u0, f, n, dz)
        plt.plot(jnp.linspace(0, n*dz, n), uz[:,0], color = c)
    plt.figtext(0.1, 0.9-i*0.05, f'rs = {rsi:.3f} M_sun/pc^3', color = c)
    i += 1
ax.grid()
ax.legend()
fig.tight_layout()

