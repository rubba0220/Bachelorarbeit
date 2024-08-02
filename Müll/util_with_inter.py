import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import jit
from functools import partial
from scipy import constants as const
import diffrax as dif
from jax.scipy.integrate import trapezoid
from jax.scipy.interpolate import RegularGridInterpolator

jax.config.update("jax_enable_x64", True)

G = const.G / (3.0857E+16)**3 * 1.989E+30 * (3.0857E+13)**2
                            #Umrechnung in pc^3/M_sun/s^2 (grav pot in (km/s)^2)
                            #Umrechnung, sodass z in parsec

f = lambda rho_dm, params, z, u: jnp.array([u[1], \
            4*jnp.pi*G * (jnp.sum(params[:,0]*jnp.exp(-u[0]/params[:,1]**2)) + rho_dm)])
z0 = 0.
u0 = jnp.array([0.,0.]) #freie Nullpunktswahl/Symmetrie

#numerische Lösung (mittels Dopri5/rk4)
@partial(jit, static_argnames=['n']) 
def diffraxDopri5(rho_dm, params, z1, n):

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
                            stepsize_controller=stepsize_controller) 
                            #throw=False, max_steps=None

    zs = sol.ts
    uz = sol.ys

    return uz, zs

@partial(jit, static_argnames=['n'])
def eigenerSolverV2(rho_dm, params, z1, n):
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

z_vel = jnp.array([350, 450, 550, 650, 750, 850, 950, 1050, 1150])
s_vel = jnp.array([21., 27., 27., 27., 28., 30., 33., 36., 36.])
sig = RegularGridInterpolator((z_vel,), s_vel)

#Berechnung des tracer density drop off
@partial(jit, static_argnames=['n', 'z1', 'z2', 'mock', 'inter'])
def vdfo_norm(z2, z1, zs, uz, n, poly, mock=False, inter=False):
    dz = (z1-z0)/(n-1)

    #still mock velocity dispersion
    if mock == False:
        def sigma(z):
           return jnp.sqrt(poly[0]*z + poly[1])

    #mock velocity dispersion function
    if mock == True:
        if inter == True:
            def sigma(z):
                return sig(z)
        else:
            def sigma(z):
                return 17. + 20.*z/1200. #z in pc, sigma in km/s

    i_s = int((z2-z0)/(z1-z0) * (n-1))
    
    z = zs[i_s:]
    sigma_sq_norm = (sigma(z)/sigma(jnp.array([z0+i_s*dz])))**(2)

    exp_int = 1.

    def exp_int_step(exp_int, i):
        carry = exp_int * jnp.exp(-sigma(jnp.array([z0+i*dz]))**(-2) * (jnp.array(uz)[i,1]+jnp.array(uz)[i+1,1])/2 * dz)
        return *carry, *carry

    _, exp_int_list = lax.scan(exp_int_step, exp_int, jnp.arange(i_s, n-1, 1)) #letzter Punkt wird nicht genutzt (Riemannsumme links)

    exp_int_list = jnp.concatenate([jnp.array([exp_int]), exp_int_list], axis=0)
    vdfo_norm_calc = jnp.multiply(sigma_sq_norm**(-1), jnp.array(exp_int_list))

    return vdfo_norm_calc, z

@partial(jit, static_argnames=['n', 'n_bins', 'z1', 'z2'])
def binning(vdfo_norm_calc, z, z2, z1, n, n_bins):
    i_s = int((z2-z0)/(z1-z0) * (n-1))
    l = int((n-i_s-1)/n_bins)
    z_borders = z[0::l]

    integral = []
    for i in range(n_bins):
        integral += [trapezoid(vdfo_norm_calc[i*l:(i+1)*l], z[i*l:(i+1)*l])]
    integral = jnp.array(integral)

    return integral, z_borders

# @partial(jit, static_argnames=['n', 'n_bins', 'z1', 'z2', 'vdfo_norm_calc'])
# def binning(vdfo_norm_calc, z, z2, z1, n, n_bins):
#     i_s = int((z2-z0)/(z1-z0) * (n-1))
#     l = int((n-i_s-1)/n_bins)
#     z_borders = z[0::l]

#     def calculate_trapezoid_integral(i, carry):
#         start_idx = i * l
#         end_idx = (i + 1) * l
#         y_slice = lax.dynamic_slice(vdfo_norm_calc, [start_idx], [l])
#         x_slice = lax.dynamic_slice(z, [start_idx], [l])
#         # integral_value = trapezoid(vdfo_norm_calc[i*l:(i+1)*l], z[i*l:(i+1)*l])
#         integral_value = trapezoid(y_slice, x_slice)
#         return carry, integral_value

#     _, integral = lax.scan(calculate_trapezoid_integral, 1, jnp.arange(n_bins))

#     return jnp.array(integral), z_borders

# @jit
# def surface_density(params, uz):

#     def test(params, u):
#         return jnp.sum(params[:,0]*jnp.exp(-u[0]/params[:,1]**2))

#     sd = jnp.array([test(params, u) for u in uz])
#     return sd

@partial(jit, static_argnames=['n', 'z1'])
def surface_density(params, uz, z1, n):

    def term(params, u):
        return jnp.sum(params[:, 0] * jnp.exp(-u[0] / params[:, 1]**2))

    def scan_fn(carry, u):
        result = term(params, u)
        return carry, result

    _, sd = lax.scan(scan_fn, None, uz)
    return 2*jnp.sum(sd*(z1-z0)/(n-1))

@jit
def Solver(rho_dm, params, points):

    vector_field = lambda z, y, args: f(args[0], args[1], z, y) #wrapper für reihenfolge
    term = dif.ODETerm(vector_field)
    solver = dif.Dopri5()
    saveat = dif.SaveAt(ts=points)
    stepsize_controller = dif.PIDController(rtol=1e-3, atol=1e-6)
    adjoint = dif.DirectAdjoint()

    sol = dif.diffeqsolve(  term, solver, 
                            t0=z0, t1=points[-1], dt0=None, y0=u0, args=(rho_dm, params), 
                            saveat=saveat,
                            adjoint=adjoint,
                            stepsize_controller=stepsize_controller) 
                            #throw=False, max_steps=None

    zs = sol.ts
    uz = sol.ys

    return uz, zs