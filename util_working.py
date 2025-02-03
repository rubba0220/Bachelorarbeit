# Here, all the functions from the forward model are implemented.

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

G = const.G / (3.085677581E+16)**3 * 1.988416E+30 * (3.085677581E+13)**2 # in pc/M_sun*(km/s)^2

f = lambda rho_dm, params, z, u: jnp.array([u[1], \
            4*jnp.pi*G * (jnp.sum(params[:,0]*jnp.exp(-u[0]/params[:,1]**2)) + rho_dm)]) # Poisson equation formulated as system of ODEs of order 1: u[0] = gravitational potential at pos z, u[1] = z-derivative of u[0] at position z, params = [visible mass density, squared velocity dispersion in the galactic midplane] (in components)
z0 = 0. # starting point z=0 (galactic midplane)
u0 = jnp.array([0.,0.]) # initial conditions on gravitational potential [free choice of ground level for condition on u[0] and symmetry condition (reflection at galactic midplane) on u[1]]

f_tilt = lambda rho_dm, params, z, u, integral_tilt, domain: jnp.array([u[1], \
            4*jnp.pi*G * (jnp.sum(params[:,0]*jnp.exp(-u[0]/params[:,1]**2)) * jnp.exp(-RegularGridInterpolator(domain, integral_tilt)(jnp.array([[z,]]))[0]) + rho_dm)]) #integral is a correlated field modelling the tilt term in the Jeans equation, z is in 1 pc steps, so use z as index for integral

# numerical solution of f from z=0 to z1 with n steps (in analysis realized to be 1pc steps and z1 to be integer)
@partial(jit, static_argnames=['n']) 
def diffraxDopri5(rho_dm, params, z1, n, integral_tilt, domain): # routine using Dopri5 from diffrax

    vector_field = lambda z, y, args: f_tilt(args[0], args[1], z, y, args[2], args[3]) #wrapper function
    term = dif.ODETerm(vector_field)
    solver = dif.Dopri5()
    saveat = dif.SaveAt(ts=jnp.linspace(0, z1, n))
    stepsize_controller = dif.PIDController(rtol=1e-3, atol=1e-6)
    adjoint = dif.DirectAdjoint()

    sol = dif.diffeqsolve(  term, solver, 
                            t0=z0, t1=z1, dt0=None, y0=u0, args=(rho_dm, params, integral_tilt, domain), 
                            saveat=saveat,
                            adjoint=adjoint,
                            stepsize_controller=stepsize_controller) 
                            #throw=False, max_steps=None

    zs = sol.ts
    uz = sol.ys

    return uz, zs # returns positions of the approximated gravitational potential and [gravitational potential, derivative]-array

@partial(jit, static_argnames=['n'])
def eigenerSolverV2(rho_dm, params, z1, n): # RK4 routine (used to benchmark Dopri5)
    dz = (z1-z0)/(n-1)

    # Runge-Kutta 4. order
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

    return uz, zs # returns positions of the approximated gravitational potential and [gravitational potential, derivative]-array

# solver to append solution of gravitational potential up to z3 by n3 additional steps (chosen larger than 1pc steps)
@partial(jit, static_argnames=['n3'])
def Solver(rho_dm, params, z1, z3, u3, n3): # u3 is the initial condition on u at z1 (despite the misleading name)

    vector_field = lambda z, y, args: f(args[0], args[1], z, y) #wrapper für reihenfolge
    term = dif.ODETerm(vector_field)
    solver = dif.Dopri5()
    saveat = dif.SaveAt(ts=jnp.linspace(z1, z3, n3)[1:])
    stepsize_controller = dif.PIDController(rtol=1e-3, atol=1e-6)
    adjoint = dif.DirectAdjoint()

    sol = dif.diffeqsolve(  term, solver, 
                            t0=z1, t1=z3, dt0=None, y0=u3, args=(rho_dm, params), 
                            saveat=saveat,
                            adjoint=adjoint,
                            stepsize_controller=stepsize_controller) 
                            #throw=False, max_steps=None

    zs_ = sol.ts
    uz_ = sol.ys

    return uz_, zs_

# calculation of integral from correlated fields
def integrate(cf_vrz_R, cf_sig2, domain):
    y = cf_vrz_R/cf_sig2
    x = domain
    integral_tilt = jnp.array([0])
    for i in range(len(list(y))-1):
        integral_tilt = jnp.append(integral_tilt, trapezoid(y[:2+i], x[:2+i]))

    return integral_tilt, domain

# calculation of tracer density drop off from z2 to z1
@partial(jit, static_argnames=['n', 'z1', 'z2'])
def vdfo_norm(z2, z1, zs, uz, n, poly, cf, z_v2): # cf=squared velocity dispersion at positions zs, z_v2 = z-values of squared velocity dispersion of the actual data

    dz = (z1-z0)/(n-1)
    
    sigma_sq = RegularGridInterpolator((zs,), cf) # interpolation of squared velocity dispersion
    sig2 = sigma_sq(z_v2) #evaluate squared velocity dispersion at positions z_v2 (of the actual data)

    # calculation of the normalized density drop off
    i_s = int((z2-z0)/(z1-z0) * (n-1))
    
    z = zs[i_s:]
    sigma_sq_norm = (sigma_sq(z)/sigma_sq(jnp.array([z0+i_s*dz])))

    exp_int = 1.

    def exp_int_step(exp_int, i):
        carry = exp_int * jnp.exp(-sigma_sq(jnp.array([z0+i*dz]))**(-1) * (jnp.array(uz)[i,1]+jnp.array(uz)[i+1,1])/2 * dz)
        return *carry, *carry

    _, exp_int_list = lax.scan(exp_int_step, exp_int, jnp.arange(i_s, n-1, 1)) # last point not used (Riemann sum left-sided)

    exp_int_list = jnp.concatenate([jnp.array([exp_int]), exp_int_list], axis=0)
    vdfo_norm_calc = jnp.multiply(sigma_sq_norm**(-1), jnp.array(exp_int_list))

    return vdfo_norm_calc, z, sig2 # returns normalized density fall off at positions z, z [so the zs that are between z2 and z1], squared velocity dispersion at positions z_v2 (actual data)

# adjusting the data structure of density fall off to match actual data via numerical integration (to get equivalent to histogrammized data)
@partial(jit, static_argnames=['n', 'n_bins', 'z1', 'z2'])
def binning(vdfo_norm_calc, z, z2, z1, n, n_bins): # n_bins = number of bins for the density fall off, 
    i_s = int((z2-z0)/(z1-z0) * (n-1))
    l = int((n-i_s-1)/n_bins)
    z_borders = z[0::l]

    integral = []
    for i in range(n_bins):
        integral += [trapezoid(vdfo_norm_calc[i*l:(i+1)*l], z[i*l:(i+1)*l])]
    integral = jnp.array(integral)

    return integral, z_borders # returns the integral of the density fall off over the bins and the bin edges

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


# calculation of the surface density from the numerical approximation of the gravitational potential (up to z1) via numerical integration
@jit
def surface_density(params, uz, zs): 

    def term(params, u):
        return jnp.sum(params[:, 0] * jnp.exp(-u[0] / params[:, 1]**2))

    def scan_fn(carry, u):
        result = term(params, u)
        return carry, result

    _, sd = lax.scan(scan_fn, None, uz)
    return 2*jnp.sum(sd[:-1]*(zs[1:]-zs[:-1]))
