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

G = const.G / (3.0857E+16)**3 * 1.989E+30 * (3.0857E+13)**2
                            #Umrechnung in pc^3/M_sun/s^2 (gravitational Potential in (km/s)^2)
                            #Umrechnung, sodass z in parsec

f = lambda rho_dm, params, z, u: jnp.array([u[1], \
            4*jnp.pi*G * (jnp.sum(params[:,0]*jnp.exp(-u[0]/params[:,1]**2)) + rho_dm)]) #Poisson equation formulated as system of ODEs of order 1: u[0] = gravitational Potential, u[1] = z- derivative of u[0], params = [rho_visible, sigma^2_visible(z=0)]
z0 = 0. # sarting point z=0 (galactic midplane)
u0 = jnp.array([0.,0.]) # initial conditions on gravitational Potential (free choice of ground level for condition on u[0] and symmetry condition on u[1])

#numerical solution (using Dopri5/RK4) from z=0 to z1 with n steps (ijn MGVI final realized to be 1pc steps)
@partial(jit, static_argnames=['n']) 
def diffraxDopri5(rho_dm, params, z1, n): #using Dopri5 see diffrax documentation

    vector_field = lambda z, y, args: f(args[0], args[1], z, y) #wrapper für order
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
def eigenerSolverV2(rho_dm, params, z1, n): #self implemented primitive RK4 (only required for Bachelor´s thesis)
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

#calculation of tracer density drop off from z2 to z1 (still same data structure as gravitational potential): see paper
@partial(jit, static_argnames=['n', 'z1', 'z2'])
def vdfo_norm(z2, z1, zs, uz, n, poly, cf, z_v2):  #cf = sigma^2(z), z_v2 = z-values for which sigma^2 same data structure as gravitational potential
    dz = (z1-z0)/(n-1)
    
    sigma_sq = RegularGridInterpolator((zs,), cf)
    sig2 = sigma_sq(z_v2)

    i_s = int((z2-z0)/(z1-z0) * (n-1))
    
    z = zs[i_s:]
    sigma_sq_norm = (sigma_sq(z)/sigma_sq(jnp.array([z0+i_s*dz])))

    exp_int = 1.

    def exp_int_step(exp_int, i):
        carry = exp_int * jnp.exp(-sigma_sq(jnp.array([z0+i*dz]))**(-1) * (jnp.array(uz)[i,1]+jnp.array(uz)[i+1,1])/2 * dz)
        return *carry, *carry

    _, exp_int_list = lax.scan(exp_int_step, exp_int, jnp.arange(i_s, n-1, 1)) #last point not used (Riemann Sum left)

    exp_int_list = jnp.concatenate([jnp.array([exp_int]), exp_int_list], axis=0)
    vdfo_norm_calc = jnp.multiply(sigma_sq_norm**(-1), jnp.array(exp_int_list))

    return vdfo_norm_calc, z, sig2

# adjusting the data structure of density fall off to match data via numerical integration
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

# calculation of surface density from gravitational potential
@jit
def surface_density(params, uz, zs):

    def term(params, u):
        return jnp.sum(params[:, 0] * jnp.exp(-u[0] / params[:, 1]**2))

    def scan_fn(carry, u):
        result = term(params, u)
        return carry, result

    _, sd = lax.scan(scan_fn, None, uz)
    return 2*jnp.sum(sd[:-1]*(zs[1:]-zs[:-1]))

# solver to append solution of gravitational potential up to z3: does the same as solver above
@partial(jit, static_argnames=['n3'])
def Solver(rho_dm, params, z1, z3, u3, n3):

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