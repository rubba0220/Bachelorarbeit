import jax
import jax.numpy as jnp
import numpy as np
import jax.lax as lax
from jax import jit, random
from functools import partial
from matplotlib import pyplot as plt
from scipy import constants as const
import nifty8.re as jft
import diffrax as dif
import pandas as pd
from jax.scipy.integrate import trapezoid

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

#Berechnung des tracer density drop off
@partial(jit, static_argnames=['n', 'i_s', 'i_n'])
def vdfo_norm(z1, i_s, i_n, uz, n, poly):
    dz = (z1-z0)/(n-1)

    #still mock velocity dispersion
    def sigma(z):
        return jnp.sqrt(poly[0]*z + poly[1])
    
    z = jnp.linspace(z0+i_s*dz, z1, n-i_s)
    sigma_sq_norm = (sigma(z)/sigma(z0+i_n*dz))**(2)
    z_ns = jnp.linspace(z0+i_n*dz, z0+(i_s-1)*dz, i_s-i_n)
    sigmass = sigma(z_ns)

    exp_int = jnp.exp(-jnp.sum(\
                sigmass**(-2) \
                * jnp.array(uz)[i_n:i_s,1] * dz))

    def exp_int_step(exp_int, i):
        return exp_int * jnp.exp(-sigma(z0+i*dz)**(-2) * jnp.array(uz)[i,1] * dz), \
                exp_int * jnp.exp(-sigma(z0+i*dz)**(-2) * jnp.array(uz)[i,1] * dz)

    _, exp_int_list = lax.scan(exp_int_step, exp_int, jnp.arange(i_s, n-1, 1)) #letzter Punkt wird nicht genutzt (Riemannsumme links)

    exp_int_list = jnp.concatenate([jnp.array([exp_int]), exp_int_list], axis=0)

    vdfo_norm_calc = jnp.multiply(sigma_sq_norm**(-1), jnp.array(exp_int_list))

    return vdfo_norm_calc, z

@partial(jit, static_argnames=['n', 'i_s', 'n_bins'])
def binning(vdfo_norm_calc, z, n, i_s, n_bins):
    l = int((n-i_s-1)/n_bins)
    z_borders = z[0::l]

    integral = []
    for i in range(n_bins):
        integral += [trapezoid(vdfo_norm_calc[i*l:(i+1)*l], z[i*l:(i+1)*l])]
    integral = jnp.array(integral)

    # def bin(vdfo_norm_calc, i):
    #     integral = simpson(vdfo_norm_calc[i*l:(i+1)*l], z[i*l:(i+1)*l])
    #     vdfo_norm_calc[i*l:(i+1)*l] = 0
    #     return vdfo_norm_calc, integral
    
    # _, integral = lax.scan(bin, vdfo_norm_calc, jnp.arange(10))

    return integral, z_borders