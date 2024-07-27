import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random
from jax import jit
from jax.scipy.interpolate import RegularGridInterpolator
import pandas as pd
import numpy as np
import nifty8.re as jft
import util_working

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)

''' Data '''
df = pd.read_csv("real data/v2_57.txt")
z = jnp.array(df["z"].values)
sorted_indices = jnp.argsort(z)
z = z[sorted_indices]
print('Größtes z in Geschwindigkeitsdaten: ', max(z))
v2 = jnp.array(df["v2"].values)
v2 = v2[sorted_indices]
poly = np.loadtxt(f'real data/poly_57.txt')

z1 = 1800
n = int(z1)+1

seeds = [42, 12 , 34, 56, 78, 93, 102, 400, 234]
for s in seeds:
    seed = s
    key = random.PRNGKey(seed)

    ''' Model '''
    dims = (n, )

    cf_zm = dict(offset_mean=0., offset_std=(1., 1.))
    cf_fl = dict(   fluctuations=(1., 1.), #7 direkt conjugate gradient failed
                    loglogavgslope=(-3., 3.),
                    flexibility=(1e-3, 1e-16),
                    asperity=(1e-3, 1e-16),)
    # cf_zm = dict(offset_mean=6.5, offset_std=(2.5, 2.5))
    # cf_fl = dict(   fluctuations=(1.5, 1.5), #7 direkt conjugate gradient failed
    #                 loglogavgslope=(-3., 3.),
    #                 flexibility=(1e-3, 1e-16),
    #                 asperity=(1e-3, 1e-16),)
    # cf_zm = dict(offset_mean=900, offset_std=(900, 900))
    # cf_fl = dict(   fluctuations=(1000, 1000),
    #                 loglogavgslope=(-20., 5.),
    #                 flexibility=(1e-3, 1e-16),
    #                 asperity=(1e-3, 1e-16),)

    cfm = jft.CorrelatedFieldMaker("cf")
    cfm.set_amplitude_total_offset(**cf_zm)
    cfm.add_fluctuations(dims, distances=1.0, **cf_fl, prefix="ax1", non_parametric_kind="power")
    correlated_field = cfm.finalize()

    class Signal(jft.Model):
        def __init__(self, correlated_field):
            self.correlated_field = correlated_field
            
            super().__init__(init=self.correlated_field.init)
        
        def __call__(self, x):
            cf = ( 20. + 10./1200 * jnp.linspace(0, z1, n) )**2 * jnp.exp(self.correlated_field(x))
            # cf = jnp.exp(self.correlated_field(x))
            # cf = self.correlated_field(x)
            grid = jnp.linspace(0, z1, n)
            sig = RegularGridInterpolator((grid,), cf)
            return sig(z)

    signal = Signal(correlated_field)
    
    signal_response = signal
    noise_cov = lambda x: 1100**2 * x
    noise_cov_inv = lambda x:1100**(-2) * x

    # Create synthetic data
    # key, subkey = random.split(key)
    # pos_truth = jft.random_like(subkey, signal_response.domain)
    # signal_response_truth = signal_response(pos_truth)
    # key, subkey = random.split(key)
    # noise_truth = ((noise_cov(jft.ones_like(signal_response.target))) ** 0.5) * jft.random_like(key, signal_response.target)
    # data = signal_response_truth + noise_truth

    lh = jft.Gaussian(v2, noise_cov_inv).amend(signal_response)

    ''' Inference '''
    n_vi_iterations = 6
    delta = 1e-4
    n_samples = 4

    key, k_i, k_o = random.split(key, 3)
    samples, state = jft.optimize_kl(
        lh,
        jft.Vector(lh.init(k_i)),
        n_total_iterations=n_vi_iterations,
        n_samples=lambda i: n_samples // 2 if i < 2 else n_samples,
        key=k_o,
        # Names of parameters that should not be sampled but still optimized
        # can be specified as point_estimates (effectively we are doing MAP for
        # these degrees of freedom).
        # point_estimates=("cfax1flexibility", "cfax1asperity"),
        draw_linear_kwargs=dict(
            cg_name="SL",
            cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=100),
        ),
        nonlinearly_update_kwargs=dict(
            minimize_kwargs=dict(
                name="SN",
                xtol=delta,
                cg_kwargs=dict(name=None),
                maxiter=5,
            )
        ),
        kl_kwargs=dict(
            minimize_kwargs=dict(
                name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=35
            )
        ),
        sample_mode="nonlinear_resample",
        odir="results_intro",
        resume=False,
    )

    # key, subkey = random.split(key)
    # samples = [jft.random_like(subkey, signal_response.domain)]

    ''' Auswertung '''
    namps = cfm.get_normalized_amplitudes()
    post_sr_mean = jft.mean(tuple(signal(s) for s in samples))
    corrfield = jft.mean(tuple(correlated_field(s) for s in samples))
    sigma_sq = jft.mean(tuple(( 20.+10./1200. * jnp.linspace(0, z1, n) )**2 * jnp.exp(correlated_field(s)) for s in samples))
    # sigma_sq = jft.mean(tuple(jnp.exp(correlated_field(s)) for s in samples))
    # sigma_sq = jft.mean(tuple(correlated_field(s) for s in samples))
    post_a_mean = jft.mean(tuple(cfm.amplitude(s)[1:] for s in samples))
    grid_ = correlated_field.target_grids[0]

    to_plot = [ ("Data", data, 'scatter'), 
                ("Reconstruction", post_sr_mean, 'plot'), 
                ("Correlated Field", corrfield, 'plot2'),
                ('Sigma_sq', sigma_sq, 'plot2'),
                ("Amplitude spectrum", (grid_.harmonic_grid.mode_lengths[1:],
                                        post_a_mean), "loglog")]

    fig, axs = plt.subplots(5, 1, figsize=(20, 20))
    for ax, v in zip(axs.flat, to_plot):
        title, field, tp = v
        ax.set_title(title)
        ax.grid()
        if tp == 'scatter':
            ax.scatter(z, field, marker='.')
            ax.plot(z, poly[0]*z+poly[1])
            ax.sharex(axs[0])
        elif tp == 'plot':
            ax.plot(z, field)
            ax.plot(z, poly[0]*z+poly[1])
            ax.sharex(axs[0])
        elif tp == 'plot2':
            ax.plot(jnp.linspace(0, z1, n), field)
            ax.sharex(axs[0])
        elif tp == 'loglog':
            x = field[0]
            ax.loglog(x, field[1], alpha=0.7)
    fig.tight_layout()
    plt.show()
