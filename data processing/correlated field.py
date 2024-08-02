import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random
from jax import jit
from jax.scipy.interpolate import RegularGridInterpolator
import pandas as pd
import numpy as np
import nifty8.re as jft

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)

''' Domain '''
am_min = 6.0
# am_max = 6.724
# am_min = 6.724
am_max = 7.4
z1 = 1800
n = int(z1)+1
interval ='both'

''' Data '''
df = pd.read_csv(f"../real data new intervals/v2_{am_min*1000:.0f}{am_max*1000:.0f}.txt")
if interval == 'both':
    z = abs(jnp.array(df["z"].values))
    where = jnp.where(z)
elif interval == 'pos':
    z = jnp.array(df["z"].values)
    where = jnp.where(z>=0)
    z = z[where]
elif interval == 'neg':
    z = jnp.array(df["z"].values)
    where = jnp.where(z<0)
    z = abs(z[where])
    print('Größtes z in Geschwindigkeitsdaten: ', max(z))

sorted_indices = jnp.argsort(z)
z = z[sorted_indices]
print('Größtes z in Geschwindigkeitsdaten: ', max(z))
v2 = jnp.array(df["v2"].values)[where]
v2 = v2[sorted_indices]

poly = np.loadtxt(f'../real data new intervals/poly_{am_min*1000:.0f}{am_max*1000:.0f}.txt')

''' Run '''
# run = 'rough_func'
# rough_func = poly[0]*jnp.linspace(0, z1, n)+poly[1]
# label = 'fit'
# rough_func = (17. + 10./1200. * jnp.linspace(0, z1, n) )**2
# label = 'read'
# rough_func = jnp.sqrt(300.**2 + 400.**2/1200. * jnp.linspace(0, z1, n))
# run = 'exp(cf)'
run = 'cf'
label=''

inference = True
seed = 42
key = random.PRNGKey(seed)
seeds = random.randint(key, (5,), 0, 1000)


fig, axs = plt.subplots(4, 1, figsize=(20, 20))
colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'yellow']
i=0
axs.flat[0].set_title('Data')
axs.flat[0].grid()
axs.flat[0].scatter(z, v2, marker='.', color='black')
axs.flat[0].plot(z, poly[0]*z+poly[1], color='red')
axs.flat[2].plot(np.linspace(0,z1,n), poly[0]*np.linspace(0,z1,n)+poly[1], color='red')
if run == 'rough_func' and label == 'read':
    axs.flat[2].plot(np.linspace(0,z1,n), rough_func, color='red')
axs.flat[0].sharex(axs[0])

axs.flat[1].set_title('Correlated Field')
axs.flat[2].set_title('Sigma_sq')
axs.flat[3].set_title('Amplitude spectrum')
axs.flat[1].grid()
axs.flat[2].grid()
axs.flat[3].grid()

for s in seeds:
    seed = s
    key = random.PRNGKey(seed)

    ''' Model '''
    dims = (n, )

    if run == 'rough_func':
        cf_zm = dict(offset_mean=0., offset_std=(0.3, 0.3)) #0.5 0.5
        cf_fl = dict(   fluctuations=(0.5, 0.3), #1. 1.
                        loglogavgslope=(-3., 0.5),
                        flexibility=(1e-3, 1e-16),
                        asperity=(1e-3, 1e-16),)
    elif run == 'exp(cf)':
        cf_zm = dict(offset_mean=6.5, offset_std=(2.5, 2.5))
        cf_fl = dict(   fluctuations=(1.5, 1.5), #7 direkt conjugate gradient failed
                        loglogavgslope=(-3., 3.),
                        flexibility=(1e-3, 1e-16),
                        asperity=(1e-3, 1e-16),)
    elif run == 'cf':
        cf_zm = dict(offset_mean=800, offset_std=(500, 500))
        cf_fl = dict(   fluctuations=(300, 300),
                        loglogavgslope=(-3., 3.),
                        flexibility=(1e-3, 1e-16),
                        asperity=(1e-3, 1e-16),)

    cfm = jft.CorrelatedFieldMaker("cf")
    cfm.set_amplitude_total_offset(**cf_zm)
    cfm.add_fluctuations(dims, distances=1.0, **cf_fl, prefix="ax1", non_parametric_kind="power")
    correlated_field = cfm.finalize()

    class Signal(jft.Model):
        def __init__(self, correlated_field):
            self.correlated_field = correlated_field
            
            super().__init__(init=self.correlated_field.init)
        
        def __call__(self, x):
            if run == 'rough_func':
                cf = rough_func * jnp.exp(self.correlated_field(x))
            elif run == 'exp(cf)':
                cf = jnp.exp(self.correlated_field(x))
            elif run == 'cf':
                cf = self.correlated_field(x)
            grid = jnp.linspace(0, z1, n)
            sig = RegularGridInterpolator((grid,), cf)
            return sig(z)

    signal = Signal(correlated_field)
    
    signal_response = signal
    noise_cov = lambda x: poly[4]**2 * x
    noise_cov_inv = lambda x:poly[4]**(-2) * x

    # Create synthetic data
    # key, subkey = random.split(key)
    # pos_truth = jft.random_like(subkey, signal_response.domain)
    # signal_response_truth = signal_response(pos_truth)
    # key, subkey = random.split(key)
    # noise_truth = ((noise_cov(jft.ones_like(signal_response.target))) ** 0.5) * jft.random_like(key, signal_response.target)
    # data = signal_response_truth + noise_truth

    lh = jft.Gaussian(v2, noise_cov_inv).amend(signal_response)

    if inference:
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
            odir=None,#"results_intro",
            resume=False,
        )
    else:
        ''' Sampling '''
        key, subkey = random.split(key)
        samples = [jft.random_like(subkey, signal_response.domain)]

    ''' Auswertung '''
    post_sr_mean = jft.mean_and_std(tuple(signal(s) for s in samples))
    corrfield = jft.mean_and_std(tuple(correlated_field(s) for s in samples))
    if run == 'rough_func':
        sigma_sq = jft.mean_and_std(tuple(rough_func * jnp.exp(correlated_field(s)) for s in samples))
    elif run == 'exp(cf)':
        sigma_sq = jft.mean_and_std(tuple(jnp.exp(correlated_field(s)) for s in samples))
    elif run == 'cf':
        sigma_sq = jft.mean_and_std(tuple(correlated_field(s) for s in samples))
    post_a_mean = jft.mean(tuple(cfm.amplitude(s)[1:] for s in samples))
    grid_ = correlated_field.target_grids[0]

    to_plot = [ ("Correlated Field", corrfield, 'plot'),
                ('Sigma_sq', sigma_sq, 'plot'),
                ("Amplitude spectrum", (grid_.harmonic_grid.mode_lengths[1:],
                                        post_a_mean), "loglog")]

    ''' Plot '''
    for ax, v in zip(axs.flat[1:], to_plot):
        title, field, tp = v
        if tp == 'plot':
            ax.plot(jnp.linspace(0, z1, n), field[0], color=colors[i])
            ax.fill_between(jnp.linspace(0, z1, n), field[0]+field[1], field[0]-field[1], alpha=0.1, color=colors[i])
            ax.sharex(axs[0])
        elif tp == 'loglog':
            x = field[0]
            ax.loglog(x, field[1], alpha=0.7, color=colors[i])
    
    i+=1
    


fig.tight_layout()
fig.subplots_adjust(hspace=0.0)
if interval == 'both':
    fig.savefig(f'../Plots/cf/corrfield_{run}{label}.png')
else:
    fig.savefig(f'../Plots/cf/corrfield_{run}{label}_{interval}.png')