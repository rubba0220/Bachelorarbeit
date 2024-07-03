import pandas as pd
import matplotlib.pyplot as plt
from jax import numpy as jnp
import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.special import gamma

# Plot-Formatierung
plt.rcParams['font.size'] = 24.0
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.0

def plot_start_cond(y, ey, mean, name, ylim):
    fig, ax = plt.subplots(figsize=(20,10))
    plt.title(name)
    ax.set_ylim(*ylim)
    ax.set_xlabel('run')
    ax.set_ylabel('Inferred Value')
    ax.errorbar(jnp.arange(1, len(y)+1,1), y, yerr=ey, fmt='o', markeredgecolor='red', color='red', label='Inferred Value')
    ax.axhline(y=mean, color='black', linestyle='--', label='True Value')
    ax.grid()
    fig.tight_layout()

def Auswertung(file_roh, file_sigma, name):
    data_roh = pd.read_csv(file_roh, header=None)
    data_sigma = pd.read_csv(file_sigma, header=None)

    truthr = jnp.array(data_roh.iloc[:,0])
    inferredr = jnp.array(data_roh.iloc[:,1])
    stdr = jnp.array(data_roh.iloc[:,2])
        
    truths = jnp.array(data_sigma.iloc[:,0])
    inferreds = jnp.array(data_sigma.iloc[:,1])
    stds = jnp.array(data_sigma.iloc[:,2])

    meanrs = jnp.mean(inferredr[0::2])
    meanrd = jnp.mean(inferredr[1::2])
    means = jnp.mean(inferreds)

    scatrs = jnp.std(inferredr[0::2])
    scatrd = jnp.std(inferredr[1::2])
    scats = jnp.std(inferreds)

    stdrs = jnp.mean(stdr[0::2])
    stdrd = jnp.mean(stdr[1::2])
    stds = jnp.mean(stds)

    print('Mean Rho_s: ', meanrs, 'Scatter: ', scatrs, 'stddev: ', stdrs, 'true: ', truthr[0])
    print('Mean Rho_d: ', meanrd, 'Scatter: ', scatrd, 'stddev: ', stdrd, 'true: ', truthr[1])
    print('Mean Sigma: ', means, 'Scatter: ', scats, 'stddev: ', stds, 'true: ', truths[0])
    print('')

    plot_start_cond(inferredr[0::2], stdr[0::2], truthr[0], 'Rho_s ' + name, [0., 0.5])
    plot_start_cond(inferredr[1::2], stdr[1::2], truthr[1], 'Rho_d ' + name, [0., 0.2])
    plot_start_cond(inferreds, stds, truths[0], 'sigma ' + name, [3., 17.])

    return meanrs, scatrs, stdrs, truthr[0], meanrd, scatrd, stdrd, truthr[1], means, scats, stds, truths[0]

# #testdurchläufe
# Auswertung('data_roh_start_cond_102.csv', 'data_sigma_start_cond_102.csv', 'seed=102') 
# Auswertung('data_roh_start_cond_99.csv', 'data_sigma_start_cond_99.csv', 'seed=99')
# Auswertung('data_roh_start_cond_58.csv', 'data_sigma_start_cond_58.csv', 'seed=58')

seeds = [56, 81, 137, 338, 474, 571, 656, 916, 928]

values = pd.DataFrame(columns=['meanrs', 'scatrs', 'stdrs', 'truthr', 'meanrd', 'scatrd', 'stdrd', 'truthr', 'means', 'scats', 'stds', 'truths'])
ind = 0
for i in seeds:
    meanrs, scatrs, stdrs, truthr, meanrd, scatrd, stdrd, truthr, means, scats, stds, truths = Auswertung('data_roh_start_cond_' + str(i) + '.csv', 'data_sigma_start_cond_' + str(i) + '.csv', 'seed=' + str(i))

    values.loc[ind] = [meanrs, scatrs, stdrs, truthr, meanrd, scatrd, stdrd, truthr, means, scats, stds, truths]
    ind += 1
    

values.to_csv('values_start_cond.csv', index=False)


