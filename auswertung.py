import pandas as pd
import matplotlib.pyplot as plt
from jax import numpy as jnp
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



def linreg(x,y,ey,ini=[1,1]):

    def func(x, m, b):
        return m*x + b
    
    popt, pcov = curve_fit(func, x, y, p0=ini, sigma=ey, absolute_sigma=True)
    perr = jnp.sqrt(jnp.diag(pcov))

    xi = y
    Xi = func(x, *popt)
    chiq = jnp.sum((xi-Xi)**2 / ey**2)
    ndof = len(xi)-2

    return popt[0], perr[0], popt[1], perr[1], chiq, ndof       



def residuenplot(var_x, unit_x, v_x, v_s_x, var_y, unit_y, v_y, v_s_y, m, b, chiq, name='', save=False, Zoom = None):
    
    fig, axarray = plt.subplots(2, 1, figsize=(20,10), sharex=True, gridspec_kw={'height_ratios': [5, 2]})
    plt.title(name)
    if Zoom != None:
        axarray[0].set_xlim(*Zoom)
        axarray[0].set_ylim(*Zoom)
    axarray[0].errorbar(v_x, v_y, xerr=v_s_x, yerr=v_s_y, color='red', fmt='o', markeredgecolor='red', label='Messwerte')
    axarray[0].set_xlabel('${0}$ / ${1}$'.format(var_x, unit_x))
    axarray[0].set_ylabel('${0}$ / ${1}$'.format(var_y, unit_y))
    axarray[0].plot(v_x, m*v_x+b, color='green', label='${0:.2f}*x+{1:.2f}$'.format(m, b))
    sigmaRes = jnp.sqrt((m*v_s_x)**2 + v_s_y**2)
    axarray[0].plot([jnp.min(v_x), jnp.max(v_x)], [jnp.min(v_x), jnp.max(v_x)], color='black', linestyle='--', label='Erwartung')
    axarray[1].axhline(y=0., color='black', linestyle='--')
    axarray[1].errorbar(v_x, v_y-(m*v_x+b), yerr=sigmaRes, color='red', fmt='o', markeredgecolor='red')
    axarray[1].set_xlabel('${0}$ / ${1}$'.format(var_x, unit_x))
    axarray[1].set_ylabel('Residuen')
    plt.figtext(0.14,0.7,'chi2/ndf = %.2f / %.0f = %.2f' % (chiq, len(v_x)-2, chiq/(len(v_x)-2)))
    axarray[0].legend()
    axarray[0].grid()
    ymax = max([abs(x) for x in axarray[1].get_ylim()])
    axarray[1].set_ylim(-ymax, ymax)
    plt.tight_layout()
    plt.grid()
    fig.subplots_adjust(hspace=0.0)
    
    if save == True:
        plt.savefig(name)
    
    plt.show() 



def Auswertung(file_roh, file_sigma, name='', Zoom = None):
    data_roh = pd.read_csv(file_roh, header=None)
    data_sigma = pd.read_csv(file_sigma, header=None)

    truthr = jnp.array(data_roh.iloc[:,0])
    inferredr = jnp.array(data_roh.iloc[:,1])
    stdr = jnp.array(data_roh.iloc[:,2])
    
    truths = jnp.array(data_sigma.iloc[:,0])
    inferreds = jnp.array(data_sigma.iloc[:,1])
    stds = jnp.array(data_sigma.iloc[:,2])

    for i in range(16):
        mr, emr, br, ebr, chiqr, ndofr = linreg(truthr[i::16], inferredr[i::16], stdr[i::16])
        residuenplot('True Value', 'a.u.', truthr[i::16], 0, 'Inferred Value', 'a.u.', inferredr[i::16], stdr[i::16], mr, br, chiqr, name=f'Rho_{i+1} ' + name, save=False, Zoom = Zoom)

    for i in range(15):
        ms, ems, bs, ebs, chiqs, ndofs = linreg(truths[i::15], inferreds[i::15], stds[i::15])
        residuenplot('True Value', 'a.u.', truths[i::15], 0, 'Inferred Value', 'a.u.', inferreds[i::15], stds[i::15], ms, bs, chiqs, name=f'Sigma_{i+1} ' + name, save=False)

# Auswertung('data_rho.csv', 'data_sigma.csv', name='changes', Zoom = None)
# Auswertung('data_rho_eigen_uniform.csv', 'data_sigma_eigen_uniform.csv', name='changes', Zoom = None)
# Auswertung('data_rho_eigen_uniform2.csv', 'data_sigma_eigen_uniform2.csv', name='changes', Zoom = None)
# Auswertung('data_rho_eigen_lognormal.csv', 'data_sigma_eigen_lognormal.csv', name='changes', Zoom = None)
# Auswertung('data_rho_eigen_lognormal2.csv', 'data_sigma_eigen_lognormal2.csv', name='changes', Zoom = None)


#jit fixes time difference with pure/impure complicated function
#params als array
#sichergestellt, dass indizierung richtig/vgl benchmark