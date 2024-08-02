import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from scipy.optimize import curve_fit

# Plot-Formatierung
plt.rcParams['font.size'] = 24.0
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.0

''' Parameters '''
rho_dm = 0.025
rhos = jnp.array([0.021, 0.016, 0.012, 0.0009, 0.0006, 0.0031, 0.0015, 0.0020, 0.0022, 0.007, 0.0135, 0.006, 0.002, 0.0035, 0.0001])
sigmas = jnp.array([4., 7., 9., 40., 20., 7.5, 10.5, 14., 18., 18.5, 18.5, 20., 20., 37., 100.])
erhos = jnp.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * rhos
esigmas = jnp.array([1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 5., 5., 5., 10.])
params = jnp.vstack([rhos, sigmas])

''' Plotting functions '''
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



def residuenplot(param, eparam, var_x, unit_x, v_x, v_s_x, var_y, unit_y, v_y, v_s_y, m, b, chiq, name='', save=False, pfad=None, regression=True, residue=True, mean=True, ind=None):
    
    if residue:
        fig, axarray = plt.subplots(2, 1, figsize=(20,10), sharex=True, gridspec_kw={'height_ratios': [5, 2]})
        sigmaRes = jnp.sqrt((m*v_s_x)**2 + v_s_y**2)
        axarray[1].set_xlabel('{0} / {1}'.format(var_x, unit_x))
        axarray[1].set_ylabel('$residue$/{0}'.format(unit_y))
        axarray[1].axhline(y=0., color='black', linestyle='--')
        axarray[1].errorbar(v_x, v_y-(m*v_x+b), yerr=sigmaRes, color='red', fmt='o', markeredgecolor='red')
        if ind != None:
            ymax = 2*max([abs(x) for x in np.delete(v_y-(m*v_x+b), ind)])
        else:
            ymax = max([abs(x) for x in axarray[1].get_ylim()])
        axarray[1].set_ylim(-ymax, ymax)
        axarray[1].grid()
    else: 
        fig, axarray = plt.subplots(1, 1, figsize=(20,10))
        axarray = np.array([axarray])

    plt.title(name)
    axarray[0].errorbar(v_x, v_y, xerr=v_s_x, yerr=v_s_y, color='red', fmt='o', markeredgecolor='red', label='test run results')
    if ind != None:
        blue_shades = ['blue','darkblue','navy','midnightblue', 'deepskyblue','dodgerblue','cornflowerblue','royalblue','mediumblue']
        for i in range(len(ind)):
            axarray[0].errorbar(v_x[ind[i]], v_y[ind[i]], xerr=0, yerr=v_s_y[ind[i]], color=blue_shades[i], fmt='o', markeredgecolor=blue_shades[i])
    axarray[0].set_xlabel('{0} / {1}'.format(var_x, unit_x))
    axarray[0].set_ylabel('{0} / {1}'.format(var_y, unit_y))
    # axarray[0].set_ylim([jnp.min(v_y)-1.2*eparam, jnp.max(v_y)+1.2*eparam])


    axarray[0].plot([jnp.min(v_x), jnp.max(v_x)], [jnp.min(v_x), jnp.max(v_x)], color='black', linestyle='--', label='expectation')

    if mean:
        axarray[0].axhline(y=param, color='blue', linestyle='-.', label='prior mean')
        axarray[0].fill_between([jnp.min(v_x), jnp.max(v_x)], param-eparam, param+eparam, color='blue', alpha=0.1, label='prior standard deviation')

    if regression:
        if b>0:
            if abs(b)>0.1:
                axarray[0].plot(v_x, m*v_x+b, color='green', linestyle=':', label=r'${0:.2f} \cdot$ {2} $+ {1:.2f}$ {3}'.format(m, b, var_x, unit_y), alpha=0.8)
            else:
                axarray[0].plot(v_x, m*v_x+b, color='green', linestyle=':', label=r'${0:.2f} \cdot$ {2} $+ {1:.2e}$ {3}'.format(m, b, var_x, unit_y), alpha=0.8)
        else:
            if abs(b)>0.1:
                axarray[0].plot(v_x, m*v_x+b, color='green', linestyle=':', label=r'${0:.2f} \cdot$ {2} ${} {1:.2f}$ {3}'.format(m, b, var_x, unit_y), alpha=0.8)
            else:
                axarray[0].plot(v_x, m*v_x+b, color='green', linestyle=':', label=r'${0:.2f} \cdot$ {2} $ {1:.2e}$ {3}'.format(m, b, var_x, unit_y), alpha=0.8)
        plt.figtext(0.14,0.7,r'$\chi^2/ndf = %.2f / (%.0f-2) = %.2f$'% (chiq, len(v_x), chiq/(len(v_x)-2)))

    axarray[0].legend()
    axarray[0].grid()
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.0)
    
    if save == True:
        plt.savefig('Plots/Tests/'+pfad+'.png')
    
    plt.show() 



def Auswertung(file_roh, file_sigma, file_sd=None, name='', Zoom = None):
    data_roh = pd.read_csv(file_roh, header=None)
    data_sigma = pd.read_csv(file_sigma, header=None)

    truthr = jnp.array(data_roh.iloc[:,0])
    inferredr = jnp.array(data_roh.iloc[:,1])
    stdr = jnp.array(data_roh.iloc[:,2])
    
    truths = jnp.array(data_sigma.iloc[:,0])
    inferreds = jnp.array(data_sigma.iloc[:,1])
    stds = jnp.array(data_sigma.iloc[:,2])

    if file_sd != None:
        data_sd = pd.read_csv(file_sd, header=None)
        truthsd = jnp.array(data_sd.iloc[:,0])
        inferredsd = jnp.array(data_sd.iloc[:,1])
        stdsd = jnp.array(data_sd.iloc[:,2])
        msd, emsd, bsd, ebsd, chiqsd, ndofsd = linreg(truthsd, inferredsd, stdsd)
        residuenplot(0, 'True Value', 'a.u.', truthsd, 0, 'Inferred Value', 'a.u.', inferredsd, stdsd, msd, bsd, chiqsd, name='Surface Density ' + name, save=False)

    for i in range(16):
        mr, emr, br, ebr, chiqr, ndofr = linreg(truthr[i::16], inferredr[i::16], stdr[i::16])
        residuenplot(jnp.append(params[:,0],rho_dm)[i], 'True Value', 'a.u.', truthr[i::16], 0, 'Inferred Value', 'a.u.', inferredr[i::16], stdr[i::16], mr, br, chiqr, name=f'Rho_{i+1} ' + name, save=False)

    for i in range(15):
        ms, ems, bs, ebs, chiqs, ndofs = linreg(truths[i::15], inferreds[i::15], stds[i::15])
        residuenplot(params[:,1][i], 'True Value', 'a.u.', truths[i::15], 0, 'Inferred Value', 'a.u.', inferreds[i::15], stds[i::15], ms, bs, chiqs, name=f'Sigma_{i+1} ' + name, save=False)



def Auswertung2(file_roh, file_sigma, file_rd=None, file_sd=None, name='', save=False, ind=None):
    data_roh = pd.read_csv(file_roh, header=None)
    data_sigma = pd.read_csv(file_sigma, header=None)

    truthr = jnp.array(data_roh.iloc[:,0])
    inferredr = jnp.array(data_roh.iloc[:,1])
    stdr = jnp.array(data_roh.iloc[:,2])
    
    truths = jnp.array(data_sigma.iloc[:,0])
    inferreds = jnp.array(data_sigma.iloc[:,1])
    stds = jnp.array(data_sigma.iloc[:,2])

    if file_sd != None:
        data_sd = pd.read_csv(file_sd, header=None)
        truthsd = jnp.array(data_sd.iloc[:,0])
        inferredsd = jnp.array(data_sd.iloc[:,1])
        stdsd = jnp.array(data_sd.iloc[:,2])
        msd, emsd, bsd, ebsd, chiqsd, ndofsd = linreg(truthsd, inferredsd, stdsd)
        residuenplot(0, 0, 'True Value', r'$M_{sun}pc^{-2}$', truthsd, 0, 'Inferred Value', r'$M_{sun}pc^{-2}$', inferredsd, stdsd, msd, bsd, chiqsd, name=r'$\Sigma_s$ ' + name, save=save, pfad=f'surfdens_'+name, regression=True, residue=True, mean=False, ind=ind)

    if file_rd != None:
        data_rd = pd.read_csv(file_rd, header=None)
        truthrd = jnp.array(data_rd.iloc[:,0])
        inferredrd = jnp.array(data_rd.iloc[:,1])
        stdrd = jnp.array(data_rd.iloc[:,2])
        mrd, emrd, brd, ebrd, chiqrd, ndofrd = linreg(truthrd, inferredrd, stdrd)
        residuenplot(rho_dm, rho_dm, '$True Value$', r'$M_{sun}pc^{-3}$', truthrd, 0, '$Inferred Value$', r'$M_{sun}pc^{-3}$', inferredrd, stdrd, mrd, brd, chiqrd, name=r'$\rho_{dm}$ ' + name, save=save, pfad='rhodm_'+name, regression=True, residue=True, mean=False, ind=ind)

    for i in range(15):
        mr, emr, br, ebr, chiqr, ndofr = linreg(truthr[i::15], inferredr[i::15], stdr[i::15])
        residuenplot(rhos[i], erhos[i], '$True Value$', r'$M_{sun}pc^{-3}$', truthr[i::15], 0, '$Inferred Value$', r'$M_{sun}pc^{-3}$', inferredr[i::15], stdr[i::15], mr, br, chiqr, name=rf'$\rho_{{{i+1}}}$ ' + name, save = save, pfad=f'rho{i+1}_'+name, regression=False, residue=False, mean=True, ind=ind)

    for i in range(15):
        ms, ems, bs, ebs, chiqs, ndofs = linreg(truths[i::15], inferreds[i::15], stds[i::15])

        residuenplot(sigmas[i], esigmas[i], '$True Value$', r'$kms^{-1}$', truths[i::15], 0, '$Inferred Value$', r'$kms^{-1}$', inferreds[i::15], stds[i::15], ms, bs, chiqs, name=rf'$\sigma_{{{i+1}}}$ ' + name, save=save, pfad=f'sigma{i+1}_'+name, regression=False, residue=False, mean=True, ind=ind)



# Auswertung2('finale tests/rhos_vdfo_uniform.csv', 'finale tests/sigma_vdfo_uniform.csv', 'finale tests/rhodm_vdfo_uniform.csv', name='Uniforme Verteilung', save=False)




# Auswertung2('finale tests/rhos_vdfo.csv', 'finale tests/sigma_vdfo.csv', 'finale tests/rhodm_vdfo.csv', name='density fall off', save=True, ind=[51,101])

# Auswertung2('finale tests/rhos_bin.csv', 'finale tests/sigma_bin.csv', 'finale tests/rhodm_bin.csv', name='density fall off (histogram)', save=True, ind=[101])

# Auswertung2('finale tests/rhos_surfdens.csv', 'finale tests/sigma_surfdens.csv', 'finale tests/rhodm_surfdens.csv', 'finale tests/sd_surfdens.csv', name='surface density', save=True)






# Auswertung('data2/data_rho.csv', 'data2/data_sigma.csv', name='changes', Zoom = None)

# Auswertung('data2/data_rho_eigen_uniform.csv', 'data2/data_sigma_eigen_uniform.csv', name='changes', Zoom = None)
# Auswertung('data2/data_rho_eigen_uniform2.csv', 'data2/data_sigma_eigen_uniform2.csv', name='changes', Zoom = None)

# Auswertung('data2/data_rho_eigen_lognormal.csv', 'data2/data_sigma_eigen_lognormal.csv', name='changes', Zoom = None)
# Auswertung('data2/data_rho_eigen_lognormal2.csv', 'data2/data_sigma_eigen_lognormal2.csv', name='changes', Zoom = None)

# Auswertung('data3/data_rho_more.csv', 'data3/data_sigma_more.csv', name='changes', Zoom = None)

# Auswertung('data3/data_rho_neuenorm.csv', 'data3/data_sigma_neuenorm.csv', name='changes', Zoom = None)

# Auswertung('data3/data_rho_binning.csv', 'data3/data_sigma_binning.csv', name='changes', Zoom = None)
# Auswertung('data3/data_rho_binning_more.csv', 'data3/data_sigma_binning_more.csv', name='changes', Zoom = None)






# Auswertung('data4/data_rho.csv', 'data4/data_sigma.csv', 'data4/data_sd.csv', name='changes', Zoom = None)
# Auswertung('data4/data_rho_morewithnoise.csv', 'data4/data_sigma_morewithnoise.csv', 'data4/data_sd_morewithnoise.csv', name='changes', Zoom = None)
# Auswertung('data4/data_rho_morepoints.csv', 'data4/data_sigma_morepoints.csv', 'data4/data_sd_morepoints.csv', name='changes', Zoom = None)
# Auswertung('data4/data_rho_morepoints2.csv', 'data4/data_sigma_morepoints2.csv', 'data4/data_sd_morepoints2.csv', name='changes', Zoom = None)