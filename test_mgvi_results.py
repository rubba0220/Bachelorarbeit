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

def linreg(x,y,ey,ini=[1,1]):

        def func(x, m, b):
            return m*x + b
        
        popt, pcov = curve_fit(func, x, y, p0=ini, sigma=ey, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))

        xi = y
        Xi = func(x, *popt)
        chiq = jnp.sum((xi-Xi)**2 / ey**2)
        ndof = len(xi)-2

        return popt[0], perr[0], popt[1], perr[1], chiq, ndof       

def residuenplot(var_x, unit_x, v_x, v_s_x, var_y, unit_y, v_y, v_s_y, m, b, chiq, namep='', savep=False):
    
    fig, axarray = plt.subplots(2, 1, figsize=(20,10), sharex=True, gridspec_kw={'height_ratios': [5, 2]})
    plt.title(namep)
    axarray[0].errorbar(v_x, v_y, xerr=v_s_x, yerr=v_s_y, color='red', fmt='o', markeredgecolor='red', label='Messwerte')
    # axarray[0].set_xlim(0,0.02)
    axarray[0].set_xlabel('${0}$ / ${1}$'.format(var_x, unit_x))
    axarray[0].set_ylabel('${0}$ / ${1}$'.format(var_y, unit_y))
    axarray[0].plot(v_x, m*v_x+b, color='green', label='${0:.2f}*x+{1:.2f}$'.format(m, b))
    sigmaRes = np.sqrt((m*v_s_x)**2 + v_s_y**2)
    axarray[0].plot([min(v_x), max(v_x)], [min(v_x), max(v_x)], color='black', linestyle='--', label='Erwartung')
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
    
    if savep == True:
        plt.savefig(namep)
        
    plt.show() 

def histogram2(data, titel, label, name='', save=False):
        
        def gauß(x, mu, sigma):
            f = 1 / jnp.sqrt(2*jnp.pi*sigma**2) * jnp.exp(-1/2 * (x-mu)**2 / sigma**2)
            return f

        f = lambda x,ndof: 1/(2*gamma(ndof/2)) * (x/2)**((ndof-2)/2) * jnp.exp(-x/2)

        fig, axarray = plt.subplots(2, 1, figsize=(20,10), sharex=True, gridspec_kw={'height_ratios': [5, 2]})
        plt.title(titel)
        anf = np.floor(np.min(data))-0.25
        end = np.ceil(np.max(data))+0.25
        n, bins, patches = axarray[0].hist(data,bins=jnp.arange(anf, end, 0.25))

        x = np.linspace(anf, end, 100)
        A = np.sum(n)*0.25

        ind = jnp.where(n>0)
        popt, perr = curve_fit(gauß, jnp.arange(anf,end,0.25)[ind]+0.125, n[ind]/A, p0=[0, 1.], sigma=jnp.sqrt(n[ind])/A, absolute_sigma=True)
        xi = n[ind]
        Xi = A * gauß(jnp.arange(anf,end,0.25)[ind]+0.125, *popt)
        chiq = jnp.sum((xi-Xi)**2 / n[ind])
        ndof = len(xi)-3
        F, müll = quad(f, chiq, jnp.inf, args=(ndof,))
        mu = popt[0]
        sigma = popt[1]
        emu = perr[0,0]
        esigma = perr[1,1]

        axarray[0].plot(x, A*gauß(x, mu, sigma), color='black', linewidth=5, label = 'Fit aus $\chi^2$-Minimierung')
        axarray[0].plot(x, A*gauß(x, mu+emu, sigma+esigma), color='black', linestyle='--')
        axarray[0].plot(x, A*gauß(x, mu-emu, sigma-esigma), color='black', linestyle='--')
        axarray[0].plot(x, A*gauß(x, mu+emu, sigma-esigma), color='black', linestyle='--')
        axarray[0].plot(x, A*gauß(x, mu-emu, sigma+esigma), color='black', linestyle='--')
        plt.figtext(0.70,0.76,r'$\mu = %.5f +/- %.5f$' %(mu,emu))
        plt.figtext(0.70,0.72,r'$\sigma = %.3f +/- %.3f$' %(sigma,esigma))
        axarray[0].set_xlabel(label)
        axarray[0].set_ylabel('absoltue Häufigkeit')
        axarray[0].legend()
        axarray[0].grid()
        
        plt.figtext(0.70,0.56,r'$N = %i$' %(A))
        plt.figtext(0.70,0.52,r'$\chi^2/ndof = %.2f / %i = %.2f$' %(chiq, ndof, chiq/ndof))
        
        axarray[1].axhline(y=0., color='black', linestyle='--')
        axarray[1].errorbar(jnp.arange(anf, end, 0.25)[ind]+0.125, xi-Xi, yerr=np.sqrt(Xi), color='red', fmt='o', markeredgecolor='red')
        axarray[1].set_xlabel(label)
        axarray[1].set_ylabel('Residuen')
        axarray[1].grid()

        plt.tight_layout()
        fig.subplots_adjust(hspace=0.0)

        if save == True:
            plt.savefig(name)

def Auswertung(file_roh, file_sigma):
    data_roh = pd.read_csv(file_roh, header=None)
    data_sigma = pd.read_csv(file_sigma, header=None)
    abw_roh = list(data_roh.iloc[:,4])
    abw_sigma = list(data_sigma.iloc[:,4])
    abw = abw_roh + abw_sigma
  
    # #Visualisierung
    # fig, ax = plt.subplots(figsize=(20,10))
    # ax.set_xlabel('parameter sortiert nach parameter')
    # ax.set_ylabel('Abweichung in Stddevs')
    # for i in range(2):
    #     ax.scatter(jnp.arange(1 + i*10, 1 + (i+1)*10, 1), abw_roh[i::2], marker='o')
    # for i in range(1):
    #     ax.scatter(jnp.arange(1 + 10*2 + i*10, 1 + 10*2 + (i+1)*10, 1), abw_sigma[i::1], marker='x')
    # ax.grid()
    # fig.tight_layout()

    truthr = jnp.array(data_roh.iloc[:,0])
    inferredr = jnp.array(data_roh.iloc[:,1])
    stdr = jnp.array(data_roh.iloc[:,2])
    
    truths = jnp.array(data_sigma.iloc[:,0])
    inferreds = jnp.array(data_sigma.iloc[:,1])
    stds = jnp.array(data_sigma.iloc[:,2])

    mrs, emrs, brs, ebrs, chiqrs, ndofrs = linreg(truthr[0::2], inferredr[0::2], stdr[0::2])
    mrd, emrd, brd, ebrd, chiqrd, ndofrd = linreg(truthr[1::2], inferredr[1::2], stdr[1::2])
    ms, ems, bs, ebs, chiqs, ndofs = linreg(truths[::1], inferreds[::1], stds[::1])

    residuenplot('True Value', '1', truthr[0::2], 0, 'Inferred Value', '1', inferredr[0::2], stdr[0::2], mrs, brs, chiqrs, namep='Roh_s eigenerSolver(120) samples(20) Uniform', savep=False)
    residuenplot('True Value', '1', truthr[1::2], 0, 'Inferred Value', '1', inferredr[1::2], stdr[1::2], mrd, brd, chiqrd, namep='Roh_d eigenerSolver(120) samples(20) Uniform', savep=False)
    residuenplot('True Value', '1', truths[::1], 0, 'Inferred Value', '1', inferreds[::1], stds[::1], ms, bs, chiqs, namep='Sigma_s eigenerSolver(120) samples(20) Uniform', savep=False)

    histogram2(abw, '10 Testdurchläufte', 'Abweichung in Stddevs', name='Plots/Fit_Sr', save=False)


Auswertung('data_roh_unreal_u_small_better.csv', 'data_sigma_unreal_u_small_better.csv')