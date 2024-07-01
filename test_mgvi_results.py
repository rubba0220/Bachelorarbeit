import pandas as pd
import matplotlib.pyplot as plt
from jax import numpy as jnp
import numpy as np
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

def Auswertung(file_roh, file_sigma):
    data_roh = pd.read_csv(file_roh, header=None)
    data_sigma = pd.read_csv(file_sigma, header=None)
    abw_roh = list(data_roh.iloc[:,4])
    abw_sigma = list(data_sigma.iloc[:,4])
    abw = abw_roh + abw_sigma
  
    #Visualisierung
    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_xlabel('parameter sortiert nach parameter')
    ax.set_ylabel('Abweichung in Stddevs')
    for i in range(16):
        ax.scatter(jnp.arange(1 +i*41, 1 + (i+1)*41, 1), abw_roh[i::16], marker='o')
    for i in range(15):
        ax.scatter(jnp.arange(1 + 41*16 + i*41, 1 + 41*16 + (i+1)*41, 1), abw_sigma[i::15], marker='x')
    ax.grid()
    fig.tight_layout()

    truth = list(data_roh.iloc[:,0])
    inferred = list(data_roh.iloc[:,1])
    sigma = list(data_roh.iloc[:,2])

    # #Visualisierung
    # fig, ax = plt.subplots(figsize=(20,10))
    # ax.set_xlabel('parameter sortiert nach Testdurchlauf')
    # ax.set_ylabel('Abweichung in Stddevs')
    # for i in range(10):
    #     ax.scatter(jnp.arange(1 + i*(16+15), 1 + (i+1)*(16) + i*15, 1), abw_roh[i::10], marker='o')
    #     ax.scatter(jnp.arange(1 + (i+1)*16 + i*15, 1 + (i+1)*16 + (i+1)*15, 1), abw_sigma[i::10], marker='x')
    # for i in range(10):
    #      ax.scatter(jnp.arange(1 + 10*16 + i*15,1 + 10*16 + (i+1)*15, 1), abw_sigma[i::10], marker='x')
    # ax.grid()
    # fig.tight_layout()

    #Visualisierung
    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_xlabel('True Value')
    ax.set_ylabel('Inferred Value')
    ax.errorbar(truth[::16], inferred[::16], yerr = sigma[::16], fmt='o')
    ax.plot([min(truth[::16]+inferred[::16]),max(truth[::16]+inferred[::16])],[min(truth[::16]+inferred[::16]),max(truth[::16]+inferred[::16])], color='black', linestyle='--')
    ax.grid()
    fig.tight_layout()


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

        print('')
        print('Der Fit liefert')
        print('\u03BC = %.3f +/- %.3f'%(popt[0], perr[0,0]))
        print('\u03C3 = %.3f +/- %.3f'%(popt[1], perr[1,1]))
        print('\u03C7\u00B2/ndof = %.2f/%.2f = %.2f'%(chiq,ndof, chiq/ndof))
        print('F = %.3f'%F)

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

    histogram2(abw, '10 Testdurchläufte', 'Abweichung in Stddevs', name='Plots/Fit_Sr', save=False)


Auswertung('data_roh_6_un_50.csv', 'data_sigma_6_un_50.csv')