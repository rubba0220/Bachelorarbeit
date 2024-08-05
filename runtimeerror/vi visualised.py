# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2021 Max-Planck-Society
# Authors: Reimar Leike, Philipp Arras, Philipp Frank
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

###############################################################################
# Variational Inference (VI)
#
# This script demonstrates how MGVI, GeoVI, MeanfieldVI and FullCovarianceVI
# work for an inference problem with only two real quantities of interest. This
# enables us to plot the posterior probability density as two-dimensional plot.
###############################################################################

import numpy as np
import pylab as plt
from matplotlib.colors import LogNorm

import nifty8 as ift


def main():
    dom = ift.UnstructuredDomain(1)
    scale = 10

    a = ift.FieldAdapter(dom, 'a')
    b = ift.FieldAdapter(dom, 'b')
    lh = (a.adjoint @ a).scale(scale) + (b.adjoint @ b).scale(-1.35*2).exp()
    lh = ift.VariableCovarianceGaussianEnergy(dom, 'a', 'b', np.float64) @ lh
    icsamp = ift.AbsDeltaEnergyController(deltaE=0.1, iteration_limit=2)
    ham = ift.StandardHamiltonian(lh, icsamp)

    x_limits = [-8/scale, 8/scale]
    x_limits_scaled = [-8, 8]
    y_limits = [-4, 4]
    x = np.linspace(*x_limits, num=401)
    y = np.linspace(*y_limits, num=401)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    def np_ham(x, y):
        prior = x**2 + y**2
        mean = x*scale
        lcov = 1.35*2*y
        lh = .5*(mean**2*np.exp(-lcov) + lcov)
        return lh + prior

    z = np.exp(-1*np_ham(xx, yy))

    mapx = xx[z == np.max(z)]
    mapy = yy[z == np.max(z)]
    meanx = (xx*z).sum()/z.sum()
    meany = (yy*z).sum()/z.sum()

    n_samples = 100
    minimizer = ift.NewtonCG(
        ift.GradientNormController(iteration_limit=3, name='Mini'))
    posmg = ift.from_random(ham.domain, 'normal')

    fig, axs = plt.subplots(1, 1, figsize=[12, 8])
    axs = [axs]
    def update_plot(runs):
        for axx, (nn, kl) in zip(axs, runs):
            axx.clear()
            axx.imshow(z.T, origin='lower',  cmap='gist_earth_r',
                       norm=LogNorm(vmin=1e-3, vmax=np.max(z)),
                       extent=x_limits_scaled + y_limits)
            xs, ys = [], []
            if isinstance(kl, ift.SampledKLEnergyClass):
                samples = kl.samples.iterator()
            else:
                samples = (kl.draw_sample() for _ in range(n_samples))
            mx, my = 0., 0.
            for samp in samples:
                a = samp.val['a']
                xs.append(a)
                mx += a
                b = samp.val['b']
                ys.append(b)
                my += b
            mx /= n_samples
            my /= n_samples
            axx.scatter(np.array(xs)*scale, np.array(ys),
                        label=f'{nn} samples')
            axx.scatter(mx*scale, my, label=f'{nn} mean')
            axx.scatter(mapx*scale, mapy, label='MAP')
            axx.scatter(meanx*scale, meany, label='Posterior mean')
            axx.set_title(nn)
            axx.set_xlim(x_limits_scaled)
            axx.set_ylim(y_limits)
            axx.legend(loc='lower right')
        axs[0].xaxis.set_visible(False)
        plt.tight_layout()
        plt.draw()
        plt.pause(2.)

    for ii in range(10):
        if ii % 2 == 0:
            # Resample GeoVI and MGVI
            mgkl = ift.SampledKLEnergy(posmg, ham, n_samples, None, False)

            runs = (("MGVI", mgkl),)
            update_plot(runs)

        mgkl, _ = minimizer(mgkl)
        posmg = mgkl.position
        runs = (("MGVI", mgkl),)
        update_plot(runs)
    ift.logger.info('Finished')
    plt.show()


if __name__ == '__main__':
    main()
