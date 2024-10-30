import unittest
from src.splines import hats as hats
from src.splines import monotonic_splines as monotonic_splines

from src.BoundedGP import BoundedGP
from src.kernels import gaussian_kernel

from src.vis import Visualize

import numpy as np

import matplotlib.pyplot as plt
import scienceplots

import functools

plt.style.use(['science', 'ieee'])

class TestConstrainedSampling(unittest.TestCase):
    def setUp(self):
        self.vis = Visualize()
        
    def testHatsLine(self):
        pass

    def test_sampling_rate_wrt_constraints(self):
        a = np.linspace(-1., 0.95, 20)
        n_samples = 10

        rejection_rate_constraint = np.zeros(a.shape)
        rejection_rate_neumann = np.zeros(a.shape)
        
        x_sample = np.linspace(0., 1., 100)
        
        means_inter = np.zeros((a.shape[0], x_sample.shape[0]))
        means_constr = np.zeros((a.shape[0], x_sample.shape[0]))
         
        interpol_x = np.linspace(0.1, 0.9, 4)
        interpol_y = np.ones_like(interpol_x)
        

        bgp = None
        bgp_init = None
        for i, a_ in enumerate(a):
            bgp = BoundedGP(50, 0., 1., gaussian_kernel({'theta': 0.1, 'sigma': 1.}), interpol_x=interpol_x, interpol_y=interpol_y, lowerbound=a_, upperbound=10000)

            if i == 0:
                bgp_init = bgp
                
            _, stats = bgp.sample_constrained(x_sample = bgp.splines.splines_x, n_samples=n_samples, return_stats=True, throw_exception_if_failed=False)

            rejection_rate_constraint[i] = stats['n_rejected_constraint']/(stats['n_rejected_constraint'] + stats['n_rejected_neumann'] + stats['n_accepted'])
            rejection_rate_neumann[i] = stats['n_rejected_neumann']/(stats['n_rejected_neumann'] + stats['n_accepted'])
            
            means_inter[i, :], _ = bgp.statistics_interpolation(x=x_sample)
            means_constr[i, :], _ = bgp.statistics_constrained(x=x_sample)
        
        self.vis.plot_interpolated(bgp, filename='test_tight_constraints_interpolation.png')        
        self.vis.plot_constrained(bgp, filename='test_tight_constraints_constrained.png', n_samples=None)
        fig, ax = plt.subplots(1, 1, figsize=(5, 2))


        ax.plot(a, 100.*rejection_rate_constraint, label='Rejection due to constraint')
        ax.plot(a, 100.*rejection_rate_neumann, label='Rejection due to Neumann sampling')

        means_diff = np.abs(means_inter - means_constr)
        means_diff = np.max(means_diff, axis=1)

        ax.axvline(a[means_diff > 0.001].min(), color='red', label=r'First time: $\mu_I\neq \mu_C$')
        ax.axvline(1., color='blue', label=r'Exact function')
        
        ax.set_title('Rejection rate vs tightness of constraint. Interpolation points are (x_i, 1).')
        
        ax.set_xlabel('lowerbound of boundedness constraint')
        ax.set_ylabel('%%')
        
        ax.legend()
        ax.grid()

        plt.tight_layout()
        plt.savefig('img/test_tight_constraints_rejection_rates.png')


    def test_sampling_wrt_constraints_ET(self):
        a = np.linspace(-1., 1.5, 20)
        n_samples = 100
        
        interpol_x = np.linspace(0.1, 0.9, 4)
        interpol_y = 1.5*np.ones_like(interpol_x)
        
        x_sample = np.linspace(0., 1., 51)
         
        samples_ET = np.zeros((a.shape[0], n_samples, x_sample.shape[0]))
        
        for i, a_ in enumerate(a):
            bgp = BoundedGP(50, 0., 1., gaussian_kernel({'theta': 0.1, 'sigma': 1.}), interpol_x=interpol_x, interpol_y=interpol_y, lowerbound=a_, upperbound=10000)
            x_sample = bgp.splines.splines_x
            samples_ET[i, :, :] = bgp.sample_constrained_ET(x_sample=x_sample, n_samples=n_samples)


        fig, ax = plt.subplots(3, 3, figsize=(15, 15))
        
        ax = ax.flatten()

        for i in range(9):
            idx = -int(1+i*samples_ET.shape[0]//9)
            for j in range(n_samples):
                ax[-i-1].plot(x_sample, samples_ET[idx, j, :], color='red', linewidth=0.3, alpha=0.7)
            ax[-i-1].axhline(a[idx])
        plt.tight_layout()
        plt.savefig('img/test_tiling_constrained_sampling.png')

