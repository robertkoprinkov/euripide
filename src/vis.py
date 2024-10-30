import matplotlib.pyplot as plt
import numpy as np

class Visualize():

    def __init__(self, directory='./img/'):
        self.directory = directory

    def plot_interpolated(self, boundedGP, n_samples=100, filename='interpolated.png', return_ax=False):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), width_ratios=[2, 1])

        mean, cov = boundedGP.statistics_interpolation(x=boundedGP.splines.splines_x)
        print(np.diag(cov))
        cov = np.sqrt(np.diag(cov))
        print(cov.shape)
        y1, = ax[0].plot(boundedGP.splines.splines_x, mean)
        y2, = ax[0].plot(boundedGP.splines.splines_x, mean - 2*cov, color='red', linestyle='--')
        y3, = ax[0].plot(boundedGP.splines.splines_x, mean + 2*cov, color='red', linestyle='--')
        
        y_sample = boundedGP.sample_interpolation(x_sample=boundedGP.splines.splines_x, n_samples=n_samples)
        line_samples = []
        for i in range(n_samples):
            line_samples.append(ax[0].plot(boundedGP.splines.splines_x, y_sample[i, :], color='gray', linestyle='-', linewidth=0.5, alpha=0.3)[0])
        
        ax[0].scatter(boundedGP.interpol_x, boundedGP.interpol_y, marker='x', color='red')
        
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[0].grid()

        ax[1].legend([y1, y2, y3] + line_samples, [r'$\mu_I$', r'$\mu_I - 2\sigma_I$', r'$\mu_I + 2\sigma_I$', 'samples'])
        ax[1].axis('off')
        
        plt.tight_layout()
        if filename is not None:
            plt.savefig(self.directory + filename, dpi=300)
        if return_ax:
            return fig, ax
    def plot_constrained(self, boundedGP, n_samples=100, filename='constrained.png', return_ax=False):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), width_ratios=[2, 1])

        mean_constrained, _ = boundedGP.statistics_constrained(boundedGP.splines.splines_x)
        mean_interpol, _ = boundedGP.statistics_interpolation(boundedGP.splines.splines_x)
        phi_T = boundedGP.splines.eval_splines(boundedGP.splines.splines_x)
        
        
        y1, = ax[0].plot(boundedGP.splines.splines_x, mean_constrained)
        y2, = ax[0].plot(boundedGP.splines.splines_x, mean_interpol, linestyle='--')
        
        y3 = None
        if boundedGP.mean_constrained_prior is not None:
            # sometimes calculating mean with prior fails
            mean_constrained_prior = phi_T @ boundedGP.mean_constrained_prior
            y3, = ax[0].plot(boundedGP.splines.splines_x, mean_constrained_prior, linestyle='--')
        y_sample = []
        
        if n_samples is not None:
            #samples = boundedGP.sample_constrained(x_sample=boundedGP.splines.splines_x, n_samples=n_samples)
            samples = boundedGP.sample_constrained_ET(x_sample=boundedGP.splines.splines_x, n_samples=n_samples)
            
            for i in range(n_samples):
                y_sample.append(ax[0].plot(boundedGP.splines.splines_x, samples[i], color='red', linestyle='-', linewidth=0.2, alpha=0.1)[0])
        
        ax[0].scatter(boundedGP.interpol_x, boundedGP.interpol_y, marker='x', color='red')
        
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[0].grid()

        ax[1].legend([y1, y2, y3] + y_sample, [r'$\mu_{C+I}$', r'$\mu_{I}$', r'$\mu_{I, \text{prior}}$', 'samples'])
        #ax[1].legend([y1, y2], [r'$\mu_{C+I}$', r'$\mu_{I}$'])
        ax[1].axis('off')
        
        plt.tight_layout()
        if filename is None:
            return fig, ax        
        else:
            plt.savefig(self.directory + filename, dpi=300)

    def visualize_optim_methods(self, boundedGP, filename='optim_comparison.png'):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.plot(boundedGP.splines.splines_x, boundedGP.mean_constrained, label=r'$\mu_C = \text{argmin}_{x\in\xi_I\cap\xi_C} (x-\mu_I)^T\Sigma(x-\mu)$')
        ax.plot(boundedGP.splines.splines_x, boundedGP.mean_constrained_prior, label=r'$\mu_C = \text{argmin}_{x\in\xi_I\cap\xi_C} x^T\Gamma^{N}x$')

        fig.suptitle('Optimization methods to calculate constrained + interpolated mode')
        
        ax.set_title(r'$\text{Optimizing w.r.t. prior }\Gamma^{N}\text{ or interpolation posterior }\Sigma$')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid()
        ax.legend()

        plt.tight_layout()

        if filename is not None:
            plt.savefig(self.directory + filename, dpi=300)
        if return_ax:
            return fig, ax
        
