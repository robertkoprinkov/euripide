import numpy as np
import scipy
from qpsolvers import solve_qp

from . import splines as splines
from .minimax_tilting_sampler import TruncatedMVN

class constrainedGP():
    """
    Abstract class for constrained Gaussian Processes. Monotonic and Bounded GPs inherit from this class.

    The classes that inherit from this class represent the infinite dimensional Gaussian Process with finite-dimensional representation, as a linear combination of a finite number of jointly Gaussian scalar Gaussian random variables. In other words, at every point x we have that Y(x) is represented as the linear combination \\sum_{i=1}^{n}\\phi_i(x) \\xi_i:
    Y(x) ~ \\sum_{i=1}^{n}\\phi_i(x) \\xi_i = \\Phi(x)^{T} \\xi
    
    This defines a Gaussian random variable for each point x, in other words, a Gaussian Process.
    The choice of the representation (or spline) functions \\phi_i, as well as the choice of the covariance structure of the \\xi_i, fully determines the GP representation. Depending on the chosen representation, truncating the distribution of the \\xi_i will impose different constraints (or a different regularity) of the resulting set of random variables Y(x).

    The subclasses differ in:
    - The covariance structure of the random variables \\xi
    - The choice of the representation or "linear combination" or spline functions \\phi_i
    - The type of truncation on the random variables \\xi, which depends on the role of the random variable, which is defined by its influence function \\phi_i
    """

    def __init__(self, N_splines, x_min, x_max, kernel, interpol_x=None, interpol_y=None, lowerbound=None, upperbound=None, nugget_prior=1e-3, nugget_interpol=1e-6, verbose=False):
        self.splines = self.init_splines(N_splines, x_min, x_max)

        self.kernel = kernel

        self.mean_prior, self.cov_prior = self.prior(nugget=nugget_prior, verbose=verbose)
        
        self.interpol_x = interpol_x
        self.interpol_y = interpol_y

        self.mean_interpol = None
        self.cov_interpol = None
        
        if self.interpol_x is not None:
            self.mean_interpol, self.cov_interpol = self.calc_interpolation(interpol_x, interpol_y, nugget=nugget_interpol, verbose=verbose)

            _, cov_no_nugget = self.calc_interpolation(interpol_x, interpol_y, nugget=None)
            eigv, eigvec = np.linalg.eigh(cov_no_nugget)

            nonzero = np.abs(eigv) > 1e-9
            self.map_from_reduced = eigvec.T
            
            eigv = eigv[nonzero]
            eigvec = eigvec[nonzero]
            self.map_from_reduced = self.map_from_reduced[:, nonzero]
            
            print(eigvec.shape)
            self.cov_interpol_reduced_old = np.diag(eigv) @ (eigvec @ eigvec.T)
            self.cov_interpol_reduced = self.map_from_reduced.T @ cov_no_nugget @ self.map_from_reduced

            print(np.abs(self.cov_interpol_reduced - self.cov_interpol_reduced.T).max(), 'Diff symmetry')
            self.cov_interpol_reduced = .5 * self.cov_interpol_reduced + .5*self.cov_interpol_reduced.T
            print(np.linalg.eigvals(self.cov_interpol_reduced).real.min())
            # TODO: they should be the same
            print(np.abs(self.cov_interpol_reduced - self.cov_interpol_reduced_old).max())
        if lowerbound is not None:
            self.constraint_matrix, self.constraints_max_value = self.define_constraints(lowerbound, upperbound)
            #optim_result_shifted = self.mean_interpol + solve_qp(np.linalg.inv(self.cov_interpol), np.zeros(self.splines.n_splines), self.constraint_matrix, self.constraints_max_value - self.constraint_matrix @ self.mean_interpol, None, None, solver='quadprog', verbose=True)
            optim_result = solve_qp(np.linalg.inv(self.cov_interpol), -self.mean_interpol.T @ np.linalg.inv(self.cov_interpol), self.constraint_matrix, self.constraints_max_value, None, None, solver='quadprog', verbose=True)

            #red_inv = np.linalg.inv(self.cov_interpol_reduced)
            #red_inv = .5*(red_inv + red_inv.T)
            #optim_red = self.mean_interpol + self.map_from_reduced @ solve_qp(red_inv, np.zeros(self.cov_interpol_reduced.shape[0]), self.constraint_matrix @ self.map_from_reduced, self.constraints_max_value - self.constraint_matrix @ self.mean_interpol, None, None, solver='quadprog', verbose=True)
            # shifted gets the same result
            #print('Max diff', np.abs(optim_result - optim_result_shifted).max(), np.abs(optim_red - optim_result_shifted).max())
            #print(optim_red)
            # in extreme cases, the prior solver works better
            # for some reason, the posterior one doesn't work in the monotonic case where we have an almost constant stretch
            # the constraint ends up being violated
            A = self.splines.eval_splines(interpol_x)
            optim_prior = solve_qp(np.linalg.inv(self.cov_prior), np.zeros(self.splines.n_splines), self.constraint_matrix, self.constraints_max_value, A, interpol_y, solver='quadprog', verbose=True)

            if optim_prior is None:
                print('Failed optim prior')
            self.mean_constrained = optim_result
            self.mean_constrained_prior = optim_prior
            
            if optim_prior is None:
                print('Optimization with matrix $\Gamma^N$ failed')
            else:
                print('Optim posterior vs prior difference', np.abs(optim_result - optim_prior).max())

            lb, ub = self.get_bounds(lowerbound, upperbound)
            self.ET_sampler = TruncatedMVN(self.mean_interpol, self.cov_interpol, lb, ub)
    """
        Return prior mean and covariance matrices. Must have same dimension as number of RV of representation.
    """
    def prior(self, nugget=None):
        pass
    
    """
        Return the constraint_matrix matrix and constraints_max_value vector, which specify the constraints applying to the RV.

        This representation limits the representable constraints to linear constraints in terms of the RV used in the representation.
    """
    def define_constraints(self):
        pass

    """
        Return the bounds for the RV. Returns two np.arrays, both of size self.splines.n_splines, one specifying the lowerbound
        of each RV in the internal representation, the other representing the upper bounds.
    """
    def get_bounds(self, lowerbound, upperbound):
        pass
    """
        Return the splines that will be used in this representation
    """
    def init_splines(N_splines, x_min, x_max):
        pass
    
    def calc_interpolation(self, interpol_x, interpol_y, nugget=5e-3, verbose=False):
        """
        Calculate distribution of RV $\\xi$ given the interpolation points, with the prior belonging to this GP.    
        """
        A = self.splines.eval_splines(interpol_x)
        
        mean_interpol = (A @ self.cov_prior).T @ np.linalg.solve(A @ self.cov_prior @ A.T, interpol_y)
        cov_interpol  = self.cov_prior - (A @ self.cov_prior).T @ np.linalg.inv(A @ self.cov_prior @ A.T) @ A @ self.cov_prior
        
        if verbose:
            print('[PosDef] Post-interpolation matrix:', np.all(np.linalg.eigvals(cov_interpol) > 0))
            print('[PosDef] Prior covariance matrix:', np.all(np.linalg.eigvals(self.cov_prior) > 0))

        if nugget is not None:
            cov_noise = cov_interpol + nugget * np.eye(self.splines.n_splines)# np.random.uniform(size=self.cov_interpol.shape)
            
            if verbose:
                print('[Nugget] of size %f added to post-interpolation covariance matrix. Condition number went from %d to %d.' % (nugget, np.linalg.cond(cov_interpol), np.linalg.cond(cov_noise)))
                print('[PosDef] Post-interpolation covariance matrix with nugget:', np.all(np.linalg.eigvals(cov_noise) > 0))

            # adding noise does decrease it
            cov_interpol = cov_noise

        return mean_interpol, cov_interpol

    """
        Get mean and covariance of used RV, or, if points are supplied, of the RVs Y^{N}(x_0, x_1, ..., x_n)
    """
    def statistics_prior(self, x=None):
        if x is None: # return mean and cov. matrix of RV used in finite dimensional representation of GP
            return self.mean_prior, self.cov_prior
        else:
            phi_T = self.spline.eval_splines(x) # I believe this expression is correct, but it should be tested, e.g. for hat function splines
            return phi_T @ self.mean_prior, phi_T @ self.cov_prior @ phi_T.T

    """
        Sample from prior distribution
        
    """
    def sample_prior(self, x_sample=None, n_samples=1):
        if x_sample is None:
            x_sample = self.splines.splines_x
        RV = np.random.multivariate_normal(self.mean, self.cov, n_samples).T # dim: self.splines.n_splines x n_samples
        phi_T = self.splines.eval_splines(x_sample)

        return phi_T @ RV

    """
        Mean and variance of the distribution of RV used in representation, obtained by conditioning the prior distribution on the interpolation points.
        This distribution is normal.

        Alternatively, if x is supplied, the mean and covariance of the RV given by Y^{N}(x_0), Y^{N}(x_1), ...Y^N(x_{n_x})|Y(x_I) = Y_I is returned.
        
    """
    def statistics_interpolation(self, x=None):
        if x is None:
            return self.mean_interpol, self.cov_interpol
        else:
            phi_T = self.splines.eval_splines(x)
            return phi_T @ self.mean_interpol, phi_T @ self.cov_interpol @ phi_T.T

    """
        Sample from distribution obtained by conditioning the prior distribution on the interpolation points. Note that this
        is not the constrained distribution, and that therefore the samples are not guaranteed to satisfy the desired constraints.
        
    """
    def sample_interpolation(self, x_sample=None, n_samples=1):
        assert self.mean_interpol is not None and self.cov_interpol is not None
        RV = np.random.multivariate_normal(self.mean_interpol, self.cov_interpol, n_samples) # dim: n_samples x self.splines.n_splines
        if x_sample is None:
            return RV
        phi_T = self.splines.eval_splines(x_sample)
        return RV @ phi_T.T

    def statistics_constrained(self, x=None):
        # hard. only sending mean
        if x is None: # sending mean of RV used in representation of GP
            return self.mean_interpol, None
        else:
            phi_T = self.splines.eval_splines(x)
            return phi_T @ self.mean_constrained, None

    """
        @param x_sample: points at which to return the sampled function. If x_sample is None, the original RV used to
                         represent this finite dimensional approximation to the GP are returned
    """
    def sample_constrained(self, x_sample=None, n_samples = 1, return_stats=False, throw_exception_if_failed=True):
        assert self.mean_interpol is not None and self.cov_interpol is not None
        assert self.mean_constrained is not None

        phi_T = None
        if x_sample is not None:
            print(x_sample)
            phi_T = self.splines.eval_splines(x_sample)
        samples = np.zeros((n_samples, x_sample.shape[0]))
        n_accepted = 0
        n_rejected = 0
        n_rejected_constraint = 0
        for n_iter in range(100*n_samples):
            potential = np.random.multivariate_normal(self.mean_constrained, self.cov_interpol, n_samples)
          
            accepted_convex = np.all(potential @ self.constraint_matrix.T < self.constraints_max_value, axis=1)
            
            n_rejected_constraint += n_samples - accepted_convex.sum()
            unif = np.random.uniform(size=n_samples)
            # Have to use cov_interpol, otherwise Neumann theorem doesn't apply and we can't sample from the original distribution in this way.
            accepted_neumann_sampling = unif < np.exp((potential-self.mean_constrained.T) @ np.linalg.solve(self.cov_interpol, self.mean_interpol - self.mean_constrained))
            #accepted_neumann_sampling = unif < np.exp(self.mean_constrained.T @ np.linalg.solve(self.cov_interpol, self.mean_constrained) - 
            #                       potential @ np.linalg.solve(self.cov_interpol, self.mean_constrained))
            accepted = np.logical_and(accepted_convex, accepted_neumann_sampling)
            n_newly_accepted = min(accepted.sum(), n_samples-n_accepted)
            if x_sample is None:
                samples[n_accepted:n_accepted+n_newly_accepted, :] = potential[accepted, :][:n_newly_accepted]
            else:
                samples[n_accepted:n_accepted+n_newly_accepted, :] = potential[accepted, :][:n_newly_accepted] @ phi_T.T

            n_accepted += accepted.sum()
            n_rejected += n_samples - accepted.sum()
            if n_accepted >= n_samples:
                break
        print('Sampling %.2f%% (accepted); %.2f%% (Rejected, violated constraints); %.2f%% (Rejected, Neumann sampling)' % (100.*n_accepted/(n_accepted+n_rejected),\
              100*n_rejected_constraint/(n_accepted+n_rejected), 100*(n_rejected-n_rejected_constraint)/(n_accepted+n_rejected)))
        if throw_exception_if_failed:
            assert n_accepted >= n_samples

        if return_stats:
            return samples[:n_accepted, :], {'n_rejected_constraint': n_rejected_constraint, 'n_rejected_neumann': n_rejected-n_rejected_constraint, 'n_accepted': n_accepted}
        else:
            return samples[:n_accepted, :]
    """
        @param x_sample: points at which to return the sampled function. If x_sample is None, the original RV used to
                         represent this finite dimensional approximation to the GP are returned
    """
    def sample_constrained_ET(self, x_sample=None, n_samples = 1, return_stats=False, throw_exception_if_failed=True):
        assert self.mean_interpol is not None and self.cov_interpol is not None
        assert self.mean_constrained is not None

        phi_T = None
        if x_sample is not None:
            phi_T = self.splines.eval_splines(x_sample)
        # though this isn't an MCMC sampler. I don't think there's any reason to do this.
        samples = self.ET_sampler.sample(10*n_samples)
        samples = samples[:, -10*n_samples:]
        permutation = np.random.permutation(samples.shape[1])
        samples = samples[:, permutation].T

        if x_sample is not None:
            samples = samples @ phi_T.T
        print(samples.shape)
        return samples[:n_samples, :]
        
    """
        Evaluate a given realization of the RV xi at new points x_eval
        
    """
    def eval_sample(self, x_eval, RV):
        phi_T = self.splines.eval_splines(x_eval)
        return RV @ phi_T.T

class BoundedGP(constrainedGP):
    
    def prior(self, nugget=None, verbose=False):
        cov_prior = np.zeros((self.splines.n_splines, self.splines.n_splines))
        mean_prior = np.zeros(self.splines.n_splines)        
        for i in range(self.splines.n_splines):
            for j in range(self.splines.n_splines):
                cov_prior[i, j] = self.kernel.k(self.splines.splines_x[i], self.splines.splines_x[j])

        if nugget is not None:
            old_cond = np.linalg.cond(cov_prior)
            cov_prior = cov_prior + nugget*np.eye(self.splines.n_splines)
            if verbose:
                print('[Nugget] added to prior covariance matrix. Condition number went from %d to %d.' % (old_cond, np.linalg.cond(cov_prior)))
        return mean_prior, cov_prior
    
    def define_constraints(self, lowerbound, upperbound):
        constraint_matrix = np.zeros((2*(self.splines.n_splines), self.splines.n_splines))
        constraint_matrix[:self.splines.n_splines, :] = np.eye(self.splines.n_splines)
        constraint_matrix[self.splines.n_splines:, :] = -np.eye(self.splines.n_splines)

        constraints_max_value = np.zeros(2*(self.splines.n_splines))
        constraints_max_value[:self.splines.n_splines] = upperbound
        constraints_max_value[self.splines.n_splines:] = -lowerbound
        
        return constraint_matrix, constraints_max_value

    def get_bounds(self, lowerbound, upperbound):
        return lowerbound*np.ones(self.splines.n_splines), upperbound*np.ones(self.splines.n_splines)
    
    def init_splines(self, N_splines, x_min, x_max):
        return splines.hats(N_splines, x_min, x_max)

class MonotonicGP(constrainedGP):

    def prior(self, nugget=None, verbose=False):
        mean_prior = np.zeros(self.splines.n_splines)
        cov_prior = np.zeros((self.splines.n_splines, self.splines.n_splines))

        cov_prior[0, 0] = self.kernel.k(0., 0.)

        for i in range(1, self.splines.n_splines):
            cov_prior[0, i] = self.kernel.k_der_x1(0., self.splines.splines_x[i-1])
            # symmetric kernel
            cov_prior[i, 0] = cov_prior[0, i]

        for i in range(1, self.splines.n_splines):
            for j in range(1, self.splines.n_splines):
                cov_prior[i, j] = self.kernel.k_der_x1x2(self.splines.splines_x[i-1], self.splines.splines_x[j-1])

        if nugget is not None:
            old_cond = np.linalg.cond(cov_prior)
            cov_prior = cov_prior + nugget*np.eye(self.splines.n_splines)

            if verbose:
                print('[Nugget] added to prior covariance matrix. Condition number went from %d to %d.' % (old_cond, np.linalg.cond(cov_prior)))
            
        
        return mean_prior, cov_prior
    
    def init_splines(self, N_splines, x_min, x_max):
        return splines.monotonic_splines(N_splines, x_min, x_max)

    def define_constraints(self, lowerbound, upperbound):
        constraint_matrix = np.zeros((2*(self.splines.n_splines-1), self.splines.n_splines))
        constraint_matrix[:self.splines.n_splines-1, 1:] = np.eye(self.splines.n_splines-1)
        constraint_matrix[self.splines.n_splines-1:, 1:] = -np.eye(self.splines.n_splines-1)

        constraints_max_value = np.zeros(2*(self.splines.n_splines-1))
        constraints_max_value[:self.splines.n_splines-1] = upperbound
        constraints_max_value[self.splines.n_splines-1:] = -lowerbound
        
        return constraint_matrix, constraints_max_value

    def get_bounds(self, lowerbound, upperbound):
        lb = np.ones(self.splines.n_splines)
        ub = np.ones(self.splines.n_splines)

        lb[0] = -1e-9
        ub[0] = 1e9

        lb[1:] = lowerbound
        ub[1:] = upperbound
        return lb, ub
