import src.BoundedGP as BoundedGP
import src.kernels as kernels
import numpy as np
import src.vis as vis


lowerbound = 0.
upperbound = 200.

interpol_x, interpol_y = np.array([0.0, 0.1, 0.19, 0.41, 0.52, 0.9]), np.array([10.0, 8., -9., -6., 15., 17.])

interpol_x, interpol_y = np.array([0., 0.05, 0.1, 0.3, 0.4, 0.45, 0.5, 0.8, 0.85, 0.9, 1.]), np.array([0., 0.6, 1.1, 5.5, 7.2, 8., 9.1, 15., 16.3, 17., 20.])

kernel = kernels.gaussian_kernel(params={'sigma': 20., 'theta': 0.14})

# this is going to be a problem, the knots. We will need an adaptive approach because we don't know a priori
# how the knots will be distributed. Maybe even something like multigrid.
boundedGP = BoundedGP.MonotonicGP(50, 0., 1., kernel, interpol_x, interpol_y, lowerbound=lowerbound, upperbound=upperbound, nugget_prior=1e-3, nugget_interpol=1e-3)

mean_knots, var_knots = boundedGP.statistics_prior()

#sample = boundedGP.sample_interpolation(sample_pts)

mean, var = boundedGP.statistics_interpolation()

visualize = vis.Visualize()

visualize.plot_interpolated(boundedGP, filename='interpolated_monotonic.png')

print(boundedGP.constraint_matrix.shape)
visualize.plot_constrained(boundedGP, filename='constrained_monotonic.png', n_samples=50)

