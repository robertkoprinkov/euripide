import src.BoundedGP as BoundedGP
import src.kernels as kernels
import numpy as np
import src.vis as vis

lowerbound = -15.
upperbound = 20.

interpol_x, interpol_y = np.array([0.0, 0.1, 0.19, 0.41, 0.52, 0.9]), np.array([10.0, 8., -9., -6., 12., 17.])

kernel = kernels.gaussian_kernel(params={'sigma': 25., 'theta': 0.2})

# this is going to be a problem, the knots. We will need an adaptive approach because we don't know a priori
# how the knots will be distributed. Maybe even something like multigrid.
boundedGP = BoundedGP.BoundedGP(50, 0., 1., kernel, interpol_x, interpol_y, lowerbound=lowerbound, upperbound=upperbound, nugget_interpol=1e-3)

mean_knots, var_knots = boundedGP.statistics_prior()
#sample = boundedGP.sample_interpolation(sample_pts)

mean, var = boundedGP.statistics_interpolation()

boundedGP.sample_constrained_ET(x_sample=boundedGP.splines.splines_x)

visualize = vis.Visualize()

visualize.visualize_optim_methods(boundedGP)
visualize.plot_interpolated(boundedGP)
visualize.plot_constrained(boundedGP)
