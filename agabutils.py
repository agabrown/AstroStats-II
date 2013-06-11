"""
Utility methods for use with the Astrostats-II school exercises.
"""

import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import fmin

def inverseVariance(sigma):
  """
  Convert standard deviation to the inverse variance.

  Parameters
  ----------
  sigma - value of standard deviation

  Returns
  -------

  1/sigma^2
  """
  return 1.0/np.power(sigma,2.0)

def calculateHistogram(samples, nbins, discrete=False):
  """
  Calculate a histogram to approximate the posterior probability density of the variable monitored in the
  MCMC sampling of a Bayesian model.
 
  Parameters
  ----------
  
  samples  - array of sample values (1D numpy)
  nbins    - number of bins to use
  discrete - optional argument: if True the samples represent a discrete stochastic variable.
 
  Returns
  -------

  The histogram and the bin centres

  Example
  -------

  histogram, bincentres = calculateHistogram(samples, nbins)
  """
  if (discrete):
    minSample = int(np.min(samples))
    maxSample = int(np.max(samples))
    histo, binEdges = np.histogram(samples,bins=(maxSample-minSample),density=False)
    indices = (histo>0)
    histo=((1.0*histo)/np.sum(histo))[indices]
    binCentres = (binEdges[0:len(binEdges)-1])[indices]
  else:
    histo, binEdges = np.histogram(samples,bins=nbins,density=True)
    binCentres = binEdges[0:len(binEdges)-1]+np.diff(binEdges)/2.0
  return histo, binCentres

def kdeAndMap(y, bwmethod='scott'):
  """
  Provide a kernel density estimate of the distribution p(y) of variable y and also calculate the
  location of the maximum of p(y).

  Parameters
  ----------

  y - Array of values from which the KDE is to be made
  bwmethod - optional argument, passed to the bw_method argument of gaussian_kde()

  Returns
  -------

  Function representing the kernel density estimate (see documentation of scipy.stats.gaussian_kde) and
  the location of the maximum.

  Example
  -------

  yDensity, maximum = kernelDensityEstimate(y)
  """
  density = gaussian_kde(y, bw_method=bwmethod)
  maximum = fmin(lambda x: -1.0*density(x),np.median(y),maxiter=1000,ftol=0.0001)

  return density, maximum
