__all__ = ['UniformSpaceDensityGaussianLFemcee', 'UniformSpaceDensityGaussianLFBookemcee']
"""
Provide classes and methods the represent luminosity calibration models for use with emcee.

Anthony Brown 2011-2013
"""

import numpy as np
from agabutils import inverseVariance, randOneOverXFourth, randOneOverX
from scipy.stats import gamma

def UniformSpaceDensityGaussianLFBookemcee(posWalker, posteriorDict, observations, observationalErrors):
  """
  A model for luminosity classification of stars. The stars are all assumed to be of a single class
  characterized by a mean absolute magnitude <M> and the variance on this magnitude sigma^2_M.
  Furthermore the stars are assumed to be distributed uniformly in the space around the sun. The upper
  and lower distance limits are fixed and the survey is assumed to be volume complete.

  The prior on the mean absolute magnitude is flat and the prior on the variance var of the absolute
  magnitude distribution is f(var)=1/var. This is an appropriate non-informative prior for a scale
  parameter.
 
  **** This model is for the emcee python package. ****

  Parameters
  ----------

  posWalker - The Emcee walker position. The first two entries are the hyper-parameters, mu = Mean
              Absolute Magnitude and var = Variance, the next set of parameters are the true parallaxes
              and absolute magnitudes, respectively, of the stars in the survey (so 2 times the number of
              stars in the survey.)
  posteriorDict - Dictionary with posterior probability density function parameters:
    minParallax : Lower limit on parallax distribution
    maxParallax : Upper limit on parallax distribution
    muLow       : Lower limit of uniform prior on mu.
    muHigh      : Upper limit of uniform prior on mu.
    varLow      : Lower limit for the 1/x prior on the variance.
    varHigh     : Upper limit for the 1/x prior on the variance.
  observations - Vector of observed parallaxes and apparent magnitudes (in that order).
  observationalErrors - Vector of errors (as inverse variances) on observed parallaxes and apparent
                        magnitudes (in that order).

  Returns
  -------

  Natural logarithm of the posterior probability density at the position of the walker.
  """
  lnPosterior=0.0
  numStarsInSurvey=(len(posWalker)-2)/2
  meanAbsMag=posWalker[0]
  if (meanAbsMag<posteriorDict['muLow'] or meanAbsMag>posteriorDict['muHigh']):
    return -np.inf
  #lnPosterior=lnPosterior-np.log(posteriorDict['muHigh']-posteriorDict['muLow'])

  variance=posWalker[1]
  if (variance<posteriorDict['varLow'] or variance>posteriorDict['varHigh']):
    return -np.inf
  lnPosterior=lnPosterior - np.log(variance)

  parallaxes=posWalker[2:numStarsInSurvey+2]
  if (np.any(parallaxes < posteriorDict['minParallax']) or np.any(parallaxes >
    posteriorDict['maxParallax'])):
    return -np.inf
  lnPosterior = lnPosterior - 4.0*np.sum(np.log(parallaxes))

  absoluteMagnitudes=posWalker[numStarsInSurvey+2:]
  apparentMagnitudes=absoluteMagnitudes-5.0*np.log10(parallaxes)+10.0

  y=np.concatenate((parallaxes,apparentMagnitudes,absoluteMagnitudes))
  inv_var=np.diagflat(np.concatenate((observationalErrors, 1.0/np.repeat(variance,numStarsInSurvey))))
  means=np.concatenate((observations, np.repeat(meanAbsMag,numStarsInSurvey)))
  diff=y-means
  lnPosterior = lnPosterior - 0.5*np.dot(diff,np.dot(inv_var,diff))
  lnPosterior = lnPosterior - 0.5*numStarsInSurvey*np.log(variance)
  return lnPosterior

def UniformSpaceDensityGaussianLFemcee(posWalker, posteriorDict, observations, observationalErrors):
  """
  A model for luminosity classification of stars. The stars are all assumed to be of a single class
  characterized by a mean absolute magnitude <M> and the variance on this magnitude sigma^2_M.
  Furthermore the stars are assumed to be distributed uniformly in the space around the sun. The upper
  and lower distance limits are fixed and the survey is assumed to be volume complete.

  The prior on the mean absolute magnitude is flat and the prior on the variance var of the
  absolute magnitude distribution is f(var)=Gamma(var|shape=k,scale=theta).
 
  **** This model is for the emcee python package. ****

  Parameters
  ----------

  posWalker - The Emcee walker position. The first two entries are the hyper-parameters, mu = Mean
              Absolute Magnitude and var = Variance, the next set of parameters are the true parallaxes
              and absolute magnitudes, respectively, of the stars in the survey (so 2 times the number of
              stars in the survey.)
  posteriorDict - Dictionary with posterior probability density function parameters:
    minParallax : Lower limit on parallax distribution
    maxParallax : Upper limit on parallax distribution
    muLow       : Lower limit of uniform prior on mu.
    muHigh      : Upper limit of uniform prior on mu.
    varShape    : Shape parameter for Gamma prior on the variance.
    varScale    : Scale parameter for Gamma prior on the variance.
  observations - Vector of observed parallaxes and apparent magnitudes (in that order).
  observationalErrors - Vector of errors (as inverse variances) on observed parallaxes and apparent
                        magnitudes (in that order).

  Returns
  -------

  Natural logarithm of the posterior probability density at the position of the walker.
  """
  lnPosterior=0.0
  numStarsInSurvey=(len(posWalker)-2)/2
  meanAbsMag=posWalker[0]
  if (meanAbsMag<posteriorDict['muLow'] or meanAbsMag>posteriorDict['muHigh']):
    return -np.inf
  #lnPosterior=lnPosterior-np.log(posteriorDict['muHigh']-posteriorDict['muLow'])

  variance=posWalker[1]
  if (variance<0.0):
    return -np.inf
  lnPosterior=lnPosterior + gamma.logpdf(variance, posteriorDict['varShape'],
      scale=posteriorDict['varScale'])

  parallaxes=posWalker[2:numStarsInSurvey+2]
  if (np.any(parallaxes < posteriorDict['minParallax']) or np.any(parallaxes >
    posteriorDict['maxParallax'])):
    return -np.inf
  lnPosterior = lnPosterior - 4.0*np.sum(np.log(parallaxes))

  absoluteMagnitudes=posWalker[numStarsInSurvey+2:]
  apparentMagnitudes=absoluteMagnitudes-5.0*np.log10(parallaxes)+10.0

  y=np.concatenate((parallaxes,apparentMagnitudes,absoluteMagnitudes))
  inv_var=np.diagflat(np.concatenate((observationalErrors, 1.0/np.repeat(variance,numStarsInSurvey))))
  means=np.concatenate((observations, np.repeat(meanAbsMag,numStarsInSurvey)))
  diff=y-means
  lnPosterior = lnPosterior - 0.5*np.dot(diff,np.dot(inv_var,diff))
  lnPosterior = lnPosterior - 0.5*numStarsInSurvey*np.log(variance)
  return lnPosterior
