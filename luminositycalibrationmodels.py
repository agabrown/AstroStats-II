__all__ = ['UniformSpaceDensityGaussianLFBook', 'UniformSpaceDensityGaussianLF',
'UniformSpaceDensityGaussianLFemcee']
"""
Provide classes and methods the represent luminosity calibration models for use with PyMC.

Anthony Brown 2011-2013
"""

import numpy as np
from pymc import Uniform, deterministic, Normal, Model, InverseGamma
from extrastochastics import OneOverXFourth, OneOverX
from agabutils import inverseVariance
from scipy.stats import gamma

class UniformSpaceDensityGaussianLFBook:
  """
  A model for luminosity classification of stars. The stars are all assumed to be of a single class
  characterized by a mean absolute magnitude <M> and the variance on this magnitude sigma^2_M.
  Furthermore the stars are assumed to be distributed uniformly in the space around the sun. The upper
  and lower distance limits are fixed and the survey is assumed to be volume complete.

  The prior on the mean absolute magnitude is flat and the prior on the inverse variance tau of the
  absolute magnitude distribution is f(tau)=1/tau (i.e., f(sigma^2)=1/sigma^2, with tau=1/sigma^2). This
  is an appropriate non-informative prior for a scale parameter.
  """

  def __init__(self, simulatedSurvey, lowParallax, upParallax, minMeanAbsoluteMagnitude,
      maxMeanAbsoluteMagnitude, minTau, maxTau):
    """
    simulatedSurvey - a simulated survey object generated with one of the UniverseModels classes.
    lowParallax     - assumed lower limit on parallaxes (mas)
    upParallax      - assumed upper limit on parallaxes (mas)
    minMeanAbsoluteMagnitude - lower limit on prior distribution of mean absolute magnitude
    maxMeanAbsoluteMagnitude - upper limit on prior distribution of mean absolute magnitude
    minTau                   - lower limit on prior distribution of inverse variance
    maxTau                   - upper limit on prior distribution of inverse variance
    """
    self.simulatedSurvey=simulatedSurvey
    self.numberOfStarsInSurvey=self.simulatedSurvey.numberOfStarsInSurvey
    self.lowParallax=lowParallax
    self.upParallax=upParallax
    self.minMeanAbsoluteMagnitude=minMeanAbsoluteMagnitude
    self.maxMeanAbsoluteMagnitude=maxMeanAbsoluteMagnitude
    self.minTau=minTau
    self.maxTau=maxTau
    self.pyMCModel=Model(self._buildModel())

  def _buildModel(self):
    # Lower and upper parallax limits are fixed
    @deterministic(plot=False)
    def minParallax():
      """Lower limit on true parallax values [mas]."""
      return self.lowParallax

    @deterministic(plot=False)
    def maxParallax():
      """Upper limit on true parallax values [mas]."""
      return self.upParallax

    # Calculate initial guesses for the true parallaxes and absolute magnitudes.
    clippedObservedParallaxes=self.simulatedSurvey.observedParallaxes.clip(minParallax.value,
        maxParallax.value)
    initialAbsMagGuesses = (self.simulatedSurvey.observedMagnitudes +
        5.0*np.log10(clippedObservedParallaxes) - 10.0)
    meanAbsoluteMagnitudeGuess=initialAbsMagGuesses.mean()
    
    # Prior on mean absolute magnitude
    expMagInit=(self.maxMeanAbsoluteMagnitude-self.minMeanAbsoluteMagnitude)/2.0
    meanAbsoluteMagnitude = Uniform('meanAbsoluteMagnitude', lower=self.minMeanAbsoluteMagnitude,
        upper=self.maxMeanAbsoluteMagnitude, value=expMagInit)
    
    # Prior on absolute magnitude variance. Use f(tau)=1/tau, i.e., f(sigma^2)=1/sigma^2 (tau=1/sigma^2).
    # This non-informative prior is appropriate for a scale parameter.
    expTauInit=(self.maxTau-self.minTau)/(np.log(self.maxTau)-np.log(self.minTau))
    tauAbsoluteMagnitude = OneOverX('tauAbsoluteMagnitude', lower=self.minTau, upper=self.maxTau,
        value=expTauInit)
    
    # Prior on parallax. Uniform distribution of stars in space around the Sun.
    priorParallaxes=OneOverXFourth('priorParallaxes', lower=minParallax, upper=maxParallax,
        size=self.simulatedSurvey.numberOfStarsInSurvey)
    
    # Prior on absolute magnitude
    priorAbsoluteMagnitudes=Normal('priorAbsoluteMagnitudes', mu=meanAbsoluteMagnitude,
        tau=tauAbsoluteMagnitude, size=self.simulatedSurvey.numberOfStarsInSurvey)
    
    # Apparent magnitudes depend on the parallax and absolute magnitude.
    @deterministic(plot=False)
    def apparentMagnitudes(parallaxes=priorParallaxes, absoluteMagnitudes=priorAbsoluteMagnitudes):
      return absoluteMagnitudes-5.0*np.log10(parallaxes)+10.0
    
    # The likelihood of the data
    predictedParallaxes=Normal('predictedParallaxes',mu=priorParallaxes,
        tau=inverseVariance(self.simulatedSurvey.parallaxErrors),
        value=self.simulatedSurvey.observedParallaxes, observed=True)
    predictedApparentMagnitudes=Normal('predictedApparentMagnitudes',mu=apparentMagnitudes,
        tau=inverseVariance(self.simulatedSurvey.magnitudeErrors),
        value=self.simulatedSurvey.observedMagnitudes, observed=True)
    
    return locals()
    #self.pyMCModel=Model(set([minParallax, maxParallax, meanAbsoluteMagnitude, tauAbsoluteMagnitude,
    #  priorParallaxes, priorAbsoluteMagnitudes, apparentMagnitudes, predictedParallaxes,
    #  predictedApparentMagnitudes]))

class UniformSpaceDensityGaussianLF:
  """
  A model for luminosity classification of stars. The stars are all assumed to be of a single class
  characterized by a mean absolute magnitude <M> and the variance on this magnitude sigma^2_M.
  Furthermore the stars are assumed to be distributed uniformly in the space around the sun. The upper
  and lower distance limits are fixed and the survey is assumed to be volume complete.

  The prior on the mean absolute magnitude is flat and the prior on the inverse variance tau of the
  absolute magnitude distribution is f(tau)=InvGamma(tau,shape=alpha,scale=beta), i.e.
  f(sigma^2)=Gamma(sigma^2, shape=alpha, scale=1/beta), with tau=1/sigma^2.
  """

  def __init__(self, simulatedSurvey, lowParallax, upParallax, minMeanAbsoluteMagnitude,
      maxMeanAbsoluteMagnitude, shapeTau, scaleTau):
    """
    simulatedSurvey          - a simulated survey object generated with one of the UniverseModels classes.
    lowParallax              - assumed lower limit on parallaxes (mas)
    upParallax               - assumed upper limit on parallaxes (mas)
    minMeanAbsoluteMagnitude - lower limit on prior distribution of mean absolute magnitude
    maxMeanAbsoluteMagnitude - upper limit on prior distribution of mean absolute magnitude
    shapeTau                 - shape parameter of Inverse Gamma prior distribution of inverse variance
    scaleTau                 - scale parameter of Inverse Gamma prior distribution of inverse variance
    """
    self.simulatedSurvey=simulatedSurvey
    self.numberOfStarsInSurvey=self.simulatedSurvey.numberOfStarsInSurvey
    self.lowParallax=lowParallax
    self.upParallax=upParallax
    self.minMeanAbsoluteMagnitude=minMeanAbsoluteMagnitude
    self.maxMeanAbsoluteMagnitude=maxMeanAbsoluteMagnitude
    self.shapeTau=shapeTau
    self.scaleTau=scaleTau
    self.pyMCModel=Model(self._buildModel())

  def _buildModel(self):
    # Lower and upper parallax limits are fixed
    @deterministic(plot=False)
    def minParallax():
      """Lower limit on true parallax values [mas]."""
      return self.lowParallax

    @deterministic(plot=False)
    def maxParallax():
      """Upper limit on true parallax values [mas]."""
      return self.upParallax

    # Calculate initial guesses for the true parallaxes and absolute magnitudes.
    clippedObservedParallaxes=self.simulatedSurvey.observedParallaxes.clip(minParallax.value,
        maxParallax.value)
    initialAbsMagGuesses = (self.simulatedSurvey.observedMagnitudes +
        5.0*np.log10(clippedObservedParallaxes) - 10.0)
    meanAbsoluteMagnitudeGuess=initialAbsMagGuesses.mean()
    
    # Prior on mean absolute magnitude
    expMagInit=(self.maxMeanAbsoluteMagnitude-self.minMeanAbsoluteMagnitude)/2.0
    meanAbsoluteMagnitude = Uniform('meanAbsoluteMagnitude', lower=self.minMeanAbsoluteMagnitude,
        upper=self.maxMeanAbsoluteMagnitude, value=expMagInit)
    
    # Prior on absolute magnitude variance.
    tauInit=self.scaleTau/(self.shapeTau+1)
    tauAbsoluteMagnitude = InverseGamma('tauAbsoluteMagnitude', self.shapeTau, self.scaleTau, value=tauInit)
    
    # Prior on parallax. Uniform distribution of stars in space around the Sun.
    priorParallaxes=OneOverXFourth('priorParallaxes', lower=minParallax, upper=maxParallax,
        size=self.simulatedSurvey.numberOfStarsInSurvey)
    
    # Prior on absolute magnitude
    priorAbsoluteMagnitudes=Normal('priorAbsoluteMagnitudes', mu=meanAbsoluteMagnitude,
        tau=tauAbsoluteMagnitude, size=self.simulatedSurvey.numberOfStarsInSurvey)
    
    # Apparent magnitudes depend on the parallax and absolute magnitude.
    @deterministic(plot=False)
    def apparentMagnitudes(parallaxes=priorParallaxes, absoluteMagnitudes=priorAbsoluteMagnitudes):
      return absoluteMagnitudes-5.0*np.log10(parallaxes)+10.0
    
    # The likelihood of the data
    predictedParallaxes=Normal('predictedParallaxes',mu=priorParallaxes,
        tau=inverseVariance(self.simulatedSurvey.parallaxErrors),
        value=self.simulatedSurvey.observedParallaxes, observed=True)
    predictedApparentMagnitudes=Normal('predictedApparentMagnitudes',mu=apparentMagnitudes,
        tau=inverseVariance(self.simulatedSurvey.magnitudeErrors),
        value=self.simulatedSurvey.observedMagnitudes, observed=True)
    
    return locals()

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
  lnPosterior=lnPosterior-np.log(posteriorDict['muHigh']-posteriorDict['muLow'])

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
  lnPosterior=lnPosterior-np.log(posteriorDict['muHigh']-posteriorDict['muLow'])

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
