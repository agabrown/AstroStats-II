__all__ = ['simParallaxesConstantSpaceDensity', 'simGaussianAbsoluteMagnitude', 'UniformDistributionSingleLuminosity']
"""
universemodels.py

Provide classes and methods for simulating the distribution of stars in space, luminosity, etc.

Anthony Brown 2011-2013
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from sys import stderr

# Configure matplotlib
rc('text', usetex=True)
rc('font', family='serif', size=16)
rc('xtick.major', size='12')
rc('xtick.minor', size='6')
rc('ytick.major', size='12')
rc('ytick.minor', size='6')
rc('lines', linewidth=2)
rc('axes', linewidth=2)

def _gaussian(t, mean=0.0, stdev=1.0):
  """
  Implements the standard normal distribution.

  Parameters
  ----------

  t - Abscissa value for which the value of the distribution is desired.

  Keywords
  --------

  mean  - Mean of the distribution
  stdev - Standard Deviation of the distribution
  
  Returns
  -------
  
  1.0/(sqrt(2.0*pi)*stdev)*exp(-0.5*((t-mean)/stdev)^2)
  """
  z=(t-mean)/stdev
  return np.exp(-0.5*z*z)/(np.sqrt(2.0*np.pi)*stdev)

def simParallaxesConstantSpaceDensity(numStars, minParallax, maxParallax):
  """
  Simulate parallaxes for stars distributed uniformly in space around the Sun.

  Parameters
  ----------

  numStars - number of stars to simulate
  minParallax - lower limit of the true parallax (volume limit of survey, arbitrary units)
  maxParallax - upper limit of the true parallax (closest possible star, arbitrary units)

  Returns
  -------

  Vector of parallax values
  """
  x=np.random.random_sample(numStars)
  minPMinThird=np.power(minParallax,-3.0)
  maxPMinThird=np.power(maxParallax,-3.0)
  return np.power(minPMinThird-1.0*x*(minPMinThird-maxPMinThird),-1.0/3.0)

def simGaussianAbsoluteMagnitude(numStars, mean, stddev):
  """
  Simulate absolute magnitudes following a Gaussian distribution.

  Parameters
  ----------

  numStars - number of stars to simulate
  mean - mean of the distribution
  stddev - standard deviation of the distribution

  Returns
  -------

  Vector of magnitudes.
  """
  return mean+np.random.randn(numStars)*stddev

class UniformDistributionSingleLuminosity:
  """
  Simulate stars distributed uniformly in space around the sun. The stars all
  have the same luminosity drawn from a Gaussian distribution.
  """

  def __init__(self, numberOfStars, minParallax, maxParallax, meanAbsoluteMagnitude,
      varianceAbsoluteMagnitude, surveyLimit=np.Inf):
    """
    Class constructor/initializer.

    Parameters
    ----------

    numberOfStars             - number of stars to simulate
    minParallax               - lower limit of the true parallax (volume limit of survey, mas)
    maxParallax               - upper limit of the true parallax (closest possible star, mas)
    meanAbsoluteMagnitude     - Mean of Gaussian absolute magnitude distribution
    varianceAbsoluteMagnitude - Variance of Gaussian absolute magnitude distribution

    Keywords
    --------

    surveyLimit - Apparent magnitude limit of the survey (default: no limit)
    """
    self.numberOfStars=numberOfStars
    self.numberOfStarsInSurvey=numberOfStars
    self.minParallax=minParallax
    self.maxParallax=maxParallax
    self.meanAbsoluteMagnitude=meanAbsoluteMagnitude
    self.varianceAbsoluteMagnitude=varianceAbsoluteMagnitude
    self.apparentMagnitudeLimit=surveyLimit
    self.parallaxErrorNormalizationMagnitude=5.0
    self.parallaxErrorSlope=0.2
    self.parallaxErrorCalibrationFloor=0.2
    self.magnitudeErrorNormalizationMagnitude=5.0
    self.magnitudeErrorSlope=0.006
    self.magnitudeErrorCalibrationFloor=0.001

  def setRandomNumberSeed(self, seed):
    """
    (Re-)Set the random number seed for the simulations.

    Parameters
    ----------

    seed - Value of random number seed
    """
    np.random.seed(seed)

  def _generateParallaxErrors(self):
    """
    Generate the parallax errors according to an ad-hoc function of parallax error as a function of
    magnitude.
    """
    errors = (np.power(10.0,0.2*(self.apparentMagnitudes -
      self.parallaxErrorNormalizationMagnitude))*self.parallaxErrorSlope)
    indices = (errors < self.parallaxErrorCalibrationFloor)
    errors[indices] = self.parallaxErrorCalibrationFloor
    return errors

  def _generateApparentMagnitudeErrors(self):
    """
    Generate the apparent magnitude errors to an ad-hoc function of magnitude error as a function of
    magnitude.
    """
    errors = (np.power(10.0,0.2*(self.apparentMagnitudes -
      self.magnitudeErrorNormalizationMagnitude))*self.magnitudeErrorSlope)
    indices = (errors < self.magnitudeErrorCalibrationFloor)
    errors[indices] = self.magnitudeErrorCalibrationFloor
    return errors

  def _applyApparentMagnitudeLimit(self):
    """
    Apply the apparent magnitude limit to the simulated survey.
    """
    indices=(self.observedMagnitudes <= self.apparentMagnitudeLimit)
    self.trueParallaxes=self.trueParallaxes[indices]
    self.absoluteMagnitudes=self.absoluteMagnitudes[indices]
    self.apparentMagnitudes=self.apparentMagnitudes[indices]
    self.parallaxErrors=self.parallaxErrors[indices]
    self.magnitudeErrors=self.magnitudeErrors[indices]
    self.observedParallaxes=self.observedParallaxes[indices]
    self.observedMagnitudes=self.observedMagnitudes[indices]
    self.numberOfStarsInSurvey=len(self.observedMagnitudes)

  def generateObservations(self):
    """
    Generate the simulated observations.
    """
    self.trueParallaxes = simParallaxesConstantSpaceDensity(self.numberOfStars, self.minParallax,
        self.maxParallax)
    self.absoluteMagnitudes = simGaussianAbsoluteMagnitude(self.numberOfStars, self.meanAbsoluteMagnitude,
        np.sqrt(self.varianceAbsoluteMagnitude))
    self.apparentMagnitudes = self.absoluteMagnitudes-5.0*np.log10(self.trueParallaxes)+10.0
    self.parallaxErrors = self._generateParallaxErrors()
    self.magnitudeErrors = self._generateApparentMagnitudeErrors()
    self.observedParallaxes=np.copy(self.trueParallaxes)
    self.observedMagnitudes=np.copy(self.apparentMagnitudes)
    for i in range(self.numberOfStars):
      self.observedParallaxes[i] = self.observedParallaxes[i]+np.random.randn()*self.parallaxErrors[i]
      self.observedMagnitudes[i] = self.observedMagnitudes[i]+np.random.randn()*self.magnitudeErrors[i]
    self._applyApparentMagnitudeLimit()

  def showSurveyStatistics(self, pdfFile=None, pngFile=None):
    """
    Produce a plot with the survey statistics.

    Keywords
    --------

    pdfFile - Name of optional PDF file in which to save the plot.
    pngFile - Name of optional PNG file in which to save the plot.
    """
    parLimitPlot=50.0
    fig = plt.figure(figsize=(12,8.5))
    ax = fig.add_subplot(2,2,1)
    try:
      n, bins, patches = plt.hist(self.observedParallaxes, 50,
          normed=1,range=(self.observedParallaxes.min(),parLimitPlot),
          histtype='step', edgecolor='k', facecolor='grey', label='observed', fill=True)
    except AttributeError:
      stderr.write("You have not generated the observations yet!\n")
      return
    n, bins, patches = plt.hist(self.trueParallaxes, 50, normed=1,
        range=(self.minParallax,parLimitPlot), histtype='step', color='k', label='true')
    minPMinThird=np.power(self.minParallax,-3.0)
    maxPMinThird=np.power(parLimitPlot,-3.0)
    x=np.linspace(self.minParallax,parLimitPlot,1001)
    plt.plot(x,3.0*np.power(x,-4.0)/(minPMinThird-maxPMinThird),'k-', label='model')
    plt.xlabel("$\\varpi$ [mas]")
    plt.ylabel("$P(\\varpi)$")
    plt.ylim(0,0.15)
    leg=plt.legend(loc=(0.05,0.55), handlelength=1.0)
    for t in leg.get_texts():
      t.set_fontsize(14)
  
    ax = fig.add_subplot(2,2,2)
    n, bins, patches = plt.hist(self.absoluteMagnitudes, 50, normed=1, histtype='step', fill=True,
        facecolor='grey', edgecolor='k')
    x=0.5*(bins[1:]+bins[:-1])
    stddevAbsMagnitude=np.sqrt(self.varianceAbsoluteMagnitude)
    plt.plot(x, _gaussian(x,mean=self.meanAbsoluteMagnitude, stdev=stddevAbsMagnitude), '-k')
    plt.xlabel("$M$")
    plt.ylabel("$P(M)$")
    plt.ylim((0.0,n.max()*1.03))
  
    ax = fig.add_subplot(2,2,3)
    m, mbins, mpatches = ax.hist(self.apparentMagnitudes, 50, normed=1,
        histtype='step', fill=True, label='true', facecolor='grey', edgecolor='k')
    n, bins, patches = ax.hist(self.observedMagnitudes, 50, normed=1, histtype='step', color='k',
        label='observed')
    plt.xlabel("$m$")
    plt.ylabel("$P(m)$")
    plt.ylim((0.0,np.array([n.max(),m.max()]).max()*1.03))
    leg=plt.legend(loc=(0.05,0.7), handlelength=1.0)
    for t in leg.get_texts():
      t.set_fontsize(14)
  
    ax = fig.add_subplot(2,2,4)
    plt.semilogy(self.observedMagnitudes,self.parallaxErrors,'.k',alpha=0.5, label="$\\sigma_\\varpi$ [mas]$")
    plt.semilogy(self.observedMagnitudes,self.magnitudeErrors,'+k',alpha=1.0, label="$\\sigma_m$")
    plt.xlabel("$m_\\mathrm{o}$")
    plt.ylabel("$\\sigma$")
    #plt.ylim((8.0e-4,10.0))
    leg=plt.legend(loc=(0.55,0.05), numpoints=1, handlelength=0.5, markerscale=1.5)
    for t in leg.get_texts():
      t.set_fontsize(14)
  
    plt.suptitle("Simulated survey statistics: $N_\\mathrm{stars}"+"={0}".format(self.numberOfStars)+"$, ${0}".format(self.minParallax)+"\\leq\\varpi\\leq{0}".format(self.maxParallax)+"$, $\\langle M\\rangle={0}".format(self.meanAbsoluteMagnitude)+"$, $\\sigma^2_M={0}".format(self.varianceAbsoluteMagnitude)+"$")
    
    if (pdfFile != None):
      plt.savefig(pdfFile)
    if (pngFile != None):
      plt.savefig(pngFile)
    plt.show()
