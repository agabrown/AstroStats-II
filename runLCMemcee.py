"""
Run the MCMC sampling of the luminosity calibration model using the emcee package by Dan Foreman-Mackey.

Here the Gamma-prior on the variance hyper-parameter sigma^2_M is used.
"""
import numpy as np
import emcee
import scipy.optimize

import universemodels as U
from luminositycalibrationmodels import UniformSpaceDensityGaussianLFemcee
from agabutils import inverseVariance, kdeAndMap

import matplotlib.pyplot as plt
import argparse
from matplotlib import rc, cm
from time import time as now
from scipy.stats import gamma, gaussian_kde

np.seterr(invalid='raise')

# Configure matplotlib
rc('text', usetex=True)
rc('font', family='serif', size=16)
rc('xtick.major', size='12')
rc('xtick.minor', size='6')
rc('ytick.major', size='12')
rc('ytick.minor', size='6')
rc('lines', linewidth=2)
rc('axes', linewidth=2)

def printSamplingStats(duration, samplerUsed, numStars):
  """
  Print out some MCMC sampling stats in a readable form.

  Parameters
  ----------

  duration - Time taken for the MCMC sampling.
  samplerUsed - The emcee.EnsembleSampler used.
  numStars - Number of stars in simulated survey.
  """
  print '                Time (s): {v:.2f}'.format(v=duration)
  print 'Median acceptance fraction: {v:.2f}'.format(v=np.median(samplerUsed.acceptance_fraction))
  print ('Acceptance fraction IQR: {low:.2f} -- {up:.2f}'.format(low=np.percentile(samplerUsed.acceptance_fraction,25),
    up=np.percentile(samplerUsed.acceptance_fraction,75)))
  correlationTimes = samplerUsed.acor
  print 'Autocorrelation times: '
  print '  Mean absolute magnitude: {v:.2f}'.format(v=correlationTimes[0])
  print '  Variance absolute magnitude: {v:.2f}'.format(v=correlationTimes[1])
  print '  Median for parallaxes: {v:.2f}'.format(v=np.median(correlationTimes[2:numStars+2]))
  print '  Median for magnitudes: {v:.2f}'.format(v=np.median(correlationTimes[numStars+2:]))

def runMCMCmodel(args):
  """
  Simulate the survey data and run the MCMC luminosity calibration model.

  Parameters
  ----------

  args - Command line arguments
  """
  mcmcParams=args['mcmcString']
  surveyParams=args['surveyString']
  priorParams=args['priorsString']

  maxIter, burnIter, thinFactor, walkerFactor = [int(par) for par in mcmcParams]
  minParallax, maxParallax, meanAbsoluteMagnitude, varianceAbsoluteMagnitude =[float(par) for par in surveyParams[1:5]]
  meanAbsMagLow, meanAbsMagHigh, varianceShape, varianceScale  = [float(par) for par in priorParams]
  if surveyParams[5] == 'Inf':
    magLim = np.Inf
  else:
    magLim = float(surveyParams[5])

  simulatedSurvey=U.UniformDistributionSingleLuminosity(int(surveyParams[0]), float(surveyParams[1]),
      float(surveyParams[2]), float(surveyParams[3]), float(surveyParams[4]), surveyLimit=magLim)
  #simulatedSurvey.setRandomNumberSeed(53949896)
  simulatedSurvey.generateObservations()
  numberOfStarsInSurvey=simulatedSurvey.numberOfStarsInSurvey

  # Calculate initial guesses for the true parallaxes and absolute magnitudes of the stars.
  clippedObservedParallaxes=simulatedSurvey.observedParallaxes.clip(minParallax, maxParallax)
  initialAbsMagGuesses=simulatedSurvey.observedMagnitudes+5.0*np.log10(clippedObservedParallaxes)-10.0

  # Initial guesses for hyper parameters (mean absolute magnitude and sigma^2)
  meanAbsoluteMagnitudeGuess=initialAbsMagGuesses.mean()
  varianceInit=varianceScale*(varianceShape-1)
  
  initialParameters = np.concatenate((np.array([meanAbsoluteMagnitudeGuess, varianceInit]),
    clippedObservedParallaxes, initialAbsMagGuesses))
  
  # Parameters for emcee ln-posterior function
  posteriorDict = {'minParallax':minParallax, 'maxParallax':maxParallax, 'muLow':meanAbsMagLow,
  'muHigh':meanAbsMagHigh, 'varShape':varianceShape, 'varScale':varianceScale}
  observations = np.concatenate((simulatedSurvey.observedParallaxes, simulatedSurvey.observedMagnitudes))
  observationalErrors=inverseVariance(np.concatenate((simulatedSurvey.parallaxErrors, 
    simulatedSurvey.magnitudeErrors)))
  
  # MCMC sampler parameters
  ndim = 2*numberOfStarsInSurvey+2
  nwalkers = walkerFactor*ndim
  
  # Generate initial positions for each walker
  initialPositions=[np.empty((ndim)) for i in xrange(nwalkers)]
  initialPositions[0]=initialParameters
  for i in xrange(nwalkers-1):
    ranMeanAbsMag=np.random.rand()*(meanAbsMagHigh-meanAbsMagLow)+meanAbsMagLow
    ranVariance=gamma.rvs(varianceShape,scale=varianceScale)
    ranParallaxes=np.zeros_like(clippedObservedParallaxes)
    ranParallaxes=clippedObservedParallaxes+simulatedSurvey.parallaxErrors*np.random.randn(numberOfStarsInSurvey)
    ranAbsMag=np.sqrt(ranVariance)*np.random.randn(numberOfStarsInSurvey)+ranMeanAbsMag
    initialPositions[i+1]=np.concatenate((np.array([ranMeanAbsMag, ranVariance]),
      ranParallaxes.clip(minParallax, maxParallax), ranAbsMag))
  
  print '{:*^30}'.format('Building sampler')
  sampler = emcee.EnsembleSampler(nwalkers, ndim, UniformSpaceDensityGaussianLFemcee, threads=4,
      args=[posteriorDict, observations, observationalErrors])
  # burn-in
  print '{:*^30}'.format('Burn in')
  start = now()
  pos,prob,state = sampler.run_mcmc(initialPositions, burnIter)
  print '{:*^30}'.format('Finished burning')
  printSamplingStats(now()-start, sampler, numberOfStarsInSurvey)
  print
  # final chain
  sampler.reset()
  print '{:*^30}'.format('Start sampling')
  start = now()
  sampler.run_mcmc(pos, maxIter, rstate0=state, thin=thinFactor)
  print '{:*^30}'.format('Finished sampling')
  printSamplingStats(now()-start, sampler, numberOfStarsInSurvey)

  # Extract the samples of the posterior distribution
  chain = sampler.flatchain

  # Point estimates of mean Absolute Magnitude and its standard deviation.
  meanAbsoluteMagnitudeSamples = chain[:,0].flatten()
  varAbsoluteMagnitudeSamples = chain[:,1].flatten()
  estimatedAbsMag=meanAbsoluteMagnitudeSamples.mean()
  errorEstimatedAbsMag=meanAbsoluteMagnitudeSamples.std()
  estimatedVarMag=varAbsoluteMagnitudeSamples.mean()
  errorEstimatedVarMag=varAbsoluteMagnitudeSamples.std()
  print "emcee estimates"
  print "     mu_M = {:4.2f} +/- {:4.2f}".format(estimatedAbsMag, errorEstimatedAbsMag)
  print "sigma^2_M = {:4.2f} +/- {:4.2f}".format(estimatedVarMag, errorEstimatedVarMag)
  
  # Plot results
  
  # Kernel density estimate of posterior distributions of mu_M and sigma^2_M, also obtain maximum a
  # posteriori estimate for these quantitities.
  muDensity, mapValueMu = kdeAndMap(meanAbsoluteMagnitudeSamples)
  varDensity, mapValueVar = kdeAndMap(varAbsoluteMagnitudeSamples)
  
  fig=plt.figure(figsize=(12,8.5))
  fig.add_subplot(2,2,1)
  x = np.linspace(meanAbsoluteMagnitudeSamples.min(), meanAbsoluteMagnitudeSamples.max(), 500)
  plt.plot(x,muDensity(x),'k-')
  plt.axvline(meanAbsoluteMagnitude, linewidth=2, color="red")
  plt.xlabel("$\\mu_M$")
  plt.ylabel("$P(\\mu_M)$")
  
  fig.add_subplot(2,2,2)
  x = np.linspace(varAbsoluteMagnitudeSamples.min(), varAbsoluteMagnitudeSamples.max(), 500)
  plt.plot(x,varDensity(x),'k-')
  plt.axvline(varianceAbsoluteMagnitude, linewidth=2, color="red")
  plt.xlabel("$\\sigma^2_M$")
  plt.ylabel("$P(\\sigma^2_M)$")
  
  fig.add_subplot(2,2,3)
  plt.hexbin(meanAbsoluteMagnitudeSamples,varAbsoluteMagnitudeSamples, C=None, bins='log', cmap=cm.gray_r)
  plt.xlabel("$\\mu_M$")
  plt.ylabel("$\\sigma^2_M$")
  
  plt.figtext(0.55,0.4,"$\\widetilde{{\\mu_M}}={:4.2f}\\pm{:4.2f}$".format(estimatedAbsMag,
    errorEstimatedAbsMag),ha='left')
  plt.figtext(0.75,0.4,"$\\mathrm{{MAP}}(\\widetilde{{\\mu_M}})={:4.2f}$".format(mapValueMu[0]))
  plt.figtext(0.55,0.35,"$\\widetilde{{\\sigma^2_M}}={:4.2f}\\pm{:4.2f}$".format(estimatedVarMag,
    errorEstimatedVarMag), ha='left')
  plt.figtext(0.75,0.35,"$\\mathrm{{MAP}}(\\widetilde{{\\sigma^2_M}})={:4.2f}$".format(mapValueVar[0]))
  
  titelA=("$N_\\mathrm{{stars}}={}$, True values: $\\mu_M={}$, $\\sigma^2_M={}$".format(numberOfStarsInSurvey, meanAbsoluteMagnitude, varianceAbsoluteMagnitude))
  titelB=("Iterations = {}, Burn = {}, Thin = {}".format(maxIter, burnIter, thinFactor))
  plt.suptitle(titelA+"\\quad\\quad "+titelB)

  titelC=[]
  titelC.append("MCMC sampling with emcee") 
  titelC.append("$N_\\mathrm{{walkers}}={}$, $N_\\mathrm{{dim}}={}".format(nwalkers, ndim))
  plt.figtext(0.55,0.15,titelC[0],horizontalalignment='left')
  plt.figtext(0.60,0.10,titelC[1],horizontalalignment='left')

  priorInfo=[]
  priorInfo.append("Prior on $\\mu_M$: flat $\\quad{}<\\mu_M<{}$".format(meanAbsMagLow, meanAbsMagHigh))
  priorInfo.append("Prior on $\\sigma^2_M$: $\\Gamma(\\sigma^2_M | k={},\\theta={})$".format(varianceShape, varianceScale))

  plt.figtext(0.55,0.25,priorInfo[0],horizontalalignment='left')
  plt.figtext(0.55,0.20,priorInfo[1],horizontalalignment='left')
  
  if (args['pdfOutput']):
    plt.savefig('luminosityCalibrationResultsEmcee.pdf')
  elif (args['pngOutput']):
    plt.savefig('luminosityCalibrationResultsEmcee.png')
  elif (args['epsOutput']):
    plt.savefig('luminosityCalibrationResultsEmcee.eps')
  else:
    plt.show()

def parseCommandLineArguments():
  """
  Set up command line parsing.
  """
  parser = argparse.ArgumentParser("Run the MCMC sampling of the luminosity calibration model using emcee.")
  parser.add_argument("--mcmc", dest="mcmcString", nargs=4,
      help="""White-space-separated list of MCMC parameters:
              (1) number of MCMC iterations,
              (2) number of initial iterations to discard as burn-in,
              (3) thinning factor
              (4) walker factor (nwalkers = walker_factor*ndim; ndim=2*nstars+2)""")
  parser.add_argument("--survey", dest="surveyString", nargs=6,
      help="""White-space-separated list of survey parameters:
              (1) number of stars,
              (2) lower limit parallaxes [mas],
              (3) upper limit parallaxes [mas],
              (4) mean absolute magnitude,
              (5) variance of absolute magnitudes
              (6) apparent magnitude limit of survey (Inf allowed)""")
  parser.add_argument("--priors", dest="priorsString", nargs=4,
      help="""White-space-separated list of prior ranges of luminosity distribution parameters:
              (1) lower limit of uniform prior for mean absolute magnitude,
              (2) upper limit of uniform prior for mean absolute magnitude,
              (3) shape parameter for Gamma-prior on the variance of the absolute magnitudes,
              (4) scale parameter for Gamma-prior on the variance of the absolute magnitudes""")
  parser.add_argument("-p", action="store_true", dest="pdfOutput", help="Make PDF plot")
  parser.add_argument("-e", action="store_true", dest="epsOutput", help="Make EPS plot")
  parser.add_argument("-g", action="store_true", dest="pngOutput", help="Make PNG plot")

  return vars(parser.parse_args())

if  __name__ in ('__main__'):
  args = parseCommandLineArguments()
  runMCMCmodel(args)
