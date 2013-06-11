"""
Run the MCMC sampling of the luminosity calibration model using the emcee package by Dan Foreman-Mackey.

Here the 1/x prior on the variance hyper-parameter sigma^2_M is used.
"""

import numpy as np
import emcee
import scipy.optimize

import universemodels as U
from luminositycalibrationmodels import UniformSpaceDensityGaussianLFBookemcee
from agabutils import inverseVariance, kdeAndMap
from extrastochastics import random_oneOverXFourth, random_oneOverX

import matplotlib.pyplot as plt
import argparse
from matplotlib import rc, cm
from time import time as now
from scipy.stats import gamma

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

  maxIter=int(mcmcParams[0])
  burnIter=int(mcmcParams[1])
  thinFactor=int(mcmcParams[2])
  walkerFactor=int(mcmcParams[3])

  minParallax=float(surveyParams[1])
  maxParallax=float(surveyParams[2])
  meanAbsoluteMagnitude=float(surveyParams[3])
  varianceAbsoluteMagnitude=float(surveyParams[4])

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
  meanAbsoluteMagnitudeGuess=initialAbsMagGuesses.mean()

  # Initial guesses for hyper parameters (mean absolute magnitude and sigma^2)
  #
  # Mean absolute magnitude uniform on (meanAbsMagLow, meanAbsMagHigh)
  meanAbsMagLow=float(priorParams[0])
  meanAbsMagHigh=float(priorParams[1])
  # Variance has 1/x distribution with lower and upper limit as prior
  varianceLow=float(priorParams[2])
  varianceHigh=float(priorParams[3])
  varianceInit=(varianceHigh-varianceLow)/(np.log(varianceHigh)-np.log(varianceLow))
  
  initialParameters = np.concatenate((np.array([meanAbsoluteMagnitudeGuess, varianceInit]),
    clippedObservedParallaxes, initialAbsMagGuesses))
  
  # Parameters for emcee ln-posterior function
  posteriorDict = {'minParallax':minParallax, 'maxParallax':maxParallax, 'muLow':meanAbsMagLow,
  'muHigh':meanAbsMagHigh, 'varLow':varianceLow, 'varHigh':varianceHigh}
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
    ranVariance=random_oneOverX(varianceLow,varianceHigh,1)
    ranParallaxes=np.zeros_like(clippedObservedParallaxes)
    for j in xrange(numberOfStarsInSurvey):
      #if (i<nwalkers/2):
      ranParallaxes[j]=clippedObservedParallaxes[j]+simulatedSurvey.parallaxErrors[j]*np.random.randn()
      #else:
      #  ranParallaxes[j]=random_oneOverXFourth(minParallax,maxParallax,1)
    ranAbsMag=np.sqrt(ranVariance)*np.random.randn(numberOfStarsInSurvey)+ranMeanAbsMag
    initialPositions[i+1]=np.concatenate((np.array([ranMeanAbsMag, ranVariance]),
      ranParallaxes.clip(minParallax, maxParallax), ranAbsMag))
  
  print '** Building sampler **'
  sampler = emcee.EnsembleSampler(nwalkers, ndim, UniformSpaceDensityGaussianLFBookemcee, threads=4,
      args=[posteriorDict, observations, observationalErrors])
  # burn-in
  print '** Burn in **'
  start = now()
  pos,prob,state = sampler.run_mcmc(initialPositions, burnIter)
  print '** Finished burning in **'
  print '                Time (s): ',now()-start
  print 'Median acceptance fraction: ',np.median(sampler.acceptance_fraction)
  print ('Acceptance fraction IQR: {0}'.format(np.percentile(sampler.acceptance_fraction,25)) +
      ' -- {0}'.format(np.percentile(sampler.acceptance_fraction,75)))
  correlationTimes = sampler.acor
  print 'Autocorrelation times: '
  print '  Mean absolute magnitude: ', correlationTimes[0]
  print '  Variance absolute magnitude: ', correlationTimes[1]
  print '  Median for parallaxes: ', np.median(correlationTimes[2:numberOfStarsInSurvey+2])
  print '  Median for magnitudes: ', np.median(correlationTimes[numberOfStarsInSurvey+2:])
  print
  # final chain
  sampler.reset()
  start = now()
  print '** Starting sampling **'
  sampler.run_mcmc(pos, maxIter, rstate0=state, thin=thinFactor)
  print '** Finished sampling **'
  print '                Time (s): ',now()-start
  print 'Median acceptance fraction: ',np.median(sampler.acceptance_fraction)
  print ('Acceptance fraction IQR: {0}'.format(np.percentile(sampler.acceptance_fraction,25)) +
      ' -- {0}'.format(np.percentile(sampler.acceptance_fraction,75)))
  correlationTimes = sampler.acor
  print 'Autocorrelation times: '
  print '  Mean absolute magnitude: ', correlationTimes[0]
  print '  Variance absolute magnitude: ', correlationTimes[1]
  print '  Median for parallaxes: ', np.median(correlationTimes[2:numberOfStarsInSurvey+2])
  print '  Median for magnitudes: ', np.median(correlationTimes[numberOfStarsInSurvey+2:])
  
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
  print "mu_M={:4.2f}".format(estimatedAbsMag)+" +/- {:4.2f}".format(errorEstimatedAbsMag)
  print "sigma^2_M={:4.2f}".format(estimatedVarMag)+" +/- {:4.2f}".format(errorEstimatedVarMag)
  
  
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
  
  ax=fig.add_subplot(2,2,3)
  plt.hexbin(meanAbsoluteMagnitudeSamples,varAbsoluteMagnitudeSamples, C=None, bins='log', cmap=cm.gray_r)
  ax.plot(meanAbsoluteMagnitude,varianceAbsoluteMagnitude,'or',mec='r', markersize=8, scalex=False, scaley=False)
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
  priorInfo.append("Prior on $\\sigma^2_M$: $1/\\sigma^2_M\\quad{}<\\sigma^2_M<{}$".format(varianceLow,varianceHigh))
  
  plt.figtext(0.55,0.25,priorInfo[0],horizontalalignment='left')
  plt.figtext(0.55,0.20,priorInfo[1],horizontalalignment='left')
 
  basename='luminosityCalibrationResultsEmcee'
  if (args['pdfOutput']):
    plt.savefig(basename+'.pdf')
  elif (args['pngOutput']):
    plt.savefig(basename+'.png')
  elif (args['epsOutput']):
    plt.savefig(basename+'.eps')
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
              (3) lower limit of the 1/x prior on the variance of the absolute magnitudes,
              (4) upper limit of the 1/x prior on the variance of the absolute magnitudes""")
  parser.add_argument("-p", action="store_true", dest="pdfOutput", help="Make PDF plot")
  parser.add_argument("-e", action="store_true", dest="epsOutput", help="Make EPS plot")
  parser.add_argument("-g", action="store_true", dest="pngOutput", help="Make PNG plot")

  return vars(parser.parse_args())

if  __name__ in ('__main__'):
  args = parseCommandLineArguments()
  runMCMCmodel(args)
