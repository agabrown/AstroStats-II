# Plot results of MCMC luminosity calibration obtained with PyMC.
#
# Anthony Brown 2011-2013

import numpy as np
from pymc import MAP, database
from scipy.stats import gaussian_kde
import scipy.optimize

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from matplotlib.patches import Circle
import argparse
from tables import openFile
from agabutils import calculateHistogram
from re import match

# Configure matplotlib
rc('text', usetex=True)
rc('font', family='serif', size=16)
rc('xtick.major', size='12')
rc('xtick.minor', size='6')
rc('ytick.major', size='12')
rc('ytick.minor', size='6')
rc('lines', linewidth=2)
rc('axes', linewidth=2)

def plotLCMResults(args):
  """
  Plot a summary of the results from the MCMC sampling of the luminosity calibration model.

  Parameters
  ----------

  args - command line arguments.
  """
  h5fileNameSurvey=args['surveyName']
  h5fileNameMcmc=args['mcmcName']

  h5fileSurvey=openFile(h5fileNameSurvey,'r')
  parameters=h5fileSurvey.root.survey.parameters
  data=h5fileSurvey.root.survey.data
  mcmcParams=h5fileSurvey.root.survey.mcmc
  numberOfStarsInSurvey=parameters.col('numberOfStarsInSurvey')[0]
  minParallax=parameters.col('minParallax')[0]
  maxParallax=parameters.col('maxParallax')[0]
  meanAbsoluteMagnitude=parameters.col('meanAbsoluteMagnitude')[0]
  varianceAbsoluteMagnitude=parameters.col('varianceAbsoluteMagnitude')[0]
  trueParallaxes=data.col('trueParallaxes')[0]
  observedParallaxes=data.col('observedParallaxes')[0]
  absoluteMagnitudes=data.col('absoluteMagnitudes')[0]
  apparentMagnitudes=data.col('apparentMagnitudes')[0]
  observedMagnitudes=data.col('observedMagnitudes')[0]
  parallaxErrors=data.col('parallaxErrors')[0]
  magnitudeErrors=data.col('magnitudeErrors')[0]
  minMeanAbsoluteMagnitude=mcmcParams.col('minMeanAbsoluteMagnitude')[0]
  maxMeanAbsoluteMagnitude=mcmcParams.col('maxMeanAbsoluteMagnitude')[0]
  priorTau=mcmcParams.col('priorTau')[0]
  if (match("OneOverX",priorTau)):
    tauLow=mcmcParams.col('tauLow')[0]
    tauHigh=mcmcParams.col('tauHigh')[0]
  elif (match("Inverse-Gamma",priorTau)):
    shapeTau=mcmcParams.col('shapeTau')[0]
    scaleTau=mcmcParams.col('scaleTau')[0]
  else:
    print "Unknown prior on Tau: aborting"
    exit(0)
  h5fileSurvey.close()

  M=database.hdf5.load(h5fileNameMcmc)
  maxIter=M.getstate()['sampler']['_iter']
  burnIter=M.getstate()['sampler']['_burn']
  thinFactor=M.getstate()['sampler']['_thin']

  # Point estimates of mean Absolute Magnitude and its standard deviation.
  estimatedAbsMag=M.trace('meanAbsoluteMagnitude', chain=-1)[:].mean()
  errorEstimatedAbsMag=M.trace('meanAbsoluteMagnitude', chain=-1)[:].std()
  estimatedVarMag=(1.0/M.trace('tauAbsoluteMagnitude', chain=-1)[:]).mean()
  errorEstimatedVarMag=(1.0/M.trace('tauAbsoluteMagnitude', chain=-1)[:]).std()

  # MAP estimates
  muSamples = M.trace('meanAbsoluteMagnitude', chain=-1)[:]
  muDensity = gaussian_kde(muSamples)
  mapValueMu = scipy.optimize.fmin(lambda x:
      -1.0*muDensity(x),np.median(muSamples),maxiter=1000,ftol=0.0001)

  varSamples = 1.0/M.trace('tauAbsoluteMagnitude', chain=-1)[:]
  varDensity = gaussian_kde(varSamples)
  mapValueVar = scipy.optimize.fmin(lambda x:
      -1.0*varDensity(x),np.median(varSamples),maxiter=1000,ftol=0.0001)

  # The wrong estimators:
  indices=(observedParallaxes > 0.0)
  wrongAbsMagEstimates=observedMagnitudes[indices]+5.0*np.log10(observedParallaxes[indices])-10.0
  print wrongAbsMagEstimates.size, wrongAbsMagEstimates.mean(), wrongAbsMagEstimates.std()**2.0
  indices=(parallaxErrors/observedParallaxes > 0.0) & (parallaxErrors/observedParallaxes < 0.2)
  wrongAbsMagEstimates=observedMagnitudes[indices]+5.0*np.log10(observedParallaxes[indices])-10.0
  print wrongAbsMagEstimates.size, wrongAbsMagEstimates.mean(), wrongAbsMagEstimates.std()**2.0

  fig = plt.figure(figsize=(12,8.5))
  ax = fig.add_subplot(2,2,1)
  x = np.linspace(muSamples.min(), muSamples.max(), 500)
  plt.plot(x,muDensity(x),'k-')
  plt.axvline(meanAbsoluteMagnitude, linewidth=2, color="red")
  plt.xlabel("$\\mu_M$")
  plt.ylabel("$P(\\mu_M)$")

  ax = fig.add_subplot(2,2,2)
  x = np.linspace(varSamples.min(), varSamples.max(), 500)
  plt.plot(x,varDensity(x),'k-')
  plt.axvline(varianceAbsoluteMagnitude, linewidth=2, color="red")
  plt.xlabel("$\\sigma^2_M$")
  plt.ylabel("$P(\\sigma^2_M)$")

  ax = fig.add_subplot(2,2,3)
  plt.hexbin(muSamples, varSamples,C=None,bins='log',cmap=cm.gray_r)
  ax.plot(meanAbsoluteMagnitude,varianceAbsoluteMagnitude,'or',mec='r', markersize=8, scalex=False, scaley=False)
  #ax.add_patch(Circle((meanAbsoluteMagnitude,varianceAbsoluteMagnitude),radius=0.05,fc='r',ec=None))
  plt.xlabel("$\\mu_M$")
  plt.ylabel("$\\sigma^2_M$")

  plt.figtext(0.55,0.4,"$\\widetilde{\\mu_M}="+"{:4.2f}".format(estimatedAbsMag) + 
      "$ $\\pm$ ${:4.2f}$".format(errorEstimatedAbsMag),ha='left')
  plt.figtext(0.75,0.4,"$\\mathrm{MAP}(\\widetilde{\\mu_M})="+"{:4.2f}".format(mapValueMu[0])+"$")
  plt.figtext(0.55,0.35,"$\\widetilde{\\sigma^2_M}="+"{:4.2f}".format(estimatedVarMag) + 
      "$ $\\pm$ ${:4.2f}$".format(errorEstimatedVarMag), ha='left')
  plt.figtext(0.75,0.35,"$\\mathrm{MAP}(\\widetilde{\\sigma^2_M})="+"{:4.2f}".format(mapValueVar[0])+"$")

  titelA=("$N_\\mathrm{stars}"+"={0}".format(numberOfStarsInSurvey) +
      "$, True values: $\\mu_M={0}".format(meanAbsoluteMagnitude) +
      "$, $\\sigma^2_M={0}".format(varianceAbsoluteMagnitude)+"$")
  titelB=("Iterations = {0}".format(maxIter)+", Burn = {0}".format(burnIter) + 
      ", Thin = {0}".format(thinFactor))
  plt.suptitle(titelA+"\\quad\\quad "+titelB)

  priorInfo=[]
  priorInfo.append("Prior on $\\mu_M$: flat $\\quad{0}".format(minMeanAbsoluteMagnitude) +
      "<\\mu_M<{0}".format(maxMeanAbsoluteMagnitude)+"$")
  if (match("OneOverX",priorTau)):
    priorInfo.append("Prior on $\\tau_M$: $1/\\tau_M\\quad{0}".format(tauLow) +
        "<\\tau_M<{0}".format(tauHigh)+"$")
  else:
    priorInfo.append("Prior on $\\tau_M$: $\\Gamma_\\mathrm{inverse}" +
        "(\\tau_M | \\alpha={0}".format(shapeTau) + ",\\beta={0}".format(scaleTau)+")$")

  plt.figtext(0.55,0.25,priorInfo[0],horizontalalignment='left')
  plt.figtext(0.55,0.20,priorInfo[1],horizontalalignment='left')

  titelC=[]
  titelC.append("MCMC sampling with PyMC") 
  plt.figtext(0.55,0.15,titelC[0],horizontalalignment='left')

  if (args['pdfOutput']):
    plt.savefig('luminosityCalibrationResults.pdf')
  elif (args['pngOutput']):
    plt.savefig('luminosityCalibrationResults.png')
  elif (args['epsOutput']):
    plt.savefig('luminosityCalibrationResults.eps')
  else:
    plt.show()

def parseCommandLineArguments():
  """
  Set up command line parsing.
  """
  parser = argparse.ArgumentParser("Plot the MCMC luminosity calibration results obtained with PyMC.")
  parser.add_argument("surveyName", type=str, help="Name of file with simulated survey")
  parser.add_argument("mcmcName", type=str, help="Name of file MCMC results of luminosity calibration")
  parser.add_argument("-p", action="store_true", dest="pdfOutput", help="Make PDF plot")
  parser.add_argument("-e", action="store_true", dest="epsOutput", help="Make EPS plot")
  parser.add_argument("-g", action="store_true", dest="pngOutput", help="Make PNG plot")

  return vars(parser.parse_args())

if __name__ in ('__main__'):
  args = parseCommandLineArguments()
  plotLCMResults(args)
