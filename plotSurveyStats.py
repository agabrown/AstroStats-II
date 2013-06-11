import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc, cm
import argparse
from tables import openFile
from scipy.stats import gaussian_kde
from agabutils import kdeAndMap

def gaussian(t):
  """Returns exp(-0.5*t^2)"""
  return np.exp(-0.5*t*t)

# Configure matplotlib
rc('text', usetex=True)
rc('font', family='serif', size=16)
rc('xtick.major', size='12')
rc('xtick.minor', size='6')
rc('ytick.major', size='12')
rc('ytick.minor', size='6')
rc('lines', linewidth=2)
rc('axes', linewidth=2)

def plotSurveyStats(args):
  """
  Plot the statistics of the simulated survey.

  Parameters
  ----------

  args - command line arguments.
  """
  h5fileName=args['surveyName']

  h5file=openFile(h5fileName, "r")
  parameters=h5file.root.survey.parameters
  data=h5file.root.survey.data
  numberOfStars=parameters.col('numberOfStars')[0]
  numberOfStarsInSurvey=parameters.col('numberOfStarsInSurvey')[0]
  apparentMagnitudeLimit=parameters.col('apparentMagnitudeLimit')[0]
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
  h5file.close()

  positiveParallaxes = (observedParallaxes > 0.0)
  estimatedAbsMags = (observedMagnitudes[positiveParallaxes] +
      5.0*np.log10(observedParallaxes[positiveParallaxes])-10.0)
  relParErr = (parallaxErrors[positiveParallaxes] /
      observedParallaxes[positiveParallaxes])
  deltaAbsMag = estimatedAbsMags - absoluteMagnitudes[positiveParallaxes]

  select = relParErr < 0.2
  selectedEstAbsMags=estimatedAbsMags[select]

  fig = plt.figure(figsize=(12,8.5))
  parLimitPlot=50.0

  ax = fig.add_subplot(2,2,1)
  x=np.linspace(observedParallaxes.min(),parLimitPlot,1000)
  trueParDensity, maxLocation = kdeAndMap(trueParallaxes, bwmethod=.2)
  plt.plot(x, trueParDensity(x), 'b--', label='true')
  plt.plot(x, gaussian_kde(observedParallaxes)(x), 'r-', label='observed')
  x=np.linspace(minParallax,parLimitPlot,1001)
  minPMinThird=np.power(minParallax,-3.0)
  maxPMinThird=np.power(parLimitPlot,-3.0)
  plt.plot(x,3.0*np.power(x,-4.0)/(minPMinThird-maxPMinThird),'k:', label='model')
  plt.xlabel("$\\varpi$ [mas]")
  plt.ylabel("$P(\\varpi)$")
  plt.ylim(0,trueParDensity(maxLocation))
  leg=plt.legend(loc=(0.05,0.5), fontsize=12)
  
  ax = fig.add_subplot(2,2,2)
  x=np.linspace(absoluteMagnitudes.min(), absoluteMagnitudes.max(), 1000)
  plt.plot(x, gaussian_kde(absoluteMagnitudes)(x), 'b--', label='true') 
  plt.plot(x, gaussian_kde(selectedEstAbsMags)(x), 'r-', label='naive estimate')
  stddevAbsMagnitude=np.sqrt(varianceAbsoluteMagnitude)
  plt.plot(x,
      gaussian((x-meanAbsoluteMagnitude)/stddevAbsMagnitude)/(np.sqrt(2.0*np.pi)*stddevAbsMagnitude),'k:',
      label='model')
  plt.xlabel("$M$")
  plt.ylabel("$P(M)$")
  leg=plt.legend(loc=0, fontsize=12)
  
  ax = fig.add_subplot(2,2,3)
  x=np.linspace(observedMagnitudes.min(), observedMagnitudes.max(), 1000)
  plt.plot(x, gaussian_kde(apparentMagnitudes)(x), 'b--', label='true')
  plt.plot(x, gaussian_kde(observedMagnitudes)(x), 'r-', label='observed')
  plt.xlabel("$m$")
  plt.ylabel("$P(m)$")
  leg=plt.legend(loc=0, fontsize=12)
  
  ax = fig.add_subplot(2,2,4)
  if len(relParErr) < 1000:
    plt.semilogx(relParErr,deltaAbsMag,'b.')
    plt.xlabel("$\\sigma_\\varpi/\\varpi_\\mathrm{o}$")
    plt.xlim(1.0e-3,100)
  else:
    plt.hexbin(np.log10(relParErr),deltaAbsMag,C=None, bins='log', cmap=cm.gray_r)
    plt.xlabel("$\\log\\sigma_\\varpi/\\varpi_\\mathrm{o}]$")
    plt.xlim(-3,2)
  plt.ylabel("$\\widetilde{M}-M_\\mathrm{true}$")
  plt.ylim(-10,6)
  
  plt.suptitle("Simulated survey statistics: $N_\\mathrm{{stars}}={}$, ${}\\leq\\varpi\\leq{}$, $\\mu_M={}$, $\\sigma^2_M={}$, $m_\\mathrm{{lim}}<{}$".format(numberOfStarsInSurvey, minParallax, maxParallax,
    meanAbsoluteMagnitude, varianceAbsoluteMagnitude, apparentMagnitudeLimit))

  basename='simulatedSurvey'
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
  parser = argparse.ArgumentParser("Plot the statistics for the simulated parallax survey.")
  parser.add_argument("surveyName", type=str, help="Name of file with simulated survey")
  parser.add_argument("-p", action="store_true", dest="pdfOutput", help="Make PDF plot")
  parser.add_argument("-e", action="store_true", dest="epsOutput", help="Make EPS plot")
  parser.add_argument("-g", action="store_true", dest="pngOutput", help="Make PNG plot")

  return vars(parser.parse_args())

if  __name__ in ('__main__'):
  args = parseCommandLineArguments()
  plotSurveyStats(args)
