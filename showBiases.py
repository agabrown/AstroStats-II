"""
Show the statistical biases that occur when using a naive estimate of absolute magnitudes for our
fictitious parallax survey.

Anthony Brown May 2013
"""

import numpy as np
import universemodels as U
import matplotlib.pyplot as plt
import argparse
from matplotlib import rc, cm
from scipy.stats import gaussian_kde, norm

def plotBiases(args):
  """
  Produce a plot showing the negative effects of using naive absolute magnitude estimates.

  Parameters
  ----------

  args - command line arguments
  """
  surveyParams=args['surveyString']
  if surveyParams[5] == 'Inf':
    magLim = np.Inf
  else:
    magLim = float(surveyParams[5])

  simulatedSurvey=U.UniformDistributionSingleLuminosity(int(surveyParams[0]), float(surveyParams[1]),
      float(surveyParams[2]), float(surveyParams[3]), float(surveyParams[4]), surveyLimit=magLim)
  #simulatedSurvey.setRandomNumberSeed(53949896)
  simulatedSurvey.generateObservations()
  numberOfStarsInSurvey=simulatedSurvey.numberOfStarsInSurvey

  positiveParallaxes = (simulatedSurvey.observedParallaxes > 0.0)
  estimatedAbsMags = (simulatedSurvey.observedMagnitudes[positiveParallaxes] +
      5.0*np.log10(simulatedSurvey.observedParallaxes[positiveParallaxes])-10.0)
  relParErr = (simulatedSurvey.parallaxErrors[positiveParallaxes] /
      simulatedSurvey.observedParallaxes[positiveParallaxes])
  deltaAbsMag = estimatedAbsMags - simulatedSurvey.absoluteMagnitudes[positiveParallaxes]

  fig=plt.figure(figsize=(12,5))
  fig.add_subplot(1,2,1)
  plt.hexbin(np.log10(relParErr),deltaAbsMag,C=None, bins='log', cmap=cm.gray_r)
  plt.xlabel("$\\log[\\sigma_\\varpi/\\varpi_\\mathrm{o}]$")
  plt.ylabel("$\\widetilde{M}-M_\\mathrm{true}$")
  plt.xlim(-3,2)
  plt.ylim(-10,6)

  fig.add_subplot(1,2,2)
  x=np.linspace(simulatedSurvey.absoluteMagnitudes.min(),simulatedSurvey.absoluteMagnitudes.max(),500)

  select = relParErr < 0.2
  selectedTrueAbsMags=simulatedSurvey.absoluteMagnitudes[positiveParallaxes][select]
  selTrueDensity = gaussian_kde(selectedTrueAbsMags)

  selectedEstAbsMags=estimatedAbsMags[select]
  selEstDensity = gaussian_kde(selectedEstAbsMags)

  plt.plot(x, norm.pdf(x,loc=float(surveyParams[3]), scale=np.sqrt(float(surveyParams[4]))), 'k:',
      label='$M_\\mathrm{true}$')
  plt.plot(x,selTrueDensity(x),'r-', label="$M_\\mathrm{true}:$ $0<\\frac{\\sigma_\\varpi}{\\varpi_\\mathrm{o}}<0.2$")
  plt.plot(x,selEstDensity(x),'b--', label="$\\widetilde{M}:$ $0<\\frac{\\sigma_\\varpi}{\\varpi_\\mathrm{o}}<0.2$")
  plt.xlabel('$M$')
  plt.ylabel('$p(M)$')
  plt.xlim(simulatedSurvey.absoluteMagnitudes.min(),simulatedSurvey.absoluteMagnitudes.max()+0.6)
  leg=plt.legend(loc=1, fontsize=11)

  plt.tight_layout(pad=1.2)

  basename='biasesFromNaiveEstimation'
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
  parser = argparse.ArgumentParser("""Show the statistical biases that occur when using a naive estimate
  of absolute magnitudes for our fictitious parallax survey.""")
  parser.add_argument("--survey", dest="surveyString", nargs=6,
      help="""White-space-separated list of survey parameters:
              (1) number of stars,
              (2) lower limit parallaxes [mas],
              (3) upper limit parallaxes [mas],
              (4) mean absolute magnitude,
              (5) variance of absolute magnitudes
              (6) apparent magnitude limit of survey (Inf allowed)""")
  parser.add_argument("-p", action="store_true", dest="pdfOutput", help="Make PDF plot")
  parser.add_argument("-e", action="store_true", dest="epsOutput", help="Make EPS plot")
  parser.add_argument("-g", action="store_true", dest="pngOutput", help="Make PNG plot")

  return vars(parser.parse_args())

if  __name__ in ('__main__'):
  args = parseCommandLineArguments()
  plotBiases(args)

