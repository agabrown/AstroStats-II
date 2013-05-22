#!/usr/bin/env python

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
from optparse import OptionParser
from tables import openFile

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

# Set up command line parsing
#
usageString = """usage: %prog [options] fileName\n\t
\tfileName - name of file with simulated survey
"""
parser = OptionParser(usage=usageString)
parser.add_option("-p", action="store_true", dest="pdfOutput", help="Make PDF plot")
parser.add_option("-g", action="store_true", dest="pngOutput", help="Make PNG plot")
parser.add_option("-c", action="store_true", dest="colourFigure", help="Make colour plot")
parser.add_option("-t", action="store_true", dest="forTalk",  help="make version for presentations")

# Parse the command line arguments
#
(options, args) = parser.parse_args()
if (len(args)<1):
  parser.error("Incorrect number of arguments")

h5fileName=args[0]

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

print observedParallaxes

parLimitPlot=50.0
fig = plt.figure(figsize=(12,8.5))
ax = fig.add_subplot(2,2,1)
n, bins, patches = ax.hist(trueParallaxes, 100, normed=1,
    range=(minParallax,parLimitPlot), histtype='stepfilled', alpha=0.75, label='true')
n, bins, patches = ax.hist(observedParallaxes, 100,
    normed=1,range=(observedParallaxes.min(),parLimitPlot),
    histtype='step', alpha=0.75, color='r', label='observed')
minPMinThird=np.power(minParallax,-3.0)
maxPMinThird=np.power(parLimitPlot,-3.0)
x=np.linspace(minParallax,parLimitPlot,1001)
plt.plot(x,3.0*np.power(x,-4.0)/(minPMinThird-maxPMinThird),'k-', label='model')
plt.xlabel("$\\varpi$ [mas]")
plt.ylabel("$P(\\varpi)$")
leg=plt.legend(loc=(0.05,0.5))
for t in leg.get_texts():
  t.set_fontsize(14)
  
ax = fig.add_subplot(2,2,2)
n, bins, patches = ax.hist(absoluteMagnitudes, 100, normed=1, histtype='stepfilled', alpha=0.75)
x=0.5*(bins[1:]+bins[:-1])
stddevAbsMagnitude=np.sqrt(varianceAbsoluteMagnitude)
plt.plot(x,
    gaussian((x-meanAbsoluteMagnitude)/stddevAbsMagnitude)/(np.sqrt(2.0*np.pi)*stddevAbsMagnitude),'or')
plt.xlabel("$M$")
plt.ylabel("$P(M)$")
plt.ylim(0,n.max())
  
ax = fig.add_subplot(2,2,3)
m, mbins, mpatches = ax.hist(apparentMagnitudes, 100, normed=1,
    histtype='stepfilled', alpha=0.75, label='true')
n, bins, patches = ax.hist(observedMagnitudes, 100, normed=1, histtype='step',
    alpha=0.75, color='r', label='observed')
plt.xlabel("$m$")
plt.ylabel("$P(m)$")
plt.ylim(0.0,np.array([n.max(),m.max()]).max())
leg=plt.legend(loc=(0.1,0.7))
for t in leg.get_texts():
  t.set_fontsize(14)
  
ax = fig.add_subplot(2,2,4)
#plt.semilogy(observedMagnitudes,parallaxErrors,'.',alpha=0.5, label="$\\sigma_\\varpi$ [mas]$")
#plt.semilogy(observedMagnitudes,magnitudeErrors,'.r',alpha=0.5, label="$\\sigma_m$")
#plt.xlabel("$m_\\mathrm{o}$")
#plt.ylabel("$\\sigma$")
#plt.ylim(8.0e-4,10.0)
#leg=plt.legend(loc=(0.55,0.05), numpoints=1, handlelength=0.5, markerscale=1.0)
#for t in leg.get_texts():
#  t.set_fontsize(14)
plt.plot(observedParallaxes,parallaxErrors/observedParallaxes,'.',alpha=0.5)
plt.xlabel("$\\varpi_\\mathrm{o}$")
plt.ylabel("$\\sigma_\\varpi/\\varpi_\\mathrm{o}$")
  
plt.suptitle("Simulated survey statistics: $N_\\mathrm{stars}"+"={0}".format(numberOfStarsInSurvey)+"$, ${0}".format(minParallax)+"\\leq\\varpi\\leq{0}".format(maxParallax)+"$, $\\langle M\\rangle={0}".format(meanAbsoluteMagnitude)+"$, $\\sigma^2_M={0}".format(varianceAbsoluteMagnitude)+"$, $m<{0}$".format(apparentMagnitudeLimit))

if (options.pdfOutput):
  plt.savefig('simulatedSurvey.pdf')
elif (options.pngOutput):
  plt.savefig('simulatedSurvey.png')
else:
  plt.show()
