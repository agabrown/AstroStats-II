#!/usr/bin/env python
#
# Plot results of MCMC luminosity calibration. This is the version for the textbook
# on astrometry edited by William Van Altena

import numpy as np
import pymc

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
parser = argparse.ArgumentParser("Plot the MCMC luminosity calibration results obtained with PyMC.
    Version for Van Altena book")
parser.add_argument("surveyName", type=str, help="Name of file with simulated survey")
parser.add_argument("mcmcName", type=str, help="Name of file MCMC results of luminosity calibration")
parser.add_argument("-p", action="store_true", dest="pdfOutput", help="Make PDF plot")
parser.add_argument("-e", action="store_true", dest="epsOutput", help="Make EPS plot")
parser.add_argument("-g", action="store_true", dest="pngOutput", help="Make PNG plot")
parser.add_argument("-c", action="store_true", dest="colourFigure", help="Make colour plot")
parser.add_argument("-t", action="store_true", dest="forTalk",  help="make version for presentations")

# Parse the command line arguments
#
args=vars(parser.parse_args())
h5fileNameSurvey=args[0]
h5fileNameMcmc=args[1]

h5fileSurvey=openFile(h5fileNameSurvey,'r')
parameters=h5fileSurvey.root.survey.parameters
data=h5fileSurvey.root.survey.data
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
h5fileSurvey.close()

M=pymc.database.hdf5.load(h5fileNameMcmc)
maxIter=M.getstate()['sampler']['_iter']
burnIter=M.getstate()['sampler']['_burn']
thinFactor=M.getstate()['sampler']['_thin']

# Point estimates of mean Absolute Magnitude and its standard deviation.
estimatedAbsMag=M.trace('meanAbsoluteMagnitude', chain=-1)[:].mean()
errorEstimatedAbsMag=M.trace('meanAbsoluteMagnitude', chain=-1)[:].std()
estimatedVarMag=(1.0/M.trace('tauAbsoluteMagnitude', chain=-1)[:]).mean()
errorEstimatedVarMag=(1.0/M.trace('tauAbsoluteMagnitude', chain=-1)[:]).std()
print "MCMC estimates"
print "mu_M={:4.2f}".format(estimatedAbsMag)+" +/- {:4.2f}".format(errorEstimatedAbsMag)
print "sigma^2_M={:4.2f}".format(estimatedVarMag)+" +/- {:4.2f}".format(errorEstimatedVarMag)

# The wrong estimators:
indices=(observedParallaxes > 0.0)
wrongAbsMagEstimates=observedMagnitudes[indices]+5.0*np.log10(observedParallaxes[indices])-10.0
print "Naive estimates"
print "Using only postive parallaxes"
print wrongAbsMagEstimates.size, wrongAbsMagEstimates.mean(), wrongAbsMagEstimates.std()**2.0
indices=(parallaxErrors/observedParallaxes > 0.0) & (parallaxErrors/observedParallaxes < 0.175)
wrongAbsMagEstimates=observedMagnitudes[indices]+5.0*np.log10(observedParallaxes[indices])-10.0
print "Using LK-slices"
print wrongAbsMagEstimates.size, wrongAbsMagEstimates.mean(), wrongAbsMagEstimates.std()**2.0

fig = plt.figure(figsize=(12,8.5))

parLimitPlot=50.0
ax = fig.add_subplot(2,2,1)
n, bins, patches = ax.hist(observedParallaxes, 50,
    normed=1,range=(observedParallaxes.min(),parLimitPlot),
    histtype='stepfilled', alpha=0.75, color='grey', label='observed')
m, bins, patches = ax.hist(trueParallaxes, 50, normed=1, color='k',
    range=(minParallax,parLimitPlot), histtype='step', label='true')
minPMinThird=np.power(minParallax,-3.0)
maxPMinThird=np.power(parLimitPlot,-3.0)
x=np.linspace(minParallax,parLimitPlot,101)
plt.plot(x,3.0*np.power(x,-4.0)/(minPMinThird-maxPMinThird),'k-', label='model')
plt.xlabel("$\\varpi$ [mas]")
plt.ylabel("$P(\\varpi)$")
#plt.ylim(0,m.max()*1.03)
plt.ylim(0,0.15)
leg=plt.legend(loc=(0.05,0.55), handlelength=1.0)
for t in leg.get_texts():
  t.set_fontsize(14)
  
ax = fig.add_subplot(2,2,2)
nn, bins, patches = ax.hist(absoluteMagnitudes, 50, normed=0, color='k', histtype='step', label='true')
pp, bins, patches = ax.hist(absoluteMagnitudes[(observedParallaxes > 0.0)], 50, normed=0, histtype='stepfilled',
    color='grey', alpha=0.75, label='censored')
x=np.linspace(0.5*(bins[0]+bins[1]),0.5*(bins[-2]+bins[-1]),101)
stddevAbsMagnitude=np.sqrt(varianceAbsoluteMagnitude)
plt.plot(x,numberOfStarsInSurvey*(bins[1]-bins[0])*
    gaussian((x-meanAbsoluteMagnitude)/stddevAbsMagnitude)/(np.sqrt(2.0*np.pi)*stddevAbsMagnitude),'k',
    label='model')
plt.xlabel("$M$")
plt.ylabel("$N(M)$")
plt.ylim(0,nn.max()*1.03)
leg=plt.legend(loc=(0.05,0.55), handlelength=1.0)
for t in leg.get_texts():
  t.set_fontsize(14)
  
ax = fig.add_subplot(2,2,3)
nnn, bins, patches = ax.hist(M.trace('meanAbsoluteMagnitude', chain=-1)[:], 100, normed=1,
    histtype='stepfilled', alpha=0.75, label='true', color='grey')
plt.xlabel("$\\mu_M$")
plt.ylabel("$P(\\mu_M)$")
plt.ylim(0,1.03*nnn.max())
#plt.title("$\\widetilde{\\langle M\\rangle}="+"{:4.2f}".format(estimatedAbsMag)+"$ $\\pm$ ${:4.2f}$".format(errorEstimatedAbsMag))

ax = fig.add_subplot(2,2,4)
mm, bins, patches = ax.hist(1.0/M.trace('tauAbsoluteMagnitude', chain=-1)[:], 100, normed=1,
    histtype='stepfilled', alpha=0.75, label='true', color='grey')
plt.xlabel("$\\sigma^2_M$")
plt.ylabel("$P(\\sigma^2_M)$")
plt.ylim(0,mm.max()*1.03)
#plt.title("$\\widetilde{\\sigma^2_M}="+"{:4.2f}".format(estimatedVarMag)+"$ $\\pm$ ${:4.2f}$".format(errorEstimatedVarMag))

#plt.figtext(0.6,0.4,"$\\widetilde{\\mu_M}="+"{:4.2f}".format(estimatedAbsMag)+"$ $\\pm$ ${:4.2f}$".format(errorEstimatedAbsMag),ha='left')
#plt.figtext(0.6,0.3,"$\\widetilde{\\sigma^2_M}="+"{:4.2f}".format(estimatedVarMag)+"$ $\\pm$ ${:4.2f}$".format(errorEstimatedVarMag), ha='left')

#titelA="$N_\\mathrm{stars}"+"={0}".format(numberOfStarsInSurvey)+"$, True values: $\\mu_M={0}".format(meanAbsoluteMagnitude)+"$, $\\sigma^2_M={0}".format(varianceAbsoluteMagnitude)+"$"
#titelB="Iterations = {0}".format(maxIter)+", Burn = {0}".format(burnIter)+", Thin = {0}".format(thinFactor)
#plt.suptitle(titelA+"\\quad\\quad "+titelB)

if (args['pdfOutput']):
  plt.savefig('luminosityCalibrationResults.pdf')
elif (args['pngOutput']):
  plt.savefig('luminosityCalibrationResults.png')
elif (args['epsOutput']):
  plt.savefig('luminosityCalibrationResults.eps')
else:
  plt.show()
