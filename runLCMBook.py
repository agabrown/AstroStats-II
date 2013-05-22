"""
Run the MCMC sampling of the luminosity calibration model using the emcee package by Dan Foreman-Mackey.

Here the Inverse-Gamma-prior on the variance hyper-parameter tau=1/sigma^2_M is used.
"""

import numpy as np
from pymc import MCMC, AdaptiveMetropolis, Metropolis
import luminositycalibrationmodels as L
import universemodels as U
from tables import Int32Col, Float64Col, StringCol, IsDescription, openFile
import argparse
from time import time as now

class SurveyParameters(IsDescription):
  """
  Class that holds the data model for the simulated parallax survey parameters. Intended for use with the
  HDF5 files through the pytables package.
  """
  kind = StringCol(itemsize=40)
  numberOfStars = Int32Col()
  minParallax = Float64Col()
  maxParallax = Float64Col()
  meanAbsoluteMagnitude = Float64Col()
  varianceAbsoluteMagnitude = Float64Col()
  parallaxErrorNormalizationMagnitude = Float64Col()
  parallaxErrorSlope = Float64Col()
  parallaxErrorCalibrationFloor = Float64Col()
  magnitudeErrorNormalizationMagnitude = Float64Col()
  magnitudeErrorSlope = Float64Col()
  magnitudeErrorCalibrationFloor = Float64Col()
  apparentMagnitudeLimit = Float64Col()
  numberOfStarsInSurvey = Int32Col()

class McmcParameters(IsDescription):
  """
  Class that holds the data model for the HDF5 table with the MCMC parameters.
  """
  iterations = Int32Col()
  burnIn = Int32Col()
  thin = Int32Col()
  minMeanAbsoluteMagnitude = Float64Col()
  maxMeanAbsoluteMagnitude = Float64Col()
  priorTau = StringCol(40)
  tauLow = Float64Col()
  tauHigh = Float64Col()

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

  if surveyParams[5] == 'Inf':
    magLim = np.Inf
  else:
    magLim = float(surveyParams[5])
  S=U.UniformDistributionSingleLuminosity(int(surveyParams[0]), float(surveyParams[1]),
      float(surveyParams[2]), float(surveyParams[3]), float(surveyParams[4]),
      surveyLimit=magLim)
  #S.setRandomNumberSeed(53949896)
  S.generateObservations()
  lumCalModel=L.UniformSpaceDensityGaussianLFBook(S,float(surveyParams[1]), float(surveyParams[2]),
      float(priorParams[0]), float(priorParams[1]), float(priorParams[2]), float(priorParams[3]))

  class SurveyData(IsDescription):
    """
    Class that holds the data model for the data from the simulated parallax survey. Intended for use
    with the HDF5 files through the pytables package.
    """
    trueParallaxes = Float64Col(S.numberOfStarsInSurvey)
    absoluteMagnitudes = Float64Col(S.numberOfStarsInSurvey)
    apparentMagnitudes = Float64Col(S.numberOfStarsInSurvey)
    parallaxErrors = Float64Col(S.numberOfStarsInSurvey)
    magnitudeErrors = Float64Col(S.numberOfStarsInSurvey)
    observedParallaxes = Float64Col(S.numberOfStarsInSurvey)
    observedMagnitudes = Float64Col(S.numberOfStarsInSurvey)

  baseName="LumCalSimSurvey-{0}".format(S.numberOfStars)+"-{0}".format(S.minParallax)
  baseName=baseName+"-{0}".format(S.maxParallax)+"-{0}".format(S.meanAbsoluteMagnitude)
  baseName=baseName+"-{0}".format(S.varianceAbsoluteMagnitude)

  h5file = openFile(baseName+".h5", mode = "w", title = "Simulated Survey")
  group = h5file.createGroup("/", 'survey', 'Survey parameters, data, and MCMC parameters')
  parameterTable = h5file.createTable(group, 'parameters', SurveyParameters, "Survey parameters")
  dataTable = h5file.createTable(group, 'data', SurveyData, "Survey data")
  mcmcTable = h5file.createTable(group, 'mcmc', McmcParameters, "MCMC parameters")

  surveyParams = parameterTable.row
  surveyParams['kind']=S.__class__.__name__
  surveyParams['numberOfStars']=S.numberOfStars
  surveyParams['minParallax']=S.minParallax
  surveyParams['maxParallax']=S.maxParallax
  surveyParams['meanAbsoluteMagnitude']=S.meanAbsoluteMagnitude
  surveyParams['varianceAbsoluteMagnitude']=S.varianceAbsoluteMagnitude
  surveyParams['parallaxErrorNormalizationMagnitude']=S.parallaxErrorNormalizationMagnitude
  surveyParams['parallaxErrorSlope']=S.parallaxErrorSlope
  surveyParams['parallaxErrorCalibrationFloor']=S.parallaxErrorCalibrationFloor
  surveyParams['magnitudeErrorNormalizationMagnitude']=S.magnitudeErrorNormalizationMagnitude
  surveyParams['magnitudeErrorSlope']=S.magnitudeErrorSlope
  surveyParams['magnitudeErrorCalibrationFloor']=S.magnitudeErrorCalibrationFloor
  surveyParams['apparentMagnitudeLimit']=S.apparentMagnitudeLimit
  surveyParams['numberOfStarsInSurvey']=S.numberOfStarsInSurvey
  surveyParams.append()
  parameterTable.flush()

  surveyData = dataTable.row
  surveyData['trueParallaxes']=S.trueParallaxes
  surveyData['absoluteMagnitudes']=S.absoluteMagnitudes
  surveyData['apparentMagnitudes']=S.apparentMagnitudes
  surveyData['parallaxErrors']=S.parallaxErrors
  surveyData['magnitudeErrors']=S.magnitudeErrors
  surveyData['observedParallaxes']=S.observedParallaxes
  surveyData['observedMagnitudes']=S.observedMagnitudes
  surveyData.append()
  dataTable.flush()

  mcmcParameters = mcmcTable.row
  mcmcParameters['iterations']=maxIter
  mcmcParameters['burnIn']=burnIter
  mcmcParameters['thin']=thinFactor
  mcmcParameters['minMeanAbsoluteMagnitude']=float(priorParams[0])
  mcmcParameters['maxMeanAbsoluteMagnitude']=float(priorParams[1])
  mcmcParameters['priorTau']="OneOverX"
  mcmcParameters['tauLow']=float(priorParams[2])
  mcmcParameters['tauHigh']=float(priorParams[3])
  mcmcParameters.append()
  dataTable.flush()

  h5file.close()

  # Run MCMC and store in HDF5 database
  baseName="LumCalResults-{0}".format(S.numberOfStars)+"-{0}".format(S.minParallax)
  baseName=baseName+"-{0}".format(S.maxParallax)+"-{0}".format(S.meanAbsoluteMagnitude)
  baseName=baseName+"-{0}".format(S.varianceAbsoluteMagnitude)

  M=MCMC(lumCalModel.pyMCModel, db='hdf5', dbname=baseName+".h5", dbmode='w', dbcomplevel=9,
      dbcomplib='bzip2')
  M.use_step_method(Metropolis, M.priorParallaxes)
  M.use_step_method(Metropolis, M.priorAbsoluteMagnitudes)
  start=now()
  M.sample(iter=maxIter, burn=burnIter, thin=thinFactor)
  finish=now()
  print "Elapsed time in seconds: %f" % (finish-start)
  M.db.close()

def parseCommandLineArguments():
  """
  Set up command line parsing.
  """
  parser = argparse.ArgumentParser("Run the MCMC sampling of the luminosity calibration model using PyMC.")
  parser.add_argument("--mcmc", dest="mcmcString", nargs=3,
      help="""White-space-separated list of MCMC parameters:
              (1) number of MCMC iterations,
              (2) number of initial iterations to discard as burn-in,
              (3) thinning factor""")
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
              (3) lower limit 1/x prior on inverse variance of absolute magnitudes,
              (4) upper limit 1/x prior on inverse variance of absolute magnitudes""")
  return vars(parser.parse_args())

if  __name__ in ('__main__'):
  args = parseCommandLineArguments()
  runMCMCmodel(args)
