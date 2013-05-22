"""
extrastochastics.py

Custom stochastic variables for use with pymc.
"""

from pymc import stochastic_from_dist
import numpy as np

def oneOverXFourth_like(x, lower, upper):
  R"""
  Log-likelihood for stochastic variable with 1/x^4 distribution

  .. math::
      f(x \mid lower, upper) = \frac{x^{-4}}{lower^{-3}-upper^{-3}}

  :Parameters
    x : float
     :math`lower \leq x \leq upper`
    lower : float
      Lower limit
    upper : float
      Upper limit
  """
  if np.any(x < lower) or np.any(x > upper):
    return -np.Inf
  else:
    return -4.0*np.sum(np.log(x))

def random_oneOverXFourth(lower, upper, size):
  lowerMinThird=np.power(lower,-3.0)
  upperMinThird=np.power(upper,-3.0)
  return np.power(lowerMinThird-np.random.random_sample(size)*(lowerMinThird-upperMinThird),-1.0/3.0)

OneOverXFourth=stochastic_from_dist('OneOverXFourth', oneOverXFourth_like, random_oneOverXFourth, dtype=np.float)

def oneOverXSecond_like(x, lower, upper):
  R"""
  Log-likelihood for stochastic variable with 1/x^2 distribution

  .. math::
      f(x \mid lower, upper) = \frac{x^{-2}}{lower^{-1}-upper^{-1}}

  :Parameters
    x : float
     :math`lower \leq x \leq upper`
    lower : float
      Lower limit
    upper : float
      Upper limit
  """
  if np.any(x < lower) or np.any(x > upper):
    return -np.Inf
  else:
    return -2.0*np.sum(np.log(x))

def random_oneOverXSecond(lower, upper, size):
  oneOverLower=1.0/lower
  oneOverUpper=1.0/upper
  return 1.0/(oneOverLower-np.random.random_sample(size)*(oneOverLower-oneOverUpper))

OneOverXSecond=stochastic_from_dist('OneOverXSecond', oneOverXSecond_like, random_oneOverXSecond, dtype=np.float)

def oneOverX_like(x, lower, upper):
  R"""
  Log-likelihood for stochastic variable with 1/x distribution

  .. math::
      f(x \mid lower, upper) = \frac{x^{-1}}{\ln(upper)-\ln(lower)}

  :Parameters
    x : float
     :math`lower \leq x \leq upper`
    lower : float
      Lower limit
    upper : float
      Upper limit
  """
  if np.any(x < lower) or np.any(x > upper):
    return -np.Inf
  else:
    return -np.sum(np.log(x))

def random_oneOverX(lower, upper, size):
  return np.exp(np.log(lower)+np.random.random_sample(size)*(np.log(upper)-np.log(lower)))

OneOverX=stochastic_from_dist('OneOverX', oneOverX_like, random_oneOverX, dtype=np.float)
