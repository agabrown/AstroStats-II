MCMC luminosity calibration exercise for GREAT AstroStats-II school June 2013
=============================================================================

Python code for the exercise on Bayesian luminosity calibration as part of the lectures by B. Holl.

Required python packages
------------------------

* `numpy <http://www.numpy.org/>`_

  ! Version 1.6.1 is currently the highest version available through the Canopy package installer,
  and it causes the error: "ImportError: numpy.core.multiarray failed to import".
  If you have version <= 1.6.1:
  (1) uninstall it (e.g. using the Package Manager of Canopy),
  (2) Download numpy version 1.6.2 or higher from: https://pypi.python.org/pypi/numpy, 
  (3) Install by running "python setup.py install --user" in the extracted directory (without "--user" when you have root rights).
  
* `scipy <http://www.scipy.org/>`_
  (Version 0.11.0 works, available through Canopy package installer)

* `matplotlib <http://matplotlib.org/>`_
  (Version 1.2.0 works, available through Canopy package installer)
  
* `PyMC <https://github.com/pymc-devs/pymc>`_
  (Version 2.1b0 works, available through Canopy package installer)

* `emcee <http://dan.iel.fm/emcee/>`_
  (Version emcee-1.2.0 works, download from: https://github.com/dfm/emcee)

* `acor <https://pypi.python.org/pypi/acor>`_
  (Version 1.0.2 works, download from: https://pypi.python.org/pypi/acor)

* `PyTables <http://pytables.github.io/>`_
  (Version 2.3.1 works, available through Canopy package installer)


Instructions
------------

See the Instructions.pdf file.
