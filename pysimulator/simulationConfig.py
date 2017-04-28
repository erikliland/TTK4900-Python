import os
from pysimulator.scenarios.scenarios import scenarioList

pdList = [1., 0.8, 0.6]
lambdaphiList = [0, 2e-6, 4e-6]
nList = [1, 3, 6, 9]
path = os.path.join(os.path.expanduser('~'), 'TTK4900-Python', 'data')
nMonteCarlo = 10
scenarioList = scenarioList
filePathList = [os.path.join(path, scenario.name + ".xml") for scenario in scenarioList]
baseSeed = 5446
