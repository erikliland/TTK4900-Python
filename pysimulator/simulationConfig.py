import os
from pysimulator.scenarios.scenarios import scenarioList

pdList = [1., 0.8, 0.6]
lambdaphiList = [0, 2e-6, 4e-6, 8e-6]
nList = [1, 3, 6, 9]
path = os.path.join(os.path.expanduser('~'), 'TTK4900-Python', 'data')
nMonteCarlo = 20
scenarioList = scenarioList
trackingFilePathList = [os.path.join(path, scenario.name + "_Tracking" + ".xml") for scenario in scenarioList]
initFilePathList = [os.path.join(path, scenario.name + "_Init" + ".xml") for scenario in scenarioList]
baseSeed = 5446
M_N_list = [(1,1),(1,2),(1,3),(1,4),(2,2),(2,3),(2,4),(2,5),(3,3),(3,4),(3,5),(3,6)]

acceptThreshold = 15 #meter