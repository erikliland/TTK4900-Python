import pymht.tracker as tomht
import pymht.utils.helpFunctions as hpf
import xml.etree.ElementTree as ET
from pymht.utils.xmlDefinitions import *
from pysimulator.scenarios.defaults import *
import os
import datetime


def runVariations(scenario, path, pdList, lambdaphiList, nList, nMonteCarlo):
    simList = scenario.getSimList()
    scenarioElement = ET.Element(scenarioTag)
    simList.storeGroundTruth(scenarioElement)
    scenario.storeScenarioSettings(scenarioElement)
    variationsElement = ET.SubElement(scenarioElement, variationsTag)
    for P_d in pdList:
        for lambda_phi in lambdaphiList:
            for N in nList:
                variationDict ={'P_d':P_d,
                                'lambda_phi':lambda_phi,
                                'N':N}
                variationElement = ET.SubElement(variationsElement, variationTag,
                                                 attrib={str(k):str(v) for k,v in variationDict.items()})
                runMonteCarloSimulations(variationElement,scenario,simList,nMonteCarlo,variationDict)
    if os.path.exists(path):
        modTime = os.path.getctime(path)
        timeString = datetime.datetime.fromtimestamp(modTime).strftime("%d.%m.%Y %H.%M")
        head, tail = os.path.split(path)
        newPath = os.path.join(head,tail+"_"+timeString)
        os.rename(path, newPath)
        print(newPath)
    hpf.writeElementToFile(path, scenarioElement)

def runMonteCarloSimulations(variationElement, scenario, simList, nSim, variationDict):
    # nCores = min(max(1, kwargs.get("c", os.cpu_count() - 1)), os.cpu_count())
    # pool = mp.Pool(nCores, initWorker)
    # results = pool.imap_unordered(functools.partial(runSimulation, sArgs), iIter, 1)

    P_d = variationDict['P_d']
    lambda_phi = variationDict['lambda_phi']
    N = variationDict['N']

    trackerArgs = (scenario.model,
                   scenario.radarPeriod,
                   lambda_phi,
                   lambda_nu)

    trackerKwargs = {'maxSpeedMS': maxSpeedMS,
                     'M_required': M_required,
                     'N_checks': N_checks,
                     'position': scenario.p0,
                     'radarRange': scenario.radarRange,
                     'eta2': eta2,
                     'N':N,
                     'P_d': scenario.P_d_true}

    storeTrackerData(variationElement, trackerArgs, trackerKwargs)
    for i in range(nSim):
        print("Running scenario iteration", i, end="", flush=True)
        seed = 5446 + i
        scanList, aisList = scenario.getSimulatedScenario(seed, simList, lambda_phi, P_d)
        runSimulation(variationElement, simList, scanList, aisList, trackerArgs, trackerKwargs,
                      seed=seed, markIterations=True)

def runSimulation(variationElement, simList, scanList, aisList, trackerArgs, trackerKwargs, **kwargs):
    tracker = tomht.Tracker(*trackerArgs, groundTruth=simList, **{**trackerKwargs, **kwargs})

    for measurementList in scanList:
        scanTime = measurementList.time
        aisPredictions = aisList.getMeasurements(scanTime)
        tracker.addMeasurementList(measurementList, aisPredictions)
        if kwargs.get('markIterations', False): print('.', end="", flush=True)
    tracker._storeRun(variationElement, **kwargs)
    if kwargs.get('markIterations'): print()

def storeTrackerData(variationElement, trackerArgs, trackerKwargs):
    trackersettingsElement = ET.SubElement(variationElement, trackerSettingsTag)
    for k,v in trackerKwargs.items():
        ET.SubElement(trackersettingsElement, str(k)).text = str(v)
    ET.SubElement(trackersettingsElement, "lambda_phi").text = str(trackerArgs[2])
    ET.SubElement(trackersettingsElement, "lambda_nu").text = str(trackerArgs[3])
