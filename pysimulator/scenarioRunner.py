import pymht.tracker as tomht
import pymht.utils.helpFunctions as hpf
import xml.etree.ElementTree as ET
from pymht.utils.xmlDefinitions import *
from pysimulator.scenarios.defaults import *
import os
import datetime


def runPreinitializedVariations(scenario, path, pdList, lambdaphiList, nList, nMonteCarlo,
                                baseSeed, **kwargs):
    simList = scenario.getSimList()

    try:
        scenarioElement = ET.parse(path).getroot()
    except Exception:
        scenarioElement = None

    if kwargs.get('overwrite', False) or (scenarioElement is None):
        print("Creating scenarioElement", scenario.name)
        scenarioElement = ET.Element(scenarioTag, attrib={nameTag: scenario.name})
        simList.storeGroundTruth(scenarioElement, scenario)
        scenario.storeScenarioSettings(scenarioElement)

    variationsElement = scenarioElement.find('.Variations[@preinitialized="True"]')

    if variationsElement is None:
        print("Creating variationsElement", True)
        variationsElement = ET.SubElement(
            scenarioElement,variationsTag,attrib={preinitializedTag: "True"})
    print("variationsElement",variationsElement.attrib)
    for P_d in pdList:
        for lambda_phi in lambdaphiList:
            for N in nList:
                variationDict = {pdTag: P_d,
                                 lambdaphiTag: lambda_phi,
                                 nTag: N}
                variationElement = variationsElement.find(
                    '.Variation[@N="{0:}"][@Pd="{1:}"][@lambda_phi="{2:}"]'.format(N, P_d, lambda_phi))
                if variationElement is None:
                    print("Creating variationElement", P_d, N, lambda_phi)
                    variationElement = ET.SubElement(
                        variationsElement, variationTag,
                        attrib={str(k): str(v) for k, v in variationDict.items()})
                print("variationElement", variationElement.attrib)
                runMonteCarloSimulations(
                    variationElement, scenario, simList, nMonteCarlo, baseSeed,
                    variationDict, preInitialized=True, **kwargs)

    _renameOldFiles(path)
    hpf.writeElementToFile(path, scenarioElement)

def runInitializationVariations(scenario, path, pdList, lambdaphiList, M_N_list, nMonteCarlo,
                                baseSeed, **kwargs):
    simList = scenario.getSimList()

    try:
        scenarioElement = ET.parse(path).getroot()
    except Exception:
        scenarioElement = None

    if kwargs.get('overwrite', False) or (scenarioElement is None):
        print("Creating scenarioElement", scenario.name)
        scenarioElement = ET.Element(scenarioTag, attrib={nameTag: scenario.name})
        simList.storeGroundTruth(scenarioElement, scenario)
        scenario.storeScenarioSettings(scenarioElement)

    variationsElement = scenarioElement.find('.Variations[@preinitialized="False"]')

    if variationsElement is None:
        print("Creating variationsElement", False)
        variationsElement = ET.SubElement(
            scenarioElement,variationsTag,attrib={preinitializedTag: "False"})
    print("variationsElement",variationsElement.attrib)
    for P_d in pdList:
        for lambda_phi in lambdaphiList:
            for (M, N) in M_N_list:
                variationDict = {pdTag: P_d,
                                 lambdaphiTag: lambda_phi,
                                 nTag: 6,
                                 mInitTag: M,
                                 nInitTag: N}
                variationElement = variationsElement.find(
                    '.Variation[@M_init="{0:}"][@N_init="{1:}"][@Pd="{2:}"][@lambda_phi="{3:}"]'.format(M, N, P_d, lambda_phi))
                if variationElement is None:
                    # print("Creating variationElement", P_d, N, lambda_phi)
                    variationElement = ET.SubElement(
                        variationsElement, variationTag,
                        attrib={str(k): str(v) for k, v in variationDict.items()})
                print("variationElement", variationElement.attrib)
                runMonteCarloSimulations(
                    variationElement, scenario, simList, nMonteCarlo, baseSeed,
                    variationDict, preInitialized=False, **kwargs)

    _renameOldFiles(path)
    hpf.writeElementToFile(path, scenarioElement)

def _renameOldFiles(path):
    if os.path.exists(path):
        modTime = os.path.getmtime(path)
        timeString = datetime.datetime.fromtimestamp(modTime).strftime("%d.%m.%Y %H.%M")
        head, tail = os.path.split(path)
        filename, extension = os.path.splitext(tail)
        newPath = os.path.join(head, filename + "_" + timeString + extension)
        os.rename(path, newPath)


def runMonteCarloSimulations(variationElement, scenario, simList, nSim, baseSeed,
                             variationDict, preInitialized, **kwargs):
    P_d = variationDict[pdTag]
    lambda_phi = variationDict[lambdaphiTag]
    N = variationDict[nTag]

    trackerArgs = (scenario.model,
                   scenario.radarPeriod,
                   lambda_phi,
                   lambda_nu)

    trackerKwargs = {'maxSpeedMS': maxSpeedMS,
                     'M_required': variationDict.get(mInitTag,M_required),
                     'N_checks': variationDict.get(nInitTag,N_checks),
                     'position': scenario.p0,
                     'radarRange': scenario.radarRange,
                     'eta2': eta2,
                     'N': N,
                     'P_d': scenario.P_d_true,
                     'dynamicWindow': False}
    trackersettingsElement = variationElement.find(trackerSettingsTag)
    if trackersettingsElement is None:
        storeTrackerData(variationElement, trackerArgs, trackerKwargs)
    for i in range(nSim):
        # print("i", i)
        runElement = variationElement.find('Run[@iteration="{:}"]'.format(i+1))
        if runElement is not None:
            # print("Skipping")
            continue
        if kwargs.get('printLog', True):
            print("Running scenario iteration", i, end="", flush=True)
        seed = baseSeed + i
        scanList, aisList = scenario.getSimulatedScenario(seed, simList, lambda_phi, P_d)

        try:
            # print("Running", variationDict)
            runSimulation(variationElement, simList, scanList, aisList, trackerArgs,
                          trackerKwargs, preInitialized,
                          seed=seed, **kwargs)
        except Exception as e:
            import traceback
            print("Scenario:", scenario.name)
            print("preInitialized", preInitialized)
            print("variationDict", variationDict)
            print("Iteration", i)
            print("Seed", seed)
            print(e)

def runSimulation(variationElement, simList, scanList, aisList, trackerArgs,
                  trackerKwargs, preInitialized, **kwargs):

    tracker = tomht.Tracker(*trackerArgs, **{**trackerKwargs, **kwargs})

    startIndex = 1 if preInitialized else 0
    if preInitialized:
        tracker.preInitialize(simList)

    for measurementList in scanList[startIndex:]:
        scanTime = measurementList.time
        aisPredictions = aisList.getMeasurements(scanTime)
        tracker.addMeasurementList(measurementList, aisPredictions)
        if kwargs.get('printLog', True):
            print('.', end="", flush=True)
    tracker._storeRun(variationElement, **kwargs)
    if kwargs.get('printLog', True):
        print()


def storeTrackerData(variationElement, trackerArgs, trackerKwargs):
    trackersettingsElement = ET.SubElement(variationElement, trackerSettingsTag)
    for k, v in trackerKwargs.items():
        ET.SubElement(trackersettingsElement, str(k)).text = str(v)
    ET.SubElement(trackersettingsElement, "lambda_phi").text = str(trackerArgs[2])
    ET.SubElement(trackersettingsElement, "lambda_nu").text = str(trackerArgs[3])
