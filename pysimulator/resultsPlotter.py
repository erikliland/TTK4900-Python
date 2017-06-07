import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
# from mpl_toolkits.mplot3d import Axes3D
import seaborn.apionly as sns
import matplotlib.cm as cm
import xml.etree.ElementTree as ET
from pymht.utils.xmlDefinitions import *
import os
import numpy as np
import csv
import math
import ast
from pysimulator import simulationConfig
from pysimulator.scenarios.scenarios import scenarioList

colors = sns.color_palette(n_colors=5)
linestyleList = ['-','--','-.', ':']
legendFontsize = 7
labelFontsize = 8
titleFontsize = 10
figureWidth = 13.0*0.4 #inch
fullPageHeight = 18.0*0.4 #inch
halfPageHeight = 8.0 * 0.4 #inch
lineWidth = 1
dpi = 600

def exportTrackLossImprovement(filePathList):
    print("exportTrackLossImprovement")
    assert len(filePathList) > 1
    referenceFilePath = filePathList[0]
    referenceRoot = ET.parse(referenceFilePath).getroot()
    referenceGroundTruthElement = referenceRoot.find(groundtruthTag)
    referenceVariationsElement = referenceRoot.findall('.Variations[@preinitialized="True"]')[0]
    referencePerformance = _getTrackLossPlotData(referenceGroundTruthElement, referenceVariationsElement)

    comparePerformanceList = []
    for path in filePathList[1:]:
        root = ET.parse(path).getroot()
        groundtruthElement = root.find(groundtruthTag)
        variationsElement = root.findall('.Variations[@preinitialized="True"]')[0]
        performance = _getTrackLossPlotData(groundtruthElement, variationsElement)
        comparePerformanceList.append(performance)
    nCompareScenarios = len(filePathList[1:])
    pdSet = set()
    nSet = set()
    lambdaphiSet = set()
    for Pd, d1 in referencePerformance.items():
        pdSet.add(Pd)
        for N, d2 in d1.items():
            nSet.add(N)
            for lambda_phi in d2.keys():
                lambdaphiSet.add(lambda_phi)
    filePath = os.path.join(simulationConfig.path, 'plots', "Track_Loss_Improvement.csv")
    changeMatrix = np.zeros((len(pdSet)*len(nSet), nCompareScenarios))
    pdList = sorted(list(pdSet))
    nList = sorted(list(nSet))
    lambda_phi = sorted(list(lambdaphiSet))[-1]
    for i, performance in enumerate(comparePerformanceList):
        for j, Pd in enumerate(pdList):
            for k, N in enumerate(nList):
                referenceValue = referencePerformance[Pd][N][lambda_phi]
                compareValue = performance[Pd][N][lambda_phi]
                improvement = (compareValue-referenceValue)/referenceValue
                changeMatrix[j*len(nList) + k][i] = abs(improvement * -100)
    with open(filePath, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Pd", "N"]+ ["S"+str(int(scenarioIndex + 1)) for scenarioIndex in range(nCompareScenarios)])
        for i, row in enumerate(changeMatrix):
            Pd = pdList[i//len(nList)]
            N = nList[i%len(nList)]
            strRow = [str(int(Pd*100))+'%', str(int(N))] + ["{:3.0f}%".format(change) for change in row]
            writer.writerow(strRow)

def exportTrackingPercentageImprovement(filePathList):
    print("exportTrackingPercentageImprovement")
    assert len(filePathList) > 1
    referenceFilePath = filePathList[0]
    referenceRoot = ET.parse(referenceFilePath).getroot()
    referenceGroundTruthElement = referenceRoot.find(groundtruthTag)
    referenceVariationsElement = referenceRoot.findall('.Variations[@preinitialized="True"]')[0]
    referencePerformance = _getTrackingPercentagePlotData(referenceGroundTruthElement, referenceVariationsElement)

    comparePerformanceList = []
    for path in filePathList[1:]:
        root = ET.parse(path).getroot()
        groundtruthElement = root.find(groundtruthTag)
        variationsElement = root.findall('.Variations[@preinitialized="True"]')[0]
        performance = _getTrackingPercentagePlotData(groundtruthElement, variationsElement)
        comparePerformanceList.append(performance)
    nCompareScenarios = len(filePathList[1:])
    pdSet = set()
    nSet = set()
    lambdaphiSet = set()
    for Pd, d1 in referencePerformance.items():
        pdSet.add(Pd)
        for N, d2 in d1.items():
            nSet.add(N)
            for lambda_phi in d2.keys():
                lambdaphiSet.add(lambda_phi)
    filePath = os.path.join(simulationConfig.path, 'plots', "Tracking_Percentage_Improvement.csv")
    changeMatrix = np.zeros((len(pdSet)*len(nSet), nCompareScenarios))
    pdList = sorted(list(pdSet))
    nList = sorted(list(nSet))
    lambda_phi = sorted(list(lambdaphiSet))[-1]
    for i, performance in enumerate(comparePerformanceList):
        for j, Pd in enumerate(pdList):
            for k, N in enumerate(nList):
                referenceValue = referencePerformance[Pd][N][lambda_phi]
                compareValue = performance[Pd][N][lambda_phi]
                improvement = (compareValue-referenceValue)/referenceValue
                changeMatrix[j*len(nList) + k][i] = improvement * 100
    with open(filePath, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Pd", "N"]+ ["S"+str(int(scenarioIndex + 1)) for scenarioIndex in range(nCompareScenarios)])
        for i, row in enumerate(changeMatrix):
            Pd = pdList[i//len(nList)]
            N = nList[i%len(nList)]
            strRow = [str(int(Pd*100))+'%', str(int(N))] + ["{:3.0f}%".format(change) for change in row]
            writer.writerow(strRow)

def exportInitialState():
    print("exportInitialState")
    filePath = os.path.join(simulationConfig.path, 'plots', "Scenario_Initial_State.csv")
    with open(filePath, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["T", "NP", "EP", "NS", "ES"])
        scenario = scenarioList[0]
        for i, target in enumerate(scenario.initialTargets):
            s = target.state
            row = [str(i), str(s[1]), str(s[0]), str(s[3]), str(s[2])]
            writer.writerow(row)

def exportAisState():
    print("exportAisState")
    filePath = os.path.join(simulationConfig.path, 'plots', "Scenario_AIS_State.csv")
    with open(filePath, 'w') as csvfile:
        writer = csv.writer(csvfile)
        nScenario = len(scenarioList)
        nTargets = len(scenarioList[0].initialTargets)
        headerList = ["T"] + ["S{:}".format(i) for i in range(nScenario)]
        writer.writerow(headerList)

        for i in range(nTargets):
            aisList = []
            for j in range(nScenario):
                if scenarioList[j].initialTargets[i].mmsi is not None:
                    aisList.append(scenarioList[j].initialTargets[i].aisClass)
                else:
                    aisList.append('-')
            row = [str(i)] + aisList
            writer.writerow(row)

def plotOverlaidRadarMeasurements():
    print("plotOverlaidRadarMeasurements")
    scenario = scenarioList[0]
    simList = scenario.getSimList()
    from .simulationConfig import lambdaphiList
    nPlots = len(lambdaphiList)
    figure = plt.figure(figsize=(figureWidth/(nPlots-1), fullPageHeight), dpi=dpi)
    for i, lambda_phi in enumerate(lambdaphiList):
        scanList, _ = scenario.getSimulatedScenario(9895, simList, lambda_phi, 1.)
        ax = figure.add_subplot(nPlots, 1, i+1)
        scanList.plotFast(ax, alpha=0.4, markersize=1., markeredgewidth=0.)
        ax.axis("equal")
        xMin, xMax = ax.get_xlim()
        yMin, yMax = ax.get_ylim()
        x = xMin
        y = yMin + (yMax-yMin)*0.02
        ax.text(x,y,"$\lambda_\phi$ = {:}".format(lambda_phi),
                horizontalalignment='left', fontsize=9)
        ax.tick_params(axis='both', left='off', top='off', right='off',
                        bottom='off', labelleft='off', labeltop='off',
                        labelright='off', labelbottom='off')
    filePath = os.path.join(simulationConfig.path, 'plots', "ScenarioOverlaid.pdf")
    figure.tight_layout(pad=0.8, h_pad=0.8, w_pad=0.8)
    figure.savefig(filePath)
    figure.clf()
    plt.cla()
    plt.clf()
    plt.close()

def plotTrueTracks():
    print("plotTrueTracks")
    import itertools
    plt.cla()
    plt.clf()
    plt.close()
    scenario = scenarioList[0]
    colors1 = itertools.cycle(cm.rainbow(np.linspace(0, 1, len(scenario))))
    simList = scenario.getSimList()
    figure = plt.figure(figsize=(figureWidth, figureWidth), dpi=dpi)
    sns.set_style(style='white')
    ax = figure.gca()
    simList.plot(ax, colors=colors1, label=True, markevery=15, fontsize=labelFontsize,
                 alpha=0.8, markeredgewidth=0.)
    scenario.plotRadarOutline(ax, markCenter=False)
    ax.set_title("True tracks", fontsize=titleFontsize)
    ax.set_xlabel("East [m]", fontsize=labelFontsize)
    ax.set_ylabel("North [m]", fontsize=labelFontsize)
    ax.tick_params(labelsize=labelFontsize)
    ax.grid(True)
    filePath = os.path.join(simulationConfig.path, 'plots', "ScenarioTruth.pdf")
    figure.tight_layout(pad=0.8, h_pad=0.8, w_pad=0.8)
    figure.savefig(filePath)
    figure.clf()
    plt.cla()
    plt.clf()
    plt.close()

def plotTrackingPercentageExample():
    print("plotTrackingPercentageExample")
    from scenarios.defaults import maxSpeedMS, M_required, N_checks, eta2, eta2_ais
    import pymht.tracker as tomht
    import itertools
    scenario = scenarioList[0]
    simList = scenario.getSimList()
    lambda_phi = 5e-6
    P_d = 0.6
    N = 3
    markEvery = max(1, int(scenario.radarPeriod / scenario.simulationTimeStep))
    scanList, aisList = scenario.getSimulatedScenario(65945, simList, lambda_phi, P_d)
    trackerArgs = (scenario.model,
                   scenario.radarPeriod,
                   lambda_phi,
                   scenario.lambda_nu)

    trackerKwargs = {'maxSpeedMS': maxSpeedMS,
                     'M_required': M_required,
                     'N_checks': N_checks,
                     'position': scenario.p0,
                     'radarRange': scenario.radarRange,
                     'eta2': eta2,
                     'eta2_ais': eta2_ais,
                     'N': N,
                     'P_d': P_d}
    tracker = tomht.Tracker(*trackerArgs, **trackerKwargs)
    for measurementList in scanList:
        scanTime = measurementList.time
        aisMeasurements = aisList.getMeasurements(scanTime)
        tracker.addMeasurementList(measurementList, aisMeasurements)

    figure = plt.figure(figsize=(figureWidth/2, halfPageHeight), dpi=dpi)
    ax = figure.gca()
    colors = itertools.cycle(cm.rainbow(np.linspace(0, 0, 100)))
    simList.plot(ax,markevery=markEvery, alpha=0.2)
    tracker.plotActiveTracks(ax, colors=colors, markInitial=True, markStates=False, markID=False, markEnd=False)
    tracker.plotTerminatedTracks(ax, colors=colors, markInitial=True, markStates=False, markID=False, markEnd=False)
    ax.axis("equal")
    ax.set_xlim(-2700, -2000)
    ax.set_ylim(2000, 3400)
    ax.set_xlabel("East [m]", fontsize=labelFontsize)
    ax.set_ylabel("North [m]", fontsize=labelFontsize)
    ax.tick_params(axis='both', left='on', top='off', right='off',
                   bottom='on', labelleft='on', labeltop='off',
                   labelright='off', labelbottom='on', labelsize=labelFontsize)
    ax.xaxis.set_ticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 3))
    filePath = os.path.join(simulationConfig.path, 'plots', "TrackingPercentageExample.pdf")
    figure.tight_layout(pad=0.8, h_pad=0.8, w_pad=0.8)
    figure.savefig(filePath)
    figure.clf()
    plt.cla()
    plt.clf()
    plt.close()

def plotTrackLoss(loadFilePath):
    if loadFilePath is None: return
    print("plotTrackLoss", loadFilePath)
    savePath = _getSavePath(loadFilePath, "TrackLoss")
    tree = ET.ElementTree()
    try:
        tree.parse(loadFilePath)
    except FileNotFoundError as e:
        print(e)
        return
    scenarioElement = tree.getroot()
    scenarioSettingsElement = scenarioElement.find(scenariosettingsTag)
    groundtruthElement = scenarioElement.find(groundtruthTag)
    variationsElement = scenarioElement.findall('.Variations[@preinitialized="True"]')[0]
    initTime = float(scenarioSettingsElement.find("initTime").text)
    assert np.isfinite(initTime)
    plotData = _getTrackLossPlotData(groundtruthElement, variationsElement)
    figure = _plotTrackLossPercentage(plotData)
    figure.savefig(savePath)
    plt.close()

def plotTrackingPercentage(loadFilePath):
    if loadFilePath is None: return

    print("plotTrackingPercentage", loadFilePath)
    savePath = _getSavePath(loadFilePath, "TrackingPercentage")
    tree = ET.ElementTree()
    try:
        tree.parse(loadFilePath)
    except FileNotFoundError as e:
        print(e)
        return
    scenarioElement = tree.getroot()
    groundtruthElement = scenarioElement.find(groundtruthTag)
    variationsElement = scenarioElement.findall('.Variations[@preinitialized="True"]')[0]
    plotData = _getTrackingPercentagePlotData(groundtruthElement, variationsElement)
    figure = _plotTrackingPercentage(plotData)
    figure.savefig(savePath)
    plt.close()

def plotInitializationTime(loadFilePath):
    if loadFilePath is None: return
    print("plotInitializationTime", loadFilePath)
    tree = ET.ElementTree()
    try:
        tree.parse(loadFilePath)
    except FileNotFoundError as e:
        print(e)
        return
    scenarioElement = tree.getroot()
    groundtruthElement = scenarioElement.find(groundtruthTag)
    scenarioSettingsElement = scenarioElement.find(scenariosettingsTag)
    variationsElement = scenarioElement.find('.{0:}[@{1:}="{2:}"]'.format(variationsTag, preinitializedTag, False))
    try:
        variationsInitLog = _getInitializationTimePlotData(variationsElement)
    except AttributeError:
        print("It looks like this file is not analyzed. Please run analyseResults() before plotting.")
        return
    simLength = float(scenarioSettingsElement.find("simTime").text)
    timeStep = float(scenarioSettingsElement.find("radarPeriod").text)
    nTargets = len(groundtruthElement.findall(trackTag))
    timeArray = np.arange(0, simLength, timeStep)
    plotData = _processVariationInitLog(variationsInitLog, simLength, timeStep, nTargets)
    threshold = 0.8
    _plotInitializationTime2D(plotData, loadFilePath, simLength, timeStep, nTargets)
    _plotInitializationPerformance(plotData, loadFilePath, timeArray, threshold)
    plt.cla()
    plt.clf()
    plt.close()

def plotTrackCorrectness(loadFilePath):
    if loadFilePath is None: return
    print("plotTrackCorrectness", loadFilePath)
    savePath = _getSavePath(loadFilePath, "TrackingCorrectness")
    tree = ET.ElementTree()
    try:
        tree.parse(loadFilePath)
    except FileNotFoundError as e:
        print(e)
        return
    scenarioElement = tree.getroot()
    # groundtruthElement = scenarioElement.find(groundtruthTag)
    variationsElement = scenarioElement.findall('.Variations[@preinitialized="True"]')[0]
    plotData = _getTrackingCorrectnessPlotData(variationsElement)
    figure = _plotTrackingCorrectness(plotData)
    figure.savefig(savePath)
    plt.close()

def plotRuntime(loadFilePath):
    if loadFilePath is None: return
    print("plotRuntime", loadFilePath)
    savePath = _getSavePath(loadFilePath, "Runtime")
    tree = ET.ElementTree()
    try:
        tree.parse(loadFilePath)
    except FileNotFoundError as e:
        print(e)
        return
    scenarioElement = tree.getroot()
    groundtruthElement = scenarioElement.find(groundtruthTag)
    scenarioSettingsElement = scenarioElement.find(scenariosettingsTag)
    variationsElement = scenarioElement.find(
        '.{0:}[@{1:}="{2:}"]'.format(variationsTag, preinitializedTag, True))
    runtimeLog = _getRuntimePlotData(variationsElement, percentile=10.)
    radarPeriod = float(scenarioSettingsElement.find("radarPeriod").text)
    # print("runtimeLog", *[str(k)+str(v) for k,v in runtimeLog.items()], sep="\n")
    figure = _plotTimeLog(runtimeLog, radarPeriod)
    figure.savefig(savePath)
    plt.close()

def _getTrackLossPlotData(groundtruthElement, variationsElement):
    trueIdList = [t.get(idTag)
                  for t in groundtruthElement.findall(trackTag)]
    variationList = variationsElement.findall(variationTag)
    plotData = {}
    for variation in variationList:
        N = float(variation.get(nTag))
        P_d = float(variation.get(pdTag))
        lambda_phi = float(variation.get(lambdaphiTag))
        trackLossPercentageList = []
        for run in variation.findall(runTag):
            trackList = run.findall(trackTag)
            trackLossList = np.array([t.get(losttrackTag)
                                      for t in trackList
                                      if ((t.get(matchidTag) in trueIdList) and
                                          (t.get(matchidTag) == t.get(idTag)))])
            trackLossList =  trackLossList == str(True)
            trackLossPercentageList.append((np.sum(trackLossList)/len(trackLossList))*100)

        trackLossPercentageMean = np.mean(np.array(trackLossPercentageList))
        if P_d not in plotData:
            plotData[P_d] = {}
        if N not in plotData[P_d]:
            plotData[P_d][N] = {}
        if lambda_phi not in plotData[P_d][N]:
            plotData[P_d][N][lambda_phi] = trackLossPercentageMean
        else:
            raise KeyError("Duplicate key found")
    return plotData

def _getTrackingPercentagePlotData(groundtruthElement, variationsElement):
    trueIdList = [t.get(idTag)
                  for t in groundtruthElement.findall(trackTag)]
    variationList = variationsElement.findall(variationTag)
    plotData = {}
    for variation in variationList:
        N = float(variation.get(nTag))
        P_d = float(variation.get(pdTag))
        lambda_phi = float(variation.get(lambdaphiTag))
        trackingPercentList = []
        for runElement in variation.findall(runTag):
            trackingPercentArray = np.zeros_like(trueIdList, dtype=np.float32)
            for i, id in enumerate(trueIdList):
                matchTrackList = runElement.findall('.{0:}[@{1:}="{2:}"]'.format(trackTag, matchidTag, id))
                for track in matchTrackList:
                    trackingPercentArray[i] += float(track.get(trackpercentTag,0))
            trackingPercentList.append(trackingPercentArray)

        trackingPercentList = np.array(trackingPercentList, ndmin=2, dtype=np.float32)
        runMeanArray = np.mean(trackingPercentList, axis=0)
        trackingPercentageMean = np.mean(runMeanArray)

        if P_d not in plotData:
            plotData[P_d] = {}
        if N not in plotData[P_d]:
            plotData[P_d][N] = {}
        if lambda_phi not in plotData[P_d][N]:
            plotData[P_d][N][lambda_phi] = trackingPercentageMean
        else:
            raise KeyError("Duplicate key found")
    return plotData

def _getInitializationTimePlotData(variationsElement):
    variationsInitLog = {}
    variationElementList = variationsElement.findall(variationTag)
    for variationElement in variationElementList:
        M_init = variationElement.get(mInitTag)
        N_init = variationElement.get(nInitTag)
        P_d = variationElement.get(pdTag)
        lambda_phi = variationElement.get(lambdaphiTag)
        correctInitTimeLog = dict()
        falseInitTimeLog = dict()
        runElementList = variationElement.findall(runTag)
        for run in runElementList:
            initiationLogElement = run.find(initializationLogTag)
            correctTargetsElement = initiationLogElement.find(correctInitialTargetsTag)
            correctsTargetsList = ast.literal_eval(correctTargetsElement.text)
            for time, amount in correctsTargetsList:
                if time not in correctInitTimeLog:
                    correctInitTimeLog[time] = 0
                correctInitTimeLog[time] += amount

            falseTargetsElement = initiationLogElement.find(falseTargetsTag)
            falseTargetsList = ast.literal_eval(falseTargetsElement.text)
            for time, change in falseTargetsList:
                if time not in falseInitTimeLog:
                    falseInitTimeLog[time] = [0,0]
                falseInitTimeLog[time][0] += change[0]
                falseInitTimeLog[time][1] += (change[0] + change[1])

        for k,v in correctInitTimeLog.items():
            correctInitTimeLog[k] = float(v) / float(len(runElementList))

        for k,v in falseInitTimeLog.items():
            falseInitTimeLog[k][0] = float(v[0]) / float(len(runElementList))
            falseInitTimeLog[k][1] = float(v[1]) / float(len(runElementList))

        #TODO: http://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys
        if M_init not in variationsInitLog:
            variationsInitLog[M_init] = {}
        if N_init not in variationsInitLog[M_init]:
            variationsInitLog[M_init][N_init] = {}
        if lambda_phi not in variationsInitLog[M_init][N_init]:
            variationsInitLog[M_init][N_init][lambda_phi] = {}
        if P_d not in variationsInitLog[M_init][N_init][lambda_phi]:
            variationsInitLog[M_init][N_init][lambda_phi][P_d] = (correctInitTimeLog, falseInitTimeLog)
        else:
            raise KeyError("Duplicate key found")

    return variationsInitLog

def _getTrackingCorrectnessPlotData(variationsElement):
    plotData = {}
    variationList = variationsElement.findall(variationTag)
    for variationElement in variationList:
        N = float(variationElement.get(nTag))
        P_d = float(variationElement.get(pdTag))
        lambda_phi = float(variationElement.get(lambdaphiTag))
        originalRmsList = []
        smoothRmsList = []
        runList = variationElement.findall(runTag)
        for runElement in runList:
            trackList = runElement.findall(trackTag)
            for trackElement in trackList:
                pass

    return plotData

def _getRuntimePlotData(variationsElement, percentile):
    variationList = variationsElement.findall(variationTag)
    plotData = {}
    for variation in variationList:
        N = float(variation.get(nTag))
        P_d = float(variation.get(pdTag))
        lambda_phi = float(variation.get(lambdaphiTag))
        runtimeList = []
        for runElement in variation.findall(runTag):
            runtimeElement = runElement.find(runtimeTag)
            totalElement = runtimeElement.find(totalTimeTag)
            totalTimeString = totalElement.text
            totalTimeString = totalTimeString.strip('[]')
            totalTimeStringList = totalTimeString.split()
            totalTimeList = np.array(totalTimeStringList, dtype=np.float64)
            runtimeList.append(totalTimeList)
        runtimeArray = np.array(runtimeList, ndmin=2, dtype=np.float64)
        meanRuntimeArray = np.mean(runtimeArray, axis=0)
        percentiles = np.percentile(meanRuntimeArray,[percentile, 100.-percentile])
        meanRuntime = np.mean(meanRuntimeArray)
        # print("runtimeArray", np.array_str(runtimeArray, precision=6))
        # print("meanRunetimeArray", np.array_str(meanRuntimeArray, precision=6))
        # print("Percentile", np.array_str(percentiles, precision=6))
        # print()
        # plt.hist(meanRuntimeArray)
        # plt.show()
        if P_d not in plotData:
            plotData[P_d] = {}
        if lambda_phi not in plotData[P_d]:
            plotData[P_d][lambda_phi] = {}
        if N not in plotData[P_d][lambda_phi]:
            plotData[P_d][lambda_phi][N] = (meanRuntime, percentiles)
        else:
            raise KeyError("Duplicate key found")
    return plotData

def _plotTrackLossPercentage(plotData):
    figure = plt.figure(figsize=(figureWidth,halfPageHeight), dpi=dpi)
    colors = sns.color_palette(n_colors=5)
    sns.set_style(style='white')
    ax = figure.gca()
    maxTrackloss = 0.
    lambdaPhiSet = set()

    trackLossList = []
    for j, (P_d, d1) in enumerate(plotData.items()):
        for i, (N, d2) in enumerate(d1.items()):
            x = []
            y = []
            for lambda_phi, trackLossPercentage in d2.items():
                x.append(lambda_phi)
                y.append(trackLossPercentage)
                maxTrackloss = max(maxTrackloss, trackLossPercentage)
                lambdaPhiSet.add(lambda_phi)
            x, y = (list(t) for t in zip(*sorted(zip(x, y))))
            x = np.array(x)
            y = np.array(y)
            trackLossList.append((P_d, N, x, y))

    trackLossList.sort(key=lambda tup: float(tup[1]))
    trackLossList.sort(key=lambda tup: float(tup[0]))

    pdSet = set()
    nSet = set()
    for P_d, N, x, y in trackLossList:
        if P_d not in pdSet:
            nSet.clear()
        pdSet.add(P_d)
        nSet.add(N)
        ax.plot(x, y,
                label = "$P_D$={0:}, N={1:.0f}".format(P_d, N),
                c = colors[len(nSet)-1],
                linestyle=linestyleList[len(pdSet)-1],
                linewidth = lineWidth,
                marker='*' if len(x) == 1 else None)
    lambdaPhiList = list(lambdaPhiSet)
    lambdaPhiList.sort()

    ax.legend(loc=0, ncol=len(pdSet), fontsize=legendFontsize)
    ax.set_xlabel("$\lambda_{\phi}$", fontsize=labelFontsize)
    ax.set_ylabel("Track loss (%)", fontsize=labelFontsize)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.set_ylim(-1, 100.0)
    ax.tick_params(labelsize=labelFontsize)
    yStart, yEnd = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(0., yEnd * 1.01, 10))
    ax.xaxis.set_ticks(lambdaPhiList)
    sns.despine(ax=ax, offset=0)
    figure.tight_layout(pad=0.8, h_pad=0.8, w_pad=0.8)
    return figure

def _plotTrackingPercentage(plotData):
    figure = plt.figure(figsize=(figureWidth, halfPageHeight), dpi=dpi)
    sns.set_style(style='white')
    ax = figure.gca()

    minTracking = 100.
    lambdaPhiSet = set()

    trackingPercentageList = []
    for j, (P_d, d1) in enumerate(plotData.items()):
        for i, (N, d2) in enumerate(d1.items()):
            x = []
            y = []
            for lambda_phi, trackingPercentage in d2.items():
                x.append(lambda_phi)
                y.append(trackingPercentage)
                minTracking = min(minTracking, trackingPercentage)
                lambdaPhiSet.add(lambda_phi)
            x = np.array(x)
            y = np.array(y)
            x, y = (list(t) for t in zip(*sorted(zip(x, y))))
            trackingPercentageList.append((P_d, N, x, y))

    trackingPercentageList.sort(key=lambda tup: float(tup[1]), reverse=False)
    trackingPercentageList.sort(key=lambda tup: float(tup[0]), reverse=False)

    pdSet = set()
    nSet = set()
    for P_d, N, x, y in trackingPercentageList:
        if P_d not in pdSet:
            nSet.clear()
        pdSet.add(P_d)
        nSet.add(N)
        minTracking = min(minTracking, np.min(y))
        ax.plot(x, y,
                label="$P_D$={0:}, N={1:.0f}".format(P_d, N),
                c=colors[len(nSet) - 1],
                linestyle=linestyleList[len(pdSet) - 1],
                linewidth=lineWidth,
                marker='*' if len(x)==1 else None)

    lambdaPhiList = list(lambdaPhiSet)
    lambdaPhiList.sort()

    ax.legend(loc=0, ncol=len(pdSet), fontsize=legendFontsize)
    ax.set_xlabel("$\lambda_{\phi}$", fontsize=labelFontsize)
    ax.set_ylabel("Average tracking percentage", fontsize=labelFontsize)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.set_ylim(min(30.0, minTracking), 100.01)
    ax.tick_params(labelsize=labelFontsize)
    sns.despine(ax=ax, offset=0)
    ax.xaxis.set_ticks(lambdaPhiList)
    figure.tight_layout(pad=0.8, h_pad=0.8, w_pad=0.8)
    return figure

def _plotInitializationPerformance(plotData, loadFilePath, timeArray, threshold):
    bigM = 1000
    normalizationConstant = 26.2
    savePath = _getSavePath(loadFilePath, "Performance")
    mSet = set()
    nSet = set()
    resultList = []
    for M_init, d1 in plotData.items():
        mSet.add(int(M_init))
        for N_init, d2 in d1.items():
            nSet.add(int(N_init))
            (cpfmList, _, accFalseTrackList) = d2
            costList = []
            for i, (P_d, lambda_phi, accFalseTrack) in enumerate(accFalseTrackList):
                (_, _, cpmf) = cpfmList[i]
                avgNumFalseTrackAlive = np.mean(accFalseTrack)
                if not np.any(np.array(cpmf)>threshold):
                    index = np.nan
                    initTime = bigM
                else:
                    index = np.argmax(cpmf>threshold)
                    initTime = timeArray[index]
                cost = float(initTime) * (1. + float(avgNumFalseTrackAlive))
                costList.append(cost)
            avg = np.mean(costList)
            resultList.append((int(M_init), int(N_init), float(avg)))
    plotArray = np.ones((len(mSet),len(nSet)), dtype=np.float32)*np.nan
    mList = sorted(list(mSet))
    nList = sorted(list(nSet))
    for M, N, cost in resultList:
        mIndex = mList.index(M)
        nIndex = nList.index(N)
        costNormalized = np.sqrt(cost)/ normalizationConstant
        assert costNormalized <= 1., "New normalization = {:}".format(costNormalized*normalizationConstant)
        plotArray[mIndex][nIndex] = costNormalized
    print("max cost", np.nanmax(plotArray))
    figure = plt.figure(figsize=(figureWidth, fullPageHeight/3), dpi=dpi)
    ax = figure.gca()
    cax = ax.matshow(plotArray)
    cax.set_clim(0,1)
    cax.set_cmap(cm.jet)
    colorbar = figure.colorbar(cax)
    colorbar.ax.tick_params(labelsize=labelFontsize)
    colorbar.ax.invert_yaxis()
    ax.set_xlabel("N", fontsize=labelFontsize)
    ax.set_ylabel("M", fontsize=labelFontsize)
    ax.set_xticklabels(['']+nList)
    ax.set_yticklabels(['']+mList)
    ax.tick_params(labelsize=labelFontsize)
    ax.xaxis.tick_bottom()
    ax.set_title("M/N initialization", fontsize=titleFontsize)
    figure.tight_layout()
    figure.savefig(savePath)
    figure.clf()
    plt.cla()
    plt.clf()
    plt.close()

def _plotInitializationTime2D(plotData, loadFilePath, simLength, timeStep, nTargets):
    timeArray = np.arange(0, simLength, timeStep)
    for M_init, d1 in plotData.items():
        for N_init, d2 in d1.items():
            figure = plt.figure(figsize=(figureWidth, fullPageHeight), dpi=dpi)
            ax11 = figure.add_subplot(311)
            ax12 = figure.add_subplot(312)
            ax13 = figure.add_subplot(313)
            sns.set_style(style='white')
            savePath = _getSavePath(loadFilePath, "Time({0:}-{1:})".format(M_init, N_init))
            (cpfmList, falseCPFMlist, accFalseTrackList) = d2
            pdSet = set()
            lambdaPhiSet = set()
            for P_d, lambda_phi, cpmf in cpfmList:
                if P_d not in pdSet:
                    lambdaPhiSet.clear()
                pdSet.add(P_d)
                lambdaPhiSet.add(lambda_phi)
                ax11.plot(timeArray,
                          cpmf,
                          label="$P_D$ = {0:}, $\lambda_\phi$ = {1:}".format(P_d, float(lambda_phi)),
                          c=colors[len(pdSet)-1],
                          linestyle=linestyleList[len(lambdaPhiSet)-1],
                          linewidth=lineWidth)

            pdSet = set()
            lambdaPhiSet = set()
            for P_d, lambda_phi, cpmf in falseCPFMlist:
                if P_d not in pdSet:
                    lambdaPhiSet.clear()
                pdSet.add(P_d)
                lambdaPhiSet.add(lambda_phi)
                ax12.semilogy(timeArray,
                              cpmf + (1e-10),
                              label="$P_D$ = {0:}, $\lambda_\phi$ = {1:}".format(P_d, float(lambda_phi)),
                              c=colors[len(pdSet)-1],
                              linestyle=linestyleList[len(lambdaPhiSet)-1],
                              linewidth=lineWidth)

            pdSet = set()
            lambdaPhiSet = set()
            for P_d, lambda_phi, accFalseTrack in accFalseTrackList:
                if P_d not in pdSet:
                    lambdaPhiSet.clear()
                pdSet.add(P_d)
                lambdaPhiSet.add(lambda_phi)
                ax13.plot(timeArray,
                          accFalseTrack,
                          label="$P_D$ = {0:}, $\lambda_\phi$ = {1:}".format(P_d, float(lambda_phi)),
                          c=colors[len(pdSet) - 1],
                          linestyle=linestyleList[len(lambdaPhiSet) - 1],
                          linewidth=lineWidth)

            ax11.set_xlabel("Time [s]", fontsize=labelFontsize)
            ax11.set_ylabel("Average cpfm", fontsize=labelFontsize)
            ax11.set_title("Cumulative Probability Mass Function", fontsize=titleFontsize)
            ax11.legend(loc=4, ncol=len(pdSet), fontsize=legendFontsize)
            ax11.grid(False)
            ax11.set_xlim(0, simLength)
            ax11.set_ylim(0,1)
            ax11.xaxis.set_ticks(np.arange(0,simLength+15, 15))
            ax11.tick_params(labelsize=labelFontsize)
            sns.despine(ax=ax11, offset=0)

            ax12.set_xlabel("Time [s]", fontsize=labelFontsize)
            ax12.set_ylabel("Average number of tracks", fontsize=labelFontsize)
            ax12.set_title("Accumulative number of erroneous tracks", fontsize=titleFontsize)
            ax12.grid(False)
            ax12.set_xlim(0, simLength)
            ax12.set_ylim(1e-3,1000)
            ax12.xaxis.set_ticks(np.arange(0,simLength+15, 15))
            ax12.tick_params(labelsize=labelFontsize)
            sns.despine(ax=ax12, offset=0)

            ax13.set_xlabel("Time [s]", fontsize=labelFontsize)
            ax13.set_ylabel("Average number of tracks", fontsize=labelFontsize)
            ax13.set_title("Number of erroneous tracks alive", fontsize=titleFontsize)
            ax13.grid(False)
            ax13.set_xlim(0, simLength)
            ax13.set_ylim(-0.02, max(1,ax13.get_ylim()[1]))
            ax13.xaxis.set_ticks(np.arange(0,simLength+15, 15))
            ax13.tick_params(labelsize=labelFontsize)
            sns.despine(ax=ax13, offset=0)

            figure.tight_layout(pad=0.8, h_pad=0.8, w_pad=0.8)
            figure.savefig(savePath)
            figure.clf()

            plt.close()

def _plotTrackingCorrectness(plotData):
    return plt.figure()

def _plotTimeLog(plotData, radarPeriod=None):
    figure = plt.figure(figsize=(figureWidth, halfPageHeight), dpi=dpi)
    ax = figure.gca()
    sns.set_style(style='white')

    nSet = set()
    for j, P_d in enumerate(sorted(plotData, reverse=True)):
        d1 = plotData[P_d]
        for i, lambda_phi in enumerate(sorted(d1)):
            d2 = d1[lambda_phi]
            x = []
            y = []
            for N, (meanRuntime, percentiles) in d2.items():
                x.append(N)
                y.append(meanRuntime)
                nSet.add(N)
            x = np.array(x)
            y = np.array(y)
            x, y = (list(t) for t in zip(*sorted(zip(x, y))))
            ax.plot(x, y, linestyle=linestyleList[i], color=colors[j],
                    label="$P_D$={0:}, $\lambda_\phi$={1:}".format(P_d, lambda_phi),
                    linewidth=lineWidth)
    if radarPeriod is not None:
        ax.plot([min(nSet), max(nSet)],[radarPeriod, radarPeriod],
                color='black', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xlim(ax.get_xlim()[0]-0.5, ax.get_xlim()[1]+0.5)
    ax.set_ylim(0.0,5.0)
    ax.set_title("Tracking iteration runtime", fontsize=titleFontsize)
    ax.set_xlabel("N", fontsize=labelFontsize, labelpad=0)
    ax.set_ylabel("Average iteration time [s]", fontsize=labelFontsize)
    ax.xaxis.set_ticks(sorted(list(nSet)))
    ax.tick_params(labelsize=labelFontsize)

    ax.legend(loc=2, fontsize=legendFontsize, ncol=3)
    ax.grid(False)

    sns.despine(ax=ax)
    figure.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    return figure

def _getSavePath(loadFilePath, nameAdd):
    head, tail = os.path.split(loadFilePath)
    name, extension = os.path.splitext(tail)
    savePath = os.path.join(head, 'plots')
    saveFilePath = os.path.join(savePath, name + "-" + nameAdd + ".pdf")
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    return saveFilePath

def _processVariationInitLog(variationInitLog, simLength, timeStep, nTargets):
    plotData = dict()
    timeArray = np.arange(0, simLength, timeStep)
    for M_init, d1 in variationInitLog.items():
        if M_init not in plotData:
            plotData[M_init] = dict()
        for N_init, d2 in d1.items():
            if N_init not in plotData[M_init]:
                plotData[M_init][N_init] = dict()
            cpfmList = []
            falseCPFMlist = []
            accFalseTrackList = []
            for k, (lambda_phi, d3) in enumerate(d2.items()):
                for j, (P_d, (correctInitTimeLog, falseInitTimeLog)) in enumerate(d3.items()):
                    falsePFM = np.zeros_like(timeArray)
                    pmf = np.zeros_like(timeArray)
                    falseTrackDelta = np.zeros_like(timeArray)
                    for i, time in enumerate(timeArray):
                        if str(time) in correctInitTimeLog:
                            pmf[i] = correctInitTimeLog[str(time)]
                        if str(time) in falseInitTimeLog:
                            falsePFM[i] = falseInitTimeLog[str(time)][0]
                            falseTrackDelta[i] = falseInitTimeLog[str(time)][1]
                    cpmf = np.cumsum(pmf) / float(nTargets)
                    falseCPFM = np.cumsum(falsePFM)
                    falseTrackDelta = np.cumsum(falseTrackDelta)
                    cpfmList.append((P_d, lambda_phi, cpmf))
                    falseCPFMlist.append((P_d, lambda_phi, falseCPFM))
                    accFalseTrackList.append((P_d, lambda_phi, falseTrackDelta))
            cpfmList.sort(key=lambda tup: float(tup[1]))
            cpfmList.sort(key=lambda tup: float(tup[0]), reverse=True)
            falseCPFMlist.sort(key=lambda tup: float(tup[1]))
            falseCPFMlist.sort(key=lambda tup: float(tup[0]), reverse=True)
            accFalseTrackList.sort(key=lambda tup: float(tup[1]))
            accFalseTrackList.sort(key=lambda tup: float(tup[0]), reverse=True)

            plotData[M_init][N_init] = (cpfmList, falseCPFMlist, accFalseTrackList)

    return plotData