import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import seaborn.apionly as sns
import xml.etree.ElementTree as ET
from pymht.utils.xmlDefinitions import *
import os
import numpy as np
import csv
import ast
import simulationConfig
from pysimulator.scenarios.scenarios import scenarioList


def exportInitialState():
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

def plotTrueTracks():
    import matplotlib.cm as cm
    import itertools
    import pymht.utils.helpFunctions as hpf
    scenario = scenarioList[0]
    colors1 = itertools.cycle(cm.rainbow(np.linspace(0, 1, len(scenario))))
    simList = scenario.getSimList()
    figure = plt.figure(num=1, figsize=(9, 9), dpi=90)
    hpf.plotTrueTrack(simList, colors=colors1, label=True, markevery=10)
    hpf.plotRadarOutline(scenario.p0, scenario.radarRange, markCenter=False)
    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.title("True tracks", fontsize=18)
    plt.grid(True)
    filePath = os.path.join(simulationConfig.path, 'plots', "ScenarioTruth.pdf")
    plt.tight_layout()
    figure.savefig(filePath)

def plotTrackLoss(loadFilePath):
    print("plotTrackLoss", loadFilePath)
    savePath = _getSavePath(loadFilePath, "TrackLoss")
    tree = ET.parse(loadFilePath)
    scenarioElement = tree.getroot()
    groundtruthElement = scenarioElement.find(groundtruthTag)
    variationsElement = scenarioElement.findall('.Variations[@preinitialized="True"]')[0]
    plotData = _getTrackLossPlotData(groundtruthElement, variationsElement)
    figure = _plotTrackLossPercentage(plotData)
    figure.savefig(savePath)

def plotTrackingPercentage(loadFilePath):
    print("plotTrackingPercentage", loadFilePath)
    savePath = _getSavePath(loadFilePath, "TrackingPercentage")
    tree = ET.parse(loadFilePath)
    scenarioElement = tree.getroot()
    groundtruthElement = scenarioElement.find(groundtruthTag)
    variationsElement = scenarioElement.findall('.Variations[@preinitialized="True"]')[0]
    plotData = _getTrackingPercentagePlotData(groundtruthElement, variationsElement)
    figure = _plotTrackingPercentage(plotData)
    figure.savefig(savePath)

def plotInitializationTime(loadFilePath):
    if loadFilePath is None: return
    print("plotInitializationTime", loadFilePath)
    tree = ET.parse(loadFilePath)
    scenarioElement = tree.getroot()
    groundtruthElement = scenarioElement.find(groundtruthTag)
    scenarioSettingsElement = scenarioElement.find(scenariosettingsTag)
    variationsElement = scenarioElement.find('.Variations[@preinitialized="False"]')
    variationsInitLog = _getInitializationTimePlotData(variationsElement)
    # print("variationsInitLog", *[str(k)+str(v) for k,v in variationsInitLog.items()], sep="\n")
    simLength = float(scenarioSettingsElement.find("simTime").text)
    timeStep = float(scenarioSettingsElement.find("radarPeriod").text)
    nTargets = len(groundtruthElement.findall(trackTag))
    _plotInitializationTime2D(variationsInitLog, loadFilePath, simLength, timeStep, nTargets)

def plotTrackCorrectness(loadFilePath):
    print("plotTrackCorrectness", loadFilePath)
    savePath = _getSavePath(loadFilePath, "TrackingCorrectness")
    tree = ET.parse(loadFilePath)
    scenarioElement = tree.getroot()
    groundtruthElement = scenarioElement.find(groundtruthTag)
    variationsElement = scenarioElement.findall('.Variations[@preinitialized="True"]')[0]
    plotData = _getTrackingCorrectnessPlotData(variationsElement)
    figure = _plotTrackingCorrectness(plotData)
    figure.savefig(savePath)

def plotRuntime(filePath):
    pass

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
                                      if t.get(matchidTag) in trueIdList])
            trackLossList = trackLossList == str(True)
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
        trackingLength = np.zeros(2, dtype=np.int)
        for run in variation.findall(runTag):
            trackList = run.findall(trackTag)
            trackingLengthArray = np.array([[t.get(goodtimematchlengthTag),
                                             t.get(timematchlengthTag)]
                                            for t in trackList
                                            if t.get(matchidTag) in trueIdList],
                                           ndmin=2, dtype=np.int)
            trackingLengthArray = np.sum(trackingLengthArray, axis=0)
            trackingLength +=  trackingLengthArray

        trackingPercentage = (float(trackingLength[0]) / float(trackingLength[1])) * 100

        if P_d not in plotData:
            plotData[P_d] = {}
        if N not in plotData[P_d]:
            plotData[P_d][N] = {}
        if lambda_phi not in plotData[P_d][N]:
            plotData[P_d][N][lambda_phi] = trackingPercentage
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

def _plotTrackLossPercentage(plotData):
    figure = plt.figure(figsize=(10, 10), dpi=100)
    colors = sns.color_palette(n_colors=5)
    sns.set_style(style='white')
    ax = figure.add_subplot(111, projection='3d')

    maxTrackloss = 0.
    lambdaPhiSet = set()

    for j, (P_d, d1) in enumerate(plotData.items()):
        for i, (N, d2) in enumerate(d1.items()):
            x = []
            z = []
            for lambda_phi, trackLossPercentage in d2.items():
                x.append(lambda_phi)
                z.append(trackLossPercentage)
                maxTrackloss = max(maxTrackloss, trackLossPercentage)
                lambdaPhiSet.add(lambda_phi)
            x = np.array(x)
            y = np.ones(len(x))*P_d*100
            z = np.array(z)
            ax.plot(x,y,z,
                    label = "N="+"{:.0f}".format(N) if j == 0 else None,
                    c = colors[i],
                    linewidth = 4)
    lambdaPhiList = list(lambdaPhiSet)
    lambdaPhiList.sort()

    ax.view_init(15, -165)
    ax.legend(loc='upper right', bbox_to_anchor=(0.5, 0.8), fontsize=18)
    ax.set_xlabel("$\lambda_{\phi}$", fontsize=18, labelpad=30)
    ax.set_zlabel("\nTrack loss (%)", fontsize=18, linespacing=3)
    ax.set_ylabel("\nProbability of detection (%)", fontsize=18, linespacing=2)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.set_zlim(0, maxTrackloss)
    ax.tick_params(labelsize=16, pad=8)
    yStart, yEnd = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(yStart, yEnd * 1.01, 10))
    ax.xaxis.set_ticks(lambdaPhiList)
    xTickLabels = ax.xaxis.get_ticklabels()
    for label in xTickLabels:
        label.set_verticalalignment('bottom')
        label.set_horizontalalignment('left')
        label.set_rotation(0)
    figure.tight_layout()
    return figure

def _plotTrackingPercentage(plotData):
    figure = plt.figure(figsize=(10, 10), dpi=100)
    colors = sns.color_palette(n_colors=5)
    sns.set_style(style='white')
    ax = figure.add_subplot(111, projection='3d')

    minTracking = 100.
    lambdaPhiSet = set()

    for j, (P_d, d1) in enumerate(plotData.items()):
        for i, (N, d2) in enumerate(d1.items()):
            x = []
            z = []
            for lambda_phi, trackingPercentage in d2.items():
                x.append(lambda_phi)
                z.append(trackingPercentage)
                minTracking = min(minTracking, trackingPercentage)
                lambdaPhiSet.add(lambda_phi)
            x = np.array(x)
            y = np.ones(len(x)) * P_d * 100
            z = np.array(z)
            ax.plot(x, y, z,
                    label="N=" + "{:.0f}".format(N) if j == 0 else None,
                    c=colors[i],
                    linewidth=4)
    lambdaPhiList = list(lambdaPhiSet)
    lambdaPhiList.sort()

    ax.view_init(15, -165)
    ax.legend(bbox_to_anchor=(0.4, 0.5), fontsize=18)
    ax.set_xlabel("$\lambda_{\phi}$", fontsize=18, labelpad=30)
    ax.set_zlabel("\nTracking (%)", fontsize=18, linespacing=3)
    ax.set_ylabel("\nProbability of detection (%)", fontsize=18, linespacing=2)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.set_zlim(96, 100)
    ax.tick_params(labelsize=16, pad=8)

    yStart, yEnd = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(yStart, yEnd * 1.01, 10))
    ax.xaxis.set_ticks(lambdaPhiList)
    xTickLabels = ax.xaxis.get_ticklabels()
    for label in xTickLabels:
        label.set_verticalalignment('bottom')
        label.set_horizontalalignment('left')
        label.set_rotation(0)
    figure.tight_layout()
    return figure

def _plotInitializationTime2D(plotData, loadFilePath, simLength, timeStep, nTargets):
    timeArray = np.arange(0, simLength, timeStep)
    for M_init, d1 in plotData.items():
        for N_init, d2 in d1.items():
            figure1 = plt.figure(figsize=(10, 14), dpi=100)
            # figure2 = plt.figure(figsize=(10, 6), dpi=100)

            ax11 = figure1.add_subplot(311)
            ax12 = figure1.add_subplot(312)
            ax13 = figure1.add_subplot(313)
            colors = sns.color_palette(n_colors=5)
            linestyleList = ['-','--','-.']
            sns.set_style(style='white')
            savePath1 = _getSavePath(loadFilePath, "Time({0:}-{1:})".format(M_init, N_init))
            savePath2 = _getSavePath(loadFilePath, "Time({0:}-{1:})_persistent".format(M_init, N_init))
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

            pdSet = set()
            lambdaPhiSet = set()
            for P_d, lambda_phi, cpmf in cpfmList:
                if P_d not in pdSet:
                    lambdaPhiSet.clear()
                pdSet.add(P_d)
                lambdaPhiSet.add(lambda_phi)
                ax11.plot(timeArray,
                        cpmf,
                        label="P_d = {0:}, $\lambda_\phi$ = {1:}".format(P_d, float(lambda_phi)),
                        c=colors[len(pdSet)-1],
                        linestyle=linestyleList[len(lambdaPhiSet)-1])

            pdSet = set()
            lambdaPhiSet = set()
            for P_d, lambda_phi, cpmf in falseCPFMlist:
                if P_d not in pdSet:
                    lambdaPhiSet.clear()
                pdSet.add(P_d)
                lambdaPhiSet.add(lambda_phi)
                ax12.semilogy(timeArray,
                        cpmf+(1e-10),
                        label="P_d = {0:}, $\lambda_\phi$ = {1:}".format(P_d, float(lambda_phi)),
                        c=colors[len(pdSet)-1],
                        linestyle=linestyleList[len(lambdaPhiSet)-1])

            pdSet = set()
            lambdaPhiSet = set()
            for P_d, lambda_phi, accFalseTrack in accFalseTrackList:
                if P_d not in pdSet:
                    lambdaPhiSet.clear()
                pdSet.add(P_d)
                lambdaPhiSet.add(lambda_phi)
                ax13.plot(timeArray,
                          accFalseTrack,
                         label="P_d = {0:}, $\lambda_\phi$ = {1:}".format(P_d, float(lambda_phi)),
                         c=colors[len(pdSet) - 1],
                         linestyle=linestyleList[len(lambdaPhiSet) - 1])

            ax11.set_xlabel("Time [s]", fontsize=14)
            ax11.set_ylabel("Average cpfm", fontsize=14, linespacing=3)
            ax11.set_title("Cumulative Probability Mass Function", fontsize=18)
            ax11.legend(loc=4)
            ax11.grid(False)
            ax11.set_ylim(0,1)
            sns.despine(ax=ax11, offset=0)

            ax12.set_xlabel("Time [s]", fontsize=14)
            ax12.set_ylabel("Average number of tracks", fontsize=14, linespacing=2)
            ax12.set_title("Accumulative number of erroneous tracks", fontsize=18)
            ax12.set_ylim(1e-2,60)
            ax12.grid(False)
            sns.despine(ax=ax12, offset=0)

            ax13.set_xlabel("Time [s]", fontsize=14)
            ax13.set_ylabel("Average number of tracks", fontsize=14, linespacing=2)
            ax13.set_title("Number of erroneous tracks alive", fontsize=18)
            ax13.grid(False)
            ax13.set_ylim(-0.02, max(1,ax13.get_ylim()[1]))
            sns.despine(ax=ax13, offset=0)

            figure1.tight_layout()
            figure1.savefig(savePath1)
            figure1.clf()

            plt.close()

def _plotTrackingCorrectness(plotData):
    return plt.figure()

def _getSavePath(loadFilePath, nameAdd):
    head, tail = os.path.split(loadFilePath)
    name, extension = os.path.splitext(tail)
    savePath = os.path.join(head, 'plots')
    saveFilePath = os.path.join(savePath, name + "-" + nameAdd + ".pdf")
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    return saveFilePath
