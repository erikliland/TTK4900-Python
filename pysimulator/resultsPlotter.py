import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import xml.etree.ElementTree as ET
from pymht.utils.xmlDefinitions import *
import os
import numpy as np


def plotTrackLoss(loadFilePath):
    print("plotTrackLoss", loadFilePath)
    savePath = _getSavePath(loadFilePath, "TrackLoss")
    tree = ET.parse(loadFilePath)
    scenarioElement = tree.getroot()
    groundtruthElement = scenarioElement.find(groundtruthTag)
    scenarioSettingsElement = scenarioElement.find(scenariosettingsTag)
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
    scenarioSettingsElement = scenarioElement.find(scenariosettingsTag)
    variationsElement = scenarioElement.findall('.Variations[@preinitialized="True"]')[0]
    plotData = _getTrackingPercentagePlotData(groundtruthElement, variationsElement)
    print("plotData", plotData)
    figure = _plotTrackingPercentage(plotData)
    figure.savefig(savePath)


def plotInitializationTime(loadFilePath):
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


def plotTrackCorrectness(filePath):
    pass


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
        initTimeLog = dict()
        runElementList = variationElement.findall(runTag)[0:1]
        for run in runElementList:
            initiationLogElement = run.find(initializationLogTag)
            initialTargetList = initiationLogElement.findall(initialtargetTag)
            for e in initialTargetList:
                time = e.get(timeTag)
                if time in initTimeLog:
                    initTimeLog[time] += 1
                else:
                    initTimeLog[time] = 1
        for k,v in initTimeLog.items():
            initTimeLog[k] = float(v) / float(len(runElementList))

        #TODO: http://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys
        if M_init not in variationsInitLog:
            variationsInitLog[M_init] = {}
        if N_init not in variationsInitLog[M_init]:
            variationsInitLog[M_init][N_init] = {}
        if lambda_phi not in variationsInitLog[M_init][N_init]:
            variationsInitLog[M_init][N_init][lambda_phi] = {}
        if P_d not in variationsInitLog[M_init][N_init][lambda_phi]:
            variationsInitLog[M_init][N_init][lambda_phi][P_d] = initTimeLog
        else:
            raise KeyError("Duplicate key found")

    return variationsInitLog


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
                    label = "N="+str(N) if j == 0 else None,
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
    ax.tick_params(labelsize=16, pad=1)
    yStart, yEnd = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(yStart, yEnd * 1.01, 10))
    ax.xaxis.set_ticks(lambdaPhiList)
    xTickLabels = ax.xaxis.get_ticklabels()
    for label in xTickLabels:
        label.set_verticalalignment('bottom')
        label.set_horizontalalignment('left')
        label.set_rotation(0)

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
                    label="N=" + str(N) if j == 0 else None,
                    c=colors[i],
                    linewidth=4)
    lambdaPhiList = list(lambdaPhiSet)
    lambdaPhiList.sort()

    ax.view_init(15, -165)
    ax.legend(loc='upper right', bbox_to_anchor=(0.5, 0.8), fontsize=18)
    ax.set_xlabel("$\lambda_{\phi}$", fontsize=18, labelpad=30)
    ax.set_zlabel("\nTracking (%)", fontsize=18, linespacing=3)
    ax.set_ylabel("\nProbability of detection (%)", fontsize=18, linespacing=2)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.set_zlim(96, 100)
    ax.tick_params(labelsize=16, pad=1)
    yStart, yEnd = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(yStart, yEnd * 1.01, 10))
    ax.xaxis.set_ticks(lambdaPhiList)
    xTickLabels = ax.xaxis.get_ticklabels()
    for label in xTickLabels:
        label.set_verticalalignment('bottom')
        label.set_horizontalalignment('left')
        label.set_rotation(0)

    return figure


def _plotInitializationTime3D(plotData, loadFilePath, simLength, timeStep, nTargets):
    for M_init, d1 in plotData.items():
        for N_init, d2 in d1.items():
            figure = plt.figure(figsize=(10, 10), dpi=100)
            lambdaPhiSet = set()
            colors = sns.color_palette(n_colors=5)
            sns.set_style(style='white')
            ax = figure.add_subplot(111, projection='3d')
            savePath = _getSavePath(loadFilePath, "Time({0:}-{1:})".format(M_init, N_init))
            for k, (lambda_phi, d3) in enumerate(d2.items()):
                lambdaPhiSet.add(float(lambda_phi))
                for j, (P_d, initTimeLog) in enumerate(d3.items()):
                    timeArray = np.arange(0, simLength, timeStep)
                    pmf = np.zeros_like(timeArray)
                    for i, time in enumerate(timeArray):
                        if str(time) in initTimeLog:
                            pmf[i] = initTimeLog[str(time)]
                    cpmf = np.cumsum(pmf) / float(nTargets)
                    lambdaphiArray = np.ones_like(timeArray) *float(lambda_phi)
                    ax.plot(timeArray, lambdaphiArray, cpmf,
                            label="P_d = {:}".format(P_d) if k==0 else None,
                            c=colors[j])
            ax.view_init(15, -150)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
            ax.set_zlim(0, 1)
            ax.yaxis.set_ticks(list(lambdaPhiSet))
            ax.set_xlabel("\nTime steps", fontsize=18, linespacing=2)
            ax.set_ylabel("$\lambda_{\phi}$", fontsize=18, labelpad=30)
            ax.set_zlabel("\ncpfm", fontsize=18, linespacing=3)
            plt.title("M={0:}, N={1:}".format(M_init, N_init), fontsize=18)
            plt.legend(loc=4)
            plt.savefig(savePath)
            plt.close()

def _plotInitializationTime2D(plotData, loadFilePath, simLength, timeStep, nTargets):
    for M_init, d1 in plotData.items():
        for N_init, d2 in d1.items():
            figure = plt.figure(figsize=(10, 10), dpi=100)
            ax = figure.add_subplot(111)
            lambdaPhiSet = set()
            colors = sns.color_palette(n_colors=5)
            linestyle = ['-','--','-.',]
            sns.set_style(style='white')
            savePath = _getSavePath(loadFilePath, "Time({0:}-{1:})".format(M_init, N_init))
            for k, (lambda_phi, d3) in enumerate(d2.items()):
                lambdaPhiSet.add(float(lambda_phi))
                for j, (P_d, initTimeLog) in enumerate(d3.items()):
                    timeArray = np.arange(0, simLength, timeStep)
                    pmf = np.zeros_like(timeArray)
                    for i, time in enumerate(timeArray):
                        if str(time) in initTimeLog:
                            pmf[i] = initTimeLog[str(time)]
                    cpmf = np.cumsum(pmf) / float(nTargets)
                    ax.plot(timeArray,
                            cpmf,
                            label="P_d = {0:}, $\lambda_\phi$ = {1:}".format(P_d, float(lambda_phi)),
                            c=colors[j],
                            linestyle=linestyle[k])
            ax.set_ylim(0,1)
            plt.xlabel("Time steps", fontsize=18, linespacing=2)
            plt.ylabel("cpfm", fontsize=18, linespacing=3)
            plt.title("M={0:}, N={1:}".format(M_init, N_init), fontsize=18)
            handles, labels = ax.get_legend_handles_labels()
            import operator
            hl = sorted(zip(handles, labels),
                        key=operator.itemgetter(1),
                        reverse=True)
            handles2, labels2 = zip(*hl)
            plt.grid(False)
            ax.legend(handles2, labels2, loc=4)
            sns.despine(ax=ax, offset=0)
            plt.savefig(savePath)
            plt.close()

def _getSavePath(loadFilePath, nameAdd):
    head, tail = os.path.split(loadFilePath)
    name, extension = os.path.splitext(tail)
    savePath = os.path.join(head, 'plots')
    saveFilePath = os.path.join(savePath, name + "-" + nameAdd + ".pdf")
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    return saveFilePath
