import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pymht.utils.xmlDefinitions import *
import os
import numpy as np


def plotLostTracks(loadFilePath):
    print("plotLostTracks", loadFilePath)
    savePath = _getSavePath(loadFilePath)
    tree = ET.parse(loadFilePath)
    scenarioElement = tree.getroot()
    groundtruthElement = scenarioElement.find(groundtruthTag)
    scenarioSettingsElement = scenarioElement.find(scenariosettingsTag)
    variationsElement = scenarioElement.findall('.Variations[@preinitialized="True"]')[0]
    plotData = _getPlotDataFromVariations(groundtruthElement,variationsElement)
    print("Plot data", plotData)
    figure = _plotTrackLoss(plotData)
    figure.savefig(savePath)

def plotInitializationTime(filePath):
    pass

def plotTrackCorrectness(filePath):
    pass

def plotRuntime(filePath):
    pass


def _getPlotDataFromVariations(groundtruthElement, variationsElement):
    variationList = variationsElement.findall(variationTag)
    plotData = {}
    for variation in variationList:
        N = float(variation.get(nTag))
        P_d = float(variation.get(pdTag))
        lambda_phi = float(variation.get(lambdaphiTag))
        trackLossList = []
        for run in variation.findall(runTag):
            try:
                trackLossList.append(float(run.get(tracklossTag,0)))
            except TypeError:
                pass
        trackLossMean = np.mean(np.array(trackLossList))
        if P_d not in plotData:
            plotData[P_d] = {}
        if N not in plotData[P_d]:
            plotData[P_d][N] = {}
        if lambda_phi not in plotData[P_d][N]:
            plotData[P_d][N][lambda_phi] = trackLossMean
        else:
            raise KeyError("Duplicate key found")
    return plotData

def _plotTrackLoss(plotData):
    return plt.figure()

def _getSavePath(loadFilePath):
    head, tail = os.path.split(loadFilePath)
    name, extension = os.path.splitext(tail)
    savePath = os.path.join(head, 'plots')
    saveFilePath = os.path.join(savePath, name + "-LostTracks.pdf")
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    return saveFilePath