import xml.etree.ElementTree as ET
from pymht.utils.xmlDefinitions import *
import numpy as np


def analyzeFile(filePath):
    print("File:", filePath)
    tree = ET.parse(filePath)
    scenarioElement = tree.getroot()
    groundtruthElement = scenarioElement.find(groundtruthTag)
    scenariosettingsElement = scenarioElement.find(scenariosettingsTag)
    variationsList = scenarioElement.findall(variationsTag)
    print("Scenario name:", scenariosettingsElement.find(nameTag).text)
    groundtruthList = groundtruthElement.findall(trackTag)

    for variationsElement in variationsList:
        analyzeVariations(groundtruthList, variationsElement)
    tree.write(filePath)

def analyzeVariations(groundtruthList, variationsElement):
    preInitialized = variationsElement.get(preinitializedTag)
    print("preInitialized", preInitialized)

    if preInitialized == "True":
        print("Analyzing tracking")
        analyzeVariationsTrackingPerformance(groundtruthList, variationsElement)
    else:
        print("Analyzing initialization")
        analyzeVariationsInitializationPerformance(groundtruthList, variationsElement)

def analyzeVariationsTrackingPerformance(groundtruthList, variationsElement):
    variationList = variationsElement.findall(variationTag)
    for variation in variationList:
        analyzeVariationTrackingPerformance(groundtruthList, variation)

def analyzeVariationsInitializationPerformance(groundtruthList, variationsElement):
    variationList = variationsElement.findall(variationTag)

    for variation in variationList:
        analyzeVariationInitializationPerformance(groundtruthList, variation)

def analyzeVariationTrackingPerformance(truetrackList, variation):
    print("nTrueTracks", len(truetrackList))
    runList = variation.findall(runTag)
    for run in runList[0:1]:
        estimateTrackList = run.findall(trackTag)
        print("Run iterations", run.get(iterationTag))
        print("nEstTracks", len(estimateTrackList))
        for smoothed in [False, True]:
            matchList = matchTrueWithEstimatedTracks(truetrackList, estimateTrackList, smoothed)
            storeMatchList(run, matchList, smoothed)



def analyzeVariationInitializationPerformance(groundtruthList, variation):
    pass

def matchTrueWithEstimatedTracks(truetrackList, estimateTrackList, smoothed, threshold = 30):
    import math
    resultList = []
    for trueTrack in truetrackList:
        trueTrackStatesElement = trueTrack.find(statesTag)
        trueTrackStateList = trueTrackStatesElement.findall(stateTag)
        trueTrackID = trueTrack.get(idTag)
        for estimatedTrack in estimateTrackList:
            estimatedTrackID = estimatedTrack.get(idTag)
            estimatedTrackStatesList = estimatedTrack.findall(statesTag)
            estimatedTrackStatesElement = [e for e in estimatedTrackStatesList
                                           if e.attrib[smoothedTag] == str(smoothed)][0]
            estimatedStateList = estimatedTrackStatesElement.findall(stateTag)
            timeMatch = _timeMatch(trueTrackStateList, estimatedStateList)
            trueTrackSlice = [s for s in trueTrackStateList if s.get(timeTag) in timeMatch]
            estimatedTrackSlice = [s for s in estimatedStateList if s.get(timeTag) in timeMatch]
            assert len(timeMatch) == len(trueTrackSlice) == len(estimatedTrackSlice)
            delta2List = []
            for trueState, estimatedState in zip(trueTrackSlice, estimatedTrackSlice):
                trueTrackPosition = _parsePosition(trueState.find(positionTag))
                estimatedTrackPosition = _parsePosition(estimatedState.find(positionTag))
                delta = trueTrackPosition - estimatedTrackPosition
                deltaNorm = np.linalg.norm(delta)
                deltaNorm2 = deltaNorm**2
                delta2List.append(deltaNorm2)
            assert len(timeMatch) == len(delta2List)
            meanSquaredError = np.mean(delta2List)
            assert meanSquaredError > 0.

            approvedMatch = all([math.sqrt(d2) < threshold for d2 in delta2List])

            if approvedMatch:
                resultList.append((trueTrackID,estimatedTrackID, meanSquaredError, timeMatch))
            # else:
            #     resultList.append((i, estimatedTrackID, [], float('inf')))
    # print("resultList", *resultList, sep = "\n")
    multiplePossibilities = _multiplePossibleMatches(resultList)
    # print("multiplePossibilities", multiplePossibilities)
    assert not multiplePossibilities
    return resultList

def storeMatchList(run, matchList, smoothed):
    for match in matchList:
        trueTrackID = match[0]
        estimatedTrackID = match[1]
        meanNormSquared = match[2]
        timeMatch = match[3]
        trackElement = run.findall('.Track[@id="{:}"]'.format(estimatedTrackID))[0]
        statesElement = trackElement.findall('.States[@smoothed="{:}"]'.format(smoothed))[0]
        statesElement.set(matchidTag, trueTrackID)
        statesElement.set(mismatchTag, "{:.4f}".format(meanNormSquared))
        statesElement.set(timematchTag, ", ".join(timeMatch))

def _timeMatch(trueTrackStateList, estimatedStateList):
    trueTrackTimeList = [e.get(timeTag) for e in trueTrackStateList]
    estimatedTrackTimeList = [e.get(timeTag) for e in estimatedStateList]
    commonTimes = [tT for tT in trueTrackTimeList if tT in estimatedTrackTimeList]
    return commonTimes

def _parsePosition(positionElement):
    north = positionElement.find(northTag).text
    east = positionElement.find(eastTag).text
    position = np.array([east,north], dtype=np.double)
    return position

def _multiplePossibleMatches(matchList):
    estimateIdList = [m[1] for m in matchList]
    estimateIdSet = set(estimateIdList)
    repeatingEstimates = len(estimateIdList) > len(estimateIdSet)

    return repeatingEstimates