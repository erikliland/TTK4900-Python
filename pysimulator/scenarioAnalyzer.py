import xml.etree.ElementTree as ET
from pymht.utils.xmlDefinitions import *
import numpy as np


def analyzeFile(filePath):
    tree = ET.parse(filePath)
    scenarioElement = tree.getroot()
    groundtruthElement = scenarioElement.find(groundtruthTag)
    scenariosettingsElement = scenarioElement.find(scenariosettingsTag)
    variationsElement = scenarioElement.find(variationsTag)
    print("Scenario name:", scenariosettingsElement.find(nameTag).text)

    groundtruthList = groundtruthElement.findall(trackTag)
    variationList = variationsElement.findall(variationTag)

    for variation in variationList[0:1]:
        analyzeVariation(groundtruthList, variation)


def analyzeVariation(truetrackList, variation):
    print("nTrueTracks", len(truetrackList))
    runList = variation.findall(runTag)
    for run in runList[0:1]:
        estimateTrackList = run.findall(trackTag)
        print("Run iterations", run.get(iterationTag))
        print("nEstTracks", len(estimateTrackList))
        matchList = matchingList(truetrackList, estimateTrackList)
        print("matchList", *matchList, sep = "\n")
        multiplePossibilities = _multiplePossibleMatches(matchList)
        assert not multiplePossibilities



def _multiplePossibleMatches(matchList):
    estimateIdList = [m[1] for m in matchList]
    estimateIdSet = set(estimateIdList)
    repeatingEstimates = len(estimateIdList) > len(estimateIdSet)

    return repeatingEstimates

def matchingList(truetrackList, estimateTrackList, threshold = 30):
    import math
    resultList = []
    for i, trueTrack in enumerate(truetrackList):
        trueTrackStatesElement = trueTrack.find(statesTag)
        trueTrackStateList = trueTrackStatesElement.findall(stateTag)
        for estimatedTrack in estimateTrackList:
            estimatedTrackID = estimatedTrack.get(idTag)
            estimatedTrackStatesElement = estimatedTrack.find(statesTag)
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
                resultList.append((i,estimatedTrackID, meanSquaredError, timeMatch))
            # else:
            #     resultList.append((i, estimatedTrackID, [], float('inf')))


    return resultList

def _timeMatch(trueTrackStateList, estimatedStateList):
    trueTrackTimeList = [e.get(timeTag) for e in trueTrackStateList]
    estimatedTrackTimeList = [e.get(timeTag) for e in estimatedStateList]
    commonTimes = [tT for tT in trueTrackTimeList if tT in estimatedTrackTimeList]
    return commonTimes

def  _parsePosition(positionElement):
    north = positionElement.find(northTag).text
    east = positionElement.find(eastTag).text
    position = np.array([east,north], dtype=np.double)
    return position