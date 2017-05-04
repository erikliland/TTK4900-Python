import xml.etree.ElementTree as ET
from pymht.utils.xmlDefinitions import *
import numpy as np


def analyzeTrackingFile(filePath):
    print("Starting:", filePath)
    tree = ET.parse(filePath)
    threshold = 15
    scenarioElement = tree.getroot()
    groundtruthElement = scenarioElement.find(groundtruthTag)
    scenariosettingsElement = scenarioElement.find(scenariosettingsTag)
    variationsList = scenarioElement.findall(variationsTag)
    print("Scenario name:", scenariosettingsElement.find(nameTag).text)
    groundtruthList = groundtruthElement.findall(trackTag)

    for variationsElement in variationsList:
        if variationsElement.get(preinitializedTag) != "True":
            print("Deleting", variationsElement.attrib)
            scenarioElement.remove(variationsElement)
            continue
        analyzeVariationsTrackingPerformance(groundtruthList, variationsElement, threshold)
    tree.write(filePath)
    print("Done:", filePath)

def analyzeInitFile(filePath):
    print("Starting:", filePath)
    tree = ET.parse(filePath)
    threshold = 15
    scenarioElement = tree.getroot()
    groundtruthElement = scenarioElement.find(groundtruthTag)
    scenariosettingsElement = scenarioElement.find(scenariosettingsTag)
    variationsList = scenarioElement.findall(variationsTag)
    print("Scenario name:", scenariosettingsElement.find(nameTag).text)
    groundtruthList = groundtruthElement.findall(trackTag)

    for variationsElement in variationsList:
        if variationsElement.get(preinitializedTag) != "False":
            print("Deleting", variationsElement.attrib)
            scenarioElement.remove(variationsElement)
            continue
        analyzeVariationsInitializationPerformance(groundtruthList, variationsElement, threshold)
    tree.write(filePath)
    print("Done:", filePath)

def analyzeVariationsTrackingPerformance(groundtruthList, variationsElement, threshold):
    variationList = variationsElement.findall(variationTag)
    for variation in variationList:
        analyzeVariationTrackLossPerformance(groundtruthList, variation, threshold)


def analyzeVariationsInitializationPerformance(groundtruthList, variationsElement, threshold):
    variationList = variationsElement.findall(variationTag)

    for variation in variationList[0:1]:
        print("Analyzing variation", variation.attrib)
        analyzeVariationInitializationPerformance(groundtruthList, variation, threshold)


def analyzeVariationTrackLossPerformance(groundtruthList, variation, threshold):
    runList = variation.findall(runTag)
    for run in runList:
        estimateTrackList = run.findall(trackTag)
        matchList = _matchTrueWithEstimatedTracks(groundtruthList, estimateTrackList, threshold)
        _storeMatchList(run, matchList)


def analyzeVariationInitializationPerformance(groundtruthList, variation, threshold):
    runList = variation.findall(runTag)
    for run in runList:
        estimateTrackList = run.findall(trackTag)
        matchList = _matchAndTimeInitialTracks(groundtruthList, estimateTrackList, threshold)
        # print("matchList", matchList)
        # storeMatchList(run, matchList)


def _matchTrueWithEstimatedTracks(truetrackList, estimateTrackList, threshold):
    resultList = []
    for trueTrack in truetrackList:
        trueTrackStatesElement = trueTrack.find(statesTag)
        trueTrackStateList = trueTrackStatesElement.findall(stateTag)
        trueTrackID = trueTrack.get(idTag)
        for estimatedTrack in estimateTrackList:
            estimatedTrackID = estimatedTrack.get(idTag)
            estimatedTrackStatesElement = estimatedTrack.find(statesTag)
            smoothedEstimatedTrackStatesElement = estimatedTrack.find(smoothedstatesTag)
            estimatedStateList = estimatedTrackStatesElement.findall(stateTag)
            smoothedEstimatedStateList = smoothedEstimatedTrackStatesElement.findall(stateTag)

            timeMatch, goodTimeMatch, meanSquaredError, lostTrack = _compareTrackList(
                trueTrackStateList,estimatedStateList,threshold)

            _, _, smoothedMeanSquaredError, _ = _compareTrackList(
                trueTrackStateList, smoothedEstimatedStateList, threshold)
            timeMatchLength = len(timeMatch)
            goodTimeMatchLength = len(goodTimeMatch)

            if timeMatchLength > 0:
                trackPercent = (goodTimeMatchLength / len(timeMatch))*100

                resultList.append((trueTrackID,
                                   estimatedTrackID,
                                   meanSquaredError,
                                   smoothedMeanSquaredError,
                                   lostTrack,
                                   timeMatch,
                                   goodTimeMatch,
                                   timeMatchLength,
                                   goodTimeMatchLength,
                                   trackPercent))

    _multiplePossibleMatches(resultList)

    return resultList


def _matchAndTimeInitialTracks(groundtruthList, estimateTrackList, threshold):
    for trueTrack in groundtruthList:
        trueTrackStatesElement = trueTrack.find(statesTag)
        trueTrackStateList = trueTrackStatesElement.findall(stateTag)
        trueTrackID = trueTrack.get(idTag)
        for estimatedTrack in estimateTrackList:
            estimatedTrackID = estimatedTrack.get(idTag)
            estimatedTrackStatesElement = estimatedTrack.find(statesTag)






def _compareTrackList(trueTrackStateList, estimatedStateList, threshold):
    timeMatch = _timeMatch(trueTrackStateList, estimatedStateList)

    trueTrackSlice = [s
                      for s in trueTrackStateList
                      if s.get(timeTag) in timeMatch]

    estimatedTrackSlice = [s
                           for s in estimatedStateList
                           if s.get(timeTag) in timeMatch]

    assert len(timeMatch) == len(trueTrackSlice) == len(estimatedTrackSlice)

    goodTimeMatch, meanSquaredError, lostTrack = _compareTrackSlices(
        timeMatch,trueTrackSlice,estimatedTrackSlice,threshold)

    if lostTrack is not None:
        return (timeMatch, goodTimeMatch, meanSquaredError, lostTrack)

    return ([],[],np.nan, None)


def _compareTrackSlices(timeMatch, trueTrackSlice, estimatedTrackSlice, threshold):
    assert len(trueTrackSlice) == len(estimatedTrackSlice)
    deltaList = []
    for trueState, estimatedState in zip(trueTrackSlice, estimatedTrackSlice):
        trueTrackPosition = _parsePosition(trueState.find(positionTag))
        estimatedTrackPosition = _parsePosition(estimatedState.find(positionTag))
        # if np.any(np.isnan(estimatedTrackPosition)): break
        # if np.any(np.isnan(trueTrackPosition)): break
        delta = trueTrackPosition - estimatedTrackPosition
        deltaNorm = np.linalg.norm(delta)
        deltaList.append(deltaNorm)

    goodTimeMatch = []
    delta2List = []
    for i, (time, delta) in enumerate(zip(timeMatch, deltaList)):
        if delta > threshold:
            if not any([d < threshold for d in deltaList[i+1:]]):
                break
        goodTimeMatch.append(time)
        delta2List.append(delta**2)

    if not goodTimeMatch or not delta2List:
        return (goodTimeMatch, np.nan, None)
    meanSquaredError = np.mean(np.array(delta2List))

    if (len(goodTimeMatch) > 0) and not np.isnan(meanSquaredError):
        lostTrack = goodTimeMatch < timeMatch
        return (goodTimeMatch, meanSquaredError, lostTrack)

    return ([], np.nan, None)


def _storeMatchList(run, matchList):
    for match in matchList:
        (trueTrackID,
         estimatedTrackID,
         meanSquaredError,
         smoothedMeanSquaredError,
         lostTrack,
         timeMatch,
         goodTimeMatch,
         timeMatchLength,
         goodTimeMatchLength,
         trackPercent) = match

        trackElement = run.findall('.Track[@id="{:}"]'.format(estimatedTrackID))[0]
        statesElement = trackElement.find(statesTag)
        smoothedStatesElement = trackElement.find(smoothedstatesTag)

        trackElement.set(matchidTag, trueTrackID)
        trackElement.set(timematchTag, ", ".join(timeMatch))
        trackElement.set(goodtimematchTag, ", ".join(goodTimeMatch))
        trackElement.set(losttrackTag, str(lostTrack))
        trackElement.set(timematchlengthTag, str(timeMatchLength))
        trackElement.set(goodtimematchlengthTag, str(goodTimeMatchLength))
        trackElement.set(trackpercentTag, "{:.1f}".format(trackPercent))

        statesElement.set(meansquarederrorTag, "{:.4f}".format(meanSquaredError))
        smoothedStatesElement.set(meansquarederrorTag, "{:.4f}".format(smoothedMeanSquaredError))


def _timeMatch(trueTrackStateList, estimatedStateList):
    trueTrackTimeList = [e.get(timeTag) for e in trueTrackStateList]
    estimatedTrackTimeList = [e.get(timeTag) for e in estimatedStateList]
    commonTimes = [tT for tT in trueTrackTimeList if tT in estimatedTrackTimeList]
    return commonTimes


def _parsePosition(positionElement):
    north = positionElement.find(northTag).text
    east = positionElement.find(eastTag).text
    position = np.array([east, north], dtype=np.double)
    return position


def _multiplePossibleMatches(resultList):
    import collections
    estimateIdList = [m[1] for m in resultList]
    duplicates = [item
                  for item, count in collections.Counter(estimateIdList).items()
                  if count > 1]
    for duplicate in duplicates:
        duplicateTuples = [(e, e[2]) for e in resultList if e[1] == duplicate]
        sortedDuplicates = sorted(duplicateTuples, key=lambda tup: tup[1])
        for e in sortedDuplicates[1:]:
            resultList.remove(e[0])
