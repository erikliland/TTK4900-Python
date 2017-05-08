from pysimulator.simulationConfig import (trackingFilePathList, initFilePathList, scenarioList, pdList,
                                          lambdaphiList, nList, M_N_list, nMonteCarlo, baseSeed)
import multiprocessing as mp
from pysimulator import scenarioRunner


def trackingWorker(scenarioIndex, filePath, pdList, lambdaphiList, nList, nMonteCarlo):
    print("Starting:", filePath, "Scenario index", scenarioIndex)
    try:
        scenarioRunner.runPreinitializedVariations(scenarioList[scenarioIndex],
                                                         filePath,
                                                         pdList,
                                                         lambdaphiList,
                                                         nList,
                                                         nMonteCarlo,
                                                         baseSeed,
                                                         printLog=False)
    except Exception as e:
        print(e)

def initWorker(scenarioIndex, filePath, pdList, lambdaphiList, M_N_list, nMonteCarlo):
    print("Starting:", filePath, "Scenario index", scenarioIndex)
    try:
        scenarioRunner.runInitializationVariations(scenarioList[scenarioIndex],
                                                         filePath,
                                                         pdList,
                                                         lambdaphiList,
                                                         M_N_list,
                                                         nMonteCarlo,
                                                         baseSeed,
                                                         printLog=False)
    except Exception as e:
        print(e)


def runScenariosMultiProcessing(filePathList, initFilePathList, scenarios,
                                pdList, lambdaphiList, nList, M_N_list, nMonteCarlo):
    nProcesses = 8
    pool = mp.Pool(processes=nProcesses)

    results = []
    for scenario, filePath in zip(scenarios, filePathList):
        results.append(pool.apply_async(trackingWorker,
                                        args=[scenarioList.index(scenario),
                                              filePath,
                                              pdList,
                                              lambdaphiList,
                                              nList,
                                              nMonteCarlo]))

    for scenario, filePath in zip(scenarios, initFilePathList):
        results.append(pool.apply_async(initWorker,
                                        args=[scenarioList.index(scenario),
                                              filePath,
                                              pdList,
                                              lambdaphiList,
                                              M_N_list,
                                              nMonteCarlo]))
    for r in results:
        r.get()


def runScenariosSingleProcess(trackingFilePathList, initFilePathList, scenarioList,
                              pdList, lambdaphiList, nList, M_N_list, nMonteCarlo):
    for scenario, filePath in zip(scenarioList, trackingFilePathList):
        print("Scenario path:", filePath)
        scenarioRunner.runPreinitializedVariations(
            scenario, filePath, pdList, lambdaphiList, nList, nMonteCarlo, baseSeed)

    for scenario, filePath in zip(scenarioList, initFilePathList):
        print("Scenario path:", filePath)
        scenarioRunner.runInitializationVariations(
            scenario, filePath, pdList, lambdaphiList, M_N_list, nMonteCarlo, baseSeed)


def mainSingle():
    runScenariosSingleProcess(trackingFilePathList, initFilePathList, scenarioList, pdList,
                              lambdaphiList, nList, M_N_list, nMonteCarlo)


def mainMulti():
    runScenariosMultiProcessing(trackingFilePathList, initFilePathList, scenarioList,
                                pdList, lambdaphiList, nList, M_N_list, nMonteCarlo)


if __name__ == '__main__':
    runScenariosMultiProcessing([], initFilePathList[0:1], scenarioList,
                                pdList, lambdaphiList, nList, M_N_list, nMonteCarlo)
