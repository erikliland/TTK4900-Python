from pysimulator.simulationConfig import filePathList, scenarioList, pdList, lambdaphiList, nList, nMonteCarlo, baseSeed
import multiprocessing as mp
from pysimulator import scenarioRunner


def worker(scenarioIndex, filePath,pdList, lambdaphiList, nList, nMonteCarlo):
    print("Starting:", filePath, "Scenario index", scenarioIndex)
    try:
        res = scenarioRunner.runVariations(
            scenarioList[scenarioIndex], filePath,pdList, lambdaphiList, nList, nMonteCarlo, baseSeed, printLog = False)
        return res
    except Exception as e:
        print("ScenarioIndex", scenarioIndex)
        print(e)

def runScenariosMultiProcessing(filePathList, scenarios, pdList, lambdaphiList, nList, nMonteCarlo):
    nProcesses = 4
    pool = mp.Pool(processes=nProcesses)

    results = [pool.apply_async(worker,
                                args=[scenarioList.index(scenario), filePath,pdList, lambdaphiList, nList, nMonteCarlo])
               for scenario, filePath in zip(scenarios, filePathList)]
    for r in results:
        r.get()

def runScenariosSingleProcess(filePathList, scenarioList, pdList, lambdaphiList, nList, nMonteCarlo):
    for scenario, filePath in zip(scenarioList, filePathList):
        print("Scenario path:", filePath)
        scenarioRunner.runVariations(scenario, filePath, pdList, lambdaphiList, nList, nMonteCarlo, baseSeed)


def mainSingle():
    runScenariosSingleProcess(filePathList, scenarioList, pdList, lambdaphiList, nList, nMonteCarlo)

def mainMulti():
    runScenariosMultiProcessing(filePathList, scenarioList, pdList, lambdaphiList, nList, nMonteCarlo)


if __name__ == '__main__':
    runScenariosMultiProcessing(filePathList, scenarioList, pdList, lambdaphiList, nList, nMonteCarlo)