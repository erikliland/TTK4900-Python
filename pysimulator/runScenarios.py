from pysimulator.simulationConfig import (trackingFilePathList, initFilePathList, scenarioList, pdList,
                                          lambdaphiList, nList, M_N_list, nMonteCarlo, baseSeed)
import multiprocessing as mp
from pysimulator import scenarioRunner
import psutil
import argparse
import numpy as np
import logging


def trackingWorker(scenarioIndex, pdList, lambdaphiList, nList, nMonteCarlo):
    logging.disable(logging.CRITICAL)
    try:
        filePath = trackingFilePathList[scenarioIndex]
        print("Starting:", filePath, "Scenario index", scenarioIndex)
        scenarioRunner.runPreinitializedVariations(scenarioList[scenarioIndex],
                                                   filePath,
                                                   pdList,
                                                   lambdaphiList,
                                                   nList,
                                                   nMonteCarlo,
                                                   baseSeed,
                                                   printLog=False)
        print("Done:", filePath, "Scenario index", scenarioIndex)
    except Exception as e:
        print(e)

def initWorker(scenarioIndex, pdList, lambdaphiList, M_N_list, nMonteCarlo):
    logging.disable(logging.CRITICAL)
    try:
        filePath = initFilePathList[scenarioIndex]
        print("Starting:", filePath, "Scenario index", scenarioIndex)
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


def runScenariosMultiProcessing(trackingScenarioIndices, initScenarioIndices,
                                pdList, lambdaphiList, nList, M_N_list, nMonteCarlo, **kwargs):
    logging.disable(logging.CRITICAL)
    nProcesses = kwargs.get('nProcesses', psutil.cpu_count(logical=True)-1)
    print("runScenariosMultiProcessing", kwargs)
    pool = mp.Pool(processes=nProcesses)

    results = []
    for scenarioIndex in trackingScenarioIndices:
        results.append(pool.apply_async(trackingWorker,
                                        args=[scenarioIndex,
                                              pdList,
                                              lambdaphiList,
                                              nList,
                                              nMonteCarlo]))

    for scenarioIndex in initScenarioIndices:
        results.append(pool.apply_async(initWorker,
                                        args=[scenarioIndex,
                                              pdList,
                                              lambdaphiList,
                                              M_N_list,
                                              nMonteCarlo]))
    for r in results:
        r.get()


def runScenariosSingleProcess(trackingScenarioIndices, initScenarioIndices,
                              pdList, lambdaphiList, nList, M_N_list, nMonteCarlo):
    for scenarioIndex in trackingScenarioIndices:
        scenario = scenarioList[scenarioIndex]
        filePath = trackingFilePathList[scenarioIndex]
        print("Scenario path:", filePath)
        scenarioRunner.runPreinitializedVariations(
            scenario, filePath, pdList, lambdaphiList, nList, nMonteCarlo, baseSeed)

    for scenarioIndex in initScenarioIndices:
        scenario = scenarioList[scenarioIndex]
        filePath = initFilePathList[scenarioIndex]
        print("Scenario path:", filePath)
        scenarioRunner.runInitializationVariations(
            scenario, filePath, pdList, lambdaphiList, M_N_list, nMonteCarlo, baseSeed)


def mainSingle():
    runScenariosSingleProcess(range(len(trackingFilePathList)), range(1), pdList,
                              lambdaphiList, nList, M_N_list, nMonteCarlo)


def mainMulti():
    logging.disable(logging.CRITICAL)

    parser = argparse.ArgumentParser(description="Run MHT tracker simulations concurrent", argument_default=argparse.SUPPRESS)
    # parser.add_argument('-F', help="Force run of files (if exist)", action='store_true')
    # parser.add_argument('-D', help="Discard result", action='store_true')
    # parser.add_argument('-f', help="File number to simulate", nargs='+', type=int)
    parser.add_argument('-i', help="Number of simulations", type=int)
    parser.add_argument('-c', help="Number of processes/cores to use", type=int)
    # parser.add_argument('-b', help="Batch size for accumulate mode in x*nCores, default = 1", type=int)
    # parser.add_argument('-C', help="Run compare and plot after finish", action='store_true')
    args = vars(parser.parse_args())

    n = nMonteCarlo
    kwargs = {}
    if 'c' in args:
        nCores = args.get('c')
        assert np.isfinite(nCores)
        assert type(nCores) is int
        assert nCores > 0
        kwargs['nProcesses'] = nCores
    if 'i' in args:
        nMonteCarloTemp = args.get('i')
        assert np.isfinite(nMonteCarloTemp)
        assert type(nMonteCarloTemp) is int
        assert nMonteCarloTemp > 0
        n = nMonteCarloTemp


    runScenariosMultiProcessing(range(len(trackingFilePathList)), range(1),
                                pdList, lambdaphiList, nList, M_N_list, n, **kwargs)


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    runScenariosMultiProcessing(range(len(trackingFilePathList)), range(0),
                                pdList, lambdaphiList, nList, M_N_list, 10)
