from pysimulator import scenarioAnalyzer
import multiprocessing as mp


def analyseResults(filePathList):
    with mp.Pool() as pool:
        pool.map(scenarioAnalyzer.analyzeFile, filePathList)

if __name__ == '__main__':
    from pysimulator.simulationConfig import filePathList
    analyseResults(filePathList[0:1])
