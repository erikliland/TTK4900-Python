from pysimulator import resultsPlotter
import multiprocessing as mp


def plotResults(filePathList):
    with mp.Pool() as pool:
        pool.map(resultsPlotter.plotLostTracks, filePathList)

if __name__ == '__main__':
    from pysimulator.simulationConfig import filePathList
    plotResults(filePathList[0:1])
