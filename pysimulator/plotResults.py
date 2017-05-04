from pysimulator import resultsPlotter
import multiprocessing as mp


def plotResults(filePathList):
    with mp.Pool() as pool:
        pool.map(resultsPlotter.plotTrackLoss, filePathList)
        pool.map(resultsPlotter.plotTrackingPercentage, filePathList)

if __name__ == '__main__':
    from pysimulator.simulationConfig import trackingFilePathList
    plotResults(trackingFilePathList)
