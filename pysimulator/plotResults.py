from pysimulator import resultsPlotter
import multiprocessing as mp


def plotResults(filePathList, initFilePathList):
    resultsPlotter.exportInitialState()
    resultsPlotter.exportAisState()
    resultsPlotter.plotOverlaidRadarMeasurements()
    resultsPlotter.plotTrueTracks()
    with mp.Pool() as pool:
        pool.map(resultsPlotter.plotInitializationTime, initFilePathList)
        pool.map(resultsPlotter.plotTrackLoss, filePathList)
        pool.map(resultsPlotter.plotTrackingPercentage, filePathList)
        pool.map(resultsPlotter.plotRuntime, filePathList)

if __name__ == '__main__':
    from pysimulator.simulationConfig import trackingFilePathList, initFilePathList
    plotResults(trackingFilePathList, initFilePathList)
