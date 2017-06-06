from pysimulator import resultsPlotter
import multiprocessing as mp


def plotResults(filePathList, initFilePathList):
    resultsPlotter.exportTrackingPercentageImprovement(filePathList)
    with mp.Pool() as pool:
        pool.apply_async(resultsPlotter.exportInitialState)
        pool.apply_async(resultsPlotter.exportAisState)
        pool.apply_async(resultsPlotter.plotOverlaidRadarMeasurements)
        pool.apply_async(resultsPlotter.plotTrueTracks)
        pool.apply_async(resultsPlotter.plotTrackingPercentageExample)

        pool.map(resultsPlotter.plotInitializationTime, initFilePathList)
        pool.map(resultsPlotter.plotTrackLoss, filePathList)
        pool.map(resultsPlotter.plotTrackingPercentage, filePathList)
        pool.map(resultsPlotter.plotRuntime, filePathList)

if __name__ == '__main__':
    from pysimulator.simulationConfig import trackingFilePathList, initFilePathList
    plotResults(trackingFilePathList, initFilePathList)
