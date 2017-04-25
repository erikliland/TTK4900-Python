def runScenarios(filePathList, scenarioList, pdList, lambdaphiList, nList, nMonteCarlo):
    from pysimulator import scenarioRunner
    for scenarioNumber, (scenario, filePath) in enumerate(zip(scenarioList, filePathList)):
        print("Scenario path:", filePath)
        scenarioRunner.runVariations(scenario, filePath, pdList, lambdaphiList, nList, nMonteCarlo)

if __name__ == '__main__':
    from pysimulator.simulationConfig import filePathList, scenarioList, pdList, lambdaphiList, nList, nMonteCarlo
    runScenarios(filePathList, scenarioList, pdList, lambdaphiList, nList, nMonteCarlo)