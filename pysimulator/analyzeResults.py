from pysimulator import scenarioAnalyzer

def analyseResults(filePathList):
    for filePath in filePathList:
        print("Analyzing", filePath)
        scenarioAnalyzer.analyzeFile(filePath)

if __name__ == '__main__':
    from pysimulator.simulationConfig import filePathList
    analyseResults(filePathList[0:1])