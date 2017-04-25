import pymht.utils.simulator as sim
import pymht.models.pv as model
from pymht.utils.classDefinitions import SimTarget
import xml.etree.ElementTree as ET
from pymht.utils.xmlDefinitions import *
import numpy as np

class ScenarioBase:
    def __init__(self):
        #Static scenario data
        self.staticSeed = 5447
        self.initTime = 0
        self.radarPeriod = 60. / 24.  # 24 RPM radar / 48 RPM radar
        self.radarRange = 5500.0  # meters
        self.simulationTimeStep = self.radarPeriod / 4  # sec
        self.simTime = self.radarPeriod * 30  # sec
        self.nScans = int(self.simTime / self.radarPeriod)
        self.nSimulationSteps = int(self.simTime / self.simulationTimeStep)
        self.P_d_true = 0.8  # Probability of detection
        self.sigma_Q = model.sigmaQ_true
        self.P_r = 0.9  # Probability of receive (AIS)
        self.model = model
        self.p0 = np.array([100., -100.])  # own position

        assert self.simulationTimeStep <= self.radarPeriod
        assert self.simTime >= self.simulationTimeStep
        assert self.nScans >= 1

    def storeScenarioSettings(self, scenarioElement):
        scenariosettingsElement = ET.SubElement(scenarioElement, scenariosettingsTag)
        for k,v in vars(self).items():
            ET.SubElement(scenariosettingsElement,str(k)).text = str(v)

class Scenario(ScenarioBase):
    def __init__(self, name):
        ScenarioBase.__init__(self)
        self.name = name
        self.initialTargets = []

    def __getitem__(self, item):
        return self.initialTargets.__getitem__(item)

    def __iter__(self):
        return self.initialTargets.__iter__()

    def __len__(self):
        return self.initialTargets.__len__()



    def append(self, newTarget):
        if type(newTarget) is not SimTarget:
            raise ValueError("Wrong input type. Must be SimTarget")
        self.initialTargets.append(newTarget)

    def add(self, state, **kwargs):
        default = {'probabilityOfReceive':self.P_r}
        self.initialTargets.append(SimTarget(state, self.initTime, self.P_d_true, self.sigma_Q, **{**default,**kwargs}))

    def getSimList(self):
        sim.seed_simulator(self.staticSeed)
        return sim.simulateTargets(self.initialTargets,
                                      self.simTime,
                                      self.simulationTimeStep,
                                      model)

    def getSimulatedScenario(self,seed, simList, lambda_phi, P_d):
        sim.seed_simulator(seed)

        scanList = sim.simulateScans(simList,
                                     self.radarPeriod,
                                     model.C_RADAR,
                                     model.R_RADAR(model.sigmaR_RADAR_true),
                                     lambda_phi,
                                     self.radarRange,
                                     self.p0,
                                     P_d = P_d)

        aisList = sim.simulateAIS(simList,
                                  model.Phi,
                                  model.C_AIS,
                                  model.R_AIS(model.sigmaR_AIS_true),
                                  model.GPS_COVARIANCE_PRECISE,
                                  self.radarPeriod)
        return scanList, aisList

#Scenario 1
scenario1 = Scenario("Scenario1")
scenario1.add([-2000, 2100, 4, -4])
scenario1.add([100, -2000, -2, 8])
scenario1.add([-4000, 300, 12, -1])
scenario1.add([-4000, 0, 12, 0], mmsi=257114401, aisClass='B')
scenario1.add([-4000, -200, 17, 1])
scenario1.add([4000, -2000, 1, -8])
scenario1.add([3000, 4000, 2, -8])
scenario1.add([200, 5000, 10, -1])
scenario1.add([-3500, -3500, 10, 5])
scenario1.add([-4100, 3200, 17, 2], mmsi=257114400, aisClass='B')
scenario1.add([3600, 3000, -10, 3])
scenario1.add([5000, 1000, -7, -2])
scenario1.add([2000, 100, -10, 8])
scenario1.add([0, -5000, 10, 2])
scenario1.add([-400, 300, 17, 0], mmsi=257304900, aisClass='B')
scenario1.add([0, 2000, 15, 15])

#Scenario 2
scenario2 = Scenario("Scenario2")
scenario2.add([-2000, 2100, 4, -4])
scenario2.add([100, -2000, -2, 8])
scenario2.add([-4000, 300, 12, -1])
scenario2.add([-4000, 0, 12, 0], mmsi=257114401, aisClass='B')
scenario2.add([-4000, -200, 17, 1])
scenario2.add([4000, -2000, 1, -8])
scenario2.add([3000, 4000, 2, -8])
scenario2.add([200, 5000, 10, -1])
scenario2.add([-3500, -3500, 10, 5])
scenario2.add([-4100, 3200, 17, 2], mmsi=257114400, aisClass='B')
scenario2.add([3600, 3000, -10, 3])
scenario2.add([5000, 1000, -7, -2])
scenario2.add([2000, 100, -10, 8])
scenario2.add([0, -5000, 10, 2])
scenario2.add([-400, 300, 17, 0], mmsi=257304900, aisClass='B')
scenario2.add([0, 2000, 15, 15])

#Scenario 3
scenario3 = Scenario("Scenario3")
scenario3.add([-2000, 2100, 4, -4])
scenario3.add([100, -2000, -2, 8])
scenario3.add([-4000, 300, 12, -1])
scenario3.add([-4000, 0, 12, 0], mmsi=257114401, aisClass='B')
scenario3.add([-4000, -200, 17, 1])
scenario3.add([4000, -2000, 1, -8])
scenario3.add([3000, 4000, 2, -8])
scenario3.add([200, 5000, 10, -1])
scenario3.add([-3500, -3500, 10, 5])
scenario3.add([-4100, 3200, 17, 2], mmsi=257114400, aisClass='B')
scenario3.add([3600, 3000, -10, 3])
scenario3.add([5000, 1000, -7, -2])
scenario3.add([2000, 100, -10, 8])
scenario3.add([0, -5000, 10, 2])
scenario3.add([-400, 300, 17, 0], mmsi=257304900, aisClass='B')
scenario3.add([0, 2000, 15, 15])

#Scenario 4
scenario4 = Scenario("Scenario4")
scenario4.add([-2000, 2100, 4, -4])
scenario4.add([100, -2000, -2, 8])
scenario4.add([-4000, 300, 12, -1])
scenario4.add([-4000, 0, 12, 0], mmsi=257114401, aisClass='B')
scenario4.add([-4000, -200, 17, 1])
scenario4.add([4000, -2000, 1, -8])
scenario4.add([3000, 4000, 2, -8])
scenario4.add([200, 5000, 10, -1])
scenario4.add([-3500, -3500, 10, 5])
scenario4.add([-4100, 3200, 17, 2], mmsi=257114400, aisClass='B')
scenario4.add([3600, 3000, -10, 3])
scenario4.add([5000, 1000, -7, -2])
scenario4.add([2000, 100, -10, 8])
scenario4.add([0, -5000, 10, 2])
scenario4.add([-400, 300, 17, 0], mmsi=257304900, aisClass='B')
scenario4.add([0, 2000, 15, 15])

scenarioList = [scenario1, scenario2, scenario3, scenario4]
