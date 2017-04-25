import logging
import numpy as np
import time
import pymht.tracker as tomht
from pymht.utils.classDefinitions import Position, SimTarget
from pymht.pyTarget import Target
import pymht.utils.simulator as sim
import pymht.utils.helpFunctions as hpf
import pymht.models.pv as model
from pysimulator import simulationConfig
from pysimulator.scenarios.defaults import *

def runSimulation(**kwargs):
    minToc = [float('Inf')]
    maxToc = [0]
    avgToc = []
    tic0 = time.time()
    #
    seed = 5446 + 1
    # p0 = np.array([100., -100.])
    # radarRange = 5500.0  # meters
    # maxSpeed = 22.0  # meters / second
    # radarPeriod = 60. / 24.  # 24 RPM radar / 48 RPM radar
    # simulationTimeStep = radarPeriod / 2 #sec
    # # aisPeriod = radarPeriod * 3 #sec
    # P_r = 0.9 #AIS probability of receive
    # simTime = radarPeriod * 30  # sec
    # nScans = int(simTime / radarPeriod)
    # nSimulationSteps = int(simTime / simulationTimeStep)
    lambda_phi = 1e-6
    # lambda_nu = 0.0002
    P_d = 0.8  # Probability of detection
    # N = 6  # Number of  timesteps to tail (N-scan)
    # eta2 = 5.99  # 95% confidence
    #
    # assert simulationTimeStep <= radarPeriod
    # assert simTime >= simulationTimeStep
    # assert nScans >= 1
    #
    # sim.seed_simulator(seed)
    # # nTargets = 4
    # # meanSpeed = 10.0 * scipy.constants.knot  # meters/second
    # # initialTargets = sim.generateInitialTargets(nTargets,p0, radarRange, meanSpeed)
    # initTime = 0 # time.time()
    # initialTargets = []
    # initialTargets.append(SimTarget([-2000, 2100, 4, -4], initTime, P_d, model.sigmaQ_true))
    # initialTargets.append(SimTarget([100, -2000, -2, 8], initTime, P_d, model.sigmaQ_true))
    # initialTargets.append(SimTarget([-4000, 300, 12, -1], initTime, P_d, model.sigmaQ_true))
    # initialTargets.append(SimTarget([-4000, 0, 12, 0], initTime, P_d, model.sigmaQ_true,
    #                                 mmsi=257114401, aisClass='B', probabilityOfReceive=P_r))
    # initialTargets.append(SimTarget([-4000, -200, 17, 1], initTime, P_d, model.sigmaQ_true))
    # initialTargets.append(SimTarget([4000, -2000, 1, -8], initTime, P_d, model.sigmaQ_true))
    # initialTargets.append(SimTarget([3000, 4000, 2, -8], initTime, P_d, model.sigmaQ_true))
    # initialTargets.append(SimTarget([200, 5000, 10, -1], initTime, P_d, model.sigmaQ_true))
    # initialTargets.append(SimTarget([-3500, -3500, 10, 5], initTime, P_d, model.sigmaQ_true))
    # initialTargets.append(SimTarget([-4100, 3200, 17, 2], initTime, P_d, model.sigmaQ_true,
    #                                 mmsi=257114400, aisClass='B', probabilityOfReceive=P_r))
    # initialTargets.append(SimTarget([3600, 3000, -10, 3], initTime, P_d, model.sigmaQ_true))
    # initialTargets.append(SimTarget([5000, 1000, -7, -2], initTime, P_d, model.sigmaQ_true))
    # initialTargets.append(SimTarget([2000, 100, -10, 8], initTime, P_d, model.sigmaQ_true))
    # initialTargets.append(SimTarget([0, -5000, 10, 2], initTime, P_d, model.sigmaQ_true))
    # initialTargets.append(SimTarget([-400, 300, 17, 0], initTime, P_d, model.sigmaQ_true,
    #                                 mmsi=257304900, aisClass='B', probabilityOfReceive=P_r))
    # initialTargets.append(SimTarget([0, 2000, 15, 15], initTime, P_d, model.sigmaQ_true))

    scenario = simulationConfig.scenarioList[0]


    if kwargs.get('printInitialTargets', False):
        print("Initial targets:")
        print(scenario, sep='\n', end="\n\n")

    simList = scenario.getSimList()

    if kwargs.get('printSimList', False):
        print("Sim list:")
        print(*simList, sep="\n", end="\n\n")

    scanList, aisList = scenario.getSimulatedScenario(seed, simList, lambda_phi, P_d)

    # scanList = sim.simulateScans(simList,
    #                              radarPeriod,
    #                              model.C_RADAR,
    #                              model.R_RADAR(model.sigmaR_RADAR_true),
    #                              lambda_phi,
    #                              radarRange,
    #                              p0,
    #                              localClutter=True,
    #                              globalClutter=True,
    #                              debug=False)
    #
    # aisList = sim.simulateAIS(simList,
    #                           model.Phi,
    #                           model.C_AIS,
    #                           model.R_AIS(model.sigmaR_AIS_true),
    #                           model.GPS_COVARIANCE_PRECISE,
    #                           radarPeriod,
    #                           probabilityOfReceive = P_r)

    trackerArgs = (scenario.model,
                   scenario.radarPeriod,
                   lambda_phi,
                   lambda_nu)

    trackerKwargs = {'maxSpeedMS': maxSpeedMS,
                     'M_required': M_required,
                     'N_checks': N_checks,
                     'position': scenario.p0,
                     'radarRange': scenario.radarRange,
                     'eta2': eta2,
                     'N':5,
                     'P_d': scenario.P_d_true}


    tracker = tomht.Tracker(*trackerArgs, groundTruth=simList, **{**trackerKwargs, **kwargs})

    # tracker = tomht.Tracker(model.Phi(radarPeriod), model.C_RADAR, model.Gamma, P_d, model.P0,
    #                         model.R_RADAR(), model.R_AIS(), model.Q(radarPeriod), lambda_phi, lambda_nu, eta2,
    #                         p0, radarRange,
    #                         N = N,
    #                         logTime=True,
    #                         logLevel=logging.DEBUG,
    #                         maxSpeed=maxSpeed,
    #                         radarPeriod = radarPeriod,
    #                         M_required=2,
    #                         N_checks=4,
    #                         groundTruth=simList)
    toc0 = time.time() - tic0
    print("Generating simulation data for {0:} targets for {1:} seconds / {2:} scans.  It took {3:.1f} ms".format(
        len(scenario), scenario.simTime, scenario.nScans, toc0 * 1000))

    if kwargs.get('printScanList', False):
        print("Scan list:")
        print(*scanList, sep="\n", end="\n\n")

    if kwargs.get('printAISList', False):
        aisList.print()


    tic1 = time.time()

    def simulate(tracker, initialTargets, scanList, minToc, maxToc, avgToc, **kwargs):
        print("#" * 100)
        time.sleep(0.1)
        if kwargs.get('preInitiate', False):
            for index, initialTarget in enumerate(initialTargets):
                tempTarget = Target(initialTarget.time,
                                    None,
                                    np.array(initialTarget.state),
                                    model.P0,
                                    measurementNumber=index + 1,
                                    measurement=model.C_RADAR.dot(initialTarget.state)
                                    )
                tracker.initiateTarget(tempTarget)

        for scanIndex, measurementList in enumerate(scanList):
            tic = time.time()
            scanTime = measurementList.time
            aisPredictions = aisList.getMeasurements(scanTime)

            tracker.addMeasurementList(measurementList,
                                       aisPredictions,
                                       printTime=True,
                                       printAssociation=False,
                                       printCluster=False,
                                       checkIntegrity=True,
                                       R=False,
                                       **kwargs)
            toc = time.time() - tic
            minToc[0] = toc if toc < minToc[0] else minToc[0]
            maxToc[0] = toc if toc > maxToc[0] else maxToc[0]
            avgToc.append(toc)
        print("#" * 100)

    if kwargs.get('profile', False):
        # try cProfile
        # try line_profiler
        # try memory_profiler
        import cProfile
        import pstats
        cProfile.runctx("simulate(tracker,initialTargets,scanList, minToc, maxToc, avgToc)",
                        globals(), locals(), 'mainProfile.prof')
        p = pstats.Stats('mainProfile.prof')
        p.strip_dirs().sort_stats('time').print_stats(20)
        p.strip_dirs().sort_stats('cumulative').print_stats(20)
    else:
        simulate(tracker, scenario, scanList, minToc, maxToc, avgToc, preInitiate=False)

    if kwargs.get('printTargetList', False):
        tracker.printTargetList()

    if kwargs.get('printAssociation', False):
        association = hpf.backtrackMeasurementNumbers(tracker.__trackNodes__)
        print("Association (measurement number)", *association, sep="\n")

    toc1 = time.time() - tic1

    print('Completed {0:} scans in {1:.0f} seconds. Min {2:4.1f} ms Avg {3:4.1f} ms Max {4:4.1f} ms'.format(
        scenario.nScans, toc1, minToc[0] * 1000, np.average(avgToc) * 1000, maxToc[0] * 1000))


    if 'exportPath' in kwargs:
        scenarioElement = tracker.getScenarioElement()
        simList.storeGroundTruth(scenarioElement)
        tracker._storeRun(scenarioElement)
        path = kwargs.get('exportPath')
        hpf.writeElementToFile(path, scenarioElement)

    if kwargs.get('plot'):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import itertools
        colors1 = itertools.cycle(cm.rainbow(
            np.linspace(0, 1, len(scenario))))
        colors2 = itertools.cycle(cm.rainbow(
            np.linspace(0, 1, len(tracker.__targetList__))))
        colors3 = itertools.cycle(cm.rainbow(
            np.linspace(0, 1, len(tracker.__targetList__))))
        fig1 = plt.figure(num=1, figsize=(9, 9), dpi=90)
        hpf.plotRadarOutline(scenario.p0, scenario.radarRange, markCenter=False)
        # tracker.plotInitialTargets()
        # tracker.plotVelocityArrowForTrack() # TODO: Does not work
        # tracker.plotValidationRegionFromTracks() # TODO: Does not work
        desiredPlotPeriod = scenario.radarPeriod
        markEvery = max(1, int(desiredPlotPeriod / scenario.simulationTimeStep))
        hpf.plotTrueTrack(simList, colors=colors1, markevery=markEvery)
        # tracker.plotMeasurementsFromRoot(dummy=False, real = False, includeHistory=False)
        # tracker.plotStatesFromRoot(dummy=False, real=False, ais=True)
        # tracker.plotMeasurementsFromTracks(labels = False, dummy = True, real = True)
        # tracker.plotLastScan()
        # tracker.plotAllScans()
        # tracker.plotLastAisUpdate()
        # tracker.plotAllAisUpdates()
        # tracker.plotHypothesesTrack(colors=colors3, markStates=False)  # CAN BE SLOW!
        tracker.plotActiveTracks(colors=colors2, markInitial=True, labelInitial=True, markRoot=False,
                                 markStates=True, real=False, dummy=False, ais=True, smooth=False)
        tracker.plotActiveTracks(colors=colors2, markInitial=False, markRoot=False, markStates=False, real=False,
                                 dummy=False, ais=False, smooth=True, markEnd=False)

        tracker.plotTerminatedTracks()
        plt.axis("equal")
        plt.xlim((scenario.p0[0] - scenario.radarRange * 1.05, scenario.p0[0] + scenario.radarRange * 1.05))
        plt.ylim((scenario.p0[1] - scenario.radarRange * 1.05, scenario.p0[1] + scenario.radarRange * 1.05))


        # plt.xlim((-3910,-3870))
        # plt.ylim((20,50))
        fig1.canvas.draw()
        plt.show()


if __name__ == '__main__':
    import argparse
    import sys
    import os

    print(sys.version)

    parser = argparse.ArgumentParser(description="Run MHT tracker",
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument('-R', help="Run recursive", action='store_true')
    args = vars(parser.parse_args())
    exportPath = os.path.join(os.path.expanduser('~'),'TTK4900-Python','data','test.xml')
    print("Storing at", exportPath)
    runSimulation(plot=True,
                  profile=False,
                  printInitialTargets=True,
                  printTargetList=False,
                  printScanList=False,
                  printAssociation=False,
                  printAISList=True,
                  exportPath=exportPath,
                  **args)
    print("-" * 100)
