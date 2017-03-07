import numpy as np
# import scipy.constants
import time
import pymht.tracker as tomht
from pymht.utils.classDefinitions import Position, TempTarget
import pymht.utils.simulator as sim
import pymht.utils.helpFunctions as hpf
import pymht.models.pv as model
import logging
import datetime


def runSimulation(**kwargs):
    minToc = [float('Inf')]
    maxToc = [0]
    avgToc = []

    tic0 = time.time()
    seed = 5446 + 1
    p0 = Position(100., -100.)
    radarRange = 5500.0  # meters
    maxSpeed = 21.0  # meters / second
    radarPeriod = 60. / 24.  # 24 RPM radar / 48 RPM radar
    simulationTimeStep = radarPeriod / 2
    simTime = radarPeriod * 8  # sec
    nScans = int(simTime / radarPeriod)
    nSimulationSteps = int(simTime / simulationTimeStep)
    lambda_phi = 1e-6  # Expected number of false measurements per unit
    # volume of the measurement space per scan
    lambda_nu = 0.0002  # Expected number of new targets per unit volume
    # of the measurement space per scan
    P_d = 0.8  # Probability of detection
    N = 6  # Number of  timesteps to tail (N-scan)
    eta2 = 5.99  # 95% confidence
    pruneThreshold = model.sigmaR_tracker

    assert simulationTimeStep <= radarPeriod
    assert simTime >= simulationTimeStep
    assert nScans >= 1

    # nTargets = 4
    # meanSpeed = 10.0 * scipy.constants.knot  # meters/second
    # initialTargets = sim.generateInitialTargets(seed,nTargets,p0, radarRange, meanSpeed)
    initialTargets = []
    initialTargets.append(TempTarget(np.array([-2000, 2100, 4, -4], dtype=np.double),
                                     time.time(), P_d))
    initialTargets.append(TempTarget(np.array([100, -2000, -2, 8], dtype=np.double),
                                     time.time(), P_d))
    initialTargets.append(TempTarget(np.array([-4000, 300, 10, -1], dtype=np.double),
                                     time.time(), P_d))
    initialTargets.append(TempTarget(np.array([-4000, 0, 13, 0], dtype=np.double),
                                     time.time(), P_d))
    initialTargets.append(TempTarget(np.array([-4000, -200, 17, 1], dtype=np.double),
                                     time.time(), P_d))
    initialTargets.append(TempTarget(np.array([4000, -2000, 1, -8], dtype=np.double),
                                     time.time(), P_d))
    initialTargets.append(TempTarget(np.array([3000, 4000, 2, -8], dtype=np.double),
                                     time.time(), P_d))
    initialTargets.append(TempTarget(np.array([200, 5000, 10, -1], dtype=np.double),
                                     time.time(), P_d))
    initialTargets.append(TempTarget(np.array([-3500, -3500, 10, 5], dtype=np.double),
                                     time.time(), P_d))
    initialTargets.append(TempTarget(np.array([-4100, 3200, 19, 2], dtype=np.double),
                                     time.time(), P_d, mmsi=257114400))
    initialTargets.append(TempTarget(np.array([3600, 3000, -10, 3], dtype=np.double),
                                     time.time(), P_d))
    initialTargets.append(TempTarget(np.array([5000, 1000, -5, -1], dtype=np.double),
                                     time.time(), P_d))
    initialTargets.append(TempTarget(np.array([2000, 100, -10, 8], dtype=np.double),
                                     time.time(), P_d))
    initialTargets.append(TempTarget(np.array([0, -5000, 10, 2], dtype=np.double),
                                     time.time(), P_d))
    initialTargets.append(TempTarget(np.array([-4000, 3000, 17, 0], dtype=np.double),
                                     time.time(), P_d, mmsi=257304900))

    if kwargs.get('printInitialTargets', False):
        print("Initial targets:")
        print(*initialTargets, sep='\n', end="\n\n")

    simList = sim.simulateTargets(seed,
                                  initialTargets,
                                  simTime,
                                  simulationTimeStep,
                                  model.Phi(simulationTimeStep),
                                  model.Q(simulationTimeStep, model.sigmaQ_true),
                                  model.Gamma)

    if kwargs.get('printSimList', False):
        print("Sim list:")
        print(*simList, sep="\n", end="\n\n")

    scanList = sim.simulateScans(seed,
                                 simList,
                                 radarPeriod,
                                 model.C,
                                 model.R(model.sigmaR_true),
                                 lambda_phi,
                                 radarRange,
                                 p0,
                                 shuffle=True,
                                 localClutter=True,
                                 globalClutter=True,
                                 debug=False)
    aisMeasurements = sim.simulateAIS(seed, simList)

    tracker = tomht.Tracker(model.Phi(radarPeriod), model.C, model.Gamma, P_d, model.P0,
                            model.R(), model.Q(radarPeriod), lambda_phi, lambda_nu, eta2, N,
                            p0, radarRange, "CBC",
                            logTime=True,
                            period=radarPeriod,
                            logLevel=logging.DEBUG,
                            M_required=2,
                            N_checks=4)
    toc0 = time.time() - tic0
    print("Generating simulation data for {0:} targets for {1:} seconds / {2:} scans.  It took {3:.1f} ms".format(
        len(initialTargets), simTime, nScans, toc0 * 1000))

    if kwargs.get('printScanList', False):
        print("Scan list:")
        print(*scanList, sep="\n", end="\n\n")

    if kwargs.get('printAISList', False):
        print("aisMeasurements:")
        print(*aisMeasurements, sep="\n", end="\n\n")

    tic1 = time.time()
    tomht._setHighPriority()

    def simulate(tracker, initialTargets, scanList, minToc, maxToc, avgToc, **kwargs):
        print("#" * 100)
        time.sleep(0.1)
        if kwargs.get('preInitiate', False):
            for index, initialTarget in enumerate(initialTargets):
                tracker.initiateTarget(initialTarget)

        aisIterator = (m for m in aisMeasurements)
        aisMeasurementList = next(aisIterator, None)
        for scanIndex, measurementList in enumerate(scanList):
            tic = time.time()
            scanTime = measurementList.time

            if aisMeasurementList is not None:
                if all((m.time <= scanTime) and (m.time - scanTime) <= radarPeriod
                       for m in aisMeasurementList):
                    aisPredictions = hpf.predictAisMeasurements(scanTime, aisMeasurementList)
                    aisMeasurementList = next(aisIterator, None)
                else:
                    aisPredictions = None
            else:
                aisPredictions = None

            tracker.addMeasurementList(measurementList,
                                       aisPredictions,
                                       printTime=True,
                                       checkIntegrity=False,
                                       R=False,
                                       **kwargs)
            toc = time.time() - tic
            minToc[0] = toc if toc < minToc[0] else minToc[0]
            maxToc[0] = toc if toc > maxToc[0] else maxToc[0]
            avgToc.append(toc)
        print("#" * 100)

    if kwargs.get('profile', False):
        import cProfile
        import pstats
        cProfile.runctx("simulate(tracker,initialTargets,scanList, minToc, maxToc, avgToc)",
                        globals(), locals(), 'mainProfile.prof')
        p = pstats.Stats('mainProfile.prof')
        p.strip_dirs().sort_stats('time').print_stats(20)
        p.strip_dirs().sort_stats('cumulative').print_stats(20)
    else:
        simulate(tracker, initialTargets, scanList, minToc, maxToc, avgToc, )

    if kwargs.get('printTargetList', False):
        tracker.printTargetList()

    if kwargs.get('printAssociation', False):
        association = hpf.backtrackMeasurementNumbers(tracker.__trackNodes__)
        print("Association (measurement number)", *association, sep="\n")

    toc1 = time.time() - tic1

    print('Completed {0:} scans in {1:.0f} seconds. Min {2:4.1f} ms Avg {3:4.1f} ms Max {4:4.1f} ms'.format(
        nScans, toc1, minToc[0] * 1000, np.average(avgToc) * 1000, maxToc[0] * 1000))

    if kwargs.get('plot'):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import itertools
        colors1 = itertools.cycle(cm.rainbow(
            np.linspace(0, 1, len(initialTargets))))
        colors2 = itertools.cycle(cm.rainbow(
            np.linspace(0, 1, len(tracker.__targetList__))))
        colors3 = itertools.cycle(cm.rainbow(
            np.linspace(0, 1, len(tracker.__targetList__))))
        fig1 = plt.figure(num=1, figsize=(9, 9), dpi=120)
        hpf.plotRadarOutline(p0, radarRange, markCenter=False)
        # tracker.plotInitialTargets()
        # tracker.plotVelocityArrowForTrack()
        # tracker.plotValidationRegionFromRoot() # TODO: Does not work
        # tracker.plotValidationRegionFromTracks() # TODO: Does not work
        # tracker.plotMeasurementsFromRoot(dummy=False, includeHistory=False)
        desiredPlotPeriod = radarPeriod * 4
        markEvery = max(1, int(desiredPlotPeriod / radarPeriod))
        hpf.plotTrueTrack(simList, colors=colors1, markevery=markEvery)
        # tracker.plotMeasurementsFromTracks(labels = False, dummy = True, real = True)
        # tracker.plotLastScan()
        tracker.plotAllScans()
        tracker.plotHypothesesTrack(colors=colors3)  # SLOW!
        tracker.plotActiveTracks(colors=colors2, markInitial=True)
        tracker.plotTerminatedTracks()
        plt.axis("equal")
        plt.xlim((p0.x() - radarRange * 1.05, p0.x() + radarRange * 1.05))
        plt.ylim((p0.y() - radarRange * 1.05, p0.y() + radarRange * 1.05))
        fig1.canvas.draw()
        plt.show()


if __name__ == '__main__':
    import argparse
    import sys

    print(sys.version)

    parser = argparse.ArgumentParser(description="Run MHT tracker",
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument('-R', help="Run recursive", action='store_true')
    args = vars(parser.parse_args())
    runSimulation(plot=False,
                  profile=False,
                  printInitialTargets=False,
                  printTargetList=False,
                  printAssociation=False,
                  printAISList=False,
                  **args)
    print("-" * 100)
