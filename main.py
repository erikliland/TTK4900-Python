import numpy as np
# import scipy.constants
import time
import pymht.tracker as tomht
from pymht.utils.classDefinitions import Position, TempTarget
import pymht.utils.simulator as sim
import pymht.utils.helpFunctions as hpf
import pymht.models.pv as model


def runSimulation(plot=False, **kwargs):
    minToc = [float('Inf')]
    maxToc = [0]
    avgToc = []

    tic0 = time.time()
    seed = 5446 + 1
    p0 = Position(100., -100.)
    radarRange = 5500.0  # meters
    maxSpeed = 21.0  # meters / second
    radarPeriod = 60. / 24.  # 24 RPM radar / 48 RPM radar
    simulationTimeStep = radarPeriod / 10
    simTime = 60 * 1  # sec
    nScans = int(simTime / radarPeriod)
    nSimulationSteps = int(simTime / simulationTimeStep)
    lambda_phi = 1e-6  # Expected number of false measurements per unit
    # volume of the measurement space per scan
    lambda_nu = 0.0002  # Expected number of new targets per unit volume
    # of the measurement space per scan
    P_d = 0.9  # Probability of detection
    N = 7  # Number of  timesteps to tail (N-scan)
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

    simList = sim.simulateTargets(seed, initialTargets, nScans, radarPeriod, model.Phi(
        radarPeriod), model.Q(radarPeriod, model.sigmaQ_true), model.Gamma)

    if kwargs.get('printSimList', False):
        print("Sim list:")
        print(*simList, sep="\n", end="\n\n")

    scanList = sim.simulateScans(seed, simList, model.C, model.R(model.sigmaR_true),
                                 lambda_phi, radarRange, p0,
                                 shuffle=True,
                                 localClutter=True,
                                 globalClutter=True,
                                 debug=False)

    tracker = tomht.Tracker(model.Phi(radarPeriod), model.C, model.Gamma, P_d, model.P0,
                            model.R(), model.Q(radarPeriod), lambda_phi, lambda_nu, eta2, N,
                            p0, radarRange, "CBC",
                            logTime=True,
                            period=radarPeriod)
    toc0 = time.time() - tic0
    print("Generating simulation data for {0:} targets for {1:} time steps.  It took {2:.1f} ms".format(
        len(initialTargets), nScans, toc0 * 1000))

    if kwargs.get('printScanList', False):
        print("Scan list:")
        print(*scanList, sep="\n", end="\n\n")

    tic1 = time.time()
    tomht._setHighPriority()

    def simulate(tracker, initialTargets, scanList, minToc, maxToc, avgToc, **kwargs):
        print("#" * 100)
        if kwargs.get('preInitiate', False):
            for index, initialTarget in enumerate(initialTargets):
                tracker.initiateTarget(initialTarget)
        for scanIndex, measurementList in enumerate(scanList):
            tic = time.time()
            tracker.addMeasurementList(measurementList,
                                       printTime=True,
                                       checkIntegrity=True,
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
        p.strip_dirs().sort_stats('cumulative').print_stats(10)
    else:
        simulate(tracker, initialTargets, scanList, minToc, maxToc, avgToc, )

    if kwargs.get('printTargetList', False):
        tracker.printTargetList()

    if kwargs.get('printAssociation', False):
        association = hpf.backtrackMeasurementsIndices(tracker.__trackNodes__)
        print("Association", *association, sep="\n")

    toc1 = time.time() - tic1

    print('Completed {0:} scans in {1:.0f} seconds. Min {2:4.1f} ms Avg {3:4.1f} ms Max {4:4.1f} ms'.format(
        nScans, toc1, minToc[0] * 1000, np.average(avgToc) * 1000, maxToc[0] * 1000))

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import itertools
        colors1 = itertools.cycle(cm.rainbow(
            np.linspace(0, 1, len(initialTargets))))
        colors2 = itertools.cycle(cm.rainbow(
            np.linspace(0, 1, len(tracker.__targetList__))))
        colors3 = itertools.cycle(cm.rainbow(
            np.linspace(0, 1, len(tracker.__targetList__))))
        fig1 = plt.figure(num=1, figsize=(9, 9), dpi=100)
        hpf.plotRadarOutline(p0, radarRange, markCenter=False)
        # tracker.plotInitialTargets()
        # tracker.plotVelocityArrowForTrack()
        # tracker.plotValidationRegionFromRoot() # TODO: Does not work
        # tracker.plotValidationRegionFromTracks() # TODO: Does not work
        # tracker.plotMeasurementsFromRoot(dummy=False, includeHistory=False)
        desiredPlotPeriod = 4.0
        markEvery = max(1, int(radarPeriod / desiredPlotPeriod))
        hpf.plotTrueTrack(simList, colors=colors1, markevery=markEvery)
        # tracker.plotMeasurementsFromTracks(labels = False, dummy = True, real = False)
        # tracker.plotLastScan()
        # tracker.plotAllScans()
        # tracker.plotHypothesesTrack(colors=colors3)  # SLOW!
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
    runSimulation(plot=True, profile=False, **args)
    print("-" * 100)
