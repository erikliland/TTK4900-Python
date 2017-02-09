import numpy as np
import scipy.constants
import time
import pymht.tracker as tomht
from pymht.utils.classDefinitions import Position
import pymht.utils.radarSimulator as sim
import pymht.utils.helpFunctions as hpf
import pymht.models.pv as model
import pymht.initiators.n_of_m as n_of_m


def runSimulation(plot=False, **kwargs):
    minToc = [float('Inf')]
    maxToc = [0]
    avgToc = []

    tic0 = time.time()
    seed = 5446 + 1
    # nTargets = 4
    p0 = Position(100, -100)
    radarRange = 5500.0  # meters
    meanSpeed = 10 * scipy.constants.knot  # meters/second
    simTime = 10 * 1  # sec
    timeStep = 60 / 24  # 24 RPM radar / 48 RPM radar
    nScans = int(simTime / timeStep)
    lambda_phi = 0e-6  # Expected number of false measurements per unit
    # volume of the measurement space per scan
    lambda_nu = 0.00001  # Expected number of new targets per unit volume
    # of the measurement space per scan
    P_d = 0.99  # Probability of detection
    N = 13  # Number of  timesteps to tail (N-scan)
    eta2 = 5.99  # 95% confidence
    pruneThreshold = model.sigmaR_tracker

    # initialTargets = sim.generateInitialTargets(seed,nTargets,p0, radarRange, meanSpeed)
    initialTargets = []
    initialTargets.append(sim.SimTarget(np.array([-2000, 2000, 7, -9], dtype=np.double),
                                        time.time(), P_d))
    initialTargets.append(sim.SimTarget(np.array([100, -2000, -2, 8], dtype=np.double),
                                        time.time(), P_d))
    initialTargets.append(sim.SimTarget(np.array([-4000, 200, 10, -1], dtype=np.double),
                                        time.time(), P_d))
    initialTargets.append(sim.SimTarget(np.array([-4000, 0, 13, 0], dtype=np.double),
                                        time.time(), P_d))
    initialTargets.append(sim.SimTarget(np.array([-4000, -200, 17, 1], dtype=np.double),
                                        time.time(), P_d))
    initialTargets.append(sim.SimTarget(np.array([4000, -2000, 1, -8], dtype=np.double),
                                        time.time(), P_d))
    initialTargets.append(sim.SimTarget(np.array([3000, 4000, 2, -8], dtype=np.double),
                                        time.time(), P_d))
    initialTargets.append(sim.SimTarget(np.array([200, 5000, 10, -1], dtype=np.double),
                                        time.time(), P_d))
    initialTargets.append(sim.SimTarget(np.array([-3500, -3500, 10, 5], dtype=np.double),
                                        time.time(), P_d))
    initialTargets.append(sim.SimTarget(np.array([3600, 3000, -10, 6], dtype=np.double),
                                        time.time(), P_d))
    initialTargets.append(sim.SimTarget(np.array([5000, 1000, -5, -1], dtype=np.double),
                                        time.time(), P_d))
    initialTargets.append(sim.SimTarget(np.array([2000, 100, -10, 8], dtype=np.double),
                                        time.time(), P_d))
    initialTargets.append(sim.SimTarget(np.array([0, -5000, 10, 2], dtype=np.double),
                                        time.time(), P_d))
    initialTargets.append(sim.SimTarget(np.array([-4000, 3000, 18, 0], dtype=np.double),
                                        time.time(), P_d))
    initialTargets.append(sim.SimTarget(np.array([-4000, 3100, 20, 0], dtype=np.double),
                                        time.time(), P_d))

    # print("Initial targets:")
    # print(*initialTargets, sep='\n', end = "\n\n")

    simList = sim.simulateTargets(seed, initialTargets, nScans, timeStep, model.Phi(
        timeStep), model.Q(timeStep, model.sigmaQ_true), model.Gamma)
    # print("Sim list:")
    # print(*simList, sep = "\n", end = "\n\n")
    # sim.writeSimList(initialTargets, simList, "parallel_targets_0.5Hz.txt")

    scanList = sim.simulateScans(seed, simList, model.C, model.R(model.sigmaR_true),
                                 lambda_phi, radarRange, p0, shuffle=False)
    # solvers: CPLEX, GLPK, CBC, GUROBI
    tracker = tomht.Tracker(model.Phi(timeStep), model.C, model.Gamma, P_d, model.P0,
                            model.R(), model.Q(timeStep), lambda_phi, lambda_nu, eta2, N,
                            "CBC", logTime=True, period=timeStep)
    initiator = n_of_m.Initiator(2,3)
    toc0 = time.time() - tic0
    print("Generating simulation data for {0:} targets for {1:} time steps.  It took {2:.1f} ms".format(
        len(initialTargets), nScans, toc0 * 1000))

    # print("Scan list:")
    # print(*scanList, sep = "\n", end = "\n\n")
    tic1 = time.time()
    tomht._setHighPriority()

    def simulate(tracker, initialTargets, scanList, minToc, maxToc, avgToc):
        for index, initialTarget in enumerate(initialTargets):
            tracker.initiateTarget(initialTarget)
        for scanIndex, measurementList in enumerate(scanList):
            tic = time.time()
            tracker.addMeasurementList(measurementList, printTime=True, **kwargs)
            toc = time.time() - tic
            minToc[0] = toc if toc < minToc[0] else minToc[0]
            maxToc[0] = toc if toc > maxToc[0] else maxToc[0]
            avgToc.append(toc)
        print("#" * 100)

    if False:
        # from timeit import default_timer as timer
        import cProfile
        import pstats
        cProfile.runctx("simulate(tracker,initialTargets,scanList, minToc, maxToc, avgToc)",
                        globals(), locals(), 'mainProfile.prof')
        p = pstats.Stats('mainProfile.prof')
        p.strip_dirs().sort_stats('time').print_stats(20)
        p.strip_dirs().sort_stats('cumulative').print_stats(10)
    else:
        for scanIndex, measurementList in enumerate(scanList):
            print(scanIndex, initiator.processMeasurements(measurementList))
            if scanIndex == 1:
                break
        # simulate(tracker, initialTargets, scanList, minToc, maxToc, avgToc)

    # tracker.printTargetList()
    # association = hpf.backtrackMeasurementsIndices(tracker.__trackNodes__)
    # print("Association",*association, sep = "\n")

    toc1 = time.time() - tic1

    print('Completed {0:} scans in {1:.0f} seconds. Min {2:4.1f} ms Avg {3:4.1f} ms Max {4:4.1f} ms'.format(
        nScans, toc1, minToc[0] * 1000, np.average(avgToc) * 1000, maxToc[0] * 1000))

    if plot:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import itertools
        colors1 = itertools.cycle(cm.rainbow(
            np.linspace(0, 1, len(tracker.__targetList__))))
        colors2 = itertools.cycle(cm.rainbow(
            np.linspace(0, 1, len(tracker.__targetList__))))
        colors3 = itertools.cycle(cm.rainbow(
            np.linspace(0, 1, len(tracker.__targetList__))))
        fig1 = plt.figure(num=1, figsize=(9, 9), dpi=100)
        hpf.plotRadarOutline(p0, radarRange, center=False)
        tracker.plotInitialTargets()
        # tracker.plotVelocityArrowForTrack()
        # tracker.plotValidationRegionFromRoot()
        tracker.plotLastScan()
        # tracker.plotMeasurementsFromRoot(dummy=False, includeHistory=True)
        # tracker.plotMeasurementsFromTracks(labels = False, dummy = False)
        # tracker.plotHypothesesTrack(colors=colors3) # SLOW!
        tracker.plotActiveTracks(colors=colors2)
        hpf.plotTrueTrack(simList, colors=colors1, markevery=10)
        plt.axis("equal")
        plt.xlim((p0.x() - radarRange * 1.05, p0.x() + radarRange * 1.05))
        plt.ylim((p0.y() - radarRange * 1.05, p0.y() + radarRange * 1.05))
        fig1.canvas.draw()
        plt.show(block=True)


if __name__ == '__main__':
    import argparse
    import sys

    print(sys.version)

    parser = argparse.ArgumentParser(description="Run MHT tracker",
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument('-R', help="Run recursive", action='store_true')
    args = vars(parser.parse_args())
    runSimulation(plot=False, **args)
    print("-" * 100)
