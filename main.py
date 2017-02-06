import numpy as np
import scipy.constants
import pulp
import time
import pymht.tracker as tomht
from pymht.utils.classDefinitions import Position
import pymht.utils.radarSimulator as sim
import pymht.utils.helpFunctions as hpf
import pymht.models.pv as model


def runSimulation(**kwargs):
    tic = time.time()
    seed = 5446 + 1
    # nTargets = 4
    p0 = Position(100, -100)
    radarRange = 5500.0  # meters
    meanSpeed = 10 * scipy.constants.knot  # meters/second
    simTime = 60  # sec
    timeStep = 60 / 24  # 24 RPM radar / 48 RPM radar
    nScans = int(simTime / timeStep)
    lambda_phi = 1e-6  # Expected number of false measurements per unit
    # volume of the measurement space per scan
    lambda_nu = 0.00001  # Expected number of new targets per unit volume
    # of the measurement space per scan
    P_d = 0.9  # Probability of detection
    N = 10  # Number of  timesteps to tail (N-scan)
    eta2 = 5.99  # 95% confidence
    pruneThreshold = model.sigmaR_tracker

    # initialTargets = sim.generateInitialTargets(seed,nTargets,p0, radarRange, meanSpeed)
    initialTargets = []
    initialTargets.append(sim.SimTarget(np.array([-2000, 2000, 7, -9], dtype=np.double),
                                        time.time(),
                                        P_d))
    initialTargets.append(sim.SimTarget(np.array([100,  -2000, -2, 8], dtype=np.double),
                                        time.time(),
                                        P_d))
    initialTargets.append(sim.SimTarget(np.array([-4000,  200, 10, -1], dtype=np.double),
                                        time.time(),
                                        P_d))
    initialTargets.append(sim.SimTarget(np.array([-4000,    0, 13, 0], dtype=np.double),
                                        time.time(),
                                        P_d))
    initialTargets.append(sim.SimTarget(np.array([-4000, -200, 17, 1], dtype=np.double),
                                        time.time(), P_d))
    initialTargets.append(sim.SimTarget(np.array([4000, -2000, 1, -8], dtype=np.double),
                                        time.time(),
                                        P_d))
    initialTargets.append(sim.SimTarget(np.array([3000,  4000, 2, -8], dtype=np.double),
                                        time.time(), P_d))
    initialTargets.append(sim.SimTarget(np.array([200,   5000, 10, -1], dtype=np.double),
                                        time.time(),
                                        P_d))
    initialTargets.append(sim.SimTarget(np.array([-3500, -3500, 10,  5], dtype=np.double),
                                        time.time(),
                                        P_d))

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
    toc = time.time() - tic
    print("Generating simulation data runtime:", round(toc * 1000, 0), "ms")
    # print("Scan list:")
    # print(*scanList, sep = "\n", end = "\n\n")
    tic2 = time.time()
    tomht._setHighPriority()

    def simulate(tracker, initialTargets, scanList):
        for index, initialTarget in enumerate(initialTargets):
            tracker.initiateTarget(initialTarget)
        for scanIndex, measurementList in enumerate(scanList):
            tracker.addMeasurementList(measurementList, printTime=True, **kwargs)
            # if scanIndex == 1:
            #     break
        print("#" * 100)

    if False:
        from timeit import default_timer as timer
        import cProfile
        import pstats
        cProfile.runctx("simulate(tracker,initialTargets,scanList)",
                        globals(), locals(), 'mainProfile.prof')
        p = pstats.Stats('mainProfile.prof')
        p.strip_dirs().sort_stats('time').print_stats(20)
        p.strip_dirs().sort_stats('cumulative').print_stats(10)
    else:
        simulate(tracker, initialTargets, scanList)

    # tracker.printTargetList()
    # association = hpf.backtrackMeasurementsIndices(tracker.__trackNodes__)
    # print("Association",*association, sep = "\n")

    if kwargs.get("plot", False):
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
        tracker.plotValidationRegionFromTracks()
        tracker.plotLastScan()
        tracker.plotMeasurementsFromRoot(dummy=False, includeHistory=True)
        # tracker.plotMeasurementsFromTracks(labels = False, dummy = False)
        tracker.plotHypothesesTrack(colors=colors3)
        tracker.plotActiveTracks(colors=colors2)
        hpf.plotTrueTrack(simList, colors=colors1, markevery=10)
        plt.axis("equal")
        plt.xlim((p0.x() - radarRange * 1.05, p0.x() + radarRange * 1.05))
        plt.ylim((p0.y() - radarRange * 1.05, p0.y() + radarRange * 1.05))
        fig1.canvas.draw()
        plt.show(block=True)
    toc2 = time.time() - tic2
    print('Completed in {:.0f} seconds'.format(toc2))
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run MHT tracker",
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument('-R', help="Run recursive", action='store_true')
    args = vars(parser.parse_args())
    runSimulation(plot=False, **args)
    print("-" * 100)
