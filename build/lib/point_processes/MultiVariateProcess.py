'''
MultiVariateProcess.py
    Multi variate inhomogenous poisson process implementation
    Author: Akshay Aravamudan, January 2020
'''

# ConstantMemoryKernel, RayleighMemoryKernel, PowerLawMemoryKernel, GammaGompertzMemoryKernel
from utils.MemoryKernel import *

from utils.DataReader import createDummyDataSplitPopulation

from utils.Simulation import simulation
from point_processes.PointProcessCollection import PoissonTPP
import numpy as np
import scipy.stats
from utils.GoodnessOfFit import KSgoodnessOfFitExp1, KSgoodnessOfFitExp1MV
from matplotlib import pyplot as plt
from core.Logger import getLoggersMultivariateProcess
from core.DataStream import DataStream
from point_processes.PointProcessCollection import TrainingStatus
from point_processes.UnivariateHawkes import *

##################################################
# MultiVariateProcess Class
#
# A class designed to model a multi-variate process which has encapsulated within it
# a list of univariate temporal point processes, each being an instance of tpp
# PUBLIC ATTRIBUTES:
#
#
# PUBLIC INTERFACE:
#
#   __str__(): string; string representation of event object
#   print():   print string representation of event object
#
# USAGE EXAMPLES:
#
# DEPENDENCIES: None
#
# AUTHOR: Akshay Aravamudan, January 2020
#
##################################################


class MultiVariateProcess(object):
    def __init__(self, desc=None):

        self._setupTrainingDone = False
        self._sourceNames = ['exploit', 'twitter', 'github']
        if desc is not None:
            self.desc = desc
        else:
            self.logger.Warning(
                "No name passed to process, assigning default: Multivariate Inhomogenous Poisson")
            self.desc = 'Multivariate Inhomogenous Poisson'

        dataStream = DataStream()

        dataStream.initializeFromSimulatedSamples(
            [[]] * len(self._sourceNames), self._sourceNames)

        _, sourceNames_0 = dataStream.getDataStreamLearning(
            self._sourceNames[0])
        _, sourceNames_1 = dataStream.getDataStreamLearning(
            self._sourceNames[1])
        _, sourceNames_2 = dataStream.getDataStreamLearning(
            self._sourceNames[2])

        # gammaGompertzKernel = GammaGompertzMemoryKernel(beta=1.0, gamma=2.0)

        #  by default all the parameters are set to alpha = 1 for these processes by the PoissonTPP sub-class
        testProcess = PoissonTPP(
            mk=ConstantMemoryKernel(), sourceNames=sourceNames_0, desc='Exploit Process: Poisson TPP with a exponential pseudo kernel')
        # exploitProcess.alpha = 20
        twitterProcess = PoissonTPP(
            mk=ConstantMemoryKernel(), sourceNames=sourceNames_1, desc='Twitter Process: Poisson TPP with an exponential pseudo kernel')
        # redditProcess.alpha = 105
        githubProcess = PoissonTPP(
            mk=ExponentialPseudoMemoryKernel(beta=1.0), sourceNames=sourceNames_2, desc='Github Process: Hawkes TPP with a exponential pseudo kernel')

        self.Processes = [testProcess, twitterProcess, githubProcess]
        self._params = []
        for process in self.Processes:
            self._params.append(process.alpha)

    def getSourceNames(self):
        return self._sourceNames

    def setParams(self, params):
        self._params = params
        pass

    def getParams(self):
        return self._params, self._sourceNames

    def simulate(self, rightCensoringTime, MTPPdata, resume):
        return simulation(self.Processes, rightCensoringTime=rightCensoringTime,
                          MTPPdata=MTPPdata, resume=resume)

    def train(self):
        assert self._setupTrainingDone, "Setup for the training has to be complete before calling the train function"
        isConverged = True
        for process in self.Processes:
            status = process.train()
            if not status == TrainingStatus.CONVERGED:
                isConverged = False
                break
        paramList = []
        for process in self.Processes:
            paramList.append(process.alpha)
        self.setParams(paramList)

        if isConverged:
            return TrainingStatus.CONVERGED
        else:
            return TrainingStatus.ERROR

    def setupTraining(self, dataStream):
        assert isinstance(
            dataStream, DataStream), "Passed data object must from the datastream class"
        for process in self.Processes:
            streams, sourceNames = dataStream.getDataStreamLearning(
                process.getSourceNames()[0])
            process.setupTraining(streams)
        self._setupTrainingDone = True

    def getNumProcesses(self):
        return len(self.Processes)

    def transformEventTimes(self, MTPPdata, dataStream):
        print("Transforming times")
        IIDSamples = []
        for i in range(self.getNumProcesses()):
            IIDSamples.append([])
        for realization in MTPPdata:
            assert len(realization) == len(
                self.Processes), "Inconsistency detected in processes and number of streams"
            # iterate through available streams in realization to accordingly transform the times
            for process in self.Processes:
                sourceName = process.getSourceNames()[0]

                modifiedRealization, sourceNames = dataStream.getDataStreamLearning(
                    process.getSourceNames()[0])

                # get position of string in processes sourceNames list
                index = self._sourceNames.index(
                    sourceName) if len(self.Processes) > 1 else 0
                if not index == 0:
                    # swap 0th position and the position of the index in the realizations
                    # print(realization)
                    modifiedRealization = realization
                    modifiedRealization[0], modifiedRealization[index] = modifiedRealization[index], modifiedRealization[0]
                    # print(modifiedRealization)
                else:
                    modifiedRealization = realization

                transformed_times = list(process.transformEventTimes(
                    modifiedRealization))

                prevList = IIDSamples[index]
                prevList.extend(transformed_times)
                IIDSamples[index] = prevList.copy()

        # return all the IID samples for each process by concatenating transformed events from all realizations
        IIDSamples = [np.sort(l) for l in IIDSamples]
        return IIDSamples


################################################################################
#
# U N I T   T E S T I N G
#
################################################################################


# Some global settings for figure sizes
normalFigSize = (8, 6)  # (width,height) in inches
largeFigSize = (12, 9)
xlargeFigSize = (18, 12)


def main():
    # Defining a custom MultiVariateProcess which contains 4 univariate inhomogenous
    # poisson process, each of whose intensity remains unchanged with history
    MTPP = MultiVariateProcess(desc='Multivariate Inhomogenous Poisson')
    numProcesses = MTPP.getNumProcesses()
    numRealizations = 10
    Realizations = []
    maxNumEvents = 0
    for r in range(0, numRealizations):
        # Exponential(100)-distributed right-censoring time
        T = scipy.stats.expon.rvs(loc=0.0, scale=15.0)
        Realization = MTPP.simulate(
            T, [np.array([])] * numProcesses, resume=False)
        print("Right Censoring Time for the current realization {0}".format(T))
        Realizations.append(Realization)
        numEvents = len(Realization[0])
        if numEvents > maxNumEvents:
            maxNumEvents = numEvents
            maxNumEventsIdx = r

    dataStream = DataStream()

    dataStream.initializeFromSimulatedSamples(
        Realizations, MTPP.getSourceNames())

    IIDsamples = MTPP.transformEventTimes(Realizations, dataStream)

    ax = [0] * numProcesses

    # save simulated values back into the DataStream object

    # learning from the simulated samples
    MTPP.setupTraining(dataStream)
    status = MTPP.train()
    print("Training has completed with status: ", status)
    print(MTPP.getParams())
    # TODO: Make generation of axes returnable by a function

    # Here the organisation of the subplots is of a matrix form, we receive all the relevant
    # axes in the appropriate list of axes.

    _, ((ax[0], ax[1]), (ax[2], _)) = plt.subplots(2,
                                                   2, figsize=normalFigSize)
    KSgoodnessOfFitExp1MV(IIDsamples, ax, True)
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
