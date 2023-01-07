'''
MultiVariateProcess.py
    Multi variate inhomogenous poisson process implementation
    Author: Akshay Aravamudan, January 2020
'''

from utils.MemoryKernel import ConstantMemoryKernel, RayleighMemoryKernel, PowerLawMemoryKernel, GammaGompertzMemoryKernel
from utils.DataReader import createDummyDataSplitPopulation
from utils.DataReader import *
from utils.MemoryKernel import *
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
from point_processes.SplitPopulationTPP import *

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
# AUTHOR: Akshay Aravamudan,Xi Zhang January 2020
#
##################################################


class MultiVariateProcess(object):
    def __init__(self, desc=None):

        self._setupTrainingDone = False
        self._sourceNames = ['github', 'reddit', 'twitter']
        if desc is not None:
            self.desc = desc
        else:
            logger.Warning(
                "No name passed to process, assigning default: Multivariate Inhomogenous Poisson")
            self.desc = 'Multivariate Inhomogenous Poisson'
        stop_criteria = {'max_iter': 600,
                         'epsilon': 1e-6}

        dataStream = DataStream()

        dataStream.initializeFromSimulatedSamples(
            [] , self._sourceNames, test_integrity = False)

        _, sourceNames_0 = dataStream.getDataStreamLearning(
            self._sourceNames[0])
        _, sourceNames_1 = dataStream.getDataStreamLearning(
            self._sourceNames[1])
        _, sourceNames_2 = dataStream.getDataStreamLearning(
            self._sourceNames[2])

        mk = ExponentialPseudoMemoryKernel(beta=1.0)
        sourceNames = ['github', 'reddit', 'twitter']

        stop_criteria = {'max_iter': 10000,
                            'epsilon': 1e-6}

        # exploitProcess = SplitPopulationTPP(
        #     mkList, sourceNames, stop_criteria,
        #     desc='Split population process with multiple kernels')

        githubProcess = HawkesTPP(
            mk=mk, sourceNames=sourceNames_0, desc='Github TPP with exponential excitation kernel', stop_criteria=stop_criteria)
        redditProcess = HawkesTPP(
            mk=mk, sourceNames=sourceNames_1, desc='Hawkes TPP with exponential excitation kernel', stop_criteria=stop_criteria)
        twitterProcess = HawkesTPP(
            mk=mk, sourceNames=sourceNames_2, desc='Hawkes TPP with exponential excitation kernel', stop_criteria=stop_criteria)


        github_alpha_dict = {'github':0.602281,'reddit':0.061859,'twitter':0.003505 }
        githubProcess.mu = 0.0005202096240235445
        github_alphas = []
        for source_name in sourceNames_0:
            github_alphas.append(github_alpha_dict[source_name])
        githubProcess.alpha = pd.Series(
            data=np.array(github_alphas), index=sourceNames_0)
        
        reddit_alpha_dict = {'github':0.001409,'reddit':0.655921,'twitter':0.000756  }
        redditProcess.mu  = 1.7346912710224584e-05
        reddit_alphas = []
        for source_name in sourceNames_1:
            reddit_alphas.append(reddit_alpha_dict[source_name])
        redditProcess.alpha = pd.Series(
            data=np.array(reddit_alphas), index=sourceNames_1)
       

        twitter_alpha_dict = {'reddit':0.124354,'github':0.014489,'twitter':0.608532}
        twitterProcess.mu = 0.0011215005012222806
        twitter_alphas = []
        for source_name in sourceNames_2:
            twitter_alphas.append(twitter_alpha_dict[source_name])
        twitterProcess.alpha = pd.Series(
            data=np.array(twitter_alphas), index=sourceNames_2)
        

        # exploitProcess.alpha = np.array([0.01763168, 0.00082092, 0.00037443, 0.00015121])
        # exploitProcess.w_tilde = np.array( [4.71532895, -3.98011785, -1.26478897])



        self.Processes = [ githubProcess, redditProcess, twitterProcess]
        # self.exploitProcess = exploitProcess
        self._params = []
        for process in self.Processes:
            self._params.append(process.alpha)

    def getSourceNames(self):
        return self._sourceNames

    def setParams(self, params):
        self._params = params

    def getParams(self):
        return self._params, self._sourceNames

    def simulate(self, rightCensoringTime, MTPPdata, resume,isRightCensoringTimeAttached):
        # also simulate split population
        # simulate random uniform distribution
        # if it is susceptible simulate the whole process
        exponential_beta = 0.5
        power_law_beta = 2.0

        mkList = [WeibullMemoryKernel(0.5),
            ExponentialPseudoMemoryKernel(beta=exponential_beta),
            ExponentialPseudoMemoryKernel(beta=exponential_beta),
            PowerLawMemoryKernel(beta=power_law_beta)]

        sourceNames = ['base', 'github', 'reddit', 'twitter']

        stop_criteria = {'max_iter': 100,
                            'epsilon': 1e-16}
        features_rw, realizations_rw, isExploited_rw = generateRealWorldExploitDataset()
            
        exploitProcess = SplitPopulationTPP(
            mkList, sourceNames, stop_criteria,
            desc='Split population process with multiple kernels')

        return simulation(self.Processes, rightCensoringTime=rightCensoringTime,
                          MTPPdata=MTPPdata, resume=resume,isRightCensoringTimeAttached=isRightCensoringTimeAttached)


        
    def train(self, dataStream):
        # assert self._setupTrainingDone, "Setup for the training has to be complete before calling the train function"
        isConverged = True
        for process in self.Processes:
            streams, sourceNames = dataStream.getDataStreamLearning(
                process.getSourceNames()[0])
            status = process.train(streams, overwrite_precal=False)
            print(process.alpha)
            print(process.mu)
            if not status == TrainingStatus.CONVERGED:
                isConverged = False
                break

        paramList = []
        for process in self.Processes:
            paramList.append(process.alpha)
            # print(paramList)
        self.setParams(paramList)
        
        if isConverged:
            return TrainingStatus.CONVERGED
        else:
            return TrainingStatus.ERROR

    def setupTraining(self, dataStream):
        assert isinstance(
            dataStream, DataStream), "Passed data object must from the datastream class"
        for index, process in enumerate(self.Processes):
            streams, sourceNames = dataStream.getDataStreamLearning(
                process.getSourceNames()[index])
            self.Processes[index] = process.setupTraining(streams)
        self._setupTrainingDone = True

    def getNumProcesses(self):
        return len(self.Processes)

    def transformEventTimes(self, MTPPdata, dataStream):
        print("Transforming times")
        IIDSamples = []
        MTPPdata = np.array(MTPPdata)
        for i in range(self.getNumProcesses()):
            IIDSamples.append([])
        for index,realization in enumerate(MTPPdata[np.random.choice(range(len(MTPPdata)), 200)]):
            print(index)
            assert len(realization) == len(
                self.Processes), "Inconsistency detected in processes and number of streams"
            # iterate through available streams in realization to accordingly transform the times
            for process in self.Processes:
                sourceName = process.getSourceNames()[0]
            
                # get position of string in processes sourceNames list
                index = self._sourceNames.index(
                    sourceName) if len(self.Processes) > 1 else 0
                
                if not index == 0:
                    # swap 0th position and the position of the index in the realizations
                    # print(realization)
                    for i in range(len(modifiedRealization)):
                        modifiedRealization = realization.copy()
                        modifiedRealization[0], modifiedRealization[index] = modifiedRealization[index], modifiedRealization[0]
                else:
                    modifiedRealization = realization
                if len(modifiedRealization[0]) < 30:
                    transformed_times = list(process.GOFks(
                        [modifiedRealization]))
                    prevList = IIDSamples[index]
                    prevList.extend(transformed_times)
                    IIDSamples[index] = prevList.copy()

        # return all the IID samples for each process by concatenating transformed events from all realizations
        IIDSamples = [np.sort(l) for l in IIDSamples]
        # IIDSamples = IIDSamples[0]
        # print(len(IIDSamples))
        # input()
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
    training_features, training_realizations, training_isExploited, \
        test_features, test_realizations, test_isExploited, \
        validation_features, validation_realizations, validation_isExploited = generateExploitSocialMediaDataset()
    
    # only include social media data
    modified_realizations = []
    for index, realization in enumerate(validation_realizations):
        modified_realizations.append(np.array(realization[1:]))
    
    
    ######### learning from the training samples
    ####### save simulated values back into the DataStream object
    dataStream = DataStream()
    dataStream = dataStream.initializeFromSimulatedSamples(
        modified_realizations, MTPP.getSourceNames(), test_integrity=False)
    
    # MTPP.setupTraining(dataStream)
    # status = MTPP.train(dataStream)


    rightCensoringTime = 9000


     
    # realization =  MTPP.simulate(rightCensoringTime, modified_realizations[0], True,True)
    # print(len(realization[0]))
    # input()
    ##### Given that the system of processes have been learned, we set up problems for individual tuples (t_c, delta_t) 
    ##### For each tuple, we cut all SM event after t_c

    tc = 0
    delta_t = 30

    # MTPP.predictUsingSimulation(tc, delta_t, validation_realizations, validation_features)
    IIDsamples = MTPP.transformEventTimes(modified_realizations, dataStream)
    ax = [0] * numProcesses

    # print(MTPP.getParams())
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
