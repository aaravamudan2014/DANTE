#!/usr/bin/python
'''PointProcessCollection.py
    Temporal Point Process (TPP) sub-classes
        Author: Georgios C. Anagnostopoulos, April 2019
     Edited by: Akshay Aravamudan (added some documentation and fixed imports)
'''


##########################################################################################
# PointProcessCollection.py
#   This file contains all the sub-classes for the Temporal Point Process (tpp) base class.
#   Currently the following point processes have been implemented:
#       - Inhomogenous Poisson Process
#
##########################################################################################


from matplotlib import pyplot as plt
import numpy as np
import scipy.stats
from point_processes.TemporalPointProcess import TemporalPointProcess as TPP, TrainingStatus
from utils.MemoryKernel import *


class PoissonTPP(TPP):
    '''Poisson Temporal Point Process'''

    # Instance variables
    #
    #   sourceNames:           list of strings; names of the TPP's sources; inherited from TPP base class.
    #   desc:                  string; short textual description of the TPP; inherited from TPP base class.
    #   _logger:               logger object; for logging; inherited from TPP base class.
    #   _setupTrainingDone:    boolean; informs whether the setupTraining(); inherited from TPP base class.
    #   alpha                  float; TPP parameter.
    #   mk                     object of a MemoryKernel subclass; TPP parameter.
    #   __totalTrainingEvents: non-negative integer; a quantity required during training.
    #   __sumOfPsis:           non-negative real scalar; a quantity required during training.

    # mk:          a specific MemoryKernel object with fixed parameters.
    # sourceNames: list of strings with the source names. The list's first element is the name of the TPP's own source.
    def __init__(self, mk, sourceNames, desc='Poisson TPP', logger=None):
        self.mk = mk
        self.initParams()  # initialize alpha
        super().__init__(sourceNames, desc, logger)

    # t: float or 1-dimensional numpy.ndarray of reals; relative event time(s). Typically, larger than 0.
    # TPPdata: a list containing the process' own (possibly, empty) realization.
    def intensity(self, t, TPPdata):
        # TPPdata is actually unused for this process
        value = self.alpha * self.mk.phi(t)
        return value

    # t: float or 1-dimensional numpy.ndarray of reals; relative event time(s). Typically, larger than 0.
    # TPPdata: a list containing the process' own (possibly, empty) realization.
    def cumIntensity(self, t, TPPdata):
        Realization = TPPdata[0]
        tN = Realization[-1] if len(Realization) > 0 else 0.0
        value = self.alpha * (self.mk.psi(t) - self.mk.psi(tN))
        return value

    # Simulate the TPP via the inverse time transformation approach
    # Returns a realization, which includes the relative right-censoring time (rightCensoringTime; see below)
    # as its last event time.
    # rightCensoringTime: strictly positive float; represents the relative censoring time to be used.
    # TPPdata: List of S (possibly, empty) realizations.
    # resume:  boolean; if True, assumes that the TPP's own realization includes a relative
    #          right-censoring time and removes it in order to resume the simulation from the last recorded
    #          event time.
    def simulate(self, rightCensoringTime, TPPdata, resume=False):
        RealizationList = list(TPPdata[0])
        SimulatedRealization = RealizationList[:-1] if (
            resume == True and len(RealizationList) > 0) else RealizationList
        # time of last event of the realization
        t = SimulatedRealization[-1] if len(SimulatedRealization) > 0 else 0.0
        stdExponentialDistribution = scipy.stats.expon(loc=0.0, scale=1.0)
        while True:
            z = stdExponentialDistribution.rvs()  # draw an Exp(1) sample
            t = self.mk.psiInv(z / self.alpha + self.mk.psi(t))
            if t > rightCensoringTime:
                SimulatedRealization.append(rightCensoringTime)
                break
            else:
                SimulatedRealization.append(t)
        return np.array(SimulatedRealization)

    # Setup TPP model for training.
    # Typically used for pre-computing quantities that are necessary for training.
    # This method is expected to be called prior to a call to the train() method.
    # TPPdata: List of S lists, where each inner list contains R realizations.
    #          Each realization contains at least a relative right-censoring time as its last event time.
    def setupTraining(self, TPPdata):
        Realizations = TPPdata[0]
        self.__totalTrainingEvents = 0
        self.__sumOfPsis = 0.0
        for Realization in Realizations:
            # excludes relative right-censoring time
            self.__totalTrainingEvents += len(Realization) - 1
            T = Realization[-1]  # process' relative right-censoring time
            self.__sumOfPsis += self.mk.psi(T)
        self._setupTrainingDone = True  # indicate that setupTraining() has been ran already
        self._logger.info('PoissonTPP.setupTraining() finished.')

    # Train the TPP model.
    # Returns an element of the TrainingStatus enumeration.
    def train(self):
        if self._setupTrainingDone == False:
            raise Exception(
                'PoissonTPP: train() method is called before a call to the setupTraining() method!')

        if self.__sumOfPsis != 0.0:
            self.alpha = self.__totalTrainingEvents / self.__sumOfPsis
        else:
            self.alpha = float('inf')
            self._logger.warning('PoissonTPP.train(): infinite alpha value!')
        status = TrainingStatus.CONVERGED
        self._logger.info('PoissonTPP.train(): training converged.')

        return status

    # Is used by an Ogata's thinning algorithm to simulate the process.
    def intensityUB(self, t, rightCensoringTime, TPPdata):
        Realization = TPPdata[0]
        # time of last event of the realization
        t_final = Realization[-1] if len(Realization) > 0 else 0.0
        assert t >= t_final, "Upper bound time value must be greater than latest event in the history"
        value = self.alpha * self.mk.phiUB(t, rightCensoringTime)
        return value

    # Returns the TTP's model parameters as a tuple.
    def getParams(self):
        return self.alpha, self.mk

    # Sets a TPP's parameters.
    # params: a tuple of parameters
    def setParams(self, params):
        self.alpha = params[0]
        self.mk = params[1]

    # Initializes a TPP's parameters
    def initParams(self):
        self.alpha = 1.0  # assing a dummy positive value


################################################################################
#
# U N I T   T E S T I N G
#
################################################################################


# Some global settings for figure sizes
normalFigSize = (8, 6)  # (width,height) in inches
largeFigSize = (12, 9)
xlargeFigSize = (18, 12)

# Unit-test PoissonTPP class


def unitTest_PoissonTPP():
    # Define the ground-truth TPP as a (inhomogenous) Poisson TPP with a GammaGompertz(1.0, 2.0) memory kernel.
    mk = GammaGompertzMemoryKernel(beta=1.0, gamma=2.0)
    sourceNames = ['ownEvents']  # this TPP only uses its own events as source.
    truetpp = PoissonTPP(
        mk, sourceNames, desc='Poisson TPP with a GammaGompertz(1.0, 2.0) kernel')
    truetpp.alpha = 1.0  # set parameter for the TPP's conditional intensity.

    # Print out informormation
    print("The procces' description: ", end='')
    print(truetpp)
    print("The procces' source names: ", end='')
    print(truetpp.sourceNames)

    # Generate some realizations with random start times & right-censoring times
    numRealizations = 10
    Realizations = []
    maxNumEvents = 0
    maxNumEventsIdx = 0
    for r in range(0, numRealizations):
        # Exponential(100)-distributed right-censoring time
        T = scipy.stats.expon.rvs(loc=0.0, scale=100.0)
        Realization = truetpp.simulate(T, [np.array([])])
        Realizations.append(Realization)
        numEvents = len(Realization)
        if numEvents > maxNumEvents:
            maxNumEvents = numEvents
            maxNumEventsIdx = r

    # Define the model TPP as a (inhomogenous) Poisson TPP with a GammaGompertz(1.0, 2.0) memory kernel.
    mk = GammaGompertzMemoryKernel(beta=1.0, gamma=2.0)
    sourceNames = ['ownEvents']  # this TPP only uses its own events as source.
    modeltpp = PoissonTPP(
        mk, sourceNames, desc='Poisson TPP with a GammaGompertz(1.0, 2.0) kernel')

    # Train the model
    # pre-calculates quantities needed for training
    modeltpp.setupTraining([Realizations])
    status = modeltpp.train()
    print('Training resulted in a status of {:1}'.format(status))

    # Assess trained model's quality
    # only meaningful if both TPPs use the same kernel and the same kernel parameter values.
    print('Estimated a={:1} vs. true a={:2}'.format(
        modeltpp.alpha, truetpp.alpha))
    # pick, for example, the longest realization that was used for training.
    Realization = Realizations[maxNumEventsIdx]
    _, ax = plt.subplots(1, 1, figsize=normalFigSize)
    # generate the P-P plot
    pvalue = modeltpp.GOFks(
        [Realization], ax, showConfidenceBands=True)
    print('KS test p-value={}'.format(pvalue))
    plt.show()


# Unit testing of implemented TPPs
def main():
    unitTest_PoissonTPP()


if __name__ == '__main__':
    main()
