'''Temporal Point Process (TPP) base class
    version: 1.2
    Authored by: Akshay Aravamudan,Georgios Anagnostopoulos, January 2020
      Edited by:                                              '''

from utils import GoodnessOfFit
from abc import ABC, abstractmethod
import numpy as np
import logging
from enum import Enum
from utils.GoodnessOfFit import KSgoodnessOfFitExp1


# A Python enumaration for training algorithms to report their statuses.
# Such an algorithm is intended to return:
#   CONVERGED, if its convergence criterion has been met.
#   STOPPED, if it has reached its maximum number of iterations and has not met its convergence criterion.
#   ERROR, if it has encountered a situation, which prevents it to recover from.


class TrainingStatus(Enum):
    CONVERGED = 0
    STOPPED = 1
    ERROR = 2


################################################################################
#
# TPP abstract class definition & implementation
#
# An abstract class designed to encapsulate functionalities of uni-variate,
# temproral point process (TPPs). Sub-classes of this class embody specific
# TPPs that can feature external sources of stimulus. This class stipulates
# interfaces for the training, goodness of fit assessment and simulation of
# such TPPs.
#
# ATTRIBUTES:
#   sourceNames:           list of strings; names of the TPP's sources.
#   desc:                  string; short textual description of the TPP.
#   _logger:               logger object; for logging
#   _setupTrainingDone:    boolean; informs whether the setupTraining()
#                          has already been called before.
#
# METHODS:
#   TPP():            TPP object; constructor.
#   setupTraining()   None; setup for training the TPP model.
#   train():          TrainingStatus object; fit TPP model to training data.
#   simulate():       TPPdata; returns simulation output.
#   intensity():      float; value of the TPP's conditional intensity.
#   intensityUB():    float; upper bound of the TPP's cond. intensity within a
#                     given time interval.
#   cumIntensity():   float; value of the TPP's condit. cumulative intensity.
#   GOFplot():        float; returns p-value of KS GOF test and produces GOF
#                     plot.
#   getParams():      None; returns a tuple of TPP model parameters.
#   setParams():      None; sets the TPP model parameters from a provided
#                     tuple.
#   numSources():     integer>=1; returns the number of sources utilized by
#                     this TPP.
#   __str__():        string; TPP's short description.
#   print():          None; prints the TPP's short description.
#
# USAGE EXAMPLES:
#   This abstract class is only intended to be sub-classed;
#   objects of this class cannot be instantiated.
#
# DEPENDENCIES:
#   imports: abc.ABC, abc.abstractmethod, ksgof
#
# NOTES:
#   1) Objects of this class cannot be instantiated.
#   2) Per standard conventions, _setupTrainingDone is a protected base class
#      attribute.
#   3) In the code's documentation, R>=1 will denote the number of realizations
#      for each of the TPP's sources.
#   - "Source" refers to any entity that is associated with or produces events
#     that need to be taken into account to model the TPP.
#     One such entity is the TPP itself. The remaining sources to a TPP are
#     external to the TPP and may be other TPPs themselves. In the code's
#     documentation S>=1 stands for the number of the TPP's sources.
#   - "Realization" refers to a single sample path of a source and is a
#     (possibly, empty) 1D numpy.ndarray of floats that represent relative event
#     times that may include a relative right-censoring time depending on
#     circumstances. All these event times are chronologically ordered.
#   - "TPPdata" refers to a collection of realizations for each source.
#
# AUTHOR(S):
#   Akshay Aravamudan & Georgios Anagnostopoulos, January 2020 (ver. 1.1)
#
################################################################################
class TemporalPointProcess(ABC):
    '''Abstract base class for TPPs'''

    # Instance variables
    #
    #   sourceNames:           list of strings; names of the TPP's sources.
    #   desc:                  string; short textual description of the TPP.
    #   _logger:               logger object; for logging
    #   _setupTrainingDone:    boolean; informs whether the setupTraining()
    #                          has already been called before.

    # Construct a base class object
    def __init__(self, sourceNames=[], desc='', logger=None):
        """[summary]

        Args:
            sourceNames (list, optional): [description]. Defaults to [].
            desc (str, optional): [description]. Defaults to ''.
            logger ([type], optional): [description]. Defaults to None.
        """
        self.sourceNames = sourceNames
        self.desc = desc
        if not isinstance(logger, logging.Logger):
            self._logger = logging.getLogger('null')
            self._logger.addHandler(logging.NullHandler())
        else:
            self._logger = logger
        self._setupTrainingDone = False
        super().__init__()

    # Return the number of realization sources utilized by this TPP.
    # This is a convenience method.
    def numSources(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.sourceNames)

    # Return the sourcenames in the order convenient to this process
    # This is a convenience method
    def getSourceNames(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.sourceNames

    # Create a string representation of the TPP object
    def __str__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.desc

    # Print out the string supplied by self.__str()__
    # Usage example: print(ttpobj) or ttpobj.print()
    def print(self):
        """[summary]
        """
        print(self)

    # Setup TPP model for training.
    # Typically used for pre-computing quantities that are necessary for training.
    # This method is expected to be called prior to a call to the train() method.
    # TPPdata: List of S lists, where each inner list contains R realizations.
    #          Each realization contains at least a relative right-censoring time as its last event time.
    # filename: name of file to be used to store quantities used when training.
    # RealizationFeatures: optional list of R "features" characterizing each realization.
    @abstractmethod
    def setupTraining(self, TPPdata, filename=None, RealizationFeatures=[]):
        """[summary]

        Args:
            TPPdata ([type]): [description]
            filename ([type], optional): [description]. Defaults to None.
            RealizationFeatures (list, optional): [description]. Defaults to [].
        """
        self._setupTrainingDone = True

    # Train the TPP model.
    # trainingSettings: optional settings passed to the training method.
    # Returns an element of the TrainingStatus enumeration.
    @abstractmethod
    def train(self, trainingSettings=None):
        """[summary]

        Args:
            trainingSettings ([type], optional): [description]. Defaults to None.

        Raises:
            Exception: [description]

        Returns:
            [type]: [description]
        """
        # Sample code:
        if self._setupTrainingDone == False:
            raise Exception(
                'TPP: train() method is called before a call to the setupTraining() method!')
        return TrainingStatus.CONVERGED

    # Simulate the TPP.
    # Returns a realization, which includes the relative right-censoring time (rightCensoringTime; see below)
    # as its last event time.
    # rightCensoringTime: strictly positive float; represents the relative censoring time to be used.
    # TPPdata: List of S (possibly, empty) realizations.
    # resume:  boolean; if True, assumes that the TPP's own realization includes a relative
    #          right-censoring time and removes it in order to resume the simulation from the last recorded
    #          event time.
    @abstractmethod
    def simulate(self, rightCensoringTime, TPPdata, resume=False):
        """[summary]

        Args:
            rightCensoringTime ([type]): [description]
            TPPdata ([type]): [description]
            resume (bool, optional): [description]. Defaults to False.
        """
        pass

    # Returns the conditional intensity value of the TPP at a given time t
    # based on the events that have been observed so far.
    # t: scalar or 1D numpy.ndarray of relative times, at which the TPP's intensity is to be computed.
    # TPPdata: List of S (possibly, empty) realizations (one per source).
    # Note: t's value(s) must greater than or equal to the last relative event time of the TPP's own realization.
    @abstractmethod
    def intensity(self, t, TTPdata):
        """[summary]

        Args:
            t ([type]): [description]
            TTPdata ([type]): [description]
        """
        pass

    # Returns the value of the cumulative intensity of the process at a given time t
    # based on the events that have been observed so far.
    # t: scalar or 1D numpy.ndarray of relative times, at which the TPP's intensity is to be computed.
    # TPPdata: List of S (possibly, empty) realizations (one per source).
    # Note: t's value(s) must greater than or equal to the last relative event time of the TPP's own realization.
    @abstractmethod
    def cumIntensity(self, t, TPPdata):
        """[summary]

        Args:
            t ([type]): [description]
            TPPdata ([type]): [description]
        """
        pass

    # Returns transformed event times as numpy.ndarray of the TPP's own realization. These are typically used
    # for goodness of fit purposes.
    # TPPdata: List of S non-empty realizations (one per source).
    def transformEventTimes(self, TPPdata):
        """[summary]

        Args:
            TPPdata ([type]): [description]

        Returns:
            [type]: [description]
        """
        transformedTimes = []
        source0 = TPPdata[0]  # the process' own realization
        for n in range(len(source0[:-1])):
            
            tn = source0[n]
            TPPdataHistory = []
            for source in TPPdata:
                isLessThanTn = (tn > np.array(source[:-1]))
                TPPdataHistory.append(np.array(source[:-1])[isLessThanTn])
            cum_rc = self.cumIntensity(source0[-1], TPPdataHistory)
            cum_tc = self.cumIntensity(tn, TPPdataHistory)
        
            exponential_transform = (1-np.exp(-cum_rc))/(np.exp(-cum_tc) - np.exp(-cum_rc))
            transformedTimes.append(np.log(exponential_transform))

        transformed_time = np.log(np.abs(exponential_transform))
            
        return np.array(transformedTimes)

    # Returns the KS p-value as a measure of goodness of fit of
    # the TPP's model to one of its own realizations. Optionally,
    # provides the associated Probability-Probability (P-P) plot.
    # TPPdata: List of S non-empty realizations (one per source).
    # ax: figure axis; used to display the P-P graph, if ax != None.
    # showConfidenceBands: boolean; if =True, shows confidence bands on the P-P graph.
    def GOFks(self, TPPdata, ax=None, showConfidenceBands=False):
        """[summary]

        Args:
            TPPdata ([type]): [description]
            ax ([type], optional): [description]. Defaults to None.
            showConfidenceBands (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        transformedTimes = self.transformEventTimes(TPPdata)
        if len(transformedTimes) == 0:
            return []
        pvalue = KSgoodnessOfFitExp1(
            transformedTimes, ax, showConfidenceBands)
        if ax is not None:
            ax.set_title(
                'P-P Plot: {:1} \n p-value: {:0.3f}'.format(self.desc, pvalue))
        return transformedTimes

    # Returns the upper bound of the intensity function in the time inerval from [t, rightCensoringTime]
    # based on the events that have been observed so far.
    # t: float; a relative time.
    # rightCensoringTime: float; relative right-censoring time; must be larger than t.
    # TPPdata: List of S non-empty realizations (one per source).
    # Note: t's value(s) must greater than or equal to the last relative event time of the TPP's own realization.
    @abstractmethod
    def intensityUB(self, rightCensoringTime, TPPdata):
        pass

    # Returns a tuple containing all of a TPP's parameters.
    @abstractmethod
    def getParams(self):
        pass

    # Sets a TPP's parameters.
    # params: a tuple of parameters
    @abstractmethod
    def setParams(self, params):
        pass

    # Initializes a TPP's parameters.
    @abstractmethod
    def initParams(self):
        pass
