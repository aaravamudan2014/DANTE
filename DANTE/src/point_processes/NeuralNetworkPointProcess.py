''' Neural Network Point Process (TPP) base class
    version: 0.1
    Authored by: Akshay Aravamudan March 2020
      Edited by: ...                                             '''

from utils import GoodnessOfFit
from abc import ABC, abstractmethod
import numpy as np
import logging
from enum import Enum
from utils.GoodnessOfFit import KSgoodnessOfFitExp1
from point_processes.TemporalPointProcess import TemporalPointProcess as TPP, TrainingStatus

################################################################################
#
# NNTPP abstract class definition & implementation
#
# A abstract class designed to encapsulate functionalties of a uni-variate
# temporal point process where certain characteristic functions are modelled
# by a neural network. This class only intializes the basic keras characteristics
# for a point process. Sub classes of this class will have to have implement
# the individual layers and loss functions of the neural network. This class
# is also inherited from TPP abstract class, thereby obtaining all the basic
# functionalities associated with a point process.
#
#
# ATTRIBUTES:
#
#


#
#
#
#
#
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

class NeuralNetworkPointProcess(TPP):
    ''' Abstract base class for neural network based point processes '''

    def __init__(self):
        self.initParams()

        super.__init__()

    @abstractmethod
    def input_model(self):
        pass

    @abstractmethod
    def hiddenLayers(self):
        pass

    @abstractmethod
    def output_model(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class FullyNeuralNetworkPointProcess(NeuralNetworkPointProcess):
    '''  Sub-class for the fully neural network based point process (By Omi et al.)'''

    def __init__(self):
        super.__init__()

    def input_model(self):
        pass

    def hiddenLayers(self):
        pass

    def output_model(self):
        pass

    def predict(self):
        pass


class MVNNPP(NeuralNetworkPointProcess):
    ''' Sub-class for a fully neural network based multivariate point process '''

    def __init__(self):
        super.__init__()

    def input_model(self):
        pass

    def hiddenLayers(self):
        pass

    def output_model(self):
        pass

    def predict(self):
        pass


def main():
    pass


if __name__ == "__main__":
    main()
