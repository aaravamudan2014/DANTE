#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Temporal Point Process (TPP) sub-class for a survival process with
    a split populations. Split-population renders some of the populations
    with a property that they can never get infected.

    This class encompasses all the features for any point process:
    - simulation
    - training,
    - goodness of fit,

    Author: Akshay, Xixi.
    Version: 1.0.
    Edited by:

'''
import json
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.metrics import confusion_matrix
# import logging
import os
import os.path
import h5py
from matplotlib import pyplot as plt
import ast
from point_processes.TemporalPointProcess import TemporalPointProcess as TPP, TrainingStatus
from point_processes.PointProcessCollection import *

from utils.GoodnessOfFit import KSgoodnessOfFitExp1, KSgoodnessOfFitExp1MV
from point_processes.UnivariateHawkes import *
from utils.MemoryKernel import *
from utils.FeatureVectorCreator import *
from utils.Simulation import *
from utils.DataReader import *
from core.DataStream import DataStream
import pickle
import itertools


class SurvivalProcess(TPP):

    '''Survival Point Process'''

    # Instance variables
    #
    #   sourceNames:           list of strings; names of the TPP's sources; inherited from TPP base class.
    #   ownName:               string, name of the current TPP, must be one of the sourceNames; inherited form TPP base class
    #   desc:                  string; short textual description of the TPP; inherited from TPP base class.
    #   _logger:               logger object; for logging; inherited from TPP base class.
    #   _setupTrainingDone:    boolean; informs whether the setupTraining(); inherited from TPP base class.
    #   mu:                    float; TPP parameter: base intensity
    #   alpha                  numpy array; TPP parameter.
    #   mk                     object of a MemoryKernel subclass; TPP parameter.
    #   stop_criteria:         dict: stop criteria for learning ; TPP parameter
    #   __totalTrainingEvents: non-negative integer; a quantity required during training.
    #   __sumOfPsis:           non-negative real scalar; a quantity required during training.

    # mk:          a specific MemoryKernel object with fixed parameters.
    # sourceNames: list of strings with the source names. The list's first element is the name of the TPP's own source.
    def __init__(self, mk, sourceNames, stop_criteria, desc='Split Population TPP', logger=None):
        """[summary]

        Args:
            mk (MemoryKernel): object of a MemoryKernel subclass; TPP parameter.
            sourceNames ([type]): [description]
            stop_criteria ([type]): [description]
            desc (str, optional): [description]. Defaults to 'Split Population TPP'.
            logger ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        self.dim = len(sourceNames)
        self.desc = desc
        self.sourceNames = sourceNames
        self.ownName = sourceNames[0]

        

        # # TODO: temporary variable, will be replaces
        # self.pre_cal_path = '../temp_storage/'
        # self.pre_cal_name = '{}-pre-cal.h5'.format(self.ownName)

        # TODO: actual variable names in usage
        self.stop_criteria = stop_criteria
        assert len(mk) == len(
            sourceNames), "The list of memory kernels is inconsistent with the number of sources (self [base], external sources)"
        self.mk = mk
        self.feature_vectors = None
        self.weight_vector = None
        self.alpha = None
        self.exploited_probs = []
        self.non_exploited_probs = []

        self.initParams()
        super().__init__(sourceNames, desc, logger)

    # t: float or 1-dimensional numpy.ndarray of reals; relative event time(s). Typically, larger than 0.
    # realization: a single realization of model including all sources events

    def intensity(self, t, MTPPdata):
        """[summary]

        Args:
            t ([type]): [description]
            MTPPdata ([type]): [description]

        Returns:
            [type]: [description]
        """
        intensity_val = 0.0
        t_0 = 0.0
        hasRealization = False
        if (len(MTPPdata[0]) > 0):
            # if t is less than its own realization
            if not t < MTPPdata[0][0]:
                hasRealization = True

        if (self.isSusceptible) and (not hasRealization):
            # intensity_val = np.mat(self.alpha, self.Phi_matrix)
            phi_vector = np.zeros(len(self.sourceNames))
            # base process
            phi_vector[0] = self.mk[0].phi(t-t_0)
            # filling up Phis for every other platform in the phi_vector
            if len(MTPPdata) > 1:
                for source_index in range(len(self.sourceNames[1:])):
                    phi_vector[source_index + 1] += np.sum(np.array([self.mk[source_index + 1].phi(
                        t-t_i) for t_i in MTPPdata[source_index+1] if t_i < t]))
            intensity_val += np.dot(phi_vector, self.alpha)
        return intensity_val

    # Is used by an Ogata's thinning algorithm to simulate the process.

    def intensityUB(self, t, rightCensoringTime, MTPPdata):
        """[summary]

        Args:
            t ([type]): [description]
            rightCensoringTime ([type]): [description]
            MTPPdata ([type]): [description]

        Returns:
            [type]: [description]
        """
        intensity_ub = 0.0
        t_0 = 0.0
        hasRealization = False

        if (len(MTPPdata[0]) > 0):
            # if t is less than its own realization
            if t > MTPPdata[0][0]:
                hasRealization = True

        if (self.isSusceptible) and (not hasRealization):
            # intensity_val = np.mat(self.alpha, self.Phi_matrix)
            phiUB_vector = np.zeros(len(self.sourceNames))
            # base process
            phiUB_vector[0] = self.mk[0].phiUB(t-t_0, rightCensoringTime)
            # filling up Phis for every other platform in the phi_vector
            if len(MTPPdata) > 1:
                for source_index in range(len(self.sourceNames[1:])):
                    phiUB_vector[source_index + 1] += np.sum(
                        np.array([self.mk[source_index+1].phiUB(t-t_i, rightCensoringTime) for t_i in MTPPdata[source_index+1] if t_i < t]))
            intensity_ub += np.dot(phiUB_vector, self.alpha)


        return intensity_ub

    # t: float or 1-dimensional numpy.ndarray of reals; relative event time(s). Typically, larger than 0.
    # TPPdata: a list containing the process' own (possibly, empty) realization.

    def cumIntensity(self, t, MTPPdata):
        """[summary]

        Args:
            t ([type]): [description]
            MTPPdata ([type]): [description]

        Returns:
            [type]: [description]
        """
        cum_intensity_val = 0.0
        t_0 = 0
        hasRealization = False

        # if (len(MTPPdata[0]) > 0):
        #     # if t is less than its own realization
        #     if t > MTPPdata[0][0]:
        #         hasRealization = True

        if (self.isSusceptible) and (not hasRealization):
            # intensity_val = np.mat(self.alpha, self.Phi_matrix)
            psi_vector = np.zeros(len(self.sourceNames))
            # base process
            psi_vector[0] = self.mk[0].psi(t-t_0)
            # filling up Phis for every other platform in the phi_vector
            if len(MTPPdata) > 1:
                for source_index in range(len(self.sourceNames[1:])):
                    psi_vector[source_index + 1] += np.sum(
                        np.array([self.mk[source_index+1].psi(t-t_i) for t_i in MTPPdata[source_index+1] if t_i < t]))

            cum_intensity_val += np.dot(psi_vector, self.alpha)

        # print("psi vector: ",psi_vector)
        return cum_intensity_val

    # the transformed times are calculted based on precalculated values
    # def transformEventTimes(self, MTPPdata):
    #     assert self.Phi_matrix is not None, "Precalculated Phi values are missing, load them first.. "
    #     assert self.Psi_matrix is not None, "Precalculated Psi values are missing, load them first.. "

    #     return

    # TODO: make this the main function dealing with all precal files to produce a matrix of Phis and Psis ,
    # TODO: i.e. for n processes, for each time instance, have two associated values
    # Setup TPP model for training.
    # Typically used for pre-computing quantities that are necessary for training.
    # This method is expected to be called prior to a call to the train() method.
    # TPPdata: List of S lists, where each inner list contains R realizations.
    #          Each realization contains at least a relative right-censoring time as its last event time.

    def setupTraining(self, TPPdata, pre_cal_filename, validation_MTPPdata, verbose=1, append=True):
        """[summary]

        Args:
            TPPdata ([type]): [description]
            feature_vectors ([type]): [description]
            pre_cal_filename ([type]): [description]
            validation_MTPPdata ([type]): [description]
        """
        # first extract all the time instances available in the data

        self.__totalTrainingEvents = 0
        self.__sumOfPsis = 0.0

        # assert len(feature_vectors_training) == len(TPPdata), "inconsistent number of training features"
        # assert len(feature_vectors_validation) == len(validation_MTPPdata), "inconsstent number of validation features"
        # feature_vectors_training, feature_vectors_validation = self.setFeatureVectors(TrainingStatus, feature_vectors_validation=feature_vectors_validation, append=append)
        # if feature_vectors is not None:
        #     assert len(feature_vectors) == len(TPPdata)
        # if append:
        #     updated_feature_vectors = np.zeros(
        #         (len(feature_vectors), len(feature_vectors[0]) + 1))

        #     for index in range(len(feature_vectors)):
        #         updated_feature_vectors[index, :] = np.array(
        #             np.append(feature_vectors[index], 1.0))

        #     self.feature_vectors = updated_feature_vectors
        # else:
        #     self.feature_vectors = feature_vectors
        self._logger.info('PoissonTPP.setupTraining() finished.')

        if not os.path.isfile("../data/exploit_paper_scenarios/"+pre_cal_filename+".pickle"):
            self._setupTrainingDone = self.precalculate_vals(
                TPPdata, pre_cal_filename, validation_MTPPdata, verbose=verbose)
        else:
            if verbose == 1:
                print("Pre_cal file {0} already exists, using that..".format(
                    pre_cal_filename))

    def setFeatureVectors(self, feature_vectors_training, feature_vectors_validation=None, append=True):
        """[summary]

        Args:
            feature_vectors ([type]): [description]
        """
        if append:
            updated_feature_vectors = np.zeros(
                (len(feature_vectors_training), len(feature_vectors_training[0]) + 1))

            for index in range(len(feature_vectors_training)):
                updated_feature_vectors[index, :] = np.array(
                    np.append(feature_vectors_training[index], 1.0))

            self.feature_vectors_training = updated_feature_vectors
        else:
            self.feature_vectors_training = feature_vectors_training
        self.feature_vectors_validation = None
        if feature_vectors_validation is not None:
            if append:
                updated_feature_vectors = np.zeros(
                    (len(feature_vectors_validation), len(feature_vectors_validation[0]) + 1))
                for index in range(len(feature_vectors_validation)):
                    updated_feature_vectors[index, :] = np.array(
                        np.append(feature_vectors_validation[index], 1.0))

                self.feature_vectors_validation = updated_feature_vectors
            else:
                self.feature_vectors_validation = feature_vectors_validation
        

        return self.feature_vectors_training,self.feature_vectors_validation

    # def neg_log_likelihood_upper_bound_weight(self,A, w_tilde,H):
    #     return (H + np.dot(w_tilde, w_tilde)*A)

    # def neg_log_likelihood_upper_bound_alpha(self,B, D, alpha_dash, E):
    #     return np.dot(alpha_dash, B) - np.dot(np.log(alpha_dash), D) + (np.dot(alpha_dash, alpha_dash)*E)

    def log_likelihood(self, H, J):
        """[summary]

        Args:
            H ([type]): [description]
            J ([type]): [description]

        Returns:
            [type]: [description]
        """
        return H + J

    def calculate_terms(self, alpha_dash, num_realizations, MTPPdata, dataset_dict):
        """[summary]

        Args:
            alpha_dash ([type]): [description]
            w_dash ([type]): [description]
            num_realizations ([type]): [description]
            MTPPdata ([type]): [description]
            dataset_dict ([type]): [description]

        Returns:
            [type]: [description]
        """
        D = np.zeros(len(self.sourceNames))
        E = np.zeros(len(self.sourceNames))
        B = np.zeros(len(self.sourceNames))
        H = 0.0
        I = 0.0
        J = 0.0
        assert num_realizations == len(self.feature_vectors), "feature vector length inconsistent with number of samples"
        # For this iteration, the values of the parameters to the model are alpha_dash and w_dash
        for c in range(num_realizations):
            realization = MTPPdata[c]

            Psi_matrix = dataset_dict[str(c)]['Psi_matrix']
            Phi_matrix = dataset_dict[str(c)]['Phi_matrix']

            time_instances = dataset_dict[str(c)]['time_instances']
            # The right censoring time for this process's own realization.
            # This works because the first source realization in the list is its own.
            T_c = realization[0][-1]
            own_realization = realization[0]
            isSusceptible = False
            if len(own_realization) > 1:
                isSusceptible = True

            if not isSusceptible:
                # A += (1 / (1 + np.exp(-np.dot(alpha_dash, Psi_matrix[:, -1]) +
                #                       np.dot(self.feature_vectors[c], w_dash))))*self.feature_vectors[c]
                # E += (1 - (1 / (1 + np.exp(-np.dot(alpha_dash, Psi_matrix[:, -1]) + np.dot(
                #     self.feature_vectors[c], w_dash))))) * Psi_matrix[:, -1]
                
                J += np.dot(alpha_dash, Psi_matrix[:, -1])
            else:
                # the first realization of its own event is when the
                t_c = own_realization[0]
                req_index = 0
                for index, t in enumerate(time_instances):
                    if t == t_c:
                        req_index = index
                if np.dot(alpha_dash, Phi_matrix[:, req_index]) > 0:
                    D += (alpha_dash * Phi_matrix[:, req_index]) / \
                        (np.dot(alpha_dash, Phi_matrix[:, req_index]))
                    
                    H += np.dot(alpha_dash, Psi_matrix[:, req_index]) - np.log(
                            np.dot(alpha_dash, Phi_matrix[:, req_index]))
                    # H += np.log(1 + np.exp(-np.dot(self.feature_vectors[c], w_dash))) + \
                    #     np.dot(alpha_dash, Psi_matrix[:, req_index]) - np.log(
                    #         np.dot(alpha_dash, Phi_matrix[:, req_index]))

                B += Psi_matrix[:, req_index]

            # F += 1 / \
            #     (1+np.exp(np.dot(self.feature_vectors[c],
            #                         w_dash))) * self.feature_vectors[c]
        return B, D, H, J

    def train(self, MTPPdata, precal_filename, validation_MTPPdata, verbose = 1):
        """[summary]

        Args:
            MTPPdata ([type]): [description]
            precal_filename ([type]): [description]
            validation_MTPPdata ([type]): [description]

        Returns:
            [type]: [description]
        """
        # read from the precalc file first
        # hf = h5py.File('myfile.hdf5', 'r')

        with open("../data/exploit_paper_scenarios/"+precal_filename+".pickle", 'rb') as handle:
            dataset_training_dict = pickle.load(handle)
        with open("../data/exploit_paper_scenarios/__validation__"+precal_filename+".pickle", 'rb') as handle:
            dataset_validation_dict = pickle.load(handle)
        best_parameter_dict_training = {}
        best_parameter_dict_training['alpha'] = None
        best_parameter_dict_training['w_tilde'] = None
        best_parameter_dict_training['neg_log_likelihood_val'] = None
        best_parameter_dict_training['neg_log_likelihood_train'] = None
        
        best_parameter_dict_validation = {} 
        best_parameter_dict_validation['alpha'] = None
        best_parameter_dict_validation['w_tilde'] = None
        best_parameter_dict_validation['neg_log_likelihood_val'] = None
        best_parameter_dict_validation['neg_log_likelihood_train'] = None
        

        num_realizations = len(MTPPdata)

        init_alpha = np.random.random(len(self.alpha))
        init_weight = self.w_tilde*0.

        iterations = 0

        alpha_dash = init_alpha.copy()
        w_dash = init_weight.copy()
        if verbose == 1:
            print("Starting training...")

        lowest_neg_log_likelihood_training = np.inf
        lowest_neg_log_likelihood_validation = np.inf
        
        upper_bound_list_training = []
        upper_bound_list_validation = []


        multiple_restarts_list = [np.random.random(),np.random.random(),np.random.random(),np.random.random() ]
        multiple_restart_index = 0
        while True:
            iterations += 1
            self.feature_vectors= self.feature_vectors_training
            B, D,H, J = self.calculate_terms(
                alpha_dash, num_realizations, MTPPdata, dataset_training_dict)

            updated_alpha_dash = np.zeros(len(alpha_dash))
            updated_w_dash = np.zeros(len(w_dash))
            for index, alpha_val in enumerate(alpha_dash):
                updated_alpha_dash[index] = D[index]/(B[index])
            learning_rate = 0.1
            # if iterations % 5 == 0:
            #     learning_rate *= 10
            # updated_w_dash = w_dash + learning_rate * (F-A)
            self.feature_vectors= self.feature_vectors_training
            neg_log_likelihood = self.log_likelihood(H, J)
            updated_B, updated_D, updated_H,  updated_J = self.calculate_terms(
                updated_alpha_dash, num_realizations, MTPPdata, dataset_training_dict)

            self.feature_vectors= self.feature_vectors_training
            updated_likelihood = self.log_likelihood(
                updated_H, updated_J)

            inner_iterations = 0
            
            while updated_likelihood > neg_log_likelihood:
                inner_iterations += 1
                learning_rate /= 2
                updated_alpha_dash = np.zeros(len(alpha_dash))
                updated_w_dash = np.zeros(len(w_dash))

                for index, alpha_val in enumerate(alpha_dash):
                    updated_alpha_dash[index] = D[index]/( B[index])

                self.feature_vectors= self.feature_vectors_training
                # updated_w_dash = w_dash + learning_rate * (F-A)
                updated_B, updated_D, updated_H, updated_J = self.calculate_terms(
                    updated_alpha_dash, num_realizations, MTPPdata, dataset_training_dict)
                
                self.feature_vectors= self.feature_vectors_training
                updated_likelihood = self.log_likelihood(
                    updated_H, updated_J)

                if inner_iterations > 100:
                    break
            if inner_iterations > 100:
                break

            if multiple_restart_index < len(multiple_restarts_list) and np.linalg.norm(np.abs(alpha_dash - updated_alpha_dash)) < self.stop_criteria['epsilon']:
                if verbose == 1:
                    print("consecutive parameters too close, trying a new initial point")
                alpha_dash = multiple_restarts_list[multiple_restart_index] * np.ones(
                    len(self.sourceNames))
                multiple_restart_index += 1
            else:
                alpha_dash = updated_alpha_dash.copy()
                # w_dash = updated_w_dash.copy()

            # validation_likelihood
            self.feature_vectors= self.feature_vectors_validation
            val_B, val_D, val_H, val_J = self.calculate_terms(updated_alpha_dash, len(
                validation_MTPPdata), validation_MTPPdata, dataset_validation_dict)
            self.feature_vectors= self.feature_vectors_validation
            validation_likelihood = self.log_likelihood(val_H, val_J)
            if lowest_neg_log_likelihood_training > neg_log_likelihood:
                if verbose == 1:
                    print("lower training neg log likelihood found")
                # input()
                # save the alpha, w_tilde for the best likelihood
                # save the kernel_list as well
                best_parameter_dict_training = {}
                best_parameter_dict_training['alpha'] = alpha_dash
                best_parameter_dict_training['w_tilde'] = w_dash
                best_parameter_dict_training['neg_log_likelihood_val'] = validation_likelihood
                best_parameter_dict_training['neg_log_likelihood_train'] = neg_log_likelihood
                
                lowest_neg_log_likelihood_training = neg_log_likelihood
            if lowest_neg_log_likelihood_validation > validation_likelihood:
                if verbose == 1:
                    print("lower validation  neg log likelihood found")
                # input()
                # save the alpha, w_tilde for the best likelihood
                # save the kernel_list as well
                best_parameter_dict_validation = {}
                best_parameter_dict_validation['alpha'] = alpha_dash
                best_parameter_dict_validation['w_tilde'] = w_dash
                best_parameter_dict_validation['neg_log_likelihood_val'] = validation_likelihood
                best_parameter_dict_validation['neg_log_likelihood_train'] = neg_log_likelihood
                
                lowest_neg_log_likelihood_validation = validation_likelihood
            
            if verbose == 1:  
                print("training likeilhood:", neg_log_likelihood)
                print("validation likeilhood:", validation_likelihood)
                print("Iterations: ", iterations)
                print(alpha_dash)
                print(w_dash)
                upper_bound_list_training.append(neg_log_likelihood)
                upper_bound_list_validation.append(validation_likelihood)
                
            if iterations == self.stop_criteria['max_iter']:
                break

        self.alpha = best_parameter_dict_validation['alpha']
        self.w_tilde = best_parameter_dict_validation['w_tilde']
        # self.alpha =alpha_dash
        # self.w_tilde = w_dash
        if verbose ==1:
            print("Best alpha value is: ", self.alpha)
            print("Best w_tilde value is: ", self.w_tilde)
            print("Best validation likelihood value is: ", lowest_neg_log_likelihood_validation)
            print("Best training likelihood value is: ", lowest_neg_log_likelihood_training)
            

        plt.plot(np.log(upper_bound_list_training), label="log of Training nll")
        plt.plot(np.log(upper_bound_list_validation), label="log of Validation nll")
        plt.xlabel("Iterations")
        plt.ylabel("Negative Log Likelihood")
        plt.legend()
        # plt.show()

        return best_parameter_dict_validation

    # Returns a realization, which includes the relative right-censoring time (rightCensoringTime; see below)
    # as its last event time.
    # rightCensoringTime: strictly positive float; represents the relative censoring time to be used.
    # TPPdata: List of S (possibly, empty) realizations.
    # resume:  boolean; if True, assumes that the TPP's own realization includes a relative
    #          right-censoring time and removes it in order to resume the simulation from the last recorded
    #          event time.
    '''
    explicitly calling ogata's simulation solution because inverse transformation
     of the simulation cannot be found in closed form.
    '''

    def simulate(self, rightCensoringTime, TPPdata, resume):
        """[summary]

        Args:
            rightCensoringTime ([type]): [description]
            TPPdata ([type]): [description]
            resume ([type]): [description]

        Returns:
            [type]: [description]
        """
        self.isSusceptible = True
        # calling the simulation for a multivariate ogata's thinning algorithm for a single process
        realizations = simulation([self], rightCensoringTime=rightCensoringTime,
                                  MTPPdata=TPPdata, resume=resume)
        # realizations is a set of realizations, one for each sourceName
        return realizations

    # Returns the TTP's model parameters as a tuple.
    # TODO: change this to split population related variablSes

    def getParams(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.alpha, self.mk

    # Sets a TPP's parameters.
    # params: a tuple of parameters
    # TODO: change this to split population related variables
    def setParams(self, alpha_vals, weight_vector):
        """[summary]

        Args:
            alpha_vals ([type]): [description]
            weight_vector ([type]): [description]

        Returns:
            [type]: [description]
        """
        assert len(alpha_vals) == len(
            self.mk), "The number of parameters in the input is inconsistent with the number of memory kernels associated with the split population process"
        self.alpha = alpha_vals
        self.weight_vector = weight_vector
        return self.alpha, self.weight_vector

    # Initializes a TPP's parameters
    # TODO: change this to split population related variables

    def initParams(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        self.alpha = np.ones(len(self.mk)) * 0.05 

        self.isSusceptible = False
        self.hasRealization = False

        return self.alpha

    def getSourceNames(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.sourceNames

    def setSusceptible(self):
        """[summary]
        """
        self.isSusceptible = True

    def setNotSusceptible(self):
        """[summary]
        """
        self.isSusceptible = False

    def predict_single(self, tc, delta_t, realization, 
    feature_vector, theta, use_gt_prior):
        """[summary]

        Args:
            tc ([type]): [description]
            delta_t ([type]): [description]
            realization ([type]): [description]

        Raises:
            Exception: [description]

        Returns:
            [type]: [description]
        """
        if realization[0][-1] >= tc:
            # cut
            new_realization = [sorted([x for x in source_realization if x <= tc])
                               for source_realization in realization]
        else:
            new_realization = [sorted(x[:-1]) for x in realization]

        if len(new_realization[0]) == 0:
            if len(realization[0]) == 1 :
                gt = 0
            elif len(realization[0]) == 2 :
                if realization[0][0] < tc + delta_t:
                    gt = 1
                else:
                    gt = 0

            Psi_vector_tc = np.zeros(len(self.sourceNames))
            Psi_vector_delta_t = np.zeros(len(self.sourceNames))
            Psi_vector_tc[0] = self.mk[0].psi(tc)
            Psi_vector_delta_t[0] = self.mk[0].psi(tc + delta_t)
            for source_index in range(1, len(self.sourceNames)):
                source_events = new_realization[source_index]
                for se in source_events:
                    Psi_vector_tc[source_index] += self.mk[source_index].psi(
                        tc - se)
                    Psi_vector_delta_t[source_index] += self.mk[source_index].psi(
                        tc + delta_t - se)

            survive_tc = np.exp(-np.dot(self.alpha, Psi_vector_tc))
            survive_delta_t = np.exp(-np.dot(self.alpha, Psi_vector_delta_t))
            survive_prior = np.exp(-np.dot(self.w_tilde, feature_vector))
        
            if use_gt_prior:
                if len(realization[0]) > 1:
                    survive_prior = 0.0
                else:
                    survive_prior = np.inf
            if survive_prior != np.inf:
                prob = (survive_tc - survive_delta_t) / \
                    (survive_tc + survive_prior)
                # prob = (1/(1+ np.exp(-np.dot(self.w_tilde, feature_vector))))*(survive_tc - survive_delta_t)
            else:
                prob = 0.0
            es = 1 if prob > theta else 0
        else:
            #  it contains an exploit
            gt = np.nan
            es = np.nan
            prob = np.nan
            # raise Exception("Such a predictive scenario should never be encountered")

        return gt, es, prob

    def predict(self, tc, delta_t, MTPPData, feature_vectors):
        """[summary]

        Args:
            tc ([type]): [description]
            delta_t ([type]): [description]
            MTPPData ([type]): [description]
            feature_vectors ([type]): [description]

        Returns:
            [type]: [description]
        """
        # create ground truth labels
        gt_E = []
        es_E = []
        Prob = []

        
        theta_list = np.arange(0.0, 1.0, 0.001)
        best_results = None
        far = []
        hr = []
        precision = []
        recall = []
        print(len(theta_list))
        gt_E = []
        es_E = []
        Prob = []
        for idx, realization in enumerate(MTPPData):
            gt, es, prob = self.predict_single(
                tc, delta_t, realization,feature_vectors[idx], 
                0.5, False)
            gt_E.append(gt)
            es_E.append(es)
            Prob.append(prob)
        results = pd.DataFrame(
            data={'true_E': gt_E, 'estimated_E': es_E, 'probability': Prob})
        results = results.dropna()
        prob_list = results.probability
        for index,theta in enumerate(theta_list):
            print(index, end='\r')
        
            es = [1 if prob >= theta else 0 for prob in prob_list]

            cm = confusion_matrix(results.true_E,
                                        es)

            if len(cm) > 1:
                far_tuple = cm[0][1]/(cm[0][1] + cm[0][0])
                mr_tuple = cm[1][0]/(cm[1][0] + cm[1][1])
                precision_tuple = cm[1][1]/(cm[1][1] + cm[0][1])
                recall_tuple = cm[1][1]/(cm[1][1] + cm[1][0])
                precision.append(precision_tuple)
                recall.append(recall_tuple)

        # start_candidate = 0.2
        # recall_candidates, = np.where(np.array(precision) >= start_candidate)
        # while len(recall_candidates) == 0 :
        #     start_candidate -=0.1
        #     recall_candidates, = np.where(np.array(precision) >= start_candidate)
        # best_theta_index, = np.where(np.array(recall)[recall_candidates] == max(np.array(recall)[recall_candidates]))
        # best_theta = np.array(theta_list)[best_theta_index][0]
        best_theta = 0.15 
        es = [1 if prob >= best_theta else 0 for prob in prob_list]
        results = pd.DataFrame(
            data={'true_E': results.true_E, 'estimated_E': es, 'probability': prob_list})
        # results = results.dropna()
        
        cm = confusion_matrix(results.true_E,
                                    es)
        return results, [(x,y) for x,y in zip(recall, precision)]


    def generate_baseline_results(self, realizations, 
                                    feature_vectors, 
                                    dataset_name, 
                                    filename, 
                                    tc_list, 
                                    delta_list,
                                    training_index_set_filename,
                                    training_realizations):
        """[summary]

        Args:
            realizations ([type]): [description]
            feature_vectors ([type]): [description
            """
        assert filename is not None, "filename of file containing index set is missing"
        realization_index_set = np.load(
            filename, allow_pickle=True).item()


        if training_index_set_filename is not None:
            training_index_set = np.load(
                training_index_set_filename, allow_pickle=True).item()

        # prin      realization_index_set)
        # input()
        cnt = 0
        model_predictions = np.array([])
        new_index_dict = {}


        # generate sub samples tuple dictioanry
        for key, value in realization_index_set.items():
            (tc, delta_t) = key
            if tc in tc_list and delta_t in delta_list:
                new_index_dict[key] = value

        # print(len(new_index_dict))
        # input()
        ground_truth = np.array([])

        all_tc = []
        all_delta = []
        for key, value in new_index_dict.items():
            print("Finished {0} tuples out of {1} tuples".format(
                cnt, len(new_index_dict)))
            cnt += 1
            # if cnt == 5:
            #     break
            (tc, delta_t) = key
            all_tc.append(tc)
            all_delta.append(delta_t)

        all_tc = sorted(list(set(all_tc)))
        all_delta = sorted(list(set(all_delta)))

        matrix = np.ones((len(all_tc), len(all_delta)))


        fp_matrix = np.empty((len(all_tc), len(all_delta)))
        fp_matrix[:] = np.nan

        fn_matrix = np.empty((len(all_tc), len(all_delta)))
        fn_matrix[:] = np.nan
        

        cnt = 0
        for key, value in new_index_dict.items():
            print("Finished {0} tuples out of {1} tuples".format(
                cnt, len(new_index_dict)))
            (tc, delta_t) = key
            cnt += 1

            if tc in all_tc and delta_t in all_delta:
                selected_training_realizations = training_realizations[training_index_set[(tc,delta_t)]]
                tc *= 24.0
                delta_t *= 24.0
                realization_indices = [x for x in list(value)]

                selected_realizations = realizations[realization_indices]
                
                i=0
                illegal_indices = []
                for index, x in enumerate(selected_realizations):
                    if len(x[0]) == 2:
                        if x[0][0] <= tc or x[0][0] < 0:
                            illegal_indices.append(index)
                    i +=1
                feature_vectors = [x for i,x in enumerate(feature_vectors) if i not in illegal_indices]
                selected_realizations = [x for i,x in enumerate(selected_realizations) if i not in illegal_indices]
      
                # naive model
                num_exploited = 0
                gt_E = np.zeros(len(selected_realizations))
                es_E = None
                

                for index,realization in enumerate(selected_realizations):
                    if len(realization[0]) > 1 and realization[0][0] < tc + delta_t:
                        gt_E[index] = 1
                
                
                for index,realization in enumerate(selected_training_realizations):
                    if len(realization[0]) > 1:
                        num_exploited +=1
                
                if len(selected_training_realizations) - num_exploited > num_exploited:
                    es_E = np.zeros(len(selected_realizations))
                else:
                    es_E = np.ones(len(selected_realizations))


                results_soc = pd.DataFrame(data={'true_E': gt_E, 'estimated_E': es_E, 'probability': np.ones(len(selected_realizations))})

                cm = confusion_matrix(results_soc.true_E,
                                      results_soc.estimated_E)
                model_predictions = np.append(
                    model_predictions, np.array(results_soc.estimated_E))
                ground_truth = np.append(
                    ground_truth, np.array(results_soc.true_E))
                Accuracy = sum(results_soc.true_E ==
                               results_soc.estimated_E) / len(results_soc)
                matrix[all_tc.index(
                    tc/24.0)][all_delta.index(delta_t/24.0)] = Accuracy

                # print(np.sum(results_soc.estimated_E))
                # print(len(results_soc.estimated_E)-np.sum(results_soc.estimated_E))
                # input()
                if len(cm) > 1:
                    print(cm)
                    print(Accuracy)

                    if tc/24.0 == 21.0 and delta_t/24.0 == 31.0:
                        print(tc/24)
                        print(delta_t/24)
                        print(np.sum(gt_E))
                        # print(realization_indices)
                        input()

                    # false positive rate
                    if cm[0][0] + cm[0][1] > 0: 
                        fp_matrix[all_tc.index(tc/24.0)][all_delta.index(delta_t/24.0)] = cm[0][1]/(cm[0][1] + cm[0][0])
                    
                    # miss rate
                    if cm[1][0] + cm[1][1] > 0:
                        fn_matrix[all_tc.index(
                            tc/24.0)][all_delta.index(delta_t/24.0)] = cm[1][0]/(cm[1][0] + cm[1][1])

            # print(Accuracy)



        import itertools
        plt.imshow(matrix, interpolation='none', cmap='jet',
                   aspect='auto', origin='lower')
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, "{:,}".format(round(matrix[i, j], 2)),
                    horizontalalignment="center",
                    color="black") 
        plt.xticks(np.arange(0, len(delta_list)),
                   np.array(delta_list), rotation=45)
        plt.yticks(np.arange(0, len(tc_list)), np.array(tc_list))
        plt.xlabel("Prediction window, $\delta_{t}$ (days)")
        plt.ylabel("Prediction start time, $t_{c}$ (days)")
        plt.title("Accuracy on test dataset: naive model")
        plt.colorbar()
        plt.savefig("../data/prediction_results/Prediction_accuracy_naive_"+dataset_name+".png", dpi=600)
        with open("../data/prediction_results/Prediction_accuracy_naive_"+dataset_name+".npy", "wb") as f:
            np.save(f,matrix)

        plt.show()

        plt.imshow(fn_matrix, interpolation='none',
                   cmap='jet', aspect='auto', origin='lower')
        for i, j in itertools.product(range(fn_matrix.shape[0]), range(fn_matrix.shape[1])):
            plt.text(j, i, "{:,}".format(round(fn_matrix[i, j], 2)),
                    horizontalalignment="center",
                    color="black") 
        
        plt.xticks(np.arange(0, len(delta_list)),
                   np.array(delta_list), rotation=45)
        plt.yticks(np.arange(0, len(tc_list)), np.array(tc_list))
        plt.xlabel("Prediction window, $\delta_{t}$ (days)")
        plt.ylabel("Prediction start time, $t_{c}$ (days)")
        plt.title("miss rate on validation dataset: naive model")
        plt.clim(0.0,1.0)
        plt.colorbar()
        plt.savefig("../data/prediction_results/Prediction_fn_rate_naive_"+dataset_name+".png", dpi=600)
        with open("../data/prediction_results/Prediction_fn_rate_naive_"+dataset_name+".npy", "wb") as f:
            np.save(f,fn_matrix)
        plt.show()

        plt.imshow(fp_matrix, interpolation='none',
                   cmap='jet', aspect='auto', origin='lower')
        for i, j in itertools.product(range(fp_matrix.shape[0]), range(fp_matrix.shape[1])):
            plt.text(j, i, "{:,}".format(round(fp_matrix[i, j], 2)),
                    horizontalalignment="center",
                    color="black") 
        plt.xticks(np.arange(0, len(delta_list)),
                   np.array(delta_list), rotation=45)
        plt.yticks(np.arange(0, len(tc_list)), np.array(tc_list))
        plt.xlabel("Prediction window, $\delta_{t}$ (days)")
        plt.ylabel("Prediction start time, $t_{c}$ (days)")
        plt.title("False alarm rate on validation dataset: naive model")
        plt.clim(0.0,1.0)
        plt.colorbar()
        plt.savefig("../data/prediction_results/Prediction_fp_rate_naive_"+dataset_name+".png", dpi=600)
        with open("../data/prediction_results/Prediction_fp_rate_naive_"+dataset_name+".npy", "wb") as f:
            np.save(f,fp_matrix)
        plt.show()

   
    def predict_test_data(self, 
                            realizations, 
                            feature_vectors, 
                            filename, 
                            scenario_name, 
                            tc_list, 
                            delta_list,
                            consider_social_media_realizations_only):
        """[summary]

        Args:
            realizations ([type]): [description]
            feature_vectors ([type]): [description]
        """

        assert filename is not None, "Filename of file containing index test set is missing"
        realization_index_set = np.load(
            filename, allow_pickle=True).item()
            

        cnt = 0
        model_predictions = np.array([])
        new_index_dict = {}
        roc_curve_points = []

        # generate sub samples tuple dictioanry
        for key, value in realization_index_set.items():
            (tc, delta_t) = key
            if tc in tc_list and delta_t in delta_list:
                new_index_dict[key] = value

        ground_truth = np.array([])

        all_tc = []
        all_delta = []
        for key, value in new_index_dict.items():
            print("Finished {0} tuples out of {1} tuples".format(
                cnt, len(new_index_dict)))
            cnt += 1
            # if cnt == 5:
            #     break
            (tc, delta_t) = key
            all_tc.append(tc)
            all_delta.append(delta_t)

        all_tc = sorted(list(set(all_tc)))
        all_delta = sorted(list(set(all_delta)))

        matrix = np.ones((len(all_tc), len(all_delta)))


        fp_matrix = np.empty((len(all_tc), len(all_delta)))
        fp_matrix[:] = np.nan

        fn_matrix = np.empty((len(all_tc), len(all_delta)))
        fn_matrix[:] = np.nan
        

        cnt = 0
        for key, value in new_index_dict.items():
            print("Finished {0} tuples out of {1} tuples".format(
                cnt, len(new_index_dict)))
            (tc, delta_t) = key
            cnt += 1

            if tc in all_tc and delta_t in all_delta:
                tc *= 24.0
                delta_t *= 24.0
                # print(realizations[0])
                # print(feature_vectors[0])
                # for index,feature in enumerate(feature_vectors):
                #     if feature[0].round(4) == 0.9834 and feature[1].round(4) == 0.0166:
                #         print(realizations[index])
                
                realization_indices = [x for x in list(value)]

                selected_realizations = realizations[realization_indices]
                selected_feature_vectors = feature_vectors[realization_indices]
                
                if consider_social_media_realizations_only:
                    social_media_indices = []
                    for index, realization in enumerate(selected_realizations):
                        if len(realization[1]) > 1 or len(realization[2]) > 1 or len(realization[3]) > 1:
                            social_media_indices.append(index)

                    selected_realizations = selected_realizations[social_media_indices]
                    selected_feature_vectors = selected_feature_vectors[social_media_indices]

                # selected_realizations = []
                # selected_feature_vectors = []
                # for index, realization in enumerate(realizations):
                #     if index in realization_indices:
                #         selected_realizations.append(realization)
                #         selected_feature_vectors.append(feature_vectors[index])
                
                results_soc, roc_list = self.predict(tc, delta_t, selected_realizations, selected_feature_vectors)
                roc_curve_points.append(roc_list)
                results_soc = results_soc.dropna()
  
                cm = confusion_matrix(results_soc.true_E,
                                      results_soc.estimated_E)
                model_predictions = np.append(
                    model_predictions, np.array(results_soc.estimated_E))
                ground_truth = np.append(
                    ground_truth, np.array(results_soc.true_E))
                Accuracy = sum(results_soc.true_E ==
                               results_soc.estimated_E) / len(results_soc)
                matrix[all_tc.index(
                    tc/24.0)][all_delta.index(delta_t/24.0)] = Accuracy

                # print(np.sum(results_soc.estimated_E))
                # print(len(results_soc.estimated_E)-np.sum(results_soc.estimated_E))
                # input()
                if len(cm) > 1:
                    print(cm)
                    print(Accuracy)


                    # atleast one non-exploit
                    # recall
                    if cm[0][0] + cm[0][1] > 0: 
                        fp_matrix[all_tc.index(tc/24.0)][all_delta.index(delta_t/24.0)] = cm[1][1]/(cm[1][1] + cm[1][0])
                    
                    # precision
                    if cm[1][0] + cm[1][1] > 0:
                        fn_matrix[all_tc.index(
                            tc/24.0)][all_delta.index(delta_t/24.0)] = cm[1][1]/(cm[1][1] + cm[0][1])

        if not os.path.exists("../data/prediction_results"+scenario_name):
            os.makedirs("../data/prediction_results"+scenario_name)
    


        plt.imshow(matrix, interpolation='none', cmap='jet',
                   aspect='auto', origin='lower')
        plt.xticks(np.arange(0, len(delta_list)),
                   np.array(delta_list), rotation=45)
        plt.yticks(np.arange(0, len(tc_list)), np.array(tc_list))
        plt.xlabel("Prediction window, $\delta_{t}$ (days)")
        plt.ylabel("Prediction start time, $t_{c}$ (days)")
        plt.title("Accuracy on "+ scenario_name + "dataset")
        plt.colorbar()
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, "{:,}".format(round(matrix[i, j], 2)),
                    horizontalalignment="center",
                    color="black") 
        plt.savefig("../data/prediction_results"+scenario_name+"/Prediction_accuracy_"+scenario_name+".png", dpi=600)
        with open("../data/prediction_results"+scenario_name+"/Prediction_accuracy_"+scenario_name+".npy", "wb") as f:
            np.save(f,matrix)
        plt.figure()

        plt.imshow(fn_matrix, interpolation='none',
                   cmap='jet', aspect='auto', origin='lower')
        plt.xticks(np.arange(0, len(delta_list)),
                   np.array(delta_list), rotation=45)
        plt.yticks(np.arange(0, len(tc_list)), np.array(tc_list))
        plt.xlabel("Prediction window, $\delta_{t}$ (days)")
        plt.ylabel("Prediction start time, $t_{c}$ (days)")
        plt.title("Precision on " + scenario_name + " dataset")
        plt.clim(0.0,1.0)
        plt.colorbar()
        for i, j in itertools.product(range(fn_matrix.shape[0]), range(fn_matrix.shape[1])):
            plt.text(j, i, "{:,}".format(round(fn_matrix[i, j], 2)),
                    horizontalalignment="center",
                    color="black")
        plt.savefig("../data/prediction_results"+scenario_name+"/Prediction_fn_rate_"+scenario_name+".png", dpi=600)
        
        with open("../data/prediction_results"+scenario_name+"/Prediction_fn_rate_"+scenario_name+".npy", "wb") as f:
            np.save(f,fn_matrix)
        plt.figure()

        plt.imshow(fp_matrix, interpolation='none',
                   cmap='jet', aspect='auto', origin='lower')
        plt.xticks(np.arange(0, len(delta_list)),
                   np.array(delta_list), rotation=45)
        plt.yticks(np.arange(0, len(tc_list)), np.array(tc_list))
        plt.xlabel("Prediction window, $\delta_{t}$ (days)")
        plt.ylabel("Prediction start time, $t_{c}$ (days)")
        plt.title("Recall on " + scenario_name + " dataset")
        plt.clim(0.0,1.0)
        plt.colorbar()
        for i, j in itertools.product(range(fp_matrix.shape[0]), range(fp_matrix.shape[1])):
            plt.text(j, i, "{:,}".format(round(fp_matrix[i, j], 2)),
                    horizontalalignment="center",
                    color="black") 
        
        plt.savefig("../data/prediction_results"+scenario_name+"/Prediction_fp_rate_"+scenario_name+".png", dpi=600)
        with open("../data/prediction_results"+scenario_name+"/Prediction_fn_rate_"+scenario_name+".npy", "wb") as f:
            np.save(f,fp_matrix)
        plt.figure()

        return roc_curve_points

    # precalculate values for a single realization
    # returns a mxn matrix where m is the number of platforms (including base) for each realization
    # stored in a dictionary indicated by the index of the appropriate
    def precalculate_vals(self, training_MTPPdata, pre_cal_filename, validation_MTPPdata, verbose=1):
        """[summary]

        Args:
            training_MTPPdata ([type]): [description]
            pre_cal_filename ([type]): [description]
            validation_MTPPdata ([type]): [description]
        """

        def prec_dataset(MTPPdata, pre_cal_filename):
            """[summary]

            Args:
                MTPPdata ([type]): [description]
                pre_cal_filename ([type]): [description]

            Returns:
                [type]: [description]
            """
            dataset_dict = {}

            for realization_index in range(len(MTPPdata)):
                if verbose == 1:
                    print("Precal index is at: ", realization_index)
                realization = MTPPdata[realization_index]
                pre_calc_dict = {}

                # generate required matrices for pre-calc
                # first generate all the time instances
                # print(time_instances)
                time_instances = sorted(
                    list(set([l for source_realization in realization for l in source_realization])))

                t_0 = 0.0

                # we ignore events from itself because, intensity, integrated intensity and upperbound intensity
                # does not depend on realizations of itself.
                Phi_matrix = np.zeros(
                    (len(self.sourceNames), len(time_instances)))
                Psi_matrix = np.zeros(
                    (len(self.sourceNames), len(time_instances)))
                t_spl = None
                if len(realization[0]) > 1:
                    t_spl = realization[0][0]

                for index, t_i in enumerate(time_instances):
                    if t_i == time_instances[-1] or ((not t_spl == None) and t_i == t_spl):
                        Phi_matrix[0, index] = self.mk[0].phi(t_i-t_0)
                        Psi_matrix[0, index] = self.mk[0].psi(t_i-t_0)

                if len(self.mk) > 1:
                    for source_index in range(len(realization[1:])):
                        source_realization = realization[source_index + 1]
                        for i in range(len(time_instances)):
                            t_i = time_instances[i]
                            if t_i == time_instances[-1] or ((not t_spl == None) and t_i == t_spl):
                                for j in range(i):
                                    if time_instances[j] in source_realization:
                                        Phi_matrix[source_index + 1][i] += self.mk[source_index + 1].phi(
                                            t_i - time_instances[j])
                                        Psi_matrix[source_index + 1][i] += self.mk[source_index + 1].psi(
                                            t_i - time_instances[j])
                pre_calc_dict['Phi_matrix'] = Phi_matrix
                pre_calc_dict['Psi_matrix'] = Psi_matrix
                pre_calc_dict['time_instances'] = time_instances
                dataset_dict[str(realization_index)] = pre_calc_dict

            pre_cal_filename = "../data/exploit_paper_scenarios/" + pre_cal_filename + ".pickle"
            with open(pre_cal_filename, 'wb') as handle:
                pickle.dump(dataset_dict, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
            if verbose == 1:
                print("Pre calculation has completed and the file is located at {0}".format(
                    pre_cal_filename))
            # return true to indicate the training has been completed.
        prec_dataset(training_MTPPdata, pre_cal_filename=pre_cal_filename)
        prec_dataset(validation_MTPPdata,
                     pre_cal_filename="__validation__"+pre_cal_filename)
        return True


################################################################################
#
# U N I T   T E S T I N G
#
################################################################################


# Some global settings for figure sizes

def main():
    """[summary]
    """

    # First kernel is base rate, the rest is for the external source.
    mk = ExponentialPseudoMemoryKernel(beta=5.0)
    mkList = [ConstantMemoryKernel()]

    # this TPP only uses its own events as source.
    sourceNames = ['base']
    # sourceNames = ['base']

    stop_criteria = {'max_iter': 600,
                     'epsilon': 1e-6}
    exploitProcess = SplitPopulationTPP(
        mkList, sourceNames, stop_criteria,
        desc='Split population process with multiple kernels')
    # this TPP only uses its own events as source.
    sourceNamesSocialMedia = ['socialMediaProcess']
    # class
    socialMediaProcess = HawkesTPP(
        mk, sourceNamesSocialMedia, stop_criteria,
        desc='Hawkes TPP with exponetial(beta=1) kernel')
    # socialMediaProcess = PoissonTPP(
    #     mk=ConstantMemoryKernel(), sourceNames=sourceNamesSocialMedia, desc='Exploit Process: Poisson TPP with a exponential pseudo kernel')
    numSamples = 200
    sampleDimensionality = 2
    classPrior1 = 0.99
    misclassificationProbability = 10e-20
    X, y, w_tilde = gen2IsotropicGaussiansSamples(
        numSamples, sampleDimensionality, classPrior1, misclassificationProbability)

    exploitProcess.setFeatureVectors(X)
    exploitProcess.w_tilde = w_tilde
    # from sklearn.datasets import load_iris
    # from sklearn.linear_model import LogisticRegression
    # clf = LogisticRegression(random_state=0).fit(X, y)
    # clf.predict(X[:2, :])
    # exploitProcess.w_tilde = w_tilde
    # generating samples for the split population process
    # example: generating points for the exploit process with three social media processes
    # first create a multivariate process with 2 processes, once exploit and another

    # hardcoding this for the unit test, we know the HawkessTPP is for a single process
    numProcesses = len(sourceNames)
    numRealizations = len(X)
    Realizations = []
    maxNumEvents = 0
    maxNumEventsIdx = 0
    total_susceptible = 0

    for r in range(0, numRealizations):
        # Exponential(100)-distributed right-censoring time
        # T = scipy.stats.expon.rvs(loc=0.0, scale=20.0)
        T = 20
        processList = [exploitProcess, socialMediaProcess]
        Realization, isSusceptible = simulation_split_population(
            processList, T, [np.array([])] * numProcesses, w_tilde, X[r], resume=False,
            resume_after_split_pop=False)
        # print(Realization)
        # input()
        print(
            "Right Censoring Time for the current realization {0}, realization: {1}".format(T, r))

        if (isSusceptible == 1):
            total_susceptible += 1
        Realizations.append(Realization)
        # number of realizations of social media
        numEvents = len(Realization[0])
        if numEvents > maxNumEvents:
            maxNumEvents = numEvents
            maxNumEventsIdx = r

    print("Total susceptible among simulated population: ", total_susceptible)
    # TODO: replace this with GOFks for multivariate process including both processes.
    # IIDsamples = hawkesTPP.transformEventTimes(Realizations[maxNumEventsIdx])
    _, ax = plt.subplots(1, 1, figsize=largeFigSize)

    IIDSamples = []

    # ax.set_title(
    #     'P-P Plot: {:1} \n p-value: {:0.3f}'.format(exploitProcess.desc, pvalue))
    # save simulated values back into the DataStream object
    # TODO: pass in dataStream object instead of TPPdata into training functin
    # exploitProcess.setupTraining(Realizations, X, "simulated_data")
    # _, plot_list = exploitProcess.train(Realizations, "simulated_data")

    for realization in Realizations:
        # print(realization[0])
        # input()
        if len(realization[0]) > 1:
            exploitProcess.setSusceptible()
            IIDSamples.extend(exploitProcess.transformEventTimes(realization))
    IIDSamples = list(filter((0.0).__ne__, IIDSamples))
    # print(IIDSamples)
    pvalue = KSgoodnessOfFitExp1(
        sorted(IIDSamples), ax, showConfidenceBands=True)
    allOnes = np.ones((numSamples, 1))
    X_tilde = np.concatenate((X, allOnes), axis=1)

    y_pred = np.sign(np.dot(X_tilde, exploitProcess.w_tilde))
    errorRate = np.sum(y_pred != y) / numSamples
    print('gen2IsotropicGaussiansSamples(): errorRate={:1} should approximately match misclassificationProbability={:2} for {:3} samples'.format(
        errorRate, misclassificationProbability, numSamples))

    plt.show()
    _, ax = plt.subplots(1, 1, figsize=normalFigSize)

    plt.show()
    plt.plot(plot_list)
    plt.show()


if __name__ == '__main__':
    main()
