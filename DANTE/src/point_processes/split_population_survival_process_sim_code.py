#!/usr/bin/python
'''Temporal Point Process (TPP) sub-classes'''
import json
import pandas as pd
import numpy as np
# import scipy.stats
import random
# import logging
import os
import h5py
# from matplotlib import pyplot as plt
from point_process_abstract import TemporalPointProcess as TPP
import memory_kernel
import GoodnessOfFit
# from utils import sigmoid


def sigmoid(x):
        
    return 1 / (1 + np.exp(-x))




class SplitPopulationSurvival(TPP):
    '''Split Population Survival Process'''

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
    # sourceNames: For survival process, sourceName does not include ownName
    def __init__(self, ownName, sourceNames, mk, stop_criteria={'max_iter':200,\
                 'epsilon': 1e-4}, desc='Split Population Survival Process', logger=None):

        self.dim = len(sourceNames)
        self.desc = desc
        self.sourceNames = sourceNames
        self.ownName = ownName
        self.stop_criteria = stop_criteria
        self.mk = mk
        self.pre_cal = False
        self.pre_cal_path = 'temp_storage'
        self.pre_cal_file = ''
        self.initParams(dict.fromkeys(['base'] + sourceNames, 1.0))

        super().__init__(sourceNames, desc, logger)

    # t: float or 1-dimensional numpy.ndarray of reals; relative event time(s). Typically, larger than 0.
    # realization: a single realization of model including all sources events
    
    def susceptible_intensity(self, t, realization):
        
        if (self.ownName not in realization) or (t<realization[self.ownName][0]):               
            # calculate history excitation
            phi_vector = pd.Series(data = 0.0, index = self.para.keys(), dtype = 'float64')
    
            phi_vector['base'] = self.mk['base'].phi(t)
            
            activated_sources = [s for s in list(realization.keys())
                                 if s in self.sourceNames]
    
            for s in activated_sources:
                #  gather all history relevant to the given point process
                s_hist = [h for h in realization[s] if h < t]
    
                # iterate over each event in this specific history
                for e in s_hist:
                    phi_vector[s] += self.mk[s].phi(t-e)
                    
            # intensity value = base intensity + hist_excitaion
            value = self.para.dot(phi_vector)
            
        else:  
            value = 0.0

        return value
    
    def intensity(self, t, realization):
        assert 'susceptible' in realization
        
        if realization['susceptible'] == np.int(1):
            value = self.susceptible_intensity(t, realization)
        else:
            value = 0.0
            
        return value
            
            

    # t: float or 1-dimensional numpy.ndarray of reals; relative event time(s). Typically, larger than 0.
    # realization: a single realization of model including all sources events
    
    def susceptible_cumIntensity(self, t, realization):
        
        if (self.ownName not in realization) or (t<realization[self.ownName][0]):

            # calculate history excitation
            psi_vector = pd.Series(data = 0.0, index = self.para.keys(), dtype = 'float64')
    
            psi_vector['base'] = self.mk['base'].psi(t)
            
            activated_sources = [s for s in list(realization.keys())
                                 if s in self.sourceNames]
            
            for s in activated_sources:
                #  gather all history relevant to the given point process
                s_hist = [h for h in realization[s] if h < t]
    
                # iterate over each event in this specific history
                for e in s_hist:
                    psi_vector[s] += self.mk[s].psi(t-e)
                    
            # intensity value = base intensity + hist_excitaion
            value = self.para.dot(psi_vector)    
        else:
            value = np.nan

        return value
    
    def susceptible_survival(self, t, realization):
         
        value = np.exp(-self.susceptible_cumIntensity( t, realization) )
        
        return value
    
    def susceptible_survival_func(self, t, realization):
        
        obs_realization = realization.copy()
        if self.ownName in realization:
            obs_realization.pop(self.ownName)
            
        value = np.exp(-self.susceptible_cumIntensity( t, obs_realization) )
        
        return value
    
    def cumIntensity(self, t, realization):
        assert 'susceptible' in realization
        
        if realization['susceptible'] == np.int(1):
            value = self.susceptible_cumIntensity(t, realization)
        else:
            value = 0.0
            
        return value
    
    
    def CDF(self, t, realization, weight_vector):
        
        assert 'feature_vector' in realization
    
        survival = np.exp(-self.susceptible_cumIntensity(t, realization))
    
        prior = sigmoid(np.exp(-np.dot(weight_vector, realization['feature_vector'])))

        value = prior*(1-survival)
        
        return value


    
    def loglikelihood(self, realization, weight_vector):
        
        if self.pre_cal:
            pre_cal_f = pd.HDFStore(self.pre_cal_file) 
            structure_df = pre_cal_f[realization['cid']]
            
        else:
            structure_df = self.structure_cal(realization)
        
        prior = sigmoid(np.dot(realization['feature_vector'], weight_vector))
        
            
        if self.ownName in realization:
            activatedTerm = np.log(self.para.dot(structure_df['event']))
            survivalTerm = - self.para.dot(structure_df['survival'])
            prior_term = np.log(prior)
            
            loglikelihood_value = activatedTerm + survivalTerm + prior_term
            
        else:
            
            survivalTerm = prior * np.exp(-self.para.dot(structure_df['survival_T']))
            nonSusceptibleTerm = 1- prior
            
            loglikelihood_value = np.log(survivalTerm + nonSusceptibleTerm)
            
            
        return loglikelihood_value
    
    def loglikelihood_TPPdata(self, TPPdata, weight_vector):
        loglikelihood_value = 0
        for realization in TPPdata:
            loglikelihood_value += self.loglikelihood(realization, weight_vector)
            
        return loglikelihood_value


    
    def transformEventTimes(self, realization, weight_vector):
        assert 'feature_vector' in realization
        
        rescaled_values = []
  
        if self.ownName in realization:
            own_events = realization[self.ownName][0]
            
            prior = sigmoid(np.dot(realization['feature_vector'], weight_vector))
            cum = self.susceptible_cumIntensity(own_events - 1e-7, realization)
            
            rescaled_values.append(cum - np.log(prior))
            # rescaled_values.append(cum)
            # 
        return rescaled_values
    
    def transformEventTimes_TPPdata(self, TPPdata, weight_vector):
        pp_times = []
        for realization in TPPdata:
            pp_times.extend(self.transformEventTimes(realization, weight_vector))
            
        return pp_times
    
    
    def base_thinning(self, realization):
        '''
        To use this simulation: the process cannot contain any other sources 
                                beside base kernel
        '''
        lambda_bar = self.intensity(self.mk['base'].mode(), realization)
        s = 0.0
        # # point generating
        while (s < realization['right_censored_time']) and (self.ownName not in realization):
            u = random.uniform(0, 1)
            w = -np.log(u) / lambda_bar
            s = s + w
            
            if s < realization['right_censored_time']:
                D = np.random.uniform() # uniformly generated from [0,1)
                if D <= (self.intensity(s, realization)/lambda_bar):
                    realization[self.ownName] = [s] 
 
        return realization
    
    
    def base_simulation(self, N, stop_times, cids, add_susceptible = False, \
                   susceptible_labels = None, feature_vectors = None, \
                   method='thinning'):
        
        if np.isscalar(stop_times):
            stop_times = [stop_times] * N
            
        if method == 'thinning' :
            sim_func = self.base_thinning
            
        TPPdata = []
        
        for k in range(N):
            realization = {}
            realization['cid'] = cids[k]
            realization['right_censored_time'] = stop_times[k]
            
            if add_susceptible:
                realization['susceptible'] = np.int(susceptible_labels[k])
                realization['feature_vector'] = list(feature_vectors[k])
            
            if realization['susceptible'] == np.int(1):
                realization = sim_func(realization)
          
            TPPdata.append(realization)
                
        return TPPdata
        
    def structure_cal(self, realization):
        

        activated_sources = [s for s in list(realization.keys())
                             if s in self.sourceNames]

        if self.ownName in realization:
            own_event = realization[self.ownName][0]
            
            structure_df = pd.DataFrame(data = np.zeros((self.dim+1,2)), \
                                        index = self.para.keys(), \
                                        columns = ['event', 'survival'])

            structure_df.loc['base', 'event'] = self.mk['base'].phi(own_event)
            structure_df.loc['base', 'survival'] = self.mk['base'].psi(own_event)
            
            for s in activated_sources:
                cal_source_events = [c for c in realization[s] if c < own_event]

                for event in cal_source_events:
                   structure_df.loc[s, 'event'] += self.mk[s].phi(own_event - event)
                   structure_df.loc[s, 'survival'] += self.mk[s].psi(own_event - event)
         
        else: # if no own observed event, return cummulative intensity to right-censoring time
            if 'right_censored_time' in realization:
                T = realization['right_censored_time']
            else:
                T = max([max(realization[s]) for s in activated_sources])
                
            structure_df = pd.DataFrame(data = np.zeros(self.dim+1), \
                                        index = self.para.keys(), \
                                        columns = ['survival_T'])
                
            structure_df.loc['base', 'survival_T'] = self.mk['base'].psi(T)
            
            for s in activated_sources:
                for event in realization[s]:
                   structure_df.loc[s, 'survival_T'] += self.mk[s].psi(T - event)

        return structure_df

    
    def learning_EM_update(self, TPPdata, weight_vector, \
                           em_stop_criteria = None):
        
        print('para training')
        
        '''
        learn split population process parameters
        '''
 
        if em_stop_criteria is None:
            em_stop_criteria  = self.stop_criteria
        
        if self.pre_cal:
            pre_cal_f = pd.HDFStore(self.pre_cal_file) 
        # query from class
        epsilon = em_stop_criteria['epsilon']
        max_iter = em_stop_criteria['max_iter']

        epoch_iter = 0
        obj_diff = epsilon + 1
        # save objective value in each iteration (optional)
        epoch_obj_value = []
        
        # weight vector initialization
        # weight_vector = np.array([2.07, 0, 0])
        #np.ones(len(TPPdata[0]['feature_vector']))
        
        # keep updating until having met stopping criteria
        while obj_diff > epsilon and (epoch_iter < max_iter):
            
            para_next_numerator = pd.Series(data=np.zeros(self.dim+1), \
                                            index = self.para.keys())
            para_next_denominator = pd.Series(data=np.zeros(self.dim+1), \
                                              index = self.para.keys())

            obj_value = 0

            #-----------------------loop through cascades----------------------
            for realization in TPPdata:
                            
                if self.pre_cal:
                    structure_df = pre_cal_f[realization['cid']]
                else:
                    structure_df = self.structure_cal(realization)
                
                if self.ownName in realization: # delta = 1
                    
                    ini_df =  structure_df['event'].mul(self.para)
                    para_next_numerator += ini_df / ini_df.sum()
                    
                    para_next_denominator += structure_df['survival'] 
                    
                    
                else:
                    pnd = 1 - sigmoid(self.para.dot(structure_df['survival_T']) -\
                          np.array(realization['feature_vector']).dot(weight_vector))
                        
                    para_next_denominator += pnd * structure_df['survival_T']
                    
                
                rnl = -self.loglikelihood(realization, weight_vector)
   
                # obj_value = obj_value + self.loglikelihood(realization)
                if 'log-likelihood-weight' in realization.keys():
                    obj_value = obj_value + \
                        np.power(rnl, realization['log-likelihood-weight'])
                else:
                    obj_value = obj_value + rnl
                    
            #------------------------------------------------------------------
            # updates parameters
            self.para = para_next_numerator / para_next_denominator
            
            # objective value in current iteration
            epoch_obj_value.append(obj_value)

            # add iteration count
            epoch_iter = epoch_iter + 1

            obj_diff = abs(epoch_obj_value[-1] - epoch_obj_value[-2]) \
                if (epoch_iter > 1) else obj_diff
                    
            # print('epoch iteration number: {}'.format(epoch_iter))
            # print('para = {}'.format(self.para))
                    
                        
        return obj_value#epoch_obj_value
    
    def learning_gradient_descent(self, TPPdata, init_weight_vector,\
                                  max_learning_rate = 0.8, \
                                  gd_stop_criteria = None):
        '''
        learn weight vector
        '''
        
        print('weight training')
        if gd_stop_criteria is None:
            gd_stop_criteria  = self.stop_criteria
            
        if self.pre_cal:
            pre_cal_f = pd.HDFStore(self.pre_cal_file)
            
        # query from class
        epsilon = gd_stop_criteria['epsilon']
        max_iter = gd_stop_criteria['max_iter']
        learning_rate = max_learning_rate

        feature_dim = len(init_weight_vector)
        next_weight_vector = np.array(init_weight_vector)
        nnl= -self.loglikelihood_TPPdata(TPPdata, next_weight_vector)
        
        epoch_iter = 0
        obj_diff = epsilon + 1
        # save objective value in each iteration (optional)
        epoch_obj_value = [nnl]
        
        ls_iter_list = []
        
        
        while obj_diff > epsilon and (epoch_iter < max_iter):
            
            process_term  = np.zeros(feature_dim)
            susceptible_term = np.zeros(feature_dim)
                                     
            current_weight_vector = next_weight_vector
            cnl = nnl

            #-----------------------loop through cascades----------------------
            for realization in TPPdata:
                
                feature_vector = np.array(realization['feature_vector'])
                
                #------------------A-------------------------------------------
                if self.ownName not in realization:
                    if self.pre_cal:
                        structure_df = pre_cal_f[realization['cid']]

                    else:
                        structure_df = self.structure_cal(realization)
                                  
                    p_para = sigmoid(self.para.dot(structure_df['survival_T']) - \
                             feature_vector.dot(current_weight_vector))
                         
                    process_term += p_para * feature_vector
                    
                #--------------------------------------------------------------
                
                s_para = 1 - sigmoid(feature_vector.dot(current_weight_vector))
                
                susceptible_term += s_para * feature_vector 
            
                    
            #--------------------finish loop-----------------------------------
            # updates parameters with backtracking line search
            next_weight_vector = current_weight_vector + \
                                 learning_rate *(susceptible_term - process_term)
                                 
            nnl= -self.loglikelihood_TPPdata(TPPdata, next_weight_vector)
                                 
            ls_iter = 0 # line search iteration
            while nnl > cnl:
                ls_iter += 1
                learning_rate = learning_rate/2
                # updates 
                next_weight_vector = current_weight_vector + \
                                     learning_rate *(susceptible_term - process_term)
    
                # next negative log likelihood value: nnlw_tilde
                nnl= -self.loglikelihood_TPPdata(TPPdata, next_weight_vector)
                
            ls_iter_list.append(ls_iter)
            
            # objective value in current iterationepoch_obj_value
            epoch_obj_value.append(nnl)

            # add iteration count
            epoch_iter = epoch_iter + 1

            obj_diff = abs(epoch_obj_value[-1] - epoch_obj_value[-2]) \
                if (epoch_iter > 1) else obj_diff
                    
            # print('epoch iteration number: {}'.format(epoch_iter))
            # print('weight_vector = {}'.format(next_weight_vector))
            
            # print('line search iteration number: {}'.format(ls_iter))
            
            # print('learning_rate: {}'.format(learning_rate))
                        
        return next_weight_vector  # , epoch_obj_value
    
    def learning_joint(self, TPPdata, init_weight_vector, \
                       gd_stop_criteria=None, em_stop_criteria=None):
        
        if gd_stop_criteria is None:
            gd_stop_criteria = {'max_iter': 1, 'epsilon': 1e-4}
            
        if em_stop_criteria is None:
            em_stop_criteria = {'max_iter': 1, 'epsilon': 1e-4}
            
        current_weight_vector = init_weight_vector
        
        # query from class
        epsilon = self.stop_criteria['epsilon']
        max_iter = self.stop_criteria['max_iter']

        epoch_iter = 0
        obj_diff = epsilon + 1
        # save objective value in each iteration (optional)
        epoch_obj_value = []
        
        # keep updating until having met stopping criteria
        while obj_diff > epsilon and (epoch_iter < max_iter):
      
            # update weight vector
            next_weight_vector = self.learning_gradient_descent(TPPdata, \
                            current_weight_vector, max_learning_rate = 0.8, \
                            gd_stop_criteria = gd_stop_criteria)

                
            obj_value = self.learning_EM_update(TPPdata, next_weight_vector, \
                                                em_stop_criteria)
            
            current_weight_vector = next_weight_vector
            # objective value in current iteration
            epoch_obj_value.append(obj_value)

            # add iteration count
            epoch_iter = epoch_iter + 1

            obj_diff = abs(epoch_obj_value[-1] - epoch_obj_value[-2]) \
                       if (epoch_iter > 1) else obj_diff
                    
            print('epoch iteration number: {}'.format(epoch_iter))
            print('para = {}'.format(self.para))
            print('weight_vector = {}'.format(current_weight_vector))
            
        return current_weight_vector, epoch_obj_value      
           
    # Setup TPP model for training.
    # Typically used for pre-computing quantities that are necessary for training.
    # This method is expected to be called prior to a call to the train() method.
    # TPPdata: List of S lists, where each inner list contains R realizations.
    #          Each realization contains at least a relative right-censoring time as its last event time.

    def setupTraining(self, TPPdata):
        pass


    # Train the TPP model.
    # Returns an element of the TrainingStatus enumeration.
    def train(self, TPPdata, init_weight_vector, pre_cal_file_name='', pre_cal=False):
        
        if pre_cal:
            self.setupPreTraining(TPPdata, pre_cal_file_name)
            
        weight_vector, obj_values = self.learning_joint(TPPdata, init_weight_vector)
        
        return weight_vector, obj_values

        
    def setupPreTraining(self, TPPdata, pre_cal_file_name):
        self.pre_cal = True
        self.pre_cal_file = os.path.join(self.pre_cal_path, pre_cal_file_name)
        
        if not os.path.isfile(self.pre_cal_file):    
            # create the h5py file
            pre_cal_f = pd.HDFStore(self.pre_cal_file)        
            for realization in TPPdata:
                
                structure_df = self.structure_cal(realization)
                pre_cal_f[realization['cid']] = structure_df        
                
            pre_cal_f.close()
            
        print('Finish Pre Training')
        
        
    # prediction 
    def predict_realization(self, init_obs, pred_length, realization, weight_vector):
        
        assert 'feature_vector' in realization
        
        
        condition_1 = realization ['right_censored_time'] > init_obs
        
        condition_2 = (self.ownName in realization) and \
                      (realization[self.ownName][0] > init_obs)
                      
        condition_3 = (self.ownName not in realization) and \
                      (realization['right_censored_time'] > init_obs+pred_length)

        
        if condition_1 and (condition_2 or condition_3):
                
            # cut event time up to init_obs
            obs_realization = realization.copy()
            # for all possible source and own events
            for s in set(realization.keys()).intersection(self.sourceNames + [self.ownName]):
                obs_s = [e for e in realization[s] if e <= init_obs]
                if len(obs_s)>0:
                    obs_realization[s] = obs_s
                else:
                    obs_realization.pop(s)
            
            #--------------Keep ex events "observable"-------------------------
            # obs_realization = realization.copy()
            # if self.ownName in realization:
            #     obs_realization.pop(self.ownName)
            #------------------------------------------------------------------
                         

            survival_tc = np.exp(-self.susceptible_cumIntensity(init_obs, obs_realization))
            survival_both = np.exp(-self.susceptible_cumIntensity(init_obs+pred_length, obs_realization))
        
            prob = (survival_tc - survival_both) / (survival_tc + \
                   np.exp(-np.dot(weight_vector, obs_realization['feature_vector'])))
                
            # get ground truth value
            gt = 1 if (self.ownName in realization) and \
                      (realization[self.ownName][0]<init_obs + pred_length) \
                      else 0
                
        else:
            gt = np.nan
            prob = np.nan
            
        return prob, gt
    

    # prediction 
    def predict_realization_with_observation(self, init_obs, pred_length, realization, weight_vector):
        
        assert 'feature_vector' in realization
        
        
        condition_1 = realization ['right_censored_time'] > init_obs
        
        condition_2 = (self.ownName in realization) and \
                      (realization[self.ownName][0] > init_obs)
                      
        condition_3 = (self.ownName not in realization) and \
                      (realization['right_censored_time'] > init_obs+pred_length)

        
        if condition_1 and (condition_2 or condition_3):
                
            
            #--------------Keep ex events "observable"-------------------------
            obs_realization = realization.copy()
            if self.ownName in realization:
                obs_realization.pop(self.ownName)
            #------------------------------------------------------------------
                         

            survival_tc = np.exp(-self.susceptible_cumIntensity(init_obs, obs_realization))
            survival_both = np.exp(-self.susceptible_cumIntensity(init_obs+pred_length, obs_realization))
        
            prob = (survival_tc - survival_both) / (survival_tc + \
                   np.exp(-np.dot(weight_vector, obs_realization['feature_vector'])))
                
            # get ground truth value
            gt = 1 if (self.ownName in realization) and \
                      (realization[self.ownName][0]<init_obs + pred_length) \
                      else 0
                
        else:
            gt = np.nan
            prob = np.nan
            
        return prob, gt
    
    def get_prediction_scores(self, TPPdata, init_obs, pred_lenght, weight_vector, threshold=None):
        prediction_results = pd.DataFrame(columns=['cid', 'prob', 'gt'])
        for realization in TPPdata:
            prob, gt =  self.predict_realization(init_obs, pred_lenght, realization, weight_vector)
            prediction_results = prediction_results.append({'cid' : realization['cid'], \
                                                            'prob' : prob, 
                                                            'gt' : gt},  ignore_index = True)
        
        prediction_results = prediction_results.dropna()
        
        if threshold:
            prediction_results['es'] = prediction_results.apply(lambda x: 1 if x.prob > threshold else 0, axis=1)
    
        return prediction_results             



    # prediction 
    def predict_realization_new(self, init_obs, pred_length, realization, weight_vector):
        
        assert 'feature_vector' in realization
        
        
        condition_1 = realization ['right_censored_time'] > init_obs
        
        condition_2 = (self.ownName in realization) and \
                      (realization[self.ownName][0] > init_obs)
                      
        condition_3 = (self.ownName not in realization) and \
                      (realization['right_censored_time'] > init_obs+pred_length)

        
        if condition_1 and (condition_2 or condition_3):
                
            # cut event time up to init_obs
            obs_realization = realization.copy()
            # for all possible source and own events
            for s in set(realization.keys()).intersection(self.sourceNames + [self.ownName]):
                obs_s = [e for e in realization[s] if e <= init_obs]
                if len(obs_s)>0:
                    obs_realization[s] = obs_s
                else:
                    obs_realization.pop(s)
            
                    
            prob = self.CDF(init_obs+pred_length, obs_realization, weight_vector)
            # get ground truth value
            gt = 1 if (self.ownName in realization) and \
                      (realization[self.ownName][0]<init_obs + pred_length) \
                      else 0
                
        else:
            gt = np.nan
            prob = np.nan
            
        return prob, gt

                      

    def get_prediction_scores_new(self, TPPdata, init_obs, pred_lenght, weight_vector, threshold=None):
        prediction_results = pd.DataFrame(columns=['cid', 'prob', 'gt'])
        for realization in TPPdata:
            prob, gt =  self.predict_realization_new(init_obs, pred_lenght, realization, weight_vector)
            prediction_results = prediction_results.append({'cid' : realization['cid'], \
                                                            'prob' : prob, 
                                                            'gt' : gt},  ignore_index = True)
        
        prediction_results = prediction_results.dropna()
        
        if threshold:
            prediction_results['es'] = prediction_results.apply(lambda x: 1 if x.prob > threshold else 0, axis=1)
    
        return prediction_results  

                
    # Returns the TTP's model parameters as a tuple.
    def getParams(self):
        return self.para.to_dict()

    # Sets a TPP's parameters.
    # params: a tuple of parameters
    def setParams(self, para_true):
        self.para = pd.Series(para_true)
    
    # Initializes a TPP's parameters
    def initParams(self, para_init):
        self.para = pd.Series(para_init)

    def getSourceNames(self):
        return self.sourceNames
    
    def simulation(self):
        pass

################################################################################
#
# U N I T   T E S T I N G
#
################################################################################


# Some global settings for figure sizes
normalFigSize = (8, 6)  # (width,height) in inches
largeFigSize = (12, 9)
xlargeFigSize = (18, 12)


if __name__=='__main__':
    majorPath = 'C:/Users/cczha/Documents/Point_Process/'
    
    dataPath = 'data/simulated_data'
    dataFile = 'exploit_hawkes_3_dim.json'

    data = os.path.join(majorPath, dataPath, dataFile)
    with open(data, 'r') as f:
        TPPdata = json.load(f)
            

    ownName = 'exploit'
    sourceNames = ['u0', 'u1']
    
    mk = {'base': memory_kernel.ConstantMemoryKernel(), \
          'u0': memory_kernel.ExponentialPseudoMemoryKernel(beta=1.0),\
          'u1': memory_kernel.ExponentialPseudoMemoryKernel(beta=1.0)}
        
    para_true = {'base': 0.01, \
                  'u0': 0.02, \
                  'u1': 0.06}
        
          
    w_tilde = [2.07, 0, 0]
    
    # class
    ep = SplitPopulationSurvival(ownName, sourceNames, mk)
    # ep.setParams(para_true)
    
    init_weight_vector = np.ones(len(w_tilde))
    # learn
    # weight_vector, obj_values = ep.learning_joint(TPPdata, init_weight_vector)
    
    # pp_times = []
    # for realization in TPPdata:
    #     pp_times.extend(ep.transformEventTimes(realization, weight_vector))
        

    # from matplotlib import pyplot as plt
    # _, ax = plt.subplots(1, 1)
    # pvalue = GoodnessOfFit.KSgoodnessOfFitExp1(sorted(pp_times), \
    #                                             ax, showConfidenceBands=True)
    pre_cal_file_name = 'split_survival_precal.h5py'.format(ownName)
    weight_vector, obj_values = ep.train(TPPdata, init_weight_vector, \
                                         pre_cal_file_name, pre_cal= True)

        
        
        
        
        
        
        
        
        
        