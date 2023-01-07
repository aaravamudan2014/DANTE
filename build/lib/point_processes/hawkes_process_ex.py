#!/usr/bin/python
'''Temporal Point Process (TPP) sub-classes'''

import sys
import os
parent_path = os.path.dirname(os.getcwd())
sys.path.append(os.path.join(parent_path, 'utils'))
sys.path.append(os.path.join(parent_path, 'data'))

import json
import pandas as pd
import numpy as np
# import scipy.stats
import random
# import logging
import os
import h5py
from matplotlib import pyplot as plt
from point_processes.TemporalPointProcess import TemporalPointProcess as TPP, TrainingStatus
from point_processes.PointProcessCollection import *
import GoodnessOfFit
from utils.MemoryKernel import *


class HawkesTPP(TPP):
    '''Hawkes Temporal Point Process'''

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
    # sourceNames: excitation_sourceNames 
    def __init__(self, ownName, sourceNames, mk, stop_criteria={'max_iter':200,\
                 'epsilon': 1e-4}, desc='Hawkes TPP', logger=None):

        self.dim = len(sourceNames)
        self.desc = desc
        self.sourceNames = sourceNames
        self.ownName = ownName
        self.stop_criteria = stop_criteria
        self.mk = mk
        self.pre_cal = False
        self.pre_cal_path = 'pre_cal_storage'
        self.pre_cal_file = ''
        self.initParams(dict.fromkeys(['base'] + sourceNames, 1.0))

    
        super().__init__(sourceNames, desc, logger)

    def intensityUB():
        pass
    def simulate():
        pass

    # t: float or 1-dimensional numpy.ndarray of reals; relative event time(s). Typically, larger than 0.
    # realization: a single realization of model including all sources events
    
    def intensity(self, t, realization):
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
        return value

    # t: float or 1-dimensional numpy.ndarray of reals; relative event time(s). Typically, larger than 0.
    # realization: a single realization of model including all sources events
    
    def cumIntensity(self, t, realization):
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

        return value

    
    def loglikelihood(self, realization):
        
        if self.pre_cal:
            pre_cal_f = pd.HDFStore(self.pre_cal_file) 
            structure_df = pre_cal_f[realization['cid']]

        else:
            structure_df = self.structure_cal(realization)
  
        # locate the event df
        event_df = structure_df.loc[:, structure_df.columns != 'survival']
        activatedTerm = np.log(self.para.dot(event_df)).sum()

        # calculate the log-likelihood for survival part
        survivalTerm = structure_df.loc[:, 'survival'].dot(self.para)

        loglikelihood_value = activatedTerm - survivalTerm


        return loglikelihood_value
    
    def transformEventTimes(self, realization):

        
        rescaled_values = []
        
        if self.ownName in realization:
            own_events = list(sorted(realization[self.ownName]))
            
            for own_idx, own_event in enumerate(own_events):
                
                tk_1 = own_events[own_idx -1] if own_idx >= 1 else 0
                tk = own_event

                rescaled_value = self.cumIntensity(tk, realization) - \
                                 self.cumIntensity(tk_1, realization)
                
                rescaled_values.append(rescaled_value)

        return rescaled_values
    
    
    def uni_variate_Hawkes_simulation(self, realization, obs_t=None):
        
        if obs_t:
            t = obs_t
        else:
            t = 0.0
            
        if self.ownName in realization:
            sp_realization = realization[self.ownName]
            if not obs_t and sp_realization:
                t =  max(sp_realization)        
        else:
            sp_realization = []
        

        s = t
        delta = 1e-7
        
        T = realization['right_censored_time']
        
        # simulation
        while s < T:
            
            lambda_bar = self.intensity(s+delta, realization)
            
            # point generating
            u = random.uniform(0, 1)
            w = -np.log(u) / lambda_bar
            s = s + w
            
            # thinning
            D = random.uniform(0, 1)
            
            if D * lambda_bar <= self.intensity(s, realization):
                # accept this point 
                t = s
                sp_realization.append(t)

                
            if t > T:
                del sp_realization[-1]
                
        realization[self.ownName] = sp_realization
            
        return realization
    

    def structure_cal(self, realization):

        activated_sources = [s for s in list(realization.keys())
                             if s in self.sourceNames]

        if self.ownName in realization:
            own_events = sorted(realization[self.ownName])
            own_events_num = len(own_events)

            # Calculate Recurssive Matrix for user == userId

            event_df = pd.DataFrame(data=np.zeros((self.dim+1, own_events_num)),\
                                    index=['base'] + self.sourceNames)
            
            event_df.loc['base'] = [self.mk['base'].phi(event) for event in own_events]
            
            for s in activated_sources:
                if self.mk[s].desc.startswith('Exponential'):
                    for own_event_ind in range(own_events_num):
                        ts = own_events[own_event_ind -1] if own_event_ind >= 1 else 0
                        te = own_events[own_event_ind]

                        newEvents = [c for c in realization[s] if c < te and c >= ts]

                        newArr = 0
                        for event in newEvents:
                            newArr = newArr + self.mk[s].phi(te - event)

                        event_df.loc[s, own_event_ind] = self.mk[s].phi(te-ts) * \
                            event_df.loc[s, own_event_ind-1] + newArr \
                            if own_event_ind >= 1 else newArr

                else:
                    for own_event_ind in range(own_events_num):
                        cal_source_events = [c for c in realization[s] \
                                             if c < own_events[own_event_ind]]

                        for event in cal_source_events:
                            event_df.loc[s, own_event_ind] += \
                                self.mk[s].phi(own_events[own_event_ind] - event)
        else: # if no own event, return empty dataframe
            event_df = pd.DataFrame(index=['base'] + self.sourceNames)

        # survival terms
        if 'right_censored_time' in realization:
            T = realization['right_censored_time']
        else:
            T = max([max(realization[s]) for s in activated_sources])
            
        # survival terms
        survival_df = pd.Series(data=np.zeros(self.dim+1), \
                                index=['base']+self.sourceNames, name='survival')
            
        survival_df['base'] = self.mk['base'].psi(T)
        
        for s in activated_sources:
            for event in realization[s]:
                survival_df.loc[s] += self.mk[s].psi(T-event)

        # combine
        structure_df = pd.concat([event_df, survival_df], axis=1, sort=False)

        return structure_df

    
    def learning_EM_update(self, TPPdata):
        # query from class
        epsilon = self.stop_criteria['epsilon']
        max_iter = self.stop_criteria['max_iter']

        epoch_iter = 0
        obj_diff = epsilon + 1

        # save objective value in each iteration (optional)
        epoch_obj_value = []
        
        # keep updating until having met stopping criteria
        while obj_diff > epsilon and (epoch_iter < max_iter):
            
            para_next_numerator = pd.Series(data=np.zeros(self.dim+1), \
                                            index=['base']+self.sourceNames)
            para_next_denominator = pd.Series(data=np.zeros(self.dim+1), \
                                              index=['base'] + self.sourceNames)

            obj_value = 0
            
            if self.pre_cal:
                pre_cal_f = pd.HDFStore(self.pre_cal_file) 

            # loop through cascades
            for realization in TPPdata:
                            
                if self.pre_cal:
                    # pre_cal_f = pd.HDFStore(self.pre_cal_file) 
                    structure_df = pre_cal_f[realization['cid']]
                else:
                    structure_df = self.structure_cal(realization)
                
                
                event_df = structure_df.loc[:, structure_df.columns != 'survival']
                
                if event_df.shape[1] > 0:
                    ini_df = event_df.mul(pd.concat([self.para] * event_df.shape[1], axis=1))
                    active_df = ini_df / ini_df.sum()

                    rnl = np.log(ini_df.sum()).sum()
                    
                else:
                    active_df = pd.DataFrame(data=np.zeros((self.dim+1, 0)),\
                                             index=['base']+self.sourceNames)
                    rnl = 0

                para_next_numerator = para_next_numerator + active_df.sum(axis=1)
                para_next_denominator = para_next_denominator + structure_df.loc[:, 'survival']
                
                rnl = rnl - self.para.dot(structure_df.loc[:, 'survival']) 
                
            
                
                # obj_value = obj_value + self.loglikelihood(realization)
                if 'log-likelihood-weight' in realization.keys():
                    obj_value = obj_value + \
                        np.power(rnl, realization['log-likelihood-weight'])
                else:
                    obj_value = obj_value + rnl

            # updates parameters
            self.para = para_next_numerator / para_next_denominator
            # objective value in current iteration
            epoch_obj_value.append(obj_value)
            print(obj_value)
            # add iteration count
            epoch_iter = epoch_iter + 1

            obj_diff = abs(epoch_obj_value[-1] - epoch_obj_value[-2]) if (epoch_iter > 1) else obj_diff
                    
            print('epoch iteration number: {}'.format(epoch_iter))
            print('para = {}'.format(self.para))
                    
                    
            # # close if open
            # try:
            #     pre_cal_f.close()
            # except:
            #     pass
                        
        return epoch_obj_value

    # Setup TPP model for training.
    # Typically used for pre-computing quantities that are necessary for training.
    # This method is expected to be called prior to a call to the train() method.
    # TPPdata: List of S lists, where each inner list contains R realizations.
    #          Each realization contains at least a relative right-censoring time as its last event time.

    def setupTraining(self, TPPdata):
        pass


    # Train the TPP model.
    # Returns an element of the TrainingStatus enumeration.
    def train(self, TPPdata, pre_cal_file_name='', pre_cal=False, method='EM_update'):
        
        if pre_cal:
            self.setupPreTraining(TPPdata, pre_cal_file_name)
            
        if method == 'EM_update':
            self.learning_EM_update(TPPdata)
            
        
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

    uh_ownName = 'hw'
    uh_sourceNames = ['hw']
    
    uh_mk = {'base': memory_kernel.ConstantMemoryKernel(), \
              'hw': memory_kernel.ExponentialMemoryKernel(beta = 1.0)}
          
    # class
    uni_hawkes = HawkesTPP(uh_ownName, uh_sourceNames, uh_mk) 
    
    para_true = {'base': 0.1, \
                  'hw': 0.6} 
        
    uni_hawkes.setParams(para_true)
    
    
    # #==============================================================================
    # #============================Simulation========================================
    # #==============================================================================
    
    N = 1
    stop_times = [100]*N#np.random.randint(6, high=10, size=N).tolist()
    cascade_ids = ['c_{}'.format(k) for k in range(N)]
    
    TPPdata = []
    for k in np.arange(N):
        realization = {}
        realization['cid'] = cascade_ids[k]
        realization['right_censored_time'] = stop_times[k]
        
        realization = uni_hawkes.uni_variate_Hawkes_simulation(realization)
        
        TPPdata.append(realization)




        
  