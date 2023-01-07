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
import glob
import random
from numpy import linalg as LA
# import logging
import os
import h5py
from matplotlib import pyplot as plt
from point_processes.TemporalPointProcess import TemporalPointProcess as TPP, TrainingStatus
from point_processes.PointProcessCollection import *
import GoodnessOfFit
from utils.MemoryKernel import *

class SelfExcitingTPP(TPP):
    '''Hawkes Temporal Point Process'''

    # Instance variables
    #
    #   HDSourceName           list of history dependent sources
    #   HinDSourceName         list of history independent sources    
    #   ownName:               string, name of the current TPP, 
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
    def __init__(self, ownName, HInDSourceNames, HDSourceNames, mk, stop_criteria={'max_iter':200,\
                 'epsilon': 1e-4}, pre_cal_path = 'pre_cal_storage', desc='Hawkes TPP', logger=None):

        self.sourceDim = len(HDSourceNames) +  len(HInDSourceNames)
        self.desc = desc
        self.HDSourceNames = HDSourceNames
        self.HInDSourceNames = HInDSourceNames
        self.ownName = ownName
        self.stop_criteria = stop_criteria
        self.mk = mk
        self.pre_cal = False
        self.pre_cal_path = pre_cal_path
        self.pre_cal_file = ''
        self.initParams(dict.fromkeys(HInDSourceNames + HDSourceNames, 1.0))

        super().__init__( HInDSourceNames + HDSourceNames, desc, logger)

    # t: float or 1-dimensional numpy.ndarray of reals; relative event time(s). Typically, larger than 0.
    # realization: a single realization of model including all sources events
    
    
    def intensity(self, t, realization):
        
        # calculate history excitation
        phi_vector = pd.DataFrame({'source':self.para.keys(), 'value':0.0})
        
        
        activated_HDsources = list( set(realization.keys()).intersection( set(self.HDSourceNames)))
        
        
        phi_vector['value'] = phi_vector.apply(lambda x: pd.Series(realization[x['source']]).\
                              loc[pd.Series(realization[x['source']]) < t].\
                              apply(lambda y: self.mk[x['source']].phi(t-y)).sum() \
                              if x['source'] in activated_HDsources else 0.0 , axis=1 )
            
               
        for hind_s in self.HInDSourceNames:
            phi_vector.loc[phi_vector['source']==hind_s,'value'] = self.mk[hind_s].phi(t) 
    

        value = self.para.dot( pd.Series(data = phi_vector['value'].values,\
                                         index=phi_vector['source']) )
            
            
        return value
    # t: float or 1-dimensional numpy.ndarray of reals; relative event time(s). Typically, larger than 0.
    # realization: a single realization of model including all sources events

    
    def cumIntensity(self, t, realization):
        
        # calculate history excitation
        psi_vector = pd.DataFrame({'source':self.para.keys(), 'value':0.0})
        
        activated_HDsources = list( set(realization.keys()).intersection( set(self.HDSourceNames)))
        
        
        psi_vector['value'] = psi_vector.apply(lambda x: pd.Series(realization[x['source']]).\
                              loc[pd.Series(realization[x['source']]) < t].\
                              apply(lambda y: self.mk[x['source']].psi(t-y)).sum() \
                              if x['source'] in activated_HDsources else 0.0 , axis=1 )
            
        for hind_s in self.HInDSourceNames:
            psi_vector.loc[psi_vector['source']==hind_s,'value'] = self.mk[hind_s].psi(t) 
        
        value = self.para.dot( pd.Series(data = psi_vector['value'].values,\
                                         index=psi_vector['source']) )
            
            
        return value
    


    
    def structure_cal(self, realization):

        activated_HDsources = list( set(realization.keys()).intersection( set(self.HDSourceNames)))
        
        if self.ownName in realization:
            own_events = sorted(realization[self.ownName])
            own_events_num = len(own_events)

            # Calculate Recurssive Matrix for user == userId

            event_df = pd.DataFrame(data=np.zeros((self.sourceDim, own_events_num)),\
                                    index= self.HInDSourceNames + self.HDSourceNames)
                                
            for hind_s in self.HInDSourceNames:
                event_df.loc[hind_s] = [self.mk[hind_s].phi(event) for event in own_events]            
            
            for s in activated_HDsources:
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
                            if own_event_ind > 1 else newArr

                else:
                    for own_event_ind in range(own_events_num):
                        cal_source_events = [c for c in realization[s] \
                                             if c < own_events[own_event_ind]]

                        for event in cal_source_events:
                            event_df.loc[s, own_event_ind] += \
                                self.mk[s].phi(own_events[own_event_ind] - event)
        else: # if no own event, return empty dataframe
            event_df = pd.DataFrame(index= self.HInDSourceNames + self.HDSourceNames)

        # survival terms
        if 'right_censored_time' in realization:
            T = realization['right_censored_time']
        elif (realization[self.ownName] in realization) and (len(realization[self.ownName]) > 0):
            T = max(realization[self.ownName])
        else:
            T = 0.0
            
        # survival terms
        survival_df = pd.Series(data=np.zeros(self.sourceDim), \
                                index= self.HInDSourceNames + self.HDSourceNames, name='survival')
        
        for hind_s in self.HInDSourceNames:
            survival_df[hind_s] = self.mk[hind_s].psi(T)
        
        for s in activated_HDsources:
            for event in realization[s]:
                survival_df.loc[s] += self.mk[s].psi(T-event)

        # combine
        structure_df = pd.concat([event_df, survival_df], axis=1, sort=False)

        return structure_df

    
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
            cum_T = self.cumIntensity(realization['right_censored_time'], realization)
           
            for own_idx, own_event in enumerate(own_events):
                
                tk_1 = own_events[own_idx -1] if own_idx >= 1 else 0
                tk = own_event

                cum = self.cumIntensity(tk, realization) - \
                      self.cumIntensity(tk_1, realization)
                                 
                value = np.log( (1-np.exp(-cum_T)) / (np.exp(-cum) - np.exp(-cum_T)) ) 
                
                rescaled_values.append(value)
        

        return rescaled_values

    def transformEventTimes_TPPdata(self, TPPdata):
        pp_times = []
        for realization in TPPdata:
            pp_times.extend(self.transformEventTimes(realization))
            
        return pp_times
    


    
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
            
            para_next_numerator = pd.Series(data=np.zeros(self.sourceDim), \
                                            index= self.HInDSourceNames + self.HDSourceNames)
            para_next_denominator = pd.Series(data=np.zeros(self.sourceDim), \
                                              index=self.HInDSourceNames + self.HDSourceNames)

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
                    
                    llh = np.log(ini_df.sum()).sum()
                    
                else:
                    active_df = pd.DataFrame(data=np.zeros((self.sourceDim, 0)),\
                                             index= self.HInDSourceNames + self.HDSourceNames)
                    llh = 0

                para_next_numerator = para_next_numerator + active_df.sum(axis=1)
                para_next_denominator = para_next_denominator + structure_df.loc[:, 'survival']
                
                llh = llh - self.para.fillna(0).dot(structure_df.loc[:, 'survival']) 
                
                # obj_value = obj_value + self.loglikelihood(realization)
                if 'log-likelihood-weight' in realization.keys():
                    obj_value = obj_value - \
                        np.power(llh, realization['log-likelihood-weight'])
                else:
                    obj_value = obj_value - llh

            # updates parameters
            self.para = para_next_numerator / para_next_denominator
            
            if self.para[self.ownName] >= 1.000:

                self.para[self.ownName] = 0.999
                print('bounded')
                

                
                
            # objective value in current iteration
            epoch_obj_value.append(obj_value)

            # add iteration count
            epoch_iter = epoch_iter + 1

            obj_diff = abs(epoch_obj_value[-1] - epoch_obj_value[-2]) \
                if (epoch_iter > 1) else obj_diff
                    
            # print('epoch iteration number: {}'.format(epoch_iter))
            # print('para = {}'.format(self.para))
                    
                
        # close if open
        try:
            pre_cal_f.close()
        except:
            pass
              
        self.para  = self.para.fillna(0)     
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
            print('done')
                
    def uni_variate_Hawkes_simulation(self, realization, obs_t=None):
        
        if obs_t:
            t = obs_t    
        elif (not obs_t) and  (self.ownName in realization) and (len(realization[self.ownName]) > 0):
            t =  max(realization[self.ownName])        
        else:
            t = 0.0
            
        if self.ownName not in realization:
            realization[self.ownName] = []

        s = t
        delta = 1e-7
        
        T = realization['right_censored_time']
        
        # simulation
        while s < T:
            
            lambda_bar = self.intensity(s+delta, realization)
            
            # point generating
            w = -np.log(random.uniform(0, 1)) / lambda_bar
            s = s + w
            
            # thinning
            D = random.uniform(0, 1)
            
            if D * lambda_bar <= self.intensity(s, realization):
                # accept this point 
                t = s
                realization[self.ownName].append(t)
            # else:
            #     print('reject')
                
        if t > T:
            del realization[self.ownName][-1]
        
        if len(realization[self.ownName]) == 0:        
            del realization[self.ownName]
            
        return realization
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
        return self.HDSourceNames + self.HInDSourceNames


    def most_probably_trigger(self, realization, pre_cal_file_name):

        pre_cal_file = os.path.join(self.pre_cal_path, pre_cal_file_name)
            
        if os.path.isfile(pre_cal_file):
            pre_cal_f = pd.HDFStore(pre_cal_file) 
            structure_df = pre_cal_f[realization['cid']]
        else:
            structure_df = self.structure_cal(realization)
        
                
        event_df = structure_df.loc[:, structure_df.columns != 'survival']
                
        if event_df.shape[1] > 0:
            
            temp_df = pd.concat([self.para] * event_df.shape[1], axis=1).\
                set_axis(np.arange(event_df.shape[1]), axis='columns')
                
            ini_df = event_df.mul(temp_df)
            
            prob_df = ini_df / ini_df.sum()
            
            trigger_list = list(prob_df.apply(lambda x: x.idxmax()))
        else:
            
            trigger_list = []
            

                
        # close if open
        try:
            pre_cal_f.close()
        except:
            pass
                        
        return trigger_list
    
    
    # # Delate after running
    # def learning_EM_update(self, TPPdata):
    #     # query from class
    #     epsilon = self.stop_criteria['epsilon']
    #     max_iter = self.stop_criteria['max_iter']

    #     epoch_iter = 0
    #     obj_diff = epsilon + 1

        
    #     # keep updating until having met stopping criteria
    #     while obj_diff > epsilon and (epoch_iter < max_iter):
            
    #         para_next_numerator = pd.Series(data=np.zeros(self.sourceDim), \
    #                                         index= self.HInDSourceNames + self.HDSourceNames)
    #         para_next_denominator = pd.Series(data=np.zeros(self.sourceDim), \
    #                                           index=self.HInDSourceNames + self.HDSourceNames)
            
    #         if self.pre_cal:
    #             pre_cal_f = pd.HDFStore(self.pre_cal_file) 

    #         # loop through cascades
    #         for realization in TPPdata:
                            
    #             if self.pre_cal:
    #                 # pre_cal_f = pd.HDFStore(self.pre_cal_file) 
    #                 structure_df = pre_cal_f[realization['cid']]
    #             else:
    #                 structure_df = self.structure_cal(realization)
                
                
    #             event_df = structure_df.loc[:, structure_df.columns != 'survival']
                
    #             if event_df.shape[1] > 0:
                    
    #                 ini_df = event_df.mul(pd.concat([self.para] * event_df.shape[1], axis=1))
    #                 active_df = ini_df / ini_df.sum()
                                        
    #             else:
    #                 active_df = pd.DataFrame(data=np.zeros((self.sourceDim, 0)),\
    #                                          index= self.HInDSourceNames + self.HDSourceNames)

    #             para_next_numerator = para_next_numerator + active_df.sum(axis=1)
    #             para_next_denominator = para_next_denominator + structure_df.loc[:, 'survival']
                
                

    #         para_pre = self.para
    #         # updates parameters
    #         self.para = (para_next_numerator / para_next_denominator).fillna(0)

            

    #         # add iteration count
    #         epoch_iter = epoch_iter + 1

    #         obj_diff = LA.norm(para_pre.fillna(0) - self.para.fillna(0) )
                    
    #         print('epoch iteration number: {}'.format(epoch_iter))
    #         print('para = {}'.format(self.para))
    #         print('obj_dff = {}'.format(obj_diff))
                    
                    
    #         # close if open
    #         try:
    #             pre_cal_f.close()
    #         except:
    #             pass
                        
    #     return obj_diff


###############################################################################
#
# U N I T   T E S T I N G
#
################################################################################


# Some global settings for figure sizes
normalFigSize = (8, 6)  # (width,height) in inches
largeFigSize = (12, 9)
xlargeFigSize = (18, 12)


if __name__=='__main__':


    ownName = 'h1'
    HInDSourceNames = ['ex1', 'ex2']
    HDSourceNames = []

    
    mk = {'ex1': memory_kernel.ConstantMemoryKernel(), \
          'ex2': memory_kernel.ExponentialMemoryKernel(beta = 0.6)} 
          # 'h1': memory_kernel.ExponentialMemoryKernel(beta = 1.0)}
          
    # class
    hw = SelfExcitingTPP(ownName, HInDSourceNames, HDSourceNames, mk) 
    
    para_true = {'ex1': 0.1, \
                 'ex2' : 0.3}
                  # 'h1'  : 0.2} 
        
    hw.setParams(para_true)
    
    realization = {}
    realization['cid'] = 'c9'
    realization['right_censored_time'] = 40.00
    
    self = hw
    
    #==============================================================================
    #============================Simulation========================================
    #==============================================================================
    
    N = 40000
    stop_times = [20]*N#np.random.randint(6, high=10, size=N).tolist()
    cascade_ids = ['c_{}'.format(k) for k in range(N)]
    
    TPPdata = []
    for k in np.arange(N):
        realization = {}
        realization['cid'] = cascade_ids[k]
        realization['right_censored_time'] = stop_times[k]
        
        realization = hw.uni_variate_Hawkes_simulation(realization)
        
        TPPdata.append(realization)

    obj_value = hw.learning_EM_update(TPPdata)
    
    hw.initParams(dict.fromkeys(HInDSourceNames + HDSourceNames, 1.0))
    
    obj_v = hw.learning_EM_update(TPPdata)
    
    
    pp_times = hw.transformEventTimes_TPPdata(TPPdata)
    
    # maxN = np.int(1e4)
    
    # fig, ax = plt.subplots()    
    # pvalue = GoodnessOfFit.KSgoodnessOfFitExp1(\
    #             random.sample(pp_times, min(maxN, len(pp_times))), \
    #             ax, showConfidenceBands=True)
        
    # ax.set_title('uni_hawkes')
    # ax.text(0.5, 0, 'p={:.02f}'.format(pvalue), fontsize=12)


    plt.figure()  
    import scipy.stats as stats
    stats.probplot(pp_times, dist="expon", plot=plt)
  