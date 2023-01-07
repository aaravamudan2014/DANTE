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
import scipy.stats
import random
import h5py
import time
from matplotlib import pyplot as plt
from point_processes.TemporalPointProcess import TemporalPointProcess as TPP, TrainingStatus
from point_processes.PointProcessCollection import *
import GoodnessOfFit
from utils.MemoryKernel import *
from hawkes_process_ex import HawkesTPP
from self_exciting_process import SelfExcitingTPP
from split_population_survival_process import SplitPopulationSurvival 

from FeatureVectorCreator import gen2IsotropicGaussiansSamples
import GoodnessOfFit


class multi_variate_process():
    # index: process ownName
    def __init__(self, processes):

        self.processes = processes
        self.dim = len(self.processes)
        self.get_sourceNames()
        # self.get_para()
        
    def get_sourceNames(self):
        self.sourceNames = [p.ownName for p in self.processes]
        
        return self.sourceNames
        

    def intensity(self, t, realization):
        intensity = pd.Series(index=self.sourceNames,dtype = 'float64')
        for p in self.processes:
            intensity[p.ownName] = p.intensity(t, realization)
            
        return intensity
    
    def ogata_thinning_simulation(self, realization, t_obs = None, start_time=None):
        
        assert start_time is not None, "please enter start time"
        
        if t_obs:
            s = t_obs
        else:
            s=0
                       
        delta = 1e-7  # s+
        stopTime = realization['right_censored_time']
        san_c = 0
        #Run until reach T
        while s < stopTime:
            end_time = time.time()
            if (end_time - start_time)/3600 > 5 :
                return None
            lambda_bar = sum(self.intensity(s+delta, realization)) # upper bound intensity
            
            
            # homogeneous poisson simulation with intensity lambda_bar
            s = s + (-np.log(random.uniform(0, 1))) / lambda_bar
        
            # accepting with probability  sum(curr_intensity.values()) / lambda_bar
            next_intensity = self.intensity(s, realization)
            
            next_intensity = next_intensity[next_intensity > 0] # Temporial fix
            
            D = random.uniform(0, 1) # uniformly generated from [0,1)
            if D * lambda_bar <= sum(next_intensity):
                
                san_c = san_c + 1
                # accept this point and assign to a dimension
                intensity_cumsum = pd.Index(next_intensity.cumsum().values)
                assign_to = intensity_cumsum.get_loc(min(i for i in intensity_cumsum if i > D*lambda_bar))
                assign_source = next_intensity.index[assign_to]
                
                if assign_source in realization:
                    realization[assign_source].append(s)
                else:
                    realization[assign_source] = [s]
                     
                if 'exploit' in realization.keys(): # Temporial fix
                    break
                    
        # if the last event out of time range, exclude that event
        if (san_c > 0) and (realization[assign_source][-1] > stopTime):
            del realization[assign_source][-1]
        
        for k, v in realization.copy().items():
            if (k in self.sourceNames) and (len(v) == 0):
                del realization[k]

        # # # only keep key value pair with non zero value length 
        # # # if user has no event, remove from the dictionary
        # realization = {k: v for k, v in realization.items() if \
        #                 k in self.sourceNames and len(v) > 0 }
                                   
        return realization
    
    def simulation(self, N, stop_times, cids, add_susceptible = False, \
                   susceptible_labels = None, feature_vectors = None, \
                   method='ogata_thinning'):
        
        # if np.isscalar(stop_times):
        #     stop_times = [stop_times] * N
        TPPdata = []

        if method == 'ogata_thinning':
            
            for k in range(N):
                print(k)
                realization = {}
                realization['right_censored_time'] = stop_times[k]
                if add_susceptible:
                    realization['susceptible'] = np.int(susceptible_labels[k])
                    realization['feature_vector'] = list(feature_vectors[k])
                else:
                    realization['susceptible'] = np.int(1)
                       
                realization = self.ogata_thinning_simulation(realization)
                realization['cid'] = cids[k]
                
                TPPdata.append(realization)
                
        return TPPdata
    

        
    
    
    def train(self, TPPdata):
        # train for each process
        for p in self.processes:
            p.train(TPPdata)
            
            
            
            
        
        
        
# Some global settings for figure sizes
normalFigSize = (8, 6)  # (width,height) in inches
largeFigSize = (12, 9)
xlargeFigSize = (18, 12)

if __name__=='__main__':
    #--------------------------------------------------------------------------
    # hawkes process: u0
    sm1_ownName = 'social_media_1'
    sm1_sourceNames = ['social_media_1', 'social_media_2']
    
    sm1_mk = {'base': memory_kernel.ConstantMemoryKernel(), \
              'social_media_1': memory_kernel.ExponentialMemoryKernel(beta=1.0), \
              'social_media_2': memory_kernel.ExponentialMemoryKernel(beta=1.0)}          
    # class
    sm1 = HawkesTPP(sm1_ownName, sm1_sourceNames, sm1_mk) 
    
    sm1_para_true = {'base': 0.1, \
                     'social_media_1': 0.4, \
                     'social_media_2': 0.7}
        
    sm1.setParams(sm1_para_true)
    #--------------------------------------------------------------------------
    # hawkes process: u1
    sm2_ownName = 'social_media_2'
    sm2_HInDSourceNames = ['ex1', 'ex2']
    sm2_HDsourceNames = ['social_media_1']
    
    sm2_mk = {'ex1': memory_kernel.ConstantMemoryKernel(), \
              'ex2': memory_kernel.WeibullMemoryKernel(gamma=0.8),\
              'social_media_1': memory_kernel.ExponentialMemoryKernel(beta=1.0)}     
          
    # class
    sm2 = SelfExcitingTPP(sm2_ownName, sm2_HInDSourceNames, sm2_HDsourceNames, sm2_mk) 
    
    # setup para
    sm2_para_true = {'ex1': 0.2, \
                     'ex2': 0.6, \
                     'social_media_1': 0.2 }
    sm2.setParams(sm2_para_true)
    
    #================simulation================================================
    
    processes = [sm1, sm2]
    mp = multi_variate_process(processes)
    
    N = 10000
    stop_times = [20]*N#np.random.randint(20, high=120, size=N).tolist()
    cascade_ids = ['c_{}'.format(k) for k in range(N)]
    
    TPPdata = mp.simulation(N, stop_times, cascade_ids)
        
    # Shuffle TPPdata:
    random.shuffle(TPPdata)

    
    #==============================================================================
    #======================GOF Plot================================================
    #==============================================================================
    pp_times_1 = sm1.transformEventTimes_TPPdata(TPPdata)
    pp_times_2 = sm2.transformEventTimes_TPPdata(TPPdata)
    # pp_times_e = ep.transformEventTimes_TPPdata(TPPdata)
    
    maxN = np.int(1e4)
    fig, ax = plt.subplots()    
    pvalue = GoodnessOfFit.KSgoodnessOfFitExp1(\
                random.sample(pp_times_1, min(maxN, len(pp_times_1))), \
                ax, showConfidenceBands=True)
    ax.set_title('sm1')
    ax.text(0.5, 0, 'p={:.02f}'.format(pvalue), fontsize=12)
    
    # #==============================================================================
    
    
    fig, ax = plt.subplots()    
    pvalue = GoodnessOfFit.KSgoodnessOfFitExp1(\
                random.sample(pp_times_2, min(maxN, len(pp_times_2))), \
                ax, showConfidenceBands=True)
    ax.set_title('sm2')
    ax.text(0.5, 0, 'p={:.02f}'.format(pvalue), fontsize=12)
    
    
    import scipy.stats as stats
    stats.probplot(pp_times_1, dist="expon", plot=plt)

        
    # # ========modification data format for Akshay ==============================
    # with open(data, 'r') as f:
    #     TPPdata = json.load(f)
        
        
    # # akshay: list of list data format
    # newTPPdata = []
    # newFeatureVector = np.zeros((len(TPPdata), 3))
    # newSusceptibleLabel = np.zeros(len(TPPdata))
    # for c, realization in enumerate(TPPdata):
    #     new_realization = []
    #     Tc = realization['right_censored_time']
        
    #     for s in mp.sourceNames:
    #         if s in realization:
    #             new_realization.append(realization[s] + [Tc])
    #         else:
    #             new_realization.append([Tc])    
    #     newTPPdata.append(new_realization)        
    #     newFeatureVector[c,:] = realization['feature_vector'] 
    #     newSusceptibleLabel[c] = realization['susceptible']

  
    # np.save('simulated_realization.npy', newTPPdata)    
    # np.save('feature_vector.npy', newFeatureVector)   
    # np.save('gt_y.npy', newSusceptibleLabel)     
    # np.save('gt_w_tilde.npy', w_tilde)   
    # np.save('gt_exploit_alpha.npy', mp.para.loc['exploit', :])
      

        
    # allOnes = np.ones((numSamples,1))
    # X_tilde = np.concatenate((X, allOnes), axis=1)

    # y_pred = np.sign(np.dot(X_tilde, w_tilde))
    # errorRate = np.sum(y_pred != y) / numSamples
    

    # processes = [hawkes0, hawkes1]
        
    # multi_hawkes = multi_variate_process(processes)
    # a = multi_hawkes.ogata_thinning_simulation(100)
    # N = 400
    # cascade_ids = ['c_{}'.format(k) for k in range(1, N+1)]
    # simulated_TPPdata = multi_hawkes.simulation(N, [40]*N, cascade_ids)
    
    # realization = TPPdata[0]
    # a = multi_hawkes.intensity(2, realization)
    # a = exploitp.transformEventTimes_noSplit(TPPdata[2])
    # b = exploitp.transformEventTimes_b(TPPdata[2], w_tilde)
    # b1 = exploitp.transformEventTimes(TPPdata[2], w_tilde)

















# #==============================================================================

# T = 10

# in
    
# # simulation_events = dict.fromkeys(keys, []) # history events
# s = 0  # homogeneous poission time
   
# delta = 1e-7  # s+

# #Run until reach T
# while s < T:
    
#     curr_intensity = self.intensity_calculation(simulation_events, s+delta)
#     lambda_bar = sum(curr_intensity.values()) # upper bound intensity
    
#     # homogeneous poisson simulation with intensity lambda_bar
#     s = s + (-np.log(np.random.uniform(size = 1)[0])) / lambda_bar

#     # accepting with probability  sum(curr_intensity.values()) / lambda_bar
#     next_intensity = self.intensity_calculation(simulation_events, s)
    
#     D = np.random.uniform(size = 1)[0] # uniformly generated from [0,1)
#     if D * lambda_bar <= sum(next_intensity.values()):
#         # accept this point and assign to a dimension
#         m = 0
#         while D * lambda_bar > sum(next_intensity[x] for x in range(m+1)):
#             m = m + 1
        
#         # save the point to the corresponding dimension
#         temp = simulation_events[m].copy()
#         temp.append(s)
#         simulation_events[m] = temp
# # if the last event out of time range, exclude that event
# if simulation_events[m][-1] > stopTime:
#     del simulation_events[m][-1]
   
# # only keep key value pair with non zero value length 
# # if user has no event, remove from the dictionary
# simulation_events = {k: v for k, v in simulation_events.items() if len(v) > 0}


# return simulation_events




        
  