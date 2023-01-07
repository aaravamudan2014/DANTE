#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 15:55:13 2020

@author: social-sim

Testification with simulation data
"""

#!/usr/bin/python
'''Temporal Point Process (TPP) sub-classes'''


import sys
import os
from utils.DataReader import generateSyntheticDataset
from os.path import dirname, abspath

import time


dirPath = dirname(abspath(os.getcwd()))

sys.path.append(os.path.join(dirPath, 'processes'))


from hawkes_process_ex import HawkesTPP
from split_population_survival_process import SplitPopulationSurvival
from multi_variate_process import multi_variate_process 

import json
import glob
import numpy as np
from dask.distributed import Client 

from utils.MemoryKernel import *
from utilities import listOfList2json_withFeature

#==============================================================================
#==============================SetUp===========================================
#==============================================================================
sm_sourceNames =  ['github','reddit', 'twitter']
sourceNames = ['exploit'] + sm_sourceNames


# split-population survival process: exploit

exploit_mk = {'base': WeibullMemoryKernel(0.8), \
              'github': ExponentialPseudoMemoryKernel(beta=1.0),\
              'reddit': ExponentialPseudoMemoryKernel(beta=1.0), \
              'twitter':PowerLawMemoryKernel(beta=1.0)}


exploit_process = SplitPopulationSurvival('exploit', sm_sourceNames, exploit_mk)

exploit_para = {'base'   : 0.01,  'github' : 0.02, \
                'reddit' : 0.005, 'twitter': 0.01}

exploit_process.setParams(exploit_para)

#-----------Social Media Processes---------------------------------------------

sm_mk = {'base'   : ConstantMemoryKernel(), \
         'github' : ExponentialPseudoMemoryKernel(beta = 1.0), \
         'reddit' : ExponentialPseudoMemoryKernel(beta = 1.0), \
         'twitter': ExponentialPseudoMemoryKernel(beta = 1.0)}  
    
#--------- GitHub----------
# class
github_process = HawkesTPP('github', sm_sourceNames, sm_mk) 

github_para = {'base'  :0.06,  'github': 0.6, \
               'reddit': 0.01, 'twitter': 0.09 }
github_process.setParams(github_para)

#--------- Reddit----------
# class
reddit_process = HawkesTPP('reddit', sm_sourceNames, sm_mk) 

reddit_para = {'base'  :0.02,  'github': 0.07, \
               'reddit': 0.6,  'twitter': 0.01}
reddit_process.setParams(reddit_para)

#--------- Twitter----------
# class
twitter_process = HawkesTPP('twitter', sm_sourceNames, sm_mk) 

twitter_para = {'base' : 0.1,  'github': 0.05, \
                'reddit': 0.2, 'twitter': 0.6}
twitter_process.setParams(twitter_para)

#------------------Multi-Variate Process---------------------------------
processes = [exploit_process, github_process, reddit_process, twitter_process]
mp = multi_variate_process(processes)
weight_vector = np.array([ 3.00674769, 0.12732851, -1.45712857])

#==============================================================================
#==========================Load Real World Data===========================================
#==============================================================================


# dataFile = '../data/test_realizations.npy'
# listOfList_data =  np.load(dataFile, allow_pickle=True)

# feature_array = np.load('../data/test_feature_vectors.npy', allow_pickle=True)
# feature_array = np.concatenate((feature_array, np.ones((len(feature_array),1))), axis=1)
# TPPdata = listOfList2json_withFeature(listOfList_data, feature_array, sourceNames)

#==============================================================================
#==========================Load Synthetic Data===========================================
#==============================================================================

synthetic_realizations, synthetic_features = generateSyntheticDataset()
listOfList_data =  synthetic_realizations


TPPdata = listOfList2json_withFeature(synthetic_realizations, synthetic_features, sourceNames)


#==============================================================================
#==========================Simulate============================================
#==============================================================================
# tc_list =  np.array( [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300])
tc_list =  np.array([5/24, 10/24, 20/24])
delta_t_list =  np.array([5/24, 10/24, 20/24])
    

N = 500

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulation_per_realization(complete_realization,scenario):

    realization_results = {}
    cid = complete_realization['cascade_id']
    realization_results['cid'] = cid
    # exploit_time = realization_results['exploit']
    start_time = time.time()

    for tc_h in tc_list:
        tc = tc_h * 24 # change to hourly granularity
        
        # case 1: exploited by tc
        if (len(complete_realization['exploit'])>0) and (complete_realization['exploit'][0] < tc):
            realization_results['tc_{}'.format(tc_h)] = None
            continue
        # case 2: stop observation by tc 
        if tc >= complete_realization['right_censored_time']:
            realization_results['tc_{}'.format(tc_h)] = None
            continue
        # empty list for results
        tc_results_list = []
        susceptible_prob =  sigmoid(np.dot(complete_realization['feature_vector'] , weight_vector))
        
        
        # modify realization with tc
        realization_modify = complete_realization.copy()
        for s in sourceNames:
            temp = np.array(complete_realization[s]) 
            realization_modify[s] = list(temp[temp<=tc])
            if len(realization_modify[s])==0:
                realization_modify.pop(s)
        
        # loop through trials
        for k in np.arange(N):
            realization = realization_modify.copy()
            # get suceptibility value

            susceptible =  np.random.binomial(1, susceptible_prob)
            
            if susceptible == 1:
                realization['susceptible'] = int(1)
                simulated_realization = mp.ogata_thinning_simulation(realization, t_obs = tc, start_time = start_time)
                
                
                if simulated_realization is None:
                    return None
                if 'exploit' in simulated_realization.keys():
                    tc_results_list.append(simulated_realization['exploit'][0]) 
                    
            

        realization_results['tc_{}'.format(tc_h)] = tc_results_list

    with open('backup'+scenario+'/cve_test_{}.json'.format(cid), 'w') as f:
        json.dump(realization_results, f)
        
    return realization_results

def simulate_dask(scenario):

    from os import listdir
    from os.path import isfile, join
    path = 'backup'+ scenario +'/'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    existing_indices = [x.split('_')[-1].split('.')[0] for x in files]
    # remaining_indices = np.array(remaining_indices)
    results_realization_list = []
    client = Client()
   
    for complete_realization in TPPdata:
        cid = int(complete_realization['cascade_id'])
        if (str(cid) not in existing_indices):
            results_realization =  client.submit(simulation_per_realization, complete_realization, scenario)
            results_realization_list.append(results_realization)
    
    all_results = client.gather(results_realization_list)
    
    return all_results

def analysis(scenario):
    #==============================================================================
    #==========================Analysis============================================
    #==============================================================================
        
    files = glob.glob('backup'+scenario+'/*.json')

    all_results = []
    for file in files:
        with open(file, 'r') as f:
            result = json.load(f)
            
        all_results.append(result)
        

    all_pairs = [(tc, delta_t) for tc in tc_list for delta_t in delta_t_list]

    prc_temp = []

    for cid_result in all_results:
        cid_r = {}
        
        cid = int(cid_result['cid'])
        gtemp = listOfList_data[cid][0] 
        exploit_time = gtemp[0] if len(gtemp) > 1 else np.nan

        right_censored_time = gtemp[-1]
        cid_r['right_censored_time'] = right_censored_time / 24.00
        cid_r['exploit_time'] = exploit_time / 24.00 if exploit_time is not np.nan else np.nan
        cid_r['idx'] = cid
        for tc in tc_list:
            tc_sim = cid_result['tc_{}'.format(tc)]
            # case 1: if (not exploited) and (continuing observation) by tc
            if (tc_sim is not None) and (right_censored_time > tc*24): 
                tc_sim = np.array(tc_sim)
                
                for delta_t in delta_t_list:
                    gt = 1 if( exploit_time ) and (exploit_time <= (tc+delta_t)*24.00 ) else 0
                    # case 2: if (right_censored_time > tc+delta_t) OR exploited by tc+delta_t
                    if (right_censored_time >= (tc+delta_t)*24) or (gt==1):
                    
                        exploits_prob = len(tc_sim[tc_sim < (tc+delta_t)*24.00]) / N
                    
                        cid_r[(tc, delta_t)] = (exploits_prob, gt)
                
        prc_temp.append(cid_r)
        
    prc_results = {}

    for tc in tc_list:
        for delta_t in delta_t_list:
            
            prob = []
            gt = []
            idx = []
            
            for r in prc_temp:
                if (tc, delta_t) in r:
                    prob.append(r[(tc, delta_t)][0])
                    gt.append(r[(tc, delta_t)][1])
                    idx.append(r['idx'])
        
            prc_results[(tc, delta_t)] = [prob, gt, idx]
        
    np.save('temporary_simulation_results_synthetic.npy', prc_results)
        
        


if __name__ == "__main__":
            
    scenario="synthetic+SM_updated_t_c"
    # analysis(scenario)
    all_results = simulate_dask(scenario=scenario)
    
    # with open('simulated_exploit_time'+scenario+'.json', 'w') as f:
    #     json.dump(all_results, f)
