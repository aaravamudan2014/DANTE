#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 04:13:38 2020

@author: xi zhang
"""

import json
import pandas as pd
import numpy as np
import random
import os
from utils.DataReader import *

    
def listOfList2json(oriFilePath, \
                    sourceNames=['exploit', 'github', 'reddit' ,'twitter'], \
                    time_granularity='D'):
    
    TPPdata = np.load(oriFilePath,allow_pickle=True)
    
    if time_granularity == 'D':
        tg_value = 24

    
    TPPdata_json = []
    for k, realization in enumerate(TPPdata):
        obj ={}
        obj['cid'] = k
        obj['right_censored_time'] = realization[0][-1] / tg_value
        
        for s, events in enumerate(realization):
            ev_a = np.sort(np.array(events[:-1]) / tg_value)
            ev =  ev_a[ev_a>0]
            if ev.size > 0:
                obj[sourceNames[s]] = list(ev)

        TPPdata_json.append(obj)
    
    return TPPdata_json
        
def get_index(oriFilePath, obs_gap=30, min_realization_number=10):
    
    TPPdata = listOfList2json(oriFilePath)
    
    summary_df = pd.DataFrame(TPPdata)
    summary_df = summary_df[['cid','right_censored_time','exploit']]
    summary_df['exploit'] = summary_df['exploit'].apply(lambda x: x[0] if x is not np.nan else np.nan)

    
    exploited = summary_df.dropna()
    noExploited = summary_df[summary_df.exploit.isnull()]
    
    maxTc = int(summary_df.right_censored_time.max())

    index_list = {}
    for tc in np.arange(0, maxTc, obs_gap):
        for delta_t in np.arange(0, maxTc, obs_gap):
    
            exploited_available = exploited.loc[(exploited['exploit'] > tc) ]
                
            noExploited_available = noExploited.loc[noExploited['right_censored_time'] > tc + delta_t]
            
            realization_list = exploited_available.cid.tolist() + \
                               noExploited_available.cid.tolist()
            
            if len(realization_list) > min_realization_number:
                index_list[(tc, delta_t)] = realization_list
                
    return index_list





def saveIndexSet():
    dset = ['test', 'training', 'validation' ]
    for val in dset:
        oriFilePath = '../data/{}_realizations.npy'.format(val)
        index_list = get_index(oriFilePath, obs_gap=30, min_realization_number=30)
        np.save('../data/{}_index_list.npy'.format(val),index_list)
    print("All index sets have been generated")


if __name__ == "__main__":
    saveIndexSet()











    