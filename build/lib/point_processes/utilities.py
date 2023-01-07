# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 19:22:34 2021

@author: cczha
"""

import numpy as np

def json2ListOfList(data_json, sourceNames):
    
    data_list = []       
    feature_vector_list = []
    for realization in data_json:
        
        new_list = []
        
        T = float(realization['right_censored_time'])
        feature_vector = realization['feature_vector']

        for s in sourceNames:
            new_list.append( realization[s]+ [T] ) if s in realization else new_list.append([T])       
            
        data_list.append(new_list)
        feature_vector_list.append(feature_vector)
        
    return data_list, feature_vector_list

def listOfList2json(listOfList_data, sourceNames):
    
    TPPdata = []
    source_num = len(listOfList_data[0])
    for k, dl in enumerate(listOfList_data):
        realization ={}
        realization['cascade_id'] = k #'c_{}'.format(k)
        
        #-----------exploit----------------------------------------------------        
        # exculde negative/zero day exploit
        if (len(dl[0])==1) or ( (len(dl[0])==2) and (dl[0][0]>0)) :

            realization['right_censored_time'] = dl[0][-1] 

            for k in range(source_num):
                ds = np.sort(dl[k][:-1])
                realization[sourceNames[k]] = ds[ds>0]
            
            TPPdata.append(realization)

    return TPPdata


def listOfList2json_withFeature(listOfList_data, feature_array, sourceNames):
    
    TPPdata = []
    source_num = len(listOfList_data[0])
    for k, dl in enumerate(listOfList_data):
        realization ={}
        realization['cascade_id'] = k #'c_{}'.format(k)
        realization['feature_vector'] = feature_array[k,:]
        
        #-----------exploit----------------------------------------------------        
        # exculde negative/zero day exploit
        if (len(dl[0])==1) or ( (len(dl[0])==2) and (dl[0][0]>0)) :

            realization['right_censored_time'] = dl[0][-1] 

            for k in range(source_num):
                ds = np.sort(dl[k][:-1])
                ds_valid = ds[ds>0]
                realization[sourceNames[k]] = ds_valid

                # if len(ds_valid) > 0:
                #     realization[sourceNames[k]] = ds_valid
            
            TPPdata.append(realization)

    return TPPdata