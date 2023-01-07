from utils.MemoryKernel import ConstantMemoryKernel, RayleighMemoryKernel, PowerLawMemoryKernel, \
                                GammaGompertzMemoryKernel, WeibullMemoryKernel,ExponentialPseudoMemoryKernel,GompertzMemoryKernel
from utils.DataReader import createDummyDataSplitPopulation

from utils.Simulation import simulation
from point_processes.PointProcessCollection import PoissonTPP
import numpy as np
import scipy.stats
from utils.GoodnessOfFit import KSgoodnessOfFitExp1, KSgoodnessOfFitExp1MV
from matplotlib import pyplot as plt
from core.Logger import getLoggersMultivariateProcess
from core.DataStream import DataStream
from point_processes.PointProcessCollection import TrainingStatus
import pandas as pd
import time
from scipy import linalg
import MVSPP_config
from numba import jit

####### Dask related import for parallel processing ###############

import dask.array as da
import dask
from dask.distributed import Client
import datetime


def getCascades(realization_filename, scenario):
    realizations = np.load(realization_filename, allow_pickle = True).item()
    # generate cascade data from all three platforms per CVE
    
    import datetime
    if scenario == 'github' or scenario == 'twitter':
        rc_df = pd.DataFrame({'year': [2018], 'month': [3], 'day': [31]})
        rightCensoringTime = pd.to_datetime(rc_df, unit='ns')
    elif scenario == 'lastfm' or scenario == 'irvine':
        rc = realizations['rightCensoring']
        rightCensoringTime = pd.to_datetime(rc, unit='ns')
        del realizations['rightCensoring']
    elif scenario  == 'twitter_link':
        rc = realizations['rightCensoring']
        rightCensoringTime = rc
        del realizations['rightCensoring']
        

    timestamp_list = []
    node_ids_list = []

    print("Converting realizations into timestamp and node lists...")
    for index, realization_id in enumerate(realizations):
        information_cascade = realizations[realization_id]
        timestamps = list(information_cascade['timestamps'])
        node_ids = list(information_cascade['timestamp_ids'])
        if scenario == 'github' or scenario == 'twitter':
            relative_rc_time = ((rightCensoringTime - timestamps[0])/np.timedelta64(1, 'h')).values[0]
        elif scenario == 'lastfm' or scenario == 'irvine':
            relative_rc_time = ((rightCensoringTime - timestamps[0])/np.timedelta64(1, 'h'))
        elif scenario == 'twitter_link':
            relative_rc_time = ((rightCensoringTime - timestamps[0])/3600)

        if scenario != 'twitter_link':
            timestamps = [((x-timestamps[0])/np.timedelta64(1,'h')) for x in timestamps]
        else: 
            timestamps = [((x-timestamps[0])/3600) for x in timestamps]
        
        timestamps.append(relative_rc_time)
        node_ids.append(-1)

        
        timestamp_list.append(np.array(timestamps))
        node_ids_list.append(np.array(node_ids))


        if index % 100 == 0:
            print("Processed ", index, " Out of ", len(realizations), end='\r')
    
    return timestamp_list, node_ids_list

def predict_top_k(num_users, MemoryKernel, scenario, run_name, start_sizes):
    # get feature vectors of remaining node ids
    if scenario == "irvine" or scenario == "lastfm":
        feature_vectors = np.zeros(num_users)
    else:
        feature_vectors = np.load('../data/KDD_data/'+scenario+'_user_features.npy', allow_pickle=True)
        feature_vectors /= np.max(feature_vectors, axis=0)
        
    timestamp_list, node_ids_list = getCascades(realization_filename='../data/KDD_data/test_'+scenario+'.npy', scenario=scenario)
    w_tilde = np.load('../precalc/parameters_'+scenario+run_name+'/best/w_tilde.npy', allow_pickle=True)
    # w_tilde = np.array([ -6.47734745,  -6.58854828,-15.01558793])
    # get vector of probabilities, for this iterate through every outstanding node and find probabilities
    p_list = []
    mSLE = np.zeros(len(start_sizes))
    MSE = np.zeros(len(start_sizes))
    total_predictions = np.zeros(len(start_sizes))
    for i in range(len(timestamp_list)):
        if i % 5 == 0:
            print("processing cascade ", i, " out of ", len(timestamp_list))
        for start_size_index, start_size in enumerate(start_sizes):
            if len(np.unique(timestamp_list[i])) - 1 <= start_size :
                continue 
            total_predictions[start_size_index] += 1
            t_c = sorted(np.unique((timestamp_list[i])))[start_size]
            delta_t = timestamp_list[i][-1]

            timestamps = timestamp_list[i]
            node_ids = np.array(node_ids_list[i])
            (req_indices ,)= np.where(timestamps < t_c)
            prediction_timestamps = timestamps[req_indices]
            prediction_node_ids = node_ids[req_indices]
            # print(prediction_timestamps)
            # print(timestamp_list[i])
            # input()
            
            remaining_user_ids = np.arange(num_users)
            remaining_user_ids = np.array([x for x in remaining_user_ids if x not in prediction_node_ids])
            
            pVector = []
            import math
            for node in remaining_user_ids:
                alpha_i = np.load('../precalc/parameters_'+scenario+run_name+'/'+str(node)+'.npy', allow_pickle=True)
                feature_vector_node = feature_vectors[node]
                if not isinstance(feature_vector_node, np.ndarray):
                    if feature_vector_node== 0:
                        feature_vector_node = np.array([1])
                else:
                    feature_vector_node = np.append(feature_vector_node, 1.0)
                pi_x_w = 1.0/(1.0 + np.exp(-np.dot(feature_vector_node, w_tilde)))
                psi_tc = 0.0
                psi_tc_delta_t = 0.0
                
                for event, event_node_id in zip(prediction_timestamps, prediction_node_ids):
                    psi_tc += alpha_i[event_node_id] * MemoryKernel.psi(t_c - event)
                    psi_tc_delta_t += alpha_i[event_node_id] * MemoryKernel.psi(t_c + delta_t - event)
                    

                prob = pi_x_w * (np.exp(-psi_tc) - np.exp(-psi_tc_delta_t))/\
                    (1.0 - pi_x_w + pi_x_w*np.exp(-psi_tc))
                # print(pi_x_w*(np.exp(-psi_tc) - np.exp(-psi_tc_delta_t)))
                # print((1.0 - pi_x_w + pi_x_w*np.exp(-psi_tc)))
                # input()
                pVector.append(prob)/v
            pVector = np.array(pVector)
            PMF = np.array([0.0, 1.0])
            for p in pVector:
                seq = np.array([1.0-p, p])
                PMF = np.convolve(PMF, seq)

            prediction_order = np.flip(np.argsort(pVector))
            predicted_users = remaining_user_ids[prediction_order]
            predicted_users_probs = pVector[prediction_order]


            true_remaining_node_ids = np.array([x for x in node_ids if x not in prediction_node_ids])
            predicted_indices = []
            for n in true_remaining_node_ids[:-1]:
                predicted_indices.append(list(remaining_user_ids[prediction_order]).index(n))
            print(predicted_indices)

            print(predicted_users_probs[predicted_indices])
            # plt.hist(predicted_users_probs, bins = 200)
            # plt.show()
            
            input()
            c_h = len(prediction_timestamps)
            
            predicted_count = 0.0
            for j in range(num_users-c_h):
                predicted_count += np.log(c_h +j)*PMF[j]
            
            mSLE[start_size_index] += (predicted_count - np.log(len(timestamps) - 1))**2
    
    mSLE = np.divide(mSLE,total_predictions)
    print("Mean Squared Log Error for event count based prediction ",start_sizes, " : " ,mSLE)
    print("Total predictions: ", total_predictions)
    input()

def most_popular_user(scenario):
    training_filename = '../data/KDD_data/training_'+str(scenario)+'.npy'
    test_filename = '../data/KDD_data/test_'+str(scenario)+'.npy'
    
    training_features_filename = '../data/KDD_data/'+str(scenario)+'_user_features.npy'
    
    training_realizations = np.load(training_filename, allow_pickle=True).item()
    test_realizations = np.load(test_filename, allow_pickle=True).item()

    del training_realizations['rightCensoring']
    del test_realizations['rightCensoring']

    training_node_list = []
    training_time_list = []
    test_node_list = []
    
    import collections

    for realization in  training_realizations.keys():
        training_node_list.extend(training_realizations[realization]['timestamp_ids'])
        try:
            arg_node = list(training_realizations[realization]['timestamp_ids']).index(291)
            training_time_list.append((training_realizations[realization]['timestamps'][arg_node] -training_realizations[realization]['timestamps'][0])/np.timedelta64(1,'h'))
        except:
            pass

    plt.hist(training_time_list)
    plt.show()

    for realization in  test_realizations.keys():
        test_node_list.extend(test_realizations[realization]['timestamp_ids'])
        arg_node = np.where(test_realizations[realization]['timestamp_ids'] == 291)[0]
        training_time_list.append(test_realizations[realization]['timestamps'][arg_node])

    
    training_counter = collections.Counter(training_node_list)
    test_counter = collections.Counter(test_node_list)
    print(training_counter[860])
    # print(test_counter)
    input()
    
        
    



def main():
    scenario = MVSPP_config.scenario
    MemoryKernel = MVSPP_config.memory_kernel
    run_name = MVSPP_config.run_name
    start_sizes = [10, 20]
    #  read github user feature data
    github_user_features = np.load('../data/KDD_data/github_user_features.npy' , allow_pickle = True)
    twitter_user_features = np.load('../data/KDD_data/twitter_user_features.npy', allow_pickle = True)
    twitter_link_user_features = np.load('../data/KDD_data/twitter_link_user_features.npy', allow_pickle = True)
    
    
    if scenario == 'twitter':
        feature_vector_length = len(twitter_user_features[0])
        num_users = len(twitter_user_features)
        # w_tilde = np.array([0.00250413, 0.00250144, 0.00243388, 0.00250184, 0.00510143]) # power law kernel
    elif scenario== 'github':
        num_users = len(github_user_features)
    elif scenario == 'irvine':
        num_users = 893
    elif scenario == 'lastfm':
        num_users = 1000
    elif scenario == 'twitter_link':
        num_users = len(twitter_link_user_features)
    

    most_popular_user(scenario)
    predict_top_k(num_users, MemoryKernel, scenario, run_name, start_sizes)
    
    pass

if __name__ == "__main__":
    main()