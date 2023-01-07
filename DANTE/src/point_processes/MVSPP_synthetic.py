'''
MultiVariateSurvivalSplitPopulation.py
    Multi variate split population survival process implementation
'''
from cgi import print_arguments
from utils.MemoryKernel import ConstantMemoryKernel, RayleighMemoryKernel, PowerLawMemoryKernel, \
                                GammaGompertzMemoryKernel, WeibullMemoryKernel,ExponentialPseudoMemoryKernel,GompertzMemoryKernel
from utils.DataReader import createDummyDataSplitPopulation

from utils.Simulation import *

from utils.GoodnessOfFit import *

from point_processes.PointProcessCollection import PoissonTPP
import numpy as np
import scipy.stats
from utils.GoodnessOfFit import KSgoodnessOfFitExp1, KSgoodnessOfFitExp1MV
from utils.FeatureVectorCreator import *
import matplotlib
from matplotlib import pyplot as plt
from core.Logger import getLoggersMultivariateProcess
from core.DataStream import DataStream
from point_processes.PointProcessCollection import TrainingStatus
from point_processes.bce import *
import pandas as pd
import time
from scipy import linalg
import MVSPP_config
from numba import jit

matplotlib.use("TKAgg")

####### Dask related import for parallel processing ###############

import dask.array as da
import dask
from dask.distributed import Client
import datetime



# L1 regularization term
def L1regularizer(alpha, nu):
    return nu * np.linalg.norm(alpha,ord = 1 )
import warnings
warnings.filterwarnings("ignore")

# Function to calculate gradients w.r.t alpha and w_tilde
def nll_i_gradients(phi_T,phi_t_i, psi_T, psi_t_i, x_i, alpha_i, w_s, l, rho, q):
    nll_grad_alpha_i = +psi_T*np.exp(-np.dot(alpha_i,psi_T))/\
                    (np.exp(-np.dot(alpha_i, psi_T)) + np.exp(-np.dot(x_i, w_s)))
    nll_grad_w_s = x_i*np.exp(-np.dot(x_i, w_s))/\
                    (np.exp(-np.dot(x_i, w_s)) + np.exp(-np.dot(alpha_i, psi_T))) -\
                    x_i*np.exp(-np.dot(x_i, w_s))/\
                        (1+ np.exp(-np.dot(x_i, w_s)))
    if l == 1:

        nll_grad_alpha_i   += -psi_T*np.exp(-np.dot(alpha_i,psi_T))/(np.exp(-np.dot(alpha_i, psi_T)) + np.exp(-np.dot(x_i, w_s)))  - \
                                phi_t_i/np.dot(alpha_i,phi_t_i) + psi_t_i - \
                                psi_t_i* np.exp(-np.dot(alpha_i, psi_t_i))/(np.exp(-np.dot(alpha_i, psi_t_i)) + np.exp(-np.dot(x_i, w_s)))
        nll_grad_w_s += -x_i*np.exp(-np.dot(x_i, w_s))/(np.exp(-np.dot(alpha_i, psi_T)) + np.exp(-np.dot(x_i, w_s))) + \
                            x_i* np.exp(-np.dot(x_i,w_s)/(1+np.exp(-np.dot(x_i, w_s)))) -  \
                            x_i*np.exp(-np.dot(x_i, w_s))/(np.exp(-np.dot(x_i, w_s)) + np.exp(-np.dot(alpha_i,psi_t_i)))

        if np.isneginf(nll_grad_alpha_i[0]):
            print("debugging values start here..")
            print(alpha_i)
            print(phi_t_i)
            print(np.exp(-np.dot(alpha_i, psi_t_i)) + np.exp(-np.dot(x_i, w_s)))
            print("Infinite gradients found, error")
            input()
    # nll_grad_w_s += rho * (w_s - q)
    return np.array([nll_grad_alpha_i, nll_grad_w_s])

@jit
def nll_i (phi_T,phi_t_i, psi_T, psi_t_i, x_i, alpha_i, w_s, l,):
    nll = np.log(1 + np.exp(-np.dot(x_i,w_s))) -\
                np.log(np.exp(-np.dot(x_i, w_s)) + np.exp(-np.dot(alpha_i, psi_T)))
    
    if l == 1:
        nll += np.log(np.exp(-np.dot(x_i, w_s)) + np.exp(-np.dot(alpha_i, psi_T))) - \
                np.log(1 + np.exp(-np.dot(x_i, w_s))) - np.log(np.dot(alpha_i, phi_t_i)) + \
                np.dot(alpha_i, psi_t_i) + np.log(np.exp(-np.dot(x_i, w_s)) + np.exp(-np.dot(alpha_i, psi_t_i)))
    
    return nll

@jit
def augmented_nll_i(nll_i, alpha_i, w_s, rho,q, nu, gamma):
    augmented_nll_i = nll_i + 0.5*rho*np.linalg.norm(w_s - q)**2 + L1regularizer(alpha_i,gamma)
    return augmented_nll_i

@jit
def augmented_nll_i_majorizer(nll_i_prime,grad_alpha_i_prime,grad_w_s_prime, alpha_i,alpha_i_prime, w_s, w_s_prime, rho,q, nu, gamma):
    augmented_nll_i_prime = augmented_nll_i(nll_i_prime, grad_alpha_i_prime, w_s_prime, rho,q, nu, gamma)
    augmented_nll_i_majorizer = augmented_nll_i_prime - 0.5*nu*( np.linalg.norm(grad_w_s_prime)**2 + np.linalg.norm(alpha_i_prime)**2) +\
                                    0.5*(1/nu)*(np.linalg.norm(alpha_i-alpha_i_prime + nu*grad_alpha_i_prime)**2 + \
                                    np.linalg.norm(w_s-w_s_prime + nu*grad_w_s_prime)**2 ) 
    return augmented_nll_i_majorizer

# @jit
def param_update(alpha_i, w_s ,L, rho, q, grad_alpha_i, grad_w_s, gamma):
    w_s_new = w_s - 1/L*(grad_w_s + rho*(w_s - q))
    alpha_i_new = (alpha_i -1/L*(grad_alpha_i) - (gamma/L)).clip(min=0.00001)
    return np.array([alpha_i_new, w_s_new])

# @jit
def calculate_likelihood_quantities(precalc_dict, alpha, w, x_i_bias , node_id,rho, q):
    # compute required likelihood quantities
    nll = 0.0
    grad_w = np.zeros(len(w))
    grad_alpha = np.zeros(len(alpha))
    
    for cascade_id in precalc_dict.keys():
        l = precalc_dict[cascade_id]['l']
        phi_t_i = precalc_dict[cascade_id]['phi_ti']
        psi_t_i = precalc_dict[cascade_id]['psi_ti']
        phi_T = precalc_dict[cascade_id]['phi_T']
        psi_T = precalc_dict[cascade_id]['psi_T']
        participating_nodes = precalc_dict[cascade_id]['participating_nodes']
        
        if np.sum(phi_t_i) == 0:
            # in such a case the cascade does not contribute to the learning process, 
            # it will just produce a Nan
            continue
        
        if len(participating_nodes) > 0:
            nll_i_val = nll_i(phi_T,phi_t_i, psi_T, psi_t_i, x_i_bias, alpha[participating_nodes], w, l)
            nll += nll_i_val
            gradients = nll_i_gradients(phi_T,phi_t_i, psi_T, psi_t_i, x_i_bias, alpha[participating_nodes], w, l, rho, q)
            
            grad_alpha[participating_nodes] += gradients[0]
            grad_w += gradients[1]
    
    nll /= len(precalc_dict.keys())
    grad_alpha /= len(precalc_dict.keys())
    grad_w /= len(precalc_dict.keys())

    return nll,  grad_alpha, grad_w


def train_sub_problem_bpgd(scenario_name,run_name, node_id, num_users, x_i, w_s, rho, L0, beta_u, beta_d, q, gamma):
    # load precalc file for node
    precalc_dict = np.load('../precalc/'+scenario_name+run_name+'/node_'+ str(node_id) + '.npy',allow_pickle = True).item()

    # TODO: load the saved value of the parmeters (alpha) if it exists, else, initialize it
    alpha_i = np.load('../precalc/parameters_'+scenario_name+run_name+'/'+str(node_id)+'.npy', allow_pickle=True)

    # maximum trials for nesterovs accelerated proximal gradient descent
    maxIter = MVSPP_config.sub_problem_iterations

    # epsilon for early stopping of Nesterov's accelerated proximal gradient descent
    epsilon = MVSPP_config.epsilon

    if not isinstance(x_i, np.ndarray):
        if x_i== 0:
            x_i_bias = np.array([1])
    else:
        x_i_bias = np.append(x_i, 1)
    

    theta = np.array([alpha_i, w_s])
    nll_list= []
    bce_list = []
    for t in range(1, maxIter + 1):
        
        nll, grad_alpha, grad_w  = calculate_likelihood_quantities(precalc_dict, theta[0], theta[1], x_i_bias, node_id,rho, q)
        
        theta_previous = theta.copy()
        if t == 1:
            nll_list.append(nll)
        L = L0
        
        inner_iter = 0
        while True:
            inner_iter += 1
            theta_update = param_update(theta[0], theta[1] ,L, rho, q, grad_alpha, grad_w, gamma)
            nll_new, grad_alpha_new, grad_w_new  = calculate_likelihood_quantities(precalc_dict, theta_update[0], theta_update[1], x_i_bias, node_id,rho, q)
            L = beta_u * L
            if MVSPP_config.run_test_sub_problem:
                print(inner_iter,nll, nll_new)
                
            if nll_new < nll or inner_iter == 20:
                theta = theta_update
                break
        # print("outer iteration: ", t)
        norm = np.sqrt(np.linalg.norm( theta_previous[0]- theta[0])**2 + np.linalg.norm( theta_previous[1]- theta[1])**2)
        # bce = calculate_bce_loss(node_id, x_i_bias,theta[1], theta[0], t_c=10000, delta_t=25000, threshold = 0.5)
        # print("iteration",t, "nll: ",nll, "norm: ",norm)
        
        nll_list.append(nll)
        
        if norm < epsilon:
            break
        


    # fig, ax = plt.subplots()
    # ax.plot(nll_list, color='red')
    # ax.set_xlabel('iterations')
    # ax.set_ylabel('nll', color='red')
    # ax2 = ax.twinx()
    # ax2.plot(bce_list, color='blue')
    # ax2.set_ylabel('bce', color='blue')
    # plt.savefig('figures/'+str(node_id)+'.png')
    
    # save the alpha vector back to a file since returning it takes up a lot of memory
    with open('../precalc/parameters_'+scenario_name+run_name+'/'+str(node_id)+'.npy', 'wb') as f:
        np.save(f, theta[0])
    
    # just return w_tilde vector since the alpha values are already saved 
    return theta[1]

def train_sub_problem(scenario_name,run_name, node_id, num_users, x_i, w_s, rho, L0, beta_u, beta_d, q, gamma):
    
    # load precalc file for node
    precalc_dict = np.load('../precalc/'+scenario_name+run_name+'/node_'+ str(node_id) + '.npy',allow_pickle = True).item()

    # TODO: load the saved value of the parmeters (alpha) if it exists, else, initialize it
    alpha_i = np.load('../precalc/parameters_'+scenario_name+run_name+'/'+str(node_id)+'.npy', allow_pickle=True)

    # maximum trials for nesterovs accelerated proximal gradient descent
    maxIter = MVSPP_config.sub_problem_iterations

    # epsilon for early stopping of Nesterov's accelerated proximal gradient descent
    epsilon = MVSPP_config.epsilon

    if not isinstance(x_i, np.ndarray):
        if x_i== 0:
            x_i_bias = np.array([1])
    else:
        x_i_bias = np.append(x_i, 1)

    theta_previous = np.array([alpha_i, w_s])
    theta = np.array([alpha_i, w_s])
    theta_previous_previous = np.array([alpha_i, w_s])
    theta_tilde = np.array([alpha_i, w_s])
    tau = 1.0
    tau_previous = 0.0
    L = L0
    L_previous = L0
    nll_list= []
    nll, grad_alpha, grad_w  = calculate_likelihood_quantities(precalc_dict, theta_tilde[0], theta_tilde[1], x_i_bias, node_id,rho, q)
    F_list = [augmented_nll_i(nll, theta[0], theta[1],rho, q, 1/L, gamma)]
    for t in range(1, maxIter + 1):
        theta_previous_previous = theta_previous
        theta_previous = theta
        nll, grad_alpha, grad_w  = calculate_likelihood_quantities(precalc_dict, theta_tilde[0], theta_tilde[1], x_i_bias, node_id,rho, q)
        theta = param_update(theta_tilde[0], theta_tilde[1] ,L, rho, q, grad_alpha, grad_w, gamma)
        
        # if t == 1:
            # bce = calculate_bce_loss(node_id, x_i_bias,theta[1], theta[0], t_c=10000, delta_t=25000, threshold = 0.5)
            # nll_list.append(nll)
        
        inner_iterations = 1
        while True:
            nll, grad_alpha, grad_w  = calculate_likelihood_quantities(precalc_dict, theta[0], theta[1], x_i_bias, node_id,rho, q)
            # if inner_iterations == 1:
            #     nll_list.append(nll)
            # param list for augmented_nll_i: nll_i, alpha_i, w_s, rho, q, nu, gamma
            F = augmented_nll_i(nll, theta[0], theta[1],rho, q, 1/L, gamma)
            # param list for calculate_likelihood_quantities : (precalc_dict, alpha, w, x_i_bias, node_id, rho, q)
            nll_prime, grad_alpha_prime, grad_w_prime  = calculate_likelihood_quantities(precalc_dict, theta_tilde[0], theta_tilde[1], x_i_bias, node_id,rho, q)
            # param list for augmented_nll_i_majorizer:(nll_i_prime, grad_alpha_i_prime, grad_w_s_prime, alpha_i, alpha_i_prime, w_s, w_s_prime, rho, q, nu, gamma)
            Q = augmented_nll_i_majorizer(nll_prime,grad_alpha_prime, grad_w_prime,theta[0], theta_tilde[0], theta[1], theta_tilde[1], rho, q, 1/L, gamma)
            print("inner iterations nll: ", F, Q)
            
            if F <= Q:
                break
            
            L = beta_u * L
            tau = 0.5* (1.0 + np.sqrt(1.0 + 4.0 * (L/L_previous)* tau_previous**2))
            
            theta_tilde_alpha = theta_previous[0] + ((tau_previous - 1.0)/tau) * (theta_previous[0] - theta_previous_previous[0])
            theta_tilde_w = theta_previous[1] + ((tau_previous - 1.0)/tau) * (theta_previous[1] - theta_previous_previous[1])
            theta_tilde = np.array([theta_tilde_alpha, theta_tilde_w])
            # param list for param_update: (alpha_i, w_s, L, rho, q, grad_alpha_i, grad_w_s, gamma)
            theta = param_update(theta_tilde[0], theta_tilde[1] ,L , rho, q, grad_alpha, grad_w,gamma)
            inner_iterations += 1
            
        F = augmented_nll_i(nll, theta[0], theta[1],rho, q, 1/L, gamma)
        F_list.append(F)
        
        
        nll, grad_alpha, grad_w = calculate_likelihood_quantities(precalc_dict, theta[0], theta[1], x_i_bias, node_id,rho, q)
        
        # param list for param_update: (alpha_i, w_s, L, rho, q, grad_alpha_i, grad_w_s, gamma)
        theta_next = param_update(theta[0], theta[1] ,L, rho, q, grad_alpha, grad_w,gamma)
        
        # nll, _, _ = calculate_likelihood_quantities(precalc_dict, theta_next[0], theta_next[1], x_i_bias, node_id,rho, q)
        norm = np.sqrt(np.linalg.norm(theta_next[0]- theta[0])**2 + np.linalg.norm( theta_next[1]- theta[1])**2)
        
        # print("iteration",t, "nll: ",nll, "norm: ",norm*L)

        if norm < epsilon/L:
            break
        L_previous = L
        L = L_previous/beta_d
        tau_previous = tau
        tau = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * (L/L_previous) * tau_previous**2))
        theta_tilde = theta +((tau_previous - 1.0)/tau) * (theta - theta_previous)



    # fig, ax = plt.subplots()
    # ax.plot(F_list, color='red')
    # ax.set_xlabel('iterations')
    # ax.set_ylabel('F (smooth + non smooth func)', color='red')
    # # ax2 = ax.twinx()
    # # ax2.plot(bce_list, color='blue')
    # # ax2.set_ylabel('bce', color='blue')
    # plt.savefig('figures/'+str(node_id)+'.png')
    

    
    # save the alpha vector back to a file since returning it takes up a lot of memory
    with open('../precalc/parameters_'+scenario_name+run_name+'/'+str(node_id)+'.npy', 'wb') as f:
        np.save(f, theta[0])
    
    # just return w_tilde vector since the alpha values are already saved 
    return theta[1]    
    


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
        elif scenario == "synthetic":
            relative_rc_time = realizations[realization_id]['rightCensoring']
        
                
        if scenario == 'github' or scenario == 'lastfm'or scenario == 'twitter'or scenario == 'irvine':
            timestamps = [((x-timestamps[0])/np.timedelta64(1,'h')) for x in timestamps]
        elif scenario == "digg" or scenario == "synthetic":
            timestamps = [x-timestamps[0] for x in timestamps]
        else: 
            timestamps = [((x-timestamps[0])/3600) for x in timestamps]
        
        timestamps.append(relative_rc_time)
        node_ids.append(-1)

        
        timestamp_list.append(np.array(timestamps))
        node_ids_list.append(np.array(node_ids))


        if index % 100 == 0:
            print("Processed ", index, " Out of ", len(realizations), end='\r')
    
    return timestamp_list, node_ids_list

def predict_cascade_size_tc(MSPSP, num_users, MemoryKernel, scenario, run_name, t_c_list,dataset='test', mode='discriminative'):
    
    # get feature vectors of remaining node ids
    if scenario == "irvine" or scenario == "lastfm" or scenario == "digg":
        feature_vectors = np.zeros(num_users)
    else:
        feature_vectors = np.load('../data/KDD_data/'+scenario+'_user_features.npy', allow_pickle=True)
        #feature_vectors /= np.max(feature_vectors, axis=0)
    timestamp_list, node_ids_list = getCascades(realization_filename='../data/KDD_data/'+dataset+'_'+scenario+'.npy', scenario=scenario)
    
    # get vector of probabilities, for this iterate through every outstanding node and find probabilities
    mSLE = np.zeros(len(t_c_list))
    mSLE_data_points = {}
    
    for t_c_index, t_c in enumerate(t_c_list):
        mSLE_data_points[t_c] = np.array([])
    total_predictions = np.zeros(len(t_c_list))
    if mode == 'discriminative':
        w_tilde = np.load('../precalc/parameters_'+scenario+run_name+'/best/w_tilde.npy', allow_pickle=True)
        alpha = np.zeros((num_users, num_users))
        for i in range(num_users):
            alpha[i] = np.load('../precalc/parameters_'+scenario+run_name+'/best/'+str(i)+'.npy', allow_pickle=True)
    else:
        alpha, w_tilde = MSPSP.getParams()
  
    
    for i in range(len(timestamp_list)):
        if i % 100 == 0:
            print("processing cascade ", i, " out of ", len(timestamp_list))
        
        for t_c_index, t_c in enumerate(t_c_list):
            delta_t = timestamp_list[i][-1] - t_c
            timestamps = timestamp_list[i]
            node_ids = np.array(node_ids_list[i])
            (req_indices ,) = np.where(timestamps[:-1] < t_c)
            prediction_timestamps = timestamps[req_indices]
            prediction_node_ids = node_ids[req_indices]
            # if len(prediction_timestamps) ==1:
            #     continue 
            total_predictions[t_c_index] += 1
            remaining_user_ids = np.arange(num_users)
            remaining_user_ids = np.array([x for x in remaining_user_ids if x not in prediction_node_ids])
            pVector = []
            for node in remaining_user_ids:
                alpha_i = alpha[node]
                # np.load('../precalc/parameters_'+scenario+run_name+'/best/'+str(node)+'.npy', allow_pickle=True)
                # if np.sum(alpha_i[prediction_node_ids]) == 0:
                #     continue
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
                pVector.append(prob)
            pVector = np.array(pVector)
            PMF = np.array([0.0, 1.0])

            for p in pVector:
                seq = np.array([1.0-p, p])
                PMF = np.convolve(PMF, seq)
            
            c_h = len(prediction_timestamps)
        
            predicted_count = 0.0
            for k in range(num_users-c_h):
                predicted_count += np.log(k +c_h)*PMF[k]

            predicted_count = int(np.exp(predicted_count))  
            mSLE[t_c_index] += (np.log(predicted_count) - np.log(len(timestamps[:-1])))**2
            mSLE_data_points[t_c] = np.append(mSLE_data_points[t_c], (np.log(predicted_count)- np.log(len(timestamps[:-1])))**2)
    
    for i in range(len(t_c_list)):
        mSLE[i] /=total_predictions[i]
    #print("Mean Squared Log Error for event count based prediction ",start_sizes, " : " ,mSLE)
    print("Total predictions: ", total_predictions)
    
    return mSLE_data_points

def predict_cascade_size_fixed_tc_varying_deltat(MSPSP, num_users, MemoryKernel, scenario, run_name, t_c, deltat_list,dataset='test', mode='discriminative'):
    
    # get feature vectors of remaining node ids
    if scenario == "irvine" or scenario == "lastfm" or scenario == "digg":
        feature_vectors = np.zeros(num_users)
    else:
        feature_vectors = np.load('../data/KDD_data/'+scenario+'_user_features.npy', allow_pickle=True)
        #feature_vectors /= np.max(feature_vectors, axis=0)
    timestamp_list, node_ids_list = getCascades(realization_filename='../data/KDD_data/'+dataset+'_'+scenario+'.npy', scenario=scenario)
    
    # get vector of probabilities, for this iterate through every outstanding node and find probabilities
    mSLE = np.zeros(len(deltat_list))
    mSLE_data_points = {}
    
    for delta_t_index, delta_t in enumerate(deltat_list):
        mSLE_data_points[delta_t] = np.array([])
    total_predictions = np.zeros(len(deltat_list))
    if mode == 'discriminative':
        w_tilde = np.load('../precalc/parameters_'+scenario+run_name+'/best/w_tilde.npy', allow_pickle=True)
        alpha = np.zeros((num_users, num_users))
        for i in range(num_users):
            alpha[i] = np.load('../precalc/parameters_'+scenario+run_name+'/best/'+str(i)+'.npy', allow_pickle=True)
    else:
        alpha, w_tilde = MSPSP.getParams()
  
    
    for i in range(len(timestamp_list)):
        if i % 100 == 0:
            print("processing cascade ", i, " out of ", len(timestamp_list))
        
        for delta_t_index, delta_t in enumerate(deltat_list):
            timestamps = timestamp_list[i]
            node_ids = np.array(node_ids_list[i])
            (req_indices ,) = np.where(timestamps[:-1] < t_c)
            prediction_timestamps = timestamps[req_indices]
            prediction_node_ids = node_ids[req_indices]
            # if len(prediction_timestamps) ==1:
            #     continue 
            total_predictions[delta_t_index] += 1
            remaining_user_ids = np.arange(num_users)
            remaining_user_ids = np.array([x for x in remaining_user_ids if x not in prediction_node_ids])
            pVector = []
            for node in remaining_user_ids:
                alpha_i = alpha[node]
                # np.load('../precalc/parameters_'+scenario+run_name+'/best/'+str(node)+'.npy', allow_pickle=True)
                # if np.sum(alpha_i[prediction_node_ids]) == 0:
                #     continue
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
                pVector.append(prob)
            pVector = np.array(pVector)
            PMF = np.array([0.0, 1.0])

            for p in pVector:
                seq = np.array([1.0-p, p])
                PMF = np.convolve(PMF, seq)
            
            c_h = len(prediction_timestamps)
        
            predicted_count = 0.0
            for k in range(num_users-c_h):
                predicted_count += np.log(k +c_h)*PMF[k]

            (req_indices ,) = np.where(timestamps[:-1] < t_c+delta_t)
            size_timestamps = timestamps[req_indices]
            actual_count = len(size_timestamps)
            predicted_count = int(np.exp(predicted_count))  
            mSLE[delta_t_index] += (np.log(predicted_count) - np.log(actual_count))**2
            mSLE_data_points[delta_t] = np.append(mSLE_data_points[delta_t], (np.log(predicted_count)- np.log(actual_count))**2)
    
    print("Total predictions: ", total_predictions)
    
    return mSLE_data_points

def predict_cascade_size(MSPSP, num_users, MemoryKernel, scenario, run_name, start_sizes, dataset='test', mode='discriminative'):
    
    # get feature vectors of remaining node ids
    if scenario == "irvine" or scenario == "lastfm":
        feature_vectors = np.zeros(num_users)
    else:
        feature_vectors = np.load('../data/KDD_data/'+scenario+'_user_features.npy', allow_pickle=True)
        feature_vectors /= np.max(feature_vectors, axis=0)
        
    timestamp_list, node_ids_list = getCascades(realization_filename='../data/KDD_data/'+dataset+'_'+scenario+'.npy', scenario=scenario)
    # w_tilde = np.load('../precalc/parameters_'+scenario+run_name+'/best/w_tilde.npy', allow_pickle=True)
    
    # w_tilde = np.array([ 3.347])
    # get vector of probabilities, for this iterate through every outstanding node and find probabilities
    if mode == 'discriminative':
        w_tilde = np.load('../precalc/parameters_'+scenario+run_name+'/best/w_tilde.npy', allow_pickle=True)
        alpha = np.zeros((num_users, num_users))
        for i in range(num_users):
            alpha[i] = np.load('../precalc/parameters_'+scenario+run_name+'/best/'+str(i)+'.npy', allow_pickle=True)
    else:
        alpha, w_tilde = MSPSP.getParams()
    #w_tilde = np.load('../precalc/parameters_'+scenario+run_name+'/w_tilde.npy', allow_pickle=True)
    

    mSLE = np.zeros(len(start_sizes))
    
    
    mSLE_data_points = {}
    for start_size_index, start_size in enumerate(start_sizes):
        mSLE_data_points[start_size] = np.array([])
            
    total_predictions = np.zeros(len(start_sizes))
    for i in range(len(timestamp_list)):
        if i % 100 == 0:
            print("processing cascade ", i, " out of ", len(timestamp_list))
        for start_size_index, start_size in enumerate(start_sizes):
            if len(np.unique(timestamp_list[i])) - 1 <= start_size :
                continue 
            total_predictions[start_size_index] += 1
            t_c = sorted(np.unique((timestamp_list[i])))[start_size]
            delta_t = timestamp_list[i][-1] - t_c
            
            timestamps = timestamp_list[i]
            node_ids = np.array(node_ids_list[i])
            (req_indices ,)= np.where(timestamps < t_c)
            prediction_timestamps = timestamps[req_indices]
            prediction_node_ids = node_ids[req_indices]
            remaining_user_ids = np.arange(num_users)
            remaining_user_ids = np.array([x for x in remaining_user_ids if x not in prediction_node_ids])
            pVector = []
            for node in remaining_user_ids:
                alpha_i = alpha[node]
                # np.load('../precalc/parameters_'+scenario+run_name+'/best/'+str(node)+'.npy', allow_pickle=True)
                # if np.sum(alpha_i[prediction_node_ids]) == 0:
                #     continue
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
                pVector.append(prob)
            pVector = np.array(pVector)
            PMF = np.array([0.0, 1.0])

            for p in pVector:
                seq = np.array([1.0-p, p])
                PMF = np.convolve(PMF, seq)
            
            c_h = len(prediction_timestamps)
        
            predicted_count = 0.0
            for k in range(num_users-c_h):
                predicted_count += np.log(k +c_h)*PMF[k]

            predicted_count = int(np.exp(predicted_count))
            mSLE[start_size_index] += (np.log(predicted_count) - np.log(len(timestamps) - 1))**2
            mSLE_data_points[start_size] = np.append(mSLE_data_points[start_size], (np.log(predicted_count) - np.log(len(timestamps[:-1])))**2)
            
    mSLE = np.divide(mSLE,total_predictions)
    print("Mean Squared Log Error for event count based prediction ",start_sizes, " : " ,mSLE)
    print("Total predictions: ", total_predictions)
    
    return mSLE_data_points

def predict_cascade_size_fixed_delta_t_varying_t_c(MSPSP, num_users, MemoryKernel, scenario, run_name, delta_t, t_c_list,dataset='test', mode='discriminative'):
    
    # get feature vectors of remaining node ids
    if scenario == "irvine" or scenario == "lastfm" or scenario == "digg":
        feature_vectors = np.zeros(num_users)
    else:
        feature_vectors = np.load('../data/KDD_data/'+scenario+'_user_features.npy', allow_pickle=True)
        #feature_vectors /= np.max(feature_vectors, axis=0)
    timestamp_list, node_ids_list = getCascades(realization_filename='../data/KDD_data/'+dataset+'_'+scenario+'.npy', scenario=scenario)
    
    # get vector of probabilities, for this iterate through every outstanding node and find probabilities
    mSLE = np.zeros(len(t_c_list))
    mSLE_data_points = {}
    
    for t_c_index, t_c in enumerate(t_c_list):
        mSLE_data_points[t_c] = np.array([])
    total_predictions = np.zeros(len(t_c_list))
    if mode == 'discriminative':
        w_tilde = np.load('../precalc/parameters_'+scenario+run_name+'/best/w_tilde.npy', allow_pickle=True)
        alpha = np.zeros((num_users, num_users))
        for i in range(num_users):
            alpha[i] = np.load('../precalc/parameters_'+scenario+run_name+'/best/'+str(i)+'.npy', allow_pickle=True)
    else:
        alpha, w_tilde = MSPSP.getParams()
  
    
    for i in range(len(timestamp_list)):
        if i % 100 == 0:
            print("processing cascade ", i, " out of ", len(timestamp_list))
        
        for t_c_index, t_c in enumerate(t_c_list):
            timestamps = timestamp_list[i]
            node_ids = np.array(node_ids_list[i])
            (req_indices ,) = np.where(timestamps[:-1] < t_c)
            prediction_timestamps = timestamps[req_indices]
            prediction_node_ids = node_ids[req_indices]
            # if len(prediction_timestamps) ==1:
            #     continue 
            total_predictions[t_c_index] += 1
            remaining_user_ids = np.arange(num_users)
            remaining_user_ids = np.array([x for x in remaining_user_ids if x not in prediction_node_ids])
            pVector = []
            for node in remaining_user_ids:
                alpha_i = alpha[node]
                # np.load('../precalc/parameters_'+scenario+run_name+'/best/'+str(node)+'.npy', allow_pickle=True)
                # if np.sum(alpha_i[prediction_node_ids]) == 0:
                #     continue
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
                pVector.append(prob)
            pVector = np.array(pVector)
            PMF = np.array([0.0, 1.0])

            for p in pVector:
                seq = np.array([1.0-p, p])
                PMF = np.convolve(PMF, seq)
            
            c_h = len(prediction_timestamps)
        
            predicted_count = 0.0
            for k in range(num_users-c_h):
                predicted_count += np.log(k +c_h)*PMF[k]

            (req_indices ,) = np.where(timestamps[:-1] < t_c+delta_t)
            size_timestamps = timestamps[req_indices]
            actual_count = len(size_timestamps)
            predicted_count = int(np.exp(predicted_count))  
            mSLE[t_c_index] += (np.log(predicted_count) - np.log(actual_count))**2
            mSLE_data_points[t_c] = np.append(mSLE_data_points[t_c], (np.log(predicted_count)- np.log(actual_count))**2)
    
    print("Total predictions: ", total_predictions)
    
    return mSLE_data_points
##################################################
# MultiVariateSurvivalSplitPopulation Class
#
# A class designed to model a multi-variate split population survival
# process which has encapsulated within it
# a list of univariate split population survival point processes
# Note that even though this is a general case of split population survival process
# We cannot treat it as such because of the different kind of intensity function  (user-level)
# and the training and precalculation cannot be generalized.
# PUBLIC ATTRIBUTES:
#
#
# PUBLIC INTERFACE:
#
#   __str__(): string; string representation of event object
#   print():   print string representation of event object
#
# USAGE EXAMPLES:
#
# DEPENDENCIES: None
#
# AUTHOR: Akshay Aravamudan, December 2020
#
##################################################
class MultiVariateSurvivalSplitPopulation(object):
    def __init__(self, desc=None, num_users = None, num_platforms = 1, feature_vector_length = None, MemoryKernel = None):

        self._setupTrainingDone = False


        self._sourceNames = np.arange(num_users)
        if desc is not None:
            self.desc = desc
        else:
            logger.Warning(
                "No name passed to process, assigning default: Multivariate Inhomogenous Poisson")
            self.desc = 'Multivariate Inhomogenous Poisson'

        assert num_users is not None, "Number of users needs to be input"
        assert feature_vector_length is not None, "The dimensionality of the feature vector is missing" 
        assert MemoryKernel is not None, "Memory kernel cannot be empty"
        #self.alpha = np.random.random((num_users, num_users))
        
        # self.alpha = np.zeros((num_users, num_users), int)
        # np.random.seed(20)
        # choices = np.random.choice(self.alpha.size,int(self.alpha.size/2),replace=False)
        # self.alpha.ravel()[choices] = 1
        self.alpha = np.random.random((num_users, num_users))
        
        
        #  per platforms, this is for the inputs to each sub-problem which will be used to build the consensus
        self.num_users = num_users
        self.feature_vector_length = feature_vector_length
        self.w_tilde_i = np.random.random((self.num_users, self.feature_vector_length + 1))    
        self.mk = MemoryKernel
        self.w_tilde = None

    def getSourceNames(self):
        return self._sourceNames

    def setParams(self, alpha, w_tilde):
        if alpha is not None:
            self.alpha = alpha
        self.w_tilde = w_tilde

    def getParams(self):
        return self.alpha, self.w_tilde

    def cumulativeIntensity(self, t, realization):
        intensity_vector = np.zeros(len(self.alpha))
        for i in range(len(intensity_vector)):
            # go through every event in MTPP data to construct intensity function value for 
            timestamps = realization['timestamps']
            node_ids = realization['timestamp_ids']
            assert len(timestamps) == len(node_ids), "Inconsistent length of realization"
            
            if len(timestamps) > 0:
                for j in range(len(timestamps)):
                    node_id = node_ids[j]
                    t_j = timestamps[j]
                    if t > t_j:
                        intensity_contribution = self.alpha[node_id, i]* self.mk.psi(t-t_j)
                        intensity_vector[i] += intensity_contribution
            else:
                raise Exception("For simulating, there should be at least one event in the cascade")
        return intensity_vector
        
    def intensity(self, t, realization, susceptible_labels):
        intensity_vector = np.zeros(len(self.alpha))
        susceptible_nodes = np.argwhere(susceptible_labels==1)
        susceptible_nodes = np.ndarray.flatten(susceptible_nodes)
        for i in range(len(intensity_vector)):
            # go through every event in MTPP data to construct intensity function value for 
            timestamps = realization['timestamps']
            node_ids = realization['timestamp_ids']
            assert len(timestamps) == len(node_ids), "Inconsistent length of realization"
            if i in node_ids or not (i in susceptible_nodes):
                intensity_vector[i] = 0.0
            elif len(timestamps) > 0:
                for j in range(len(timestamps)):
                    node_id = node_ids[j]
                    t_j = timestamps[j]
                    relative_t = t-t_j
                    if relative_t > 0:
                        intensity_contribution = self.alpha[node_id, i] * self.mk.phiUB(t_j, t)
                        intensity_vector[i] += intensity_contribution
            else:
                raise Exception("For simulating, there should be at least one event in the cascade")
        return intensity_vector
                        
    def train(self, reinitialize = False, scenario_name = None, run_name = ""):
        assert scenario_name is not None, "Scenario name cannot be empty"
        if scenario_name == "synthetic" or scenario_name == "github" or scenario_name == "twitter" or scenario_name == 'twitter_link':
            feature_vectors = np.load('../data/KDD_data/'+scenario_name+'_user_features.npy', allow_pickle=True)
            feature_vectors /= np.max(feature_vectors, axis= 0 )
        else:
            feature_vectors = np.zeros(self.num_users)
        

        print("saving alpha values to files...")
        # save the alpha values in a file and extract per node as necessary
        lowest_max_residual = np.inf
        best_w_tilde = None
        
        if reinitialize:
            for node_id in range(self.num_users):
                if node_id % 100 == 0:
                    print("saving alpha values ", node_id, "out of ", self.num_users, end='\r')


                # determine all the participating nodes for this sub problem
                precalc_dict = np.load('../precalc/'+scenario_name+run_name+'/node_'+ str(node_id) + '.npy',allow_pickle = True).item()
                participating_nodes = []
                for cascade in precalc_dict.keys():
                    nodes = precalc_dict[cascade]['participating_nodes']
                    if nodes is not None:
                        if len(nodes) > 0:
                            participating_nodes.extend(nodes)

                participating_nodes = np.unique(participating_nodes)
                self.alpha[node_id, list(participating_nodes)] = np.ones(len(participating_nodes))* MVSPP_config.init_alpha
                self.w_tilde_i[node_id, :] = np.ones(self.feature_vector_length + 1)* 0.00
                with open('../precalc/parameters_'+scenario_name+run_name+'/'+str(node_id)+'.npy', 'wb') as f:
                    np.save(f, self.alpha[node_id])
                        
        else:
            for node_id in range(self.num_users):
                if node_id % 100 == 0:
                    print("loading alpha values ", node_id, "out of ", self.num_users, end='\r')


                # # determine all the participating nodes for this sub problem
                # precalc_dict = np.load('../precalc/'+scenario_name+run_name+'/node_'+ str(node_id) + '.npy',allow_pickle = True).item()
                # participating_nodes = []
                # for cascade in precalc_dict.keys():
                #     nodes = precalc_dict[cascade]['participating_nodes']
                #     if nodes is not None:
                #         if len(nodes) > 0:
                #             participating_nodes.extend(nodes[1:])

                # participating_nodes = np.unique(participating_nodes)
                self.w_tilde_i[node_id, :] = np.ones(self.feature_vector_length + 1)*0.
                self.alpha[node_id,:] = np.ones(self.num_users)* MVSPP_config.init_alpha
                


        print("Starting training...")
        reacting_users = []
        
        for node_id in range(self.num_users):
            alpha_i = np.load('../precalc/parameters_'+scenario_name+run_name+'/'+str(node_id)+'.npy', allow_pickle=True)
            if np.sum(alpha_i) > 0:
                reacting_users.append(node_id)
        reacting_users = np.array(range(self.num_users))

        if MVSPP_config.users_to_run == "all":
            subproblems_to_run = len(reacting_users)
            selected_sub_problems = range(subproblems_to_run)
            
        else:
            subproblems_to_run = int(MVSPP_config.users_to_run)
            selected_sub_problems = np.random.choice(range(len(reacting_users)),subproblems_to_run)
            

        # Number of ADMM iterations
        t_max = MVSPP_config.t_max
        # consensus ADMM hyper-parameter
        rho = MVSPP_config.rho
        tol = MVSPP_config.tol
        beta_u = MVSPP_config.beta_u
        beta_d = MVSPP_config.beta_d
        tau_incr = MVSPP_config.tau_incr
        tau_decr = MVSPP_config.tau_decr
        L0 = MVSPP_config.L0
        mu = MVSPP_config.mu
        gamma = MVSPP_config.gamma

        # initializing a w_tilde values for each user. Although there is only one w_tilde 
        
        # initializing the duals 
        y_i =np.zeros((self.num_users, self.feature_vector_length + 1))
        def dual_update(y_i, delta_w, rho):
            return y_i + delta_w*rho
        
        max_residual_list = []
        for t in range(t_max):
            time_start = time.time()
            print("ADMM Iteration: ",t)
            w_tilde = np.sum(self.w_tilde_i[reacting_users[selected_sub_problems]], axis=0)/len(reacting_users[selected_sub_problems])
            futures = []
            if t % MVSPP_config.n_save_iter == 0:
                mvspp_results = pd.read_csv('MVSPP_parameters_temp.csv')
                mvspp_results = mvspp_results.append({'iter':t,'dataset':MVSPP_config.scenario, 'kernel':str(MVSPP_config.memory_kernel), 'w_tilde':w_tilde}, ignore_index=True)
                mvspp_results.to_csv('MVSPP_parameters_temp.csv', index=False)

            if MVSPP_config.run_test_sub_problem == True:
                # test_sub_problem = 693
                test_sub_problem = MVSPP_config.test_sub_problem
                
                q = self.w_tilde_i[test_sub_problem] - (1/rho)*y_i[test_sub_problem] 
                train_sub_problem_bpgd(scenario_name,run_name, test_sub_problem, self.num_users, feature_vectors[test_sub_problem], self.w_tilde_i[test_sub_problem], rho, L0, beta_u, beta_d, q, gamma ) ## debug
                print("Test problem completed")
                input()

            w_tilde_avg = np.average(self.w_tilde_i)
            # solve ith user-level problem
            for node_id in reacting_users[selected_sub_problems]:
                if np.sum(self.alpha[node_id]) > 0:
                    q = w_tilde_avg - (1/rho)*y_i[node_id]
                    # dask concurrent futures
                    # future = client.submit(train_sub_problem,scenario_name,run_name, node_id, self.num_users, feature_vectors[node_id], self.w_tilde_i[node_id], rho, L0, beta_u, beta_d, q , gamma)
                    future = client.submit(train_sub_problem_bpgd,scenario_name,run_name, node_id, self.num_users, feature_vectors[node_id], self.w_tilde_i[node_id], rho, L0, beta_u, beta_d, q , gamma)
                    futures.append(future)

                    
            self.w_tilde_i[reacting_users[selected_sub_problems]] = np.array(client.gather(futures))

            w_new = np.sum(self.w_tilde_i[reacting_users[selected_sub_problems]], axis=0)/len(reacting_users[selected_sub_problems])
            delta_w = (self.w_tilde_i[reacting_users[selected_sub_problems]] - w_new)
            
            futures = []
            # solve ith user-level problem
            for index, node_id in enumerate(reacting_users[selected_sub_problems]):
                # dask concurrent futures
                future = client.submit(dual_update, y_i[node_id], delta_w[index], rho)
                futures.append(future)
            y_i[reacting_users[selected_sub_problems]] = np.array(client.gather(futures))
            primal_residual_norm = np.linalg.norm(delta_w)
            dual_residual_norm = rho*np.linalg.norm(w_new - w_tilde)
            
            print("Max residual norm ",  max(primal_residual_norm, dual_residual_norm))
            max_residual_list.append(max(primal_residual_norm, dual_residual_norm))
            
            if max(primal_residual_norm, dual_residual_norm) <= tol:
                break
            if primal_residual_norm > mu*dual_residual_norm:
                rho = rho*tau_incr

            if dual_residual_norm > mu*primal_residual_norm:
                rho = rho/tau_decr
            w_tilde = w_new
            if max(primal_residual_norm, dual_residual_norm) < lowest_max_residual:
                best_w_tilde = w_tilde
                lowest_max_residual = max(primal_residual_norm, dual_residual_norm)
                print("Saving best parameters..")
                for node_id in range(self.num_users):
                    alpha_i = np.load('../precalc/parameters_'+scenario_name+run_name+'/'+str(node_id)+'.npy', allow_pickle=True)
                    with open('../precalc/parameters_'+scenario_name+run_name+'/best/'+str(node_id)+'.npy', 'wb') as f:
                        np.save(f, alpha_i)
                with open('../precalc/parameters_'+scenario_name+run_name+'/best/w_tilde.npy', 'wb') as f:
                    np.save(f, best_w_tilde)
                

            print("W_tilde ", w_new)
            time_elapsed =  (time.time() - time_start)
            print("Total time elapsed sub-problems", time_elapsed)
    
        # plt.xlabel("Number of consensus ADMM iterations")
        # plt.ylabel("Max residuals")
        # plt.plot(max_residual_list[1:])
        # plt.show()
        
        #self.w_tilde = w_tilde
        
        alpha = np.zeros((self.num_users,self.num_users))
        for i in range(self.num_users):
            alpha[i,:] = np.load('../precalc/parameters_'+scenario_name+run_name+'/best/'+str(i)+'.npy')
        #self.alpha = alpha
        return alpha, w_tilde

    def setupTraining(self, realization_filename, scenario_name, run_name, t_c = None, delta_t=None):
        # CascadeRealizations take the following form:
        # [c_1 = {realization: [timestamp list], ids:[id list]}]
        # Such a list format
        # length of realization and Ids have to be the same
        # last timestamp is the relative right censoring time and has user_id: 0
        realizations = np.load(realization_filename, allow_pickle = True).item()
        
        import os
        if not os.path.isdir("../precalc/"+scenario_name + run_name):
            os.makedirs("../precalc/"+scenario_name + run_name)
        if not os.path.isdir("../precalc/parameters_"+scenario_name + run_name):   
            os.makedirs("../precalc/parameters_"+scenario_name+run_name)
        if not os.path.isdir("../precalc/parameters_"+scenario_name+run_name+"/best"):   
            os.makedirs("../precalc/parameters_"+scenario_name+run_name+"/best")
            
        
        if scenario_name == 'github' or scenario_name == 'twitter':
            rc_df = pd.DataFrame({'year': [2018], 'month': [3], 'day': [31]})
            rightCensoringTime = pd.to_datetime(rc_df, unit='ns')
        elif scenario_name == 'irvine' or scenario_name == 'lastfm':
            rc = realizations['rightCensoring']
            rightCensoringTime = pd.to_datetime(rc, unit='ns')
            del realizations['rightCensoring']
        elif scenario_name == 'twitter_link':
            rc = realizations['rightCensoring']
            rightCensoringTime = rc
            del realizations['rightCensoring']
        elif scenario_name == "digg":
            rc = realizations['rightCensoring']
            rightCensoringTime = rc
            del realizations['rightCensoring']
        # elif scenario_name == "synthetic":
        #     rc = realizations['rightCensoring']
        #     rightCensoringTime = rc
        #     del realizations['rightCensoring']
            
        timestamp_list = []
        node_ids_list = []

        print("Converting realizations into timestamp and node lists...")
        for index, realization_id in enumerate(realizations):
            information_cascade = realizations[realization_id]
            
            timestamps = information_cascade['timestamps']
            node_ids = list(information_cascade['timestamp_ids'])
            if scenario_name == 'github' or scenario_name == 'twitter':
                relative_rc_time = ((rightCensoringTime - timestamps[0])/np.timedelta64(1, 'h')).values[0]
            elif scenario_name == 'lastfm' or scenario_name == 'irvine':
                relative_rc_time = ((rightCensoringTime - timestamps[0])/np.timedelta64(1, 'h'))
            elif scenario_name == 'twitter_link':
                relative_rc_time = ((rightCensoringTime - timestamps[0])/3600)
            elif scenario_name == 'digg':
                relative_rc_time = rightCensoringTime
            elif scenario_name == "synthetic":
                relative_rc_time = realizations[realization_id]['rightCensoring']
                
            if scenario_name == 'twitter_link' :
                timestamps = [((x-timestamps[0])/3600) for x in timestamps]
            elif scenario_name == "digg" or scenario_name == "synthetic":
                timestamps = [x-timestamps[0] for x in timestamps]
            else:
                timestamps = [((x-timestamps[0])/np.timedelta64(1,'h')) for x in timestamps]

            timestamps.append(relative_rc_time)
            node_ids.append(-1)
            if t_c is not None:
                (req_indices, )= np.where(np.array(timestamps[:-1]) > t_c)
                if  len(req_indices) == 0:
                    continue
                (req_indices_tc_delta_t, )= np.where(np.array(timestamps[:-1]) < t_c + delta_t)
                if  len(req_indices_tc_delta_t) - len(req_indices) < 3:
                    continue
    
            # remove all duplicate node ids from timestamp and node_id list: Duplicates should not exist
            unique_ids = []
            unique_timestamps = []
            for i, val in enumerate(node_ids):
                if val not in unique_ids:
                    unique_ids.append(val)
                    unique_timestamps.append(timestamps[i])

            timestamps = np.array(unique_timestamps)
            node_ids = np.array(unique_ids)

            timestamp_list.append(np.array(timestamps))
            node_ids_list.append(np.array(node_ids))


            if index % 100 == 0:
                print("Processed ", index, " Out of ", len(realizations), end='\r')

            
        timestamp_list = np.array(timestamp_list)
        node_ids_list = np.array(node_ids_list)
        

        # min_diff = np.inf
        # for timestamp in timestamp_list:
        #     if np.unique(timestamp)[1] - timestamp[0] < min_diff:
        #         min_diff = timestamp[1] - timestamp[0]
      
        print("Creating precalc files for each node")
        for node_id in range(self.num_users):
            node_precalc_dict = {}
            
            for i in range(len(timestamp_list)):
                # declare phi and psi arrays for the node, the +1 is for the right censoring time
                phi_array_ti = None
                psi_array_ti = None
                phi_array_T = None
                psi_array_T = None
                cascade_dict = {}
                node_index = None
                
                if node_id in node_ids_list[i] :
                    # find position of node within list
                    node_index = np.where(node_ids_list[i]== node_id)[0][0]
                    
                    # re-initialize all arrays as per length of node index
                    phi_array_ti = np.zeros(node_index)
                    psi_array_ti = np.zeros(node_index)
                    phi_array_T = np.zeros(node_index)
                    psi_array_T = np.zeros(node_index)
                    
                    # if node is participating in the cascade and is not the first event 
                    if node_index > 0:
                        # find time of occurence of the node i
                        t_i = timestamp_list[i][node_index]

                        
                        # for all nodes before 
                        phi_array_ti = self.mk.phi([t_i-x for x in timestamp_list[i][np.arange(node_index)]])
                        psi_array_ti = self.mk.psi([t_i-x for x in timestamp_list[i][np.arange(node_index)]])
                        
                        T = timestamp_list[i][-1]
                        phi_array_T = self.mk.phi([T-x for x in timestamp_list[i][np.arange(node_index)]])
                        psi_array_T = self.mk.psi([T-x for x in timestamp_list[i][np.arange(node_index)]])
                        
                else:
                    T = timestamp_list[i][-1]
                    phi_array_T = self.mk.phi([T-x for x in timestamp_list[i]])
                    psi_array_T = self.mk.psi([T-x for x in timestamp_list[i]])
                
                
                cascade_dict['phi_ti'] = phi_array_ti
                cascade_dict['psi_ti'] = psi_array_ti
                cascade_dict['phi_T'] = phi_array_T
                cascade_dict['psi_T'] = psi_array_T

                if node_index is None:
                    cascade_dict['l'] = 0
                    cascade_dict['participating_nodes'] = node_ids_list[i]
                else:
                    cascade_dict['l'] = 1
                    cascade_dict['participating_nodes'] = node_ids_list[i][np.arange(node_index)]
                    
                node_precalc_dict[i] = cascade_dict
            # save precalc dictionary for node to file
            with open('../precalc/'+scenario_name + run_name+'/node_'+str(node_id)+'.npy' , 'wb') as f:
                np.save(f, node_precalc_dict)

            print("Processing node ", node_id, " out of ", self.num_users, end='\r')

        #  save the dictionary of dictionaries as a pickle file
        self._setupTrainingDone = True

    def getNumProcesses(self):
        return len(self.Processes)

    def transformEventTimes(self, MTPPdata, dataStream):
        pass

    def simulate(self,sampleDimensionality,classPrior1,misclassificationProbability,num_users,num_realizations,rightCensoringTime):
        # simulate 
        N = num_users
        
        
        X, y, w_tilde = gen2IsotropicGaussiansSamples(N, sampleDimensionality,\
                                                    classPrior1, \
                                                    misclassificationProbability)

        allOnes = np.ones((N,1))
        X_tilde = np.concatenate((X, allOnes), axis=1)
        
        # simulate samples 
        
        TPPdata = {}
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        for realization_id in range(num_realizations):
            # draw susceptible lables
            # This will produce a list of susceptible nodes in the network: num_user labels
            susceptible_labels = np.random.binomial(1, p=sigmoid(np.dot(X_tilde, w_tilde)))
            

            
            
            realization = {}
            
            
            susceptible_nodes = np.argwhere(susceptible_labels==1)
            susceptible_nodes = np.ndarray.flatten(susceptible_nodes)
            # TODO: shuffle all the node labels
            first_node_id = np.random.choice(susceptible_nodes)
            
            # start time of the simulation
            # start_time = np.random.random() 
            start_time = 0.0
        
        
            # First node is randomly chosen from the user set
            realization['timestamp_ids'] = np.array([first_node_id])
            
            
            # First event occurs at a random time
            realization['timestamps'] = np.array([start_time])
            
            
            realization = simulation_split_population_mv(self,realization,
                                                        rightCensoringTime, 
                                                        susceptible_labels,
                                                        start_time)
            
            realization['rightCensoring'] = rightCensoringTime
            
            TPPdata[str(realization_id)] = realization
            print("Simulations completed: {}".format(str(realization_id)), end='\r')
    
        return TPPdata, X_tilde

    def gof(self, TPPdata):
        ################## perform goodness of fit on simulated data ########################
        # compute IID samples
        IID_samples = []
        for realization_id, realization in TPPdata.items():
            timestamps = realization['timestamps']
            timestamp_ids = realization['timestamp_ids']
            rightCensoringTime = realization['rightCensoring']
            for i in range(1, len(timestamps)):
                node_id = timestamp_ids[i]
                t_i = timestamps[i]
                cumulative_intensity_event = self.cumulativeIntensity(t_i, realization)[node_id]
                cumulative_intensity_rc = self.cumulativeIntensity(rightCensoringTime, realization)[node_id]
                transform = np.log((1-np.exp(-cumulative_intensity_rc))/\
                                (np.exp(-cumulative_intensity_event) - np.exp(-cumulative_intensity_rc)))
                IID_samples.append(transform)
        
        print("Total number of IID samples{}".format(len(IID_samples)))
        
        
        # perform goodness of fit
        ax = plt.gca()
        pvalue = KSgoodnessOfFitExp1(np.random.choice(IID_samples,100, replace=False), ax=ax, title="Simulated MVSPP")
        print("P-value of simulation".format(pvalue))
    
    def generate_adj_graph(self):
        adj_matrix = np.zeros_like(self.alpha)

        for i in range(len(self.alpha)):
            for j in range(len(self.alpha)):
                if self.alpha[i][j] > 0:
                    adj_matrix[i][j] = 1
        
        with open('../data/synthetic/graph_adj_matrix_'+ str(len(self.alpha)) +'.pkl', 'wb') as f:
            pickle.dump(adj_matrix, f)

        
################################################################################
#
# U N I T   T E S T I N G
#
################################################################################


# Some global settings for figure sizes
normalFigSize = (8, 6)  # (width,height) in inches
largeFigSize = (12, 9)
xlargeFigSize = (18, 12)

def plot_results(data, ticks, labels, title):
    data_a = [val for index, val in data[0].items()]
    # data_b = [[6,4,2], [1,2,5,3,2], [2,3,5,1]]
    data_b = [val for index, val in data[1].items()]

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color='white')

    plt.figure()

    bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6, patch_artist=True,showmeans=True, meanprops={'markerfacecolor':'white', 'markeredgecolor':'black'})
    bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6,patch_artist=True,showmeans=True, meanprops={'markerfacecolor':'white', 'markeredgecolor':'black'})
    set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#2C7BB6')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label=labels[0])
    plt.plot([], c='#2C7BB6', label=labels[1])
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    # plt.xlim(-2, len(ticks)*2)
    # plt.ylim(-0.05, 0.2)
    plt.xlabel('$t_c$')
    plt.ylabel('Squared Log Error')
    plt.title(title)
    plt.tight_layout()  
    
def main():
    """ Unit testing the MVSPP synthetic data processing
    """
    simulate_data = True
    num_users = 50
        
    memory_kernel = MVSPP_config.memory_kernel
    run_name = MVSPP_config.run_name
    # feature generation for split population model
    sampleDimensionality = 3
    classPrior1 = 0.8
    misclassificationProbability = 0.01
    
    num_training_realizations = 500
    num_test_realizations = 200
    num_validation_realizations = 200
    
    
    
    ########### create multivariate process
    MSPSP = MultiVariateSurvivalSplitPopulation(
        desc='Multivariate Survival split population', num_users=num_users, 
        feature_vector_length= sampleDimensionality + 1, MemoryKernel = memory_kernel)
    
    ############## generate user features ##################
    
    MSPSP.generate_adj_graph()

    if simulate_data:
        TPPdata,X_tilde = MSPSP.simulate(sampleDimensionality,
                                        classPrior1,
                                        misclassificationProbability,
                                        num_users, 
                                        num_realizations=num_training_realizations,
                                        rightCensoringTime=np.random.uniform(0.5, 10.5))

        test_TPPdata,_ = MSPSP.simulate(sampleDimensionality,
                                        classPrior1,
                                        misclassificationProbability,
                                        num_users,
                                        num_realizations=num_test_realizations,
                                        rightCensoringTime=np.random.uniform(0.5, 10.5))
        validation_TPPdata,_ = MSPSP.simulate(sampleDimensionality,
                                        classPrior1,
                                        misclassificationProbability,
                                        num_users,
                                        num_realizations=num_validation_realizations,
                                        rightCensoringTime=np.random.uniform(0.5, 10.5))
                                        
                            

        # MSPSP.gof(TPPdata)
        
        # ############################### Save data onto disk for training ######################
        # save simulated realizations
        np.save('../data/KDD_data/training_synthetic.npy',TPPdata)
        # save feature vectors
        np.save('../data/KDD_data/synthetic_user_features.npy',X_tilde )
        # save test realizations
        np.save('../data/KDD_data/test_synthetic.npy', test_TPPdata)
        np.save('../data/KDD_data/validation_synthetic.npy', validation_TPPdata)
        
    
    
    #################################### Experiment 1 #####################################
    ## To see if discriminative model performs better in predicive setting.,
    ## note: for fair comparison, re-train with generative model from scratch and use those parameters 
    ## This experiment compares the prediction performance for two models 
    ##  (i)  Split populated discriminative survival prediction model (ours) trained on the entire data 
    ##  (ii) Multivariate survival
    
    ground_truth_alphas,_ = MSPSP.getParams()
    ground_truth_alphas = ground_truth_alphas.copy()
    ground_truth_w_tilde = np.array([ 0.23166263,  0.,          0. ,        -1.38629436,1]) 
    scenario = MVSPP_config.scenario
    if MVSPP_config.train:
        if MVSPP_config.run_setup:
            MSPSP.setupTraining(realization_filename='../data/KDD_data/training_'+scenario+'.npy', scenario_name = scenario,run_name = run_name)
        alpha, w_tilde = MSPSP.train(reinitialize=MVSPP_config.reinitialize, scenario_name = scenario,run_name=run_name)
        

    #######################################################################################
    t_c_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 3.0, 4.0]
    # t_c_list = [0.6, 0.8, 1.0, 1.2]
    
    t_c = 1.0
    delta_t = 0.5
    start_size_list = [5,6,7,8,9, 10, 11,12,13,14,15, 16,17,18,19,20, 21,22,23,24,25, 26,27,28,29]
    deltat_list = [ 0.4, 0.5, 1.0, 2.0, 3.0, 4.0,5.0,6.0]
    #######################################################################################

    w_tilde = np.load('../precalc/parameters_'+scenario+run_name+'/best/w_tilde.npy', allow_pickle=True)
    MSPSP.setParams(None, w_tilde)

    mode_list = ['tc_finalsize', 'startsize_finalsize','fixed_tc_changing_delta_t', 'fixed_delta_t_changing_t_c']
    #mode_list = ['startsize_finalsize']

    for mode in mode_list: 
        if mode == 'tc_finalsize':
            mSLE_data_points_tc_discriminative = predict_cascade_size_tc(MSPSP,num_users, memory_kernel, scenario,run_name=run_name,t_c_list=t_c_list, dataset='test', mode='discriminative')
            MSPSP.setParams(ground_truth_alphas, ground_truth_w_tilde)
            mSLE_data_points_tc_generative = predict_cascade_size_tc(MSPSP,num_users, memory_kernel, scenario,run_name=run_name,t_c_list=t_c_list, dataset='test', mode='generative')
            plot_results([mSLE_data_points_tc_discriminative,mSLE_data_points_tc_generative],
                        ticks=t_c_list,
                        labels = ['Discriminative','Generative (GT)'],
                        title = 'Final size prediction error for varying $t_c$')
        
        elif mode == 'fixed_tc_changing_delta_t':
            mSLE_data_points_fixed_tc_discriminative = predict_cascade_size_fixed_tc_varying_deltat(MSPSP,num_users, memory_kernel, scenario,run_name=run_name,t_c=t_c,deltat_list = deltat_list, dataset='test', mode='discriminative')
            MSPSP.setParams(ground_truth_alphas, ground_truth_w_tilde)
            mSLE_data_points_fixed_tc_generative = predict_cascade_size_fixed_tc_varying_deltat(MSPSP,num_users, memory_kernel, scenario,run_name=run_name,t_c=t_c,deltat_list = deltat_list, dataset='test', mode='generative')
            plot_results([mSLE_data_points_fixed_tc_discriminative,mSLE_data_points_fixed_tc_generative],
                    ticks=deltat_list,
                    labels = ['Discriminative','Generative (GT)'],
                    title = 'Prediction error for fixed t_c = '+str(t_c)+' and varying $\Delta t$')
        
        elif mode == 'fixed_delta_t_changing_t_c':
            mSLE_data_points_fixed_delta_t_discriminative = predict_cascade_size_fixed_delta_t_varying_t_c(MSPSP,num_users, memory_kernel, scenario,run_name=run_name,delta_t=delta_t,t_c_list = t_c_list, dataset='test', mode='discriminative')
            MSPSP.setParams(ground_truth_alphas, ground_truth_w_tilde)
            mSLE_data_points_fixed_delta_t_generative = predict_cascade_size_fixed_delta_t_varying_t_c(MSPSP,num_users, memory_kernel, scenario,run_name=run_name,delta_t=delta_t,t_c_list = t_c_list, dataset='test', mode='generative')
            
            plot_results([mSLE_data_points_fixed_delta_t_discriminative,mSLE_data_points_fixed_delta_t_generative],
                    ticks=t_c_list,
                    labels = ['Discriminative','Generative (GT)'],
                    title = 'Prediction error for fixed delta_t = '+str(delta_t)+' and varying $t_c$')

        elif mode == 'startsize_finalsize':
            mSLE_data_points_start_size_discriminative = predict_cascade_size(MSPSP,num_users, memory_kernel, scenario, run_name=run_name,start_sizes=start_size_list,dataset='test', mode='discriminative')
            MSPSP.setParams(ground_truth_alphas, ground_truth_w_tilde)
            mSLE_data_points_start_size_generative = predict_cascade_size(MSPSP,num_users, memory_kernel, scenario, run_name=run_name,start_sizes=start_size_list,dataset='test', mode='generative')
            plot_results([mSLE_data_points_start_size_discriminative,mSLE_data_points_start_size_generative],
                    ticks=start_size_list,
                    labels = ['Discriminative','Generative (GT)'],
                    title = 'Prediction error for varying observed nodes')
        
        plt.savefig('synthetic_data_results_'+mode+'.png', dpi=300)
        plt.show()
    
    
    
if __name__ == "__main__":
    client = Client()
    print("Dask Hosted at: ", client.scheduler_info()['services'])
    main()


# %%
