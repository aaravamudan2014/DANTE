'''
MultiVariateSurvivalSplitPopulation.py
    Multi variate split population survival process implementation
'''

from utils.MemoryKernel import ConstantMemoryKernel, RayleighMemoryKernel, PowerLawMemoryKernel, \
                                GammaGompertzMemoryKernel, WeibullMemoryKernel,ExponentialPseudoMemoryKernel,GompertzMemoryKernel
from utils.DataReader import createDummyDataSplitPopulation

# from utils.Simulation import simulation


from point_processes.PointProcessCollection import PoissonTPP
import numpy as np
import scipy.stats
from utils.GoodnessOfFit import KSgoodnessOfFitExp1, KSgoodnessOfFitExp1MV
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

import warnings
warnings.filterwarnings("ignore")


# L1 regularization term
@jit
def L1regularizer(alpha, nu):
    return nu * np.linalg.norm(alpha,ord = 1 )

# Function to calculate gradients w.r.t alpha and w_tilde
@jit
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

@jit
def param_update(alpha_i, w_s ,L, rho, q, grad_alpha_i, grad_w_s, gamma):
    w_s_new = w_s - 1/L*(grad_w_s + rho*(w_s - q))
    alpha_i_new = (alpha_i -1/L*(grad_alpha_i) - (gamma/L)).clip(min=0.00001)
    return np.array([alpha_i_new, w_s_new])

@jit
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
            # if MVSPP_config.run_test_sub_problem:
            #     print(l)
            #     print(phi_t_i)
            #     print(phi_T)
            #     print(psi_t_i)
            #     print(psi_T)
                
            #     print(alpha[participating_nodes])
            #     input()
                
    
    nll /= len(precalc_dict.keys())
    grad_alpha /= len(precalc_dict.keys())
    grad_w /= len(precalc_dict.keys())

    return nll,  grad_alpha, grad_w


def train_sub_problem_adam(scenario_name,run_name, node_id, num_users, x_i, w_s, rho, L0, beta_u, beta_d, q, gamma):
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
    inner_iter_break = False

    # ADAM hyperparameters
    beta_1 = 0.005
    beta_2 = 0.005
    m_0 = np.ones(len(theta[0]))*0.05
    m_1 = np.ones(len(theta[1]))*0.05
    v_0 = np.ones(len(theta[0]))*0.05
    v_1 = np.ones(len(theta[1]))*0.05
    
    for t in range(1, maxIter + 1):
        
        nll, grad_alpha, grad_w  = calculate_likelihood_quantities(precalc_dict, theta[0], theta[1], x_i_bias, node_id,rho, q)

        nll_list.append(nll)
        

        m_0 = beta_1*m_0 + (1-beta_1)*grad_alpha
        m_1 = beta_1*m_1 + (1-beta_1)*grad_w

        v_0 = beta_2*v_0 + (1-beta_2)*np.multiply(grad_alpha,grad_alpha)
        v_1 = beta_2*v_1 + (1-beta_2)*np.multiply(grad_w, grad_w)

        m_0_cap = m_0/(1-beta_1**t)
        m_1_cap = m_1/(1-beta_1**t)

        v_0_cap = v_0/(1-beta_2**t)
        v_1_cap = v_1/(1-beta_2**t)

        theta_new = np.empty_like(theta)
        theta_new[0] = (theta[0] - (1/L0)*m_0_cap*(1/(np.sqrt(v_0_cap)+ epsilon)) - (gamma/L0)).clip(0.0000001)
        theta_new[1] = (theta[1] - (1/L0)*m_1_cap*(1/(np.sqrt(v_1_cap)+ epsilon)) - (1/L0)*(rho*(theta[1] - q)))
        
        theta_previous = theta
        theta = theta_new


        norm = np.sqrt(np.linalg.norm(theta_previous[0]- theta[0])**2 + np.linalg.norm( theta_previous[1]- theta[1])**2)

        #w_s_new = w_s - 1/L*(grad_w_s + rho*(w_s - q))
        #alpha_i_new = (alpha_i -1/L*(grad_alpha_i) - (gamma/L)).clip(min=0.00001)
    
        print(theta)
        input()
        if norm < epsilon:
            break

    if MVSPP_config.visualize_subproblems:
        fig, ax = plt.subplots()
        ax.plot(nll_list, color='red')
        ax.set_xlabel('iterations')
        ax.set_ylabel('nll', color='red')
        #ax2 = ax.twinx()
        #ax2.plot(bce_list, color='blue')
        #ax2.set_ylabel('bce', color='blue')
        plt.savefig('figures/'+str(node_id)+'.png')
    
    # save the alpha vector back to a file since returning it takes up a lot of memory
    with open('../precalc/parameters_'+scenario_name+run_name+'/'+str(node_id)+'.npy', 'wb') as f:
        np.save(f, theta[0])
    
    # just return w_tilde vector since the alpha values are already saved 
    return theta[1]



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
    inner_iter_break = False
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
            #if MVSPP_config.run_test_sub_problem:
            #    print(inner_iter,nll, nll_new)
            if nll_new < nll or inner_iter == MVSPP_config.inner_iter:
                if not inner_iter == MVSPP_config.inner_iter:
                    theta = theta_update
                    nll = nll_new
                break
        if MVSPP_config.run_test_sub_problem == True:
            print("outer iteration: ", t)
            print(nll)
        #print(theta[0])
        #input()
            
        norm = np.sqrt(np.linalg.norm(theta_previous[0]- theta[0])**2 + np.linalg.norm( theta_previous[1]- theta[1])**2)
        
        # bce = calculate_bce_loss(node_id, x_i_bias,theta[1], theta[0], t_c=10000, delta_t=25000, threshold = 0.5)
        # print("iteration",t, "nll: ",nll, "norm: ",norm)
        
        nll_list.append(nll)
        #print(nll_list)
        if norm < epsilon:
            break

    if MVSPP_config.visualize_subproblems:
        fig, ax = plt.subplots()
        ax.plot(nll_list, color='red')
        ax.set_xlabel('iterations')
        ax.set_ylabel('nll', color='red')
        plt.savefig('figures/'+str(node_id)+'.png')
    
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
    elif scenario  == 'twitter_link' or scenario == 'digg' or scenario == "memes":
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
        elif scenario == 'digg' or scenario == "memes":
            relative_rc_time = (rightCensoringTime - timestamps[0])
        elif scenario == "synthetic":
                relative_rc_time = realizations[realization_id]['rightCensoring']
                
            

        if scenario == 'twitter_link':
            timestamps = [((x-timestamps[0])/3600) for x in timestamps]
        elif scenario == 'lastfm' or scenario == 'irvine':
            timestamps = [((x-timestamps[0])/np.timedelta64(1,'h')) for x in timestamps] 
        else:
            timestamps = [(x-timestamps[0]) for x in timestamps] 
        
        timestamps.append(relative_rc_time)
        node_ids.append(-1)

        
        timestamp_list.append(np.array(timestamps))
        node_ids_list.append(np.array(node_ids))


        if index % 100 == 0:
            print("Processed ", index, " Out of ", len(realizations), end='\r')
    
    return timestamp_list, node_ids_list

def predict_cascade_size_tc(num_users, MemoryKernel, scenario, run_name, t_c_list, show_status=False, dataset_type="validation"):
    
    # get feature vectors of remaining node ids
    if scenario == "irvine" or scenario == "lastfm" :
        feature_vectors = np.zeros(num_users)
    else:
        feature_vectors = np.load('../data/KDD_data/'+scenario+'_user_features.npy', allow_pickle=True)
        feature_vectors /= np.max(feature_vectors, axis=0)
    timestamp_list, node_ids_list = getCascades(realization_filename='../data/KDD_data/'+dataset_type+'_'+scenario+'.npy', scenario=scenario)
    # get vector of probabilities, for this iterate through every outstanding node and find probabilities
    mSLE = np.zeros(len(t_c_list))
    mSLE_data_points = {}
    
    for t_c_index, t_c in enumerate(t_c_list):
        mSLE_data_points[t_c] = np.array([])
    
    total_predictions = np.zeros(len(t_c_list))
    if dataset_type == "validation":
        w_tilde = np.load('../precalc/parameters_'+scenario+run_name+'/w_tilde.npy', allow_pickle=True)
    elif dataset_type == "test":
        w_tilde = np.load('../precalc/parameters_'+scenario+run_name+'/best/w_tilde.npy', allow_pickle=True)
        
    for i in range(len(timestamp_list)):
        if i % 50 == 0 and show_status == True:
            print("processing cascade ", i, " out of ", len(timestamp_list))
        
        timestamps = timestamp_list[i]
        node_ids = np.array(node_ids_list[i])
        for t_c_index, t_c in enumerate(t_c_list):
            delta_t = timestamps[-1] - t_c
            
            (req_indices ,)= np.where(timestamps[:-1] < t_c)
            prediction_timestamps = timestamps[req_indices]
            prediction_node_ids = node_ids[ req_indices]
            if len(prediction_timestamps) ==1:
                continue 
            remaining_user_ids = np.arange(num_users)
            remaining_user_ids = np.array([x for x in remaining_user_ids if x not in prediction_node_ids])
            pVector = []
            for node in remaining_user_ids:
                if dataset_type == "validation":
                    alpha_i = np.load('../precalc/parameters_'+scenario+run_name+'/'+str(node)+'.npy', allow_pickle=True)
                elif dataset_type == "test":
                    alpha_i = np.load('../precalc/parameters_'+scenario+run_name+'/best/'+str(node)+'.npy', allow_pickle=True)
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
            
            confidence = 0.95
            sortedPMF = -np.sort(-PMF)
            sortedIndexPMF = np.argsort(-PMF)
            cumSumPMF = np.cumsum(sortedPMF)
            endIndex = np.argmax(confidence <= cumSumPMF)
            # find cumsum
            # add all indexes before 
            ret_list = []
            for k in range(endIndex):
                ret_list.append(sortedIndexPMF[k])
            ret_list = np.sort(ret_list)
            #print("95% CI bounds: ", ret_list[0], ret_list[-1])
            
            #input()
            # if t_c == 0.2:
            #     print("t_c", t_c)
            #     print("predicted count" , np.exp(predicted_count))
            #     print( "total count ", len(timestamps[:-1]))
            #     print("observation count ", len(prediction_timestamps))
            #     input()
            predicted_count = int(np.exp(predicted_count))
            mSLE[t_c_index] += (np.log(predicted_count) - np.log(len(timestamps[:-1])))**2
            mSLE_data_points[t_c] = np.append(mSLE_data_points[t_c], (np.log(predicted_count)- np.log(len(timestamps[:-1])))**2)
            total_predictions[t_c_index] += 1
            

    mSLE = np.divide(mSLE,total_predictions)
  
    print("Mean Squared Log Error for start time based prediction ("+dataset_type+")",t_c_list, " : " ,mSLE)
    print("Total predictions: ", total_predictions)
  
    return mSLE_data_points, mSLE

def predict_cascade_size(num_users, MemoryKernel, scenario, run_name, start_sizes,show_status=False, dataset_type="test"):
    
    # get feature vectors of remaining node ids
    if scenario == "irvine" or scenario == "lastfm":
        feature_vectors = np.zeros(num_users)
    else:
        feature_vectors = np.load('../data/KDD_data/'+scenario+'_user_features.npy', allow_pickle=True)
        feature_vectors /= np.max(feature_vectors, axis=0)
        
    timestamp_list, node_ids_list = getCascades(realization_filename='../data/KDD_data/'+dataset_type+'_'+scenario+'.npy', scenario=scenario)
    
    if dataset_type == 'validation':
        w_tilde = np.load('../precalc/parameters_'+scenario+run_name+'/w_tilde.npy', allow_pickle=True)
    elif dataset_type in ['test', 'training']:
        w_tilde = np.load('../precalc/parameters_'+scenario+run_name+'/best/w_tilde.npy', allow_pickle=True)
    #print(w_tilde)
    #input()
    # get vector of probabilities, for this iterate through every outstanding node and find probabilities
    p_list = []
    mSLE = np.zeros(len(start_sizes))
    mSLE_data_points = {}
    for start_size_index, start_size in enumerate(start_sizes):
        mSLE_data_points[start_size] = np.array([])

    coverage_count = np.zeros(len(start_sizes))
    interval_widths = np.zeros(len(start_sizes))
    MSE = np.zeros(len(start_sizes))
    total_predictions = np.zeros(len(start_sizes))

    prob_list  = []

    for i in range(len(timestamp_list)):
        if i % 50 == 0 and show_status==True:
            print("processing cascade ", i, " out of ", len(timestamp_list))
            #print(total_predictions)
            #input()
        for start_size_index, start_size in enumerate(start_sizes):
            timestamps = timestamp_list[i]
            node_ids = np.array(node_ids_list[i])
            
            if len(np.unique(timestamps)) - 1 <= start_size:
                #print(len((timestamp_list[i])) - 1)
                continue
            
            total_predictions[start_size_index] += 1
            t_c = sorted((np.unique(timestamps)))[start_size]
            delta_t = timestamps[-1] - t_c
            

            (req_indices ,)= np.where(timestamps < t_c)
            prediction_timestamps = timestamps[req_indices]
            prediction_node_ids = node_ids[ req_indices]
            
            remaining_user_ids = np.arange(num_users)
            remaining_user_ids = np.array([x for x in remaining_user_ids if x not in prediction_node_ids])
            
            pVector = []
            import math
            for node in remaining_user_ids:
                if dataset_type == "validation":
                    alpha_i = np.load('../precalc/parameters_'+scenario+run_name+'/'+str(node)+'.npy', allow_pickle=True)
                elif dataset_type in ['test', 'training']:
                    alpha_i = np.load('../precalc/parameters_'+scenario+run_name+'/best/'+str(node)+'.npy', allow_pickle=True)
                    
                
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
                prob_list.append(pi_x_w)
                pVector.append(prob)
                #print(pi_x_w)
            pVector = np.array(pVector)
            PMF = np.array([0.0, 1.0])

            for p in pVector:
                seq = np.array([1.0-p, p])
                PMF = np.convolve(PMF, seq)
            
            
            c_h = len(prediction_timestamps)
            
            predicted_count = 0.0
            for j in range(num_users-c_h):
                predicted_count += np.log(c_h +j)*PMF[j]
            
            confidence = 0.95
            sortedPMF = -np.sort(-PMF)
            sortedIndexPMF = np.argsort(-PMF)
            cumSumPMF = np.cumsum(sortedPMF)
            endIndex = np.argmax(confidence <= cumSumPMF)
            # find cumsum
            # add all indexes before 
            ret_list = []
            for k in range(endIndex):
                ret_list.append(sortedIndexPMF[k])
            ret_list = np.sort(ret_list)
            
            if len(ret_list) > 0:
                interval_widths[start_size_index] = ret_list[-1] - ret_list[0]
                if len(timestamps[:-1]) <= ret_list[-1]+c_h and len(timestamps[:-1]) >= ret_list[0]+c_h:
                    coverage_count[start_size_index] +=1

            else:
                interval_widths[start_size_index] = 0
            
            plot=False
            if plot==True:
                print("95% CI bounds: ", ret_list[0]+c_h, ret_list[-1]+c_h)
                fig, ax = plt.subplots()
                plt.stem(PMF)
                plt.axvline(ret_list[0], color='r', linestyle='--', label='95 % CI')
                plt.axvline(ret_list[-1], color='r', linestyle='--')
                print(c_h,len(timestamps[:-1]))
                plt.scatter(len(timestamps[:-1])-c_h,0, marker="*", label="Ground Truth", color='yellow', edgecolors='black', s=400, zorder=1)
                plt.legend()
                plt.xlim([0,20])
                plt.xlabel('count')
                plt.ylabel('probability')
                plt.title('Estimated PMF of predicted count distribution')
                plt.show()
            
            predicted_count = int(np.exp(predicted_count))
            mSLE[start_size_index] += (np.log(predicted_count) - np.log(len(timestamps) - 1))**2
            mSLE_data_points[start_size] = np.append(mSLE_data_points[start_size], (np.log(predicted_count) - np.log(len(timestamps[:-1])))**2)
            
    # plt.hist(prob_list, bins=20)
    # plt.xlabel('Probability of susceptibility')
    # plt.ylabel('Counts of user appearing in cascades')
    # plt.show()
    # input()
    mSLE = np.divide(mSLE,total_predictions)
    print("Mean Squared Log Error for event count based prediction ("+dataset_type+") ",start_sizes, " : " ,mSLE)
    print("Total predictions: ", total_predictions)
    print("Coverage Probability: ", np.divide(coverage_count,total_predictions))
    print("Interval widths: ", np.divide(interval_widths, total_predictions))
    return mSLE_data_points, mSLE


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
        assert feature_vector_length is not None, "The dimensionaloty of the feature vector is missing" 
        assert MemoryKernel is not None, "Memory kernel cannot be empty"
        self.alpha = np.zeros((num_users, num_users)) 
        #  per platforms, this is for the inputs to each sub-problem which will be used to build the consensus
        self.num_users = num_users
        self.feature_vector_length = feature_vector_length
        self.w_tilde_i = np.ones((self.num_users, self.feature_vector_length + 1))    
        self.mk = MemoryKernel

    def getSourceNames(self):
        return self._sourceNames

    def setParams(self, params):
        self._params = params
        pass

    def getParams(self):
        return self._params, self._sourceNames

    def simulate(self, rightCensoringTime, MTPPdata, resume):
        
        return simulation_split_population(self.Processes, rightCensoringTime=rightCensoringTime,
                          MTPPdata=MTPPdata, resume=resume)


    def intensity(self, t, MTPPdata):
        intensity_val = 0
        for user_realization in MTPPdata:
            pass
             
    def train(self, reinitialize = False, scenario_name = None, run_name = ""):
        

        assert scenario_name is not None, "Scenario name cannot be empty"
        if scenario_name in ["github",  "twitter", 'twitter_link', 'synthetic', 'memes', 'digg']:
            feature_vectors = np.load('../data/KDD_data/'+scenario_name+'_user_features.npy', allow_pickle=True)
            feature_vectors /= np.max(feature_vectors, axis= 0 )
        else:
            feature_vectors = np.zeros(self.num_users)

        print("saving alpha values to files...")
        # save the alpha values in a file and extract per node as necessary
        lowest_max_residual = np.inf
        lowest_val_msle = np.inf
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
                # print(participating_nodes)
                # input()
                np.random.seed(20)
                self.alpha[node_id, list(participating_nodes)] = np.ones(len(participating_nodes))* MVSPP_config.init_alpha
                self.w_tilde_i[node_id, :] = np.random.random(self.feature_vector_length + 1)* 1.0
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

                self.w_tilde_i[node_id, :] = np.load('../precalc/parameters_'+scenario_name+run_name+'/best/w_tilde.npy', allow_pickle=True)
                self.alpha[node_id,:] = np.load('../precalc/parameters_'+scenario_name+run_name+'/best/'+str(node_id)+'.npy', allow_pickle=True)
                


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
        y_i =np.ones((self.num_users, self.feature_vector_length + 1))*0.5
        def dual_update(y_i_input, delta_w, rho):
            return y_i_input + delta_w*rho
        
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
                if MVSPP_config.opt_alg == "bpgd":
                    train_sub_problem_bpgd(scenario_name,run_name, test_sub_problem, self.num_users, feature_vectors[test_sub_problem], self.w_tilde_i[test_sub_problem], rho, L0, beta_u, beta_d, q, gamma ) ## debug
                elif MVSPP_config.opt_alg == "adam":
                    train_sub_problem_adam(scenario_name,run_name, test_sub_problem, self.num_users, feature_vectors[test_sub_problem], self.w_tilde_i[test_sub_problem], rho, L0, beta_u, beta_d, q, gamma ) ## debug
                
                print("Test problem completed")
                input()

            w_tilde_avg = np.average(self.w_tilde_i)
            # solve ith user-level problem
            for node_id in reacting_users[selected_sub_problems]:
                if np.sum(self.alpha[node_id]) > 0:
                    q = w_tilde_avg - (1/rho)*y_i[node_id]
                    # dask concurrent futures
                    # future = client.submit(train_sub_problem,scenario_name,run_name, node_id, self.num_users, feature_vectors[node_id], self.w_tilde_i[node_id], rho, L0, beta_u, beta_d, q , gamma)
                    if MVSPP_config.opt_alg == "bpgd":
                        future = client.submit(train_sub_problem_bpgd,scenario_name,run_name, node_id, self.num_users, feature_vectors[node_id], self.w_tilde_i[node_id], rho, L0, beta_u, beta_d, q , gamma)
                    elif MVSPP_config.opt_alg == "adam":
                        future = client.submit(train_sub_problem_adam,scenario_name,run_name, node_id, self.num_users, feature_vectors[node_id], self.w_tilde_i[node_id], rho, L0, beta_u, beta_d, q , gamma)
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
            
            time_elapsed =  (time.time() - time_start)
            print("Total time elapsed sub-problems", time_elapsed)
            
            
            with open('../precalc/parameters_'+scenario_name+run_name+'/w_tilde.npy', 'wb') as f:
                    np.save(f, w_tilde)
            
            val_msle_list_startsize, val_msle_startsize = predict_cascade_size(self.num_users, MVSPP_config.memory_kernel, MVSPP_config.scenario, run_name=MVSPP_config.run_name,start_sizes=MVSPP_config.start_size_list,show_status=False, dataset_type='validation')
            val_msle_list_tc, val_msle_tc = predict_cascade_size_tc(self.num_users, MVSPP_config.memory_kernel, MVSPP_config.scenario, run_name=MVSPP_config.run_name, t_c_list=MVSPP_config.tc_list, show_status=False, dataset_type='validation')
            #val_msle_list_tc = 0
            #val_msle_tc = 0

            #if lowest_max_residual > max(primal_residual_norm, dual_residual_norm):
            if np.sum(val_msle_startsize) + np.sum(val_msle_tc) < lowest_val_msle:
                best_w_tilde = w_tilde
                lowest_val_msle = np.sum(val_msle_startsize) + np.sum(val_msle_tc)
                lowest_max_residual = max(primal_residual_norm, dual_residual_norm)
                print("Saving best parameters..")
                for node_id in range(self.num_users):
                    alpha_i = np.load('../precalc/parameters_'+scenario_name+run_name+'/'+str(node_id)+'.npy', allow_pickle=True)
                    with open('../precalc/parameters_'+scenario_name+run_name+'/best/'+str(node_id)+'.npy', 'wb') as f:
                        np.save(f, alpha_i)
                with open('../precalc/parameters_'+scenario_name+run_name+'/best/w_tilde.npy', 'wb') as f:
                    np.save(f, best_w_tilde)
                predict_cascade_size(self.num_users, MVSPP_config.memory_kernel, MVSPP_config.scenario, run_name=MVSPP_config.run_name,start_sizes=MVSPP_config.start_size_list, dataset_type='test')
                predict_cascade_size_tc(self.num_users, MVSPP_config.memory_kernel, MVSPP_config.scenario, run_name=MVSPP_config.run_name, t_c_list=MVSPP_config.tc_list, show_status=False, dataset_type='test')
            
            #if max(primal_residual_norm, dual_residual_norm) < lowest_max_residual:
            #    best_w_tilde = w_tilde
            #    lowest_max_residual = max(primal_residual_norm, dual_residual_norm)
            #    print("Saving best parameters..")
            #    for node_id in range(self.num_users):
            #        with open('../precalc/parameters_'+scenario_name+run_name+'/best/'+str(node_id)+'.npy', 'wb') as f:
            #            np.save(f, self.alpha[node_id])
            #    with open('../precalc/parameters_'+scenario_name+run_name+'/best/w_tilde.npy', 'wb') as f:
            #        np.save(f, best_w_tilde)
                
            
            print("W_tilde ", w_new)
            
        # plt.xlabel("Number of consensus ADMM iterations")
        # plt.ylabel("Max residuals")
        # plt.plot(max_residual_list[1:])
        # plt.show()
        return w_tilde

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
        elif scenario_name == "digg" or scenario_name == "memes":
            rc = realizations['rightCensoring']
            rightCensoringTime = rc
            del realizations['rightCensoring']

            


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
            elif scenario_name == 'digg' or scenario_name == "memes":
                relative_rc_time = rightCensoringTime
            elif scenario_name == "synthetic":
                relative_rc_time = realizations[realization_id]['rightCensoring']
                
                
            if scenario_name == 'twitter_link' :
                timestamps = [((x-timestamps[0])/3600) for x in timestamps]
            elif scenario_name == "digg" or scenario_name == "memes" or scenario_name == "synthetic":
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

################################################################################
#
# U N I T   T E S T I N G
#
################################################################################


# Some global settings for figure sizes
normalFigSize = (8, 6)  # (width,height) in inches
largeFigSize = (12, 9)
xlargeFigSize = (18, 12)


def main():
    # Defining a custom MultiVariateSurvivalSplitPopulation which contains fixed
    #  number of users, each of whom has a univariate split population survival process 
    #  that is history dependent

    #  read github user feature data
    github_user_features = np.load('../data/KDD_data/github_user_features.npy' , allow_pickle = True)
    twitter_user_features = np.load('../data/KDD_data/twitter_user_features.npy', allow_pickle = True)
    twitter_link_user_features = np.load('../data/KDD_data/twitter_link_user_features.npy', allow_pickle = True)
    
    scenario = MVSPP_config.scenario
    if scenario == 'twitter':
        feature_vector_length = len(twitter_user_features[0])
        num_users = len(twitter_user_features)
        # w_tilde = np.array([0.00250413, 0.00250144, 0.00243388, 0.00250184, 0.00510143]) # power law kernel
        w_tilde = np.array([0.00029876,  0.00026437,  0.00029347,  0.00030508, -0.0029])
    elif scenario== 'github':
        feature_vector_length = len(github_user_features[0])
        num_users = len(github_user_features)
        w_tilde = np.array([-0.01822598, -0.01822621, -0.01822697, -0.01822568, -0.0182257,  -0.01988372])
    elif scenario == 'irvine':
        feature_vector_length = 0
        num_users = 893
        #num_users = 200
        
        w_tilde = np.array([4.623])
    elif scenario == 'lastfm':
        feature_vector_length = 0
        w_tilde = np.array([0.01])  # powerlaw kernel(1.0)
        num_users = 1000
    elif scenario == 'twitter_link':
        feature_vector_length = 2
        w_tilde = np.array([1,1,1])  # powerlaw kernel(1.0)
        num_users = len(twitter_link_user_features)
    elif scenario == 'digg':
        feature_vector_length = 4
        w_tilde = np.array([0.01, 0.01,0.01, 0.01, 0.02])  
        num_users = 200
    elif scenario == 'memes':
        feature_vector_length = 4
        w_tilde = np.array([0.0001,0.0001,0.0001,0.0001])  
        num_users = 200
    elif scenario == 'synthetic':
        feature_vector_length = 4
        w_tilde = np.array([0.0001,0.0001,0.0001,0.0001])  
        num_users = 100
    
    
    

    memory_kernel = MVSPP_config.memory_kernel
    run_name = MVSPP_config.run_name
    MSPSP = MultiVariateSurvivalSplitPopulation(
        desc='Multivariate Inhomogenous Poisson', num_users=num_users, 
        feature_vector_length=feature_vector_length, MemoryKernel = memory_kernel)
    print("\n\nMemory kernel: ", memory_kernel)
    print("Dataset:", scenario)
    if MVSPP_config.train:
        if MVSPP_config.run_setup:
            MSPSP.setupTraining(realization_filename='../data/KDD_data/training_'+scenario+'.npy', scenario_name = scenario,run_name = run_name)
        w_tilde = MSPSP.train(reinitialize=MVSPP_config.reinitialize, scenario_name = scenario,run_name=run_name)
        mvspp_results = pd.read_csv('MVSPP_parameters.csv')
        mvspp_results = mvspp_results.append({'dataset':scenario, 'kernel':str(memory_kernel), 'w_tilde':w_tilde}, ignore_index=True)
        mvspp_results.to_csv('MVSPP_parameters.csv', index=False)

    if MVSPP_config.evaluate:
        test_msle_list_start_size, test_msle_start_size = predict_cascade_size(num_users, memory_kernel, scenario, run_name=run_name,start_sizes=MVSPP_config.start_size_list,show_status=True, dataset_type="test")
        
        # save the data
        print(test_msle_list_start_size.keys())
        for start_size in MVSPP_config.start_size_list:
            predictions = test_msle_list_start_size[start_size]
            np.save("../results/"+MVSPP_config.scenario+"/ours_start_size"+ str(start_size)+".npy", predictions)
        
        test_msle_list_tc, test_msle_tc  = predict_cascade_size_tc(num_users, memory_kernel, scenario,run_name=run_name,t_c_list=MVSPP_config.tc_list, show_status=True, dataset_type="test")
        
        
        for t_c in MVSPP_config.tc_list:
            predictions = test_msle_list_tc[t_c]
            np.save("../results/"+MVSPP_config.scenario+"/ours"+ str(t_c)+".npy", predictions)
        

    print("\n\nMemory kernel: ", memory_kernel)
    print("Dataset:", scenario)

    
if __name__ == "__main__":
    client = Client()
    print("Dask Hosted at: ", client.scheduler_info()['services'])
    main()

