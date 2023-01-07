#!/usr/bin/python
''' KMEstimator.py
    version: 1.2
    Authored by: Akshay Aravamudan,Georgios Anagnostopoulos, April 2020
      Edited by:                                              '''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from itertools import groupby
from point_processes.TemporalPointProcess import TemporalPointProcess as TPP, TrainingStatus
from point_processes.PointProcessCollection import *
from matplotlib import pyplot as plt
from utils.MemoryKernel import *
from utils.DataReader import *
import pandas as pd
from point_processes.SplitPopulationTPP import *
from point_processes.UnivariateHawkes import *
import numpy as np
import scipy.stats


# Fixing the seed to replicate experiments..
# np.random.seed(seed=233423)


# Some global settings for figure sizes
normalFigSize = (8, 6)  # (width,height) in inches
largeFigSize = (12, 9)
xlargeFigSize = (18, 12)

#==================================================================================================== 
#=============Non Parametric cumulative intensity estimation for split population ===================
#==================================================================================================== 

#--------------------log likelihood calculation--------------------------------


def convertRealizationsDataframeSplitPopNPE(realizations, features, w_tilde):
    realizations_df  = pd.DataFrame(columns=['cid', 'event', 'observed_time', 'prior'])
    for index, realization in enumerate(realizations):
        feature_vector = features[index]
        prior = 1/(1 + np.exp(-np.dot(w_tilde, feature_vector)))
        cid = index
        if len(realization[0]) > 1:
            event = 1
            observed_time = realization[0][0]
        else:
            event = 0
            observed_time = realization[0][-1]
        realizations_df = realizations_df.append({'cid':cid, 'event':event, 'observed_time':observed_time, 'prior': prior}, ignore_index=True)

    assert len(realizations_df) > 0, "length will not be zero"
    return realizations_df    
    


def get_loglikelihood(data_df, npe):
    '''
    Parameters
    ----------
    data_df : dataframe ['cid', 'observed_time', 'event', 'prior']
    
        cid: realization id;
        event: event(1) or not(0); 
        observed_time: event time (with event==1) or
                       right censored time (with event==0);
        prior: fixed prior probability of susceptibility
        
    npe : dataframe ['time', 'h']
    
        time: discrete times with point mass
        h: point mass at time 
    Returns
    -------
    loglikelihood : scaler
        the loglikelihood value of the data_df with npe 
    '''
    events_df = data_df.loc[data_df['event']==np.int(1)][['observed_time', 'prior']]
    
    events_df['event_prob'] = events_df['observed_time'].apply(lambda x:\
                              (1-npe['h'].loc[npe['time'] < x]).product() *\
                               npe['h'].loc[npe['time'] == x].iat[0])
    
    l1 = events_df.apply(lambda x: np.log(x['prior']*x['event_prob']) , axis=1).sum()
            
    censored_df = data_df.loc[data_df['event']==np.int(0)][['observed_time', 'prior']]
    
    censored_df['survival_prob'] = censored_df['observed_time'].\
                  apply(lambda x: (1-npe['h'].loc[npe['time'] <= x]).product())
                  
    l0 = censored_df.apply(lambda x: np.log(1-x['prior']+x['prior']*x['survival_prob']), axis=1).sum()

    return l0+l1
# #--------------------log likelihood calculation--------------------------------

def non_parametric_estimation_split_population(data_df, maxIter = 40, threshold = 1e-3):
    '''
    Parameters
    ----------
    data_df : dataframe ['cid', 'observed_time', 'event', 'prior']
    
        cid: realization id;
        event: event(1) or not(0); 
        observed_time: event time (with event==1) or
                       right censored time (with event==0);
        prior: fixed prior probability of susceptibility

    maxIter : scaler
        maximum number of allowed iteration. The default is 40.
    threshold : scaler
        the difference of 2 successive objective values need to be smaller 
        than the threshold value to be considered converged. 
        The default is 1e-4.

    Returns
    -------
    npe : dataframe ['time', 'h']
        time: discrete times with point mass
        h: point mass at time 
    obj_values : list 
        list of objective values for each iteration
    '''
    #------------------------------setup-------------------------------------------
    
    events_df = data_df.loc[data_df['event']==np.int(1)].drop(columns=['event']).\
                        sort_values(by='observed_time').reset_index(drop=True)
    events_df.loc[:, 'posterior'] = 1.0 
    
    events_df['eid'] = events_df['observed_time'].rank(method='dense').astype(int)   
    events_df['count'] = events_df.groupby('eid')['observed_time'].transform('count')
    
    
    censored_df = data_df.loc[data_df['event']==np.int(0)][['observed_time', 'prior']]
    
    # non parametric estimation
    npe_df = events_df.drop(columns=['cid', 'prior', 'posterior']).drop_duplicates().set_index('eid').\
                       rename(columns={'observed_time':'time', 'count':'d'})
    
    #-------------------------initialization---------------------------------------
    
    censored_df['posterior'] = censored_df['prior']
    iter_count = 0
    obj_diff = threshold + 1.0
    obj_values = [] # negative log likelihood
    #-------------------------Iteration--------------------------------------------
    
    while (iter_count < maxIter) and  \
                    (obj_diff > threshold):
        #---------------------------Maximization-----------------------------------
        posterior_df = pd.concat([ censored_df[['observed_time', 'posterior']], \
                                   events_df[['observed_time', 'posterior']] \
                                 ], sort=False)
            
        npe_df['y'] = npe_df['time'].apply(lambda x: \
                      sum(posterior_df['posterior'].loc[posterior_df['observed_time'] >= x]))
        
        npe_df['h'] = npe_df['d'] / npe_df['y']    
        
        #------------------------Expectation--------------------------------------
        censored_df['survival_prob'] = censored_df['observed_time'].\
                      apply(lambda x: (1-npe_df['h'].loc[npe_df['time'] <= x]).product())
                     
        censored_df['posterior'] = censored_df.\
        apply(lambda x: (x['prior']*x['survival_prob']) / (1-x['prior'] + x['prior']*x['survival_prob']), axis=1)   
        
        #--------------------------------------------------------------------------
        iter_count = iter_count + 1
        
        # objective value calculation: negative log likelihood
        events_df['event_prob'] = events_df['observed_time'].apply(lambda x:\
                                  (1-npe_df['h'].loc[npe_df['time'] < x]).product() *\
                                    npe_df['h'].loc[npe_df['time'] == x].iat[0])
        
        l1 = events_df[['prior', 'event_prob']].apply(lambda x: np.log(\
                                        x['prior']*x['event_prob']) , axis=1).sum()
                
        l0 = censored_df[['prior', 'survival_prob']].apply(lambda x: np.log(\
                          1-x['prior']+x['prior']*x['survival_prob']), axis=1).sum()
            
        obj_values.append(-l0-l1)
        
        obj_diff = abs(obj_values[-1] - obj_values[-2]) if iter_count > 1 else obj_diff
        print(iter_count)
    
    # npe = npe_df[['time', 'h']]
    return npe_df, obj_values
        



def greenwood_variance(npe):
    
    npe_df = npe.copy()
    npe_df['varh'] = npe_df.apply(lambda x:\
            ( x['d']/x['y']*((x['y']-x['d'])/x['y']) ) / (x['y']-1) , axis=1)

    return npe_df

def confidence_interval(npe_df, aa=0.05, distribution=scipy.stats.norm()):
    '''

    Parameters
    ----------
    npe_df : dataframe with required columns ['time', 'h', 'varh']
        
    aa : scalar 0 <aa < 1
        (1-a)*100 confidence level
    distribution : scipy.stats distributions
        The default is Gaussian standard normal distribution
    Returns
    -------
    npe_df : dataframe with ['time', 'h', 'lowerb', 'higherb']

    '''

    z = distribution.ppf(1-aa/2)
    
    npe_df['cumh'] = npe_df['h'].cumsum()
    npe_df['low'] = npe_df['cumh'] - z * np.sqrt(npe_df['varh'].cumsum()) 
    npe_df['high'] = npe_df['cumh'] + z * np.sqrt(npe_df['varh'].cumsum()) 
    
    return npe_df







#########################################################################################################################
# generateSyntheticData(): This function will generate synthetic data from the associated point process. Currently, this
#                          generates points from an inhomogenouse poisson process with the associate kernel. If no kernel
#                           is chosen, it defaults to a constant memory kernel.
# arguments:  generateIntensityPlot: boolean; If true, create a plot showing the intensity function, for a few points in a single realization
#               kernel:              MemoryKernel; The memory kernel to use for the inhomogenous poisson process, if empty,
#                                                   choose the constant memory kernel.


def generateSyntheticData(generateIntensityPlot=False, kernel=None):

    # Define the ground-truth TPP as a (inhomogenous) Poisson TPP with the associated memory kernel.
    if kernel is not None:
        mk = kernel
    else:
        mk = ConstantMemoryKernel()

    sourceNames = ['ownEvents']  # this TPP only uses its own events as source.
    truetpp = PoissonTPP(
        mk, sourceNames, desc='Poisson TPP with a Gamma Gompertz (1.0, 2.0) kernel')
    truetpp.alpha = 1.0  # set parameter for the TPP's conditional intensity.

    # Generate some realizations with random start times & right-censoring times
    numRealizations = 10
    Realizations = []
    maxNumEvents = 0
    maxNumEventsIdx = 0
    maxTime = 0
    minMaxTime = np.inf
    for r in range(0, numRealizations):
        # Exponential(100)-distributed right-censoring time
        # T = scipy.stats.expon.rvs(loc=0.0, scale=20.0)
        T = 20
        Realization = truetpp.simulate(T, [np.array([])])
        Realizations.append([Realization])
        # print(Realization)
        numEvents = len(Realization)
        if numEvents > maxNumEvents:
            maxNumEvents = numEvents
            maxNumEventsIdx = r
        if Realization[-1] > maxTime:
            maxTime = Realization[-1]
        if maxTime < minMaxTime:
            minMaxTime = maxTime

    # Adding the right censoring time to each realization
    for Realization in Realizations:
        Realization[0] = list(Realization[0])
        Realization[0].append(maxTime)

    if generateIntensityPlot:
        intensity_plot(truetpp, Realization, num_points=5)
        plt.show()
    return Realizations, maxNumEventsIdx, maxTime

###################################################################################################################
#  fitModelInhomogenousPoisson(): This function will fit a set of realizations to an inhomogenous process
#                                  with a kernel of our choosing.
# Arguments:   Realizations: List of realizations for fitting; The structure is the standard used for this library.
#              kernel: MemoryKernel;The kernel to be used for the inhomogenous process to which the realizations
#               will be fit. If empty, the constant kernel will be chosen.
# return:       TemporalPointProcess; The point process object for the fitted inhomogenous poisson process


def fitModelInhomogenousPoisson(Realizations, kernel=None):
    # Define the model TPP as a (inhomogenous) Poisson TPP with a GammaGompertz(1.0, 2.0) memory kernel.
    print("***************************** Fitting an inhomogenous process *************************")
    if kernel is None:
        print("\n \t No kerne has been chosen, using the constant memory kernel...")
        mk = ConstantMemoryKernel()
    else:
        mk = kernel
    sourceNames = ['ownEvents']  # this TPP only uses its own events as source.
    modeltpp = PoissonTPP(
        mk, sourceNames, desc='Poisson TPP with {0} kernel'.format(mk.desc))

    # Train the model
    # pre-calculates quantities needed for training
    modeltpp.setupTraining(Realizations)
    status = modeltpp.train()
    print('Training resulted in a status of {:1}'.format(status))
    print('Estimated a={:1}'.format(
        modeltpp.alpha))
    print("***************************** Fitting complete.... ***********************************")
    return modeltpp


################################################################################################
# fitModelHawkes(): Function to fit a set of realizations to a hawkes process with an
#                   associated memory kernel.
# Arguments:    Realizations: Set of realizations, as per standard input structure of this library
#               kernel: memory kernel for hawkes process. If empty, the exponential pseudo memory
#                       is chosen.
def fitModelHawkes(Realizations, kernel=None):

    # Realizations = [Realizations
    if kernel is None:
        mk = ExponentialPseudoMemoryKernel(beta=1.0)
    else:
        mk = kernel

    sourceNames = ['self']
    pre_cal_path = os.getcwd()
    stop_criteria = {'max_iter': 20,
                     'epsilon': 1e-4}
    modeltpp = HawkesTPP(
        mk, sourceNames, stop_criteria, pre_cal_path,
        desc='Hawkes TPP with exponetial(beta=1) kernel')
    modeltpp.setupTraining(Realizations)
    modeltpp.train(Realizations)

    return modeltpp


#################################################################################################################
# NelsonAalenEstimation(): Function used to perform nelson aalen estimation from realizations of an inhomogenous
#                          poisson process.
#                           If a set of realizations is passed, then it will consider all points from all
#                           all realizations to derive estimate for alpha * psi(), where alpha is the parameter
# Arguments:
#           Realizations: set of realizations used to derive the estimate
#           maxTime: the final time upto which the estimator function
#
def NelsonAalenEstimates(Realizations, maxTime):
    TPPdata = [Realizations]
    RealizationsList = TPPdata[0]

    # Find number J of distinct relative event times

    EventTimeListFull = []
    RightCensoringTimeList = []
    for realization in RealizationsList:
        EventTimeListFull.append(list(realization[0]))
        RightCensoringTimeList.append(realization[0][-1])
    # flatten list of lists
    EventTimeListFull = [
        val for sublist in EventTimeListFull for val in sublist]
    # eliminate duplicate event times
    EventTimeList = list(set(EventTimeListFull))
    EventTimeList.sort()
    J = len(EventTimeList)
    # Determine d_j's
    EventTimeListFull.sort()
    # d_i
    KMd = [len(list(group)) for key, group in groupby(EventTimeListFull)]
    # Determine Y_j's
    # Y_i
    KMY = []
    for tj in EventTimeList:
        # num of realizations, whose right-censoring time is not less than tj
        numRealizations = len(
            [Tr for Tr in RightCensoringTimeList if Tr >= tj])
        KMY.append(numRealizations)
    h = np.array(KMd) / np.array(KMY)

    SList = [1.0]
    hList = [0.0]
    Sj = 1.0
    lg_Sj = 0.0
    for j in range(J):
        if h[j] != 1.0:
            lg_Sj = np.log((1.0 - h[j])) + lg_Sj
            SList.append(lg_Sj)
        else:
            SList.append(SList[-1])
        hList.append(h[j])

    t_h = np.linspace(0.0, maxTime, num=len(hList))

    t = np.linspace(0.0, maxTime)

    EventTimeListTiles = np.tile(EventTimeList, (len(t), 1)).T
    tTiles = np.tile(t, (len(EventTimeList), 1))
    idx = sum(((EventTimeListTiles <= tTiles).astype(int)))

    SList = np.array(SList)[idx]
    hList = np.array(hList)

    import scipy.stats
    H_hat = np.cumsum(np.array(hList))[idx]
    VarH = np.zeros(len(H_hat))

    KMd.insert(0, 0.0)
    KMY.insert(0, 1.0)

    KMd = np.array(KMd)[idx]
    KMY = np.array(KMY)[idx]

    for i in range(len(H_hat)):
        for j in range(i):
            VarH[i] += KMd[j]*(KMY[j] - KMd[j]) / KMY[j]**3
            # VarH[i] += KMd[j] / KMY[j]**2

    stdNormDistr = scipy.stats.norm()
    aa = 0.1  # 90% CI

    q = stdNormDistr.ppf(aa/2.0)

    CI_half_width = q * np.sqrt(np.abs(VarH))
    CI_below = H_hat - CI_half_width
    CI_above = H_hat + CI_half_width

    return t, SList, H_hat, CI_below, CI_above

#################################################################################################
# estimateKernelShapeUnivariate(): This function uses nelson-aalen estimate to perform a lasso
#                                  regression over a set of pre-determined kernels. This is
#                                  in an attempt to represent unknown kernels as a linear combination
#                                  of pre-determined kernels. Currently it is implemented as a linear
#                                   combination of a family of gamma gompertz memory kernels.
# Arguments:
#           t           : np.array (np.linspace) The time  linspace that were used to generate the
#                           nelson-aalen estimates
#           h_list      : list of
# returns:
#           linear_combination: linear combination (of gamma gompertz) estimates for alpha * psi()
#           linear_combination_tpp: point process of linear combination of gamma gompertz kernels.


def estimateKernelShapeUnivariate(t, h_list):
    param_space_beta = np.linspace(1.0, 2.0, num=10)
    gamma_gompertz_beta_space, gamma_gompertz_gamma_space = np.meshgrid(
        param_space_beta, param_space_beta, indexing='ij')

    kernel_list = []

    # add constant kernel
    kernel_list.append(ConstantMemoryKernel())
    # for param in weibull_param_space:
    #     kernel_list.append(WeibullMemoryKernel(param))
    cnt = 0
    for i in range(len(gamma_gompertz_beta_space)):
        for j in range(len(gamma_gompertz_gamma_space)):
            # print(gamma_gompertz_gamma_space[i])
            if gamma_gompertz_beta_space[i][j] == 1.0 and gamma_gompertz_gamma_space[i][j] == 2.0:
                true_kernel_position = len(kernel_list)
            # print("------------:", cnt, gamma_gompertz_beta_space[i][j],
            #       " ", gamma_gompertz_gamma_space[i][j])
            cnt += 1
            kernel_list.append(GammaGompertzMemoryKernel(
                beta=gamma_gompertz_beta_space[i][j], gamma=gamma_gompertz_gamma_space[i][j]))

    Y = h_list

    nan_indices = np.isinf(Y)
    X = np.zeros((len(t), len(kernel_list)))

    for i in range(len(t)):
        for j in range(len(kernel_list)):
            X[i][j] = kernel_list[j].psi(t[i])

    lin = Lasso(alpha=0.00000001, precompute=True, max_iter=10000, fit_intercept=True,
                positive=True, random_state=9999, selection='random')
    print(X)
    lin.fit(X[:-1], Y[:-1])

    print("\n The alpha value associated with the true value is: ",
          lin.coef_[true_kernel_position])
    print("\n The maximum alpha value in the array is given by: ", max(lin.coef_))
    print("\n Sum of all the alpha values for all kernels: ", sum(lin.coef_))

    linear_combination = np.zeros(len(t))
    for i in range(len(t[:-1])):
        linear_combination[i] = np.dot(X[i], lin.coef_)

    sourceNames = ["self"]
    mk = CompositeMemoryKernel(kernel_list)
    linear_combination_tpp = PoissonTPP(
        mk, sourceNames, desc='Poisson TPP with {0} kernel'.format(mk.desc))

    return linear_combination, linear_combination_tpp

#################################################################################################
# NelsonAalenEstimation(): This function generates a plot of the nelson-aalen estimates in the
#                           time- period spanned by the data
#
# Arguments: Realizations: realizations used to generate NA estimates
#              maxTime: The maximum time spanned by the data, this is used for plotting.
#
# returns: True if succesful.


def NelsonAalenEstimation(Realizations, maxTime):

    t, phi_estimates, H_hat, CI_below, CI_above = NelsonAalenEstimates(
        Realizations, maxTime)
    _, ax = plt.subplots(1, 1, figsize=normalFigSize)
    t_h = np.linspace(0, maxTime, len(H_hat))
    ax.plot(t_h, H_hat, 'g')
    ax.plot(t_h, CI_below, 'r--')
    ax.plot(t_h, CI_above, 'r--')
    ax.legend(['alpha*Psi Estimate from H_hat',
               '90% CI', '90% CI'])

    return True

#################################################################################################
# generateSplineKernelProcess(): This function generates an inhomogenous poisson point process
#                                 with the spline kernel generates by nelson aalen estimates of
#                                   of the realizations.
# Arguments: Realizations: realizations used to generate NA estimates
#              maxTime: The maximum time spanned by the data, this is used for plotting.
#
# returns: True if succesful.


def generateSplineKernelProcess(Realizations, maxTime):
    t, phi_estimates, H_hat, CI_below, CI_above = NelsonAalenEstimates(
        Realizations, maxTime)
    t_h = np.linspace(0, maxTime, len(H_hat))

    mkS = SplineMemoryKernel(
        x_vals=t, y_vals=H_hat, timeOffset=0.0)
    new_tpp = PoissonTPP(mkS, ['ownEvents'])
    new_tpp.desc = "Spline kernel based inhomogenous poisson process"

    return new_tpp


def getSplineKernel():
    training_features, training_realizations, training_isExploited, \
    test_features, test_realizations, test_isExploited, \
    validation_features, validation_realizations, validation_isExploited = generateExploitSocialMediaDataset()
   
    training_modified_realizations = [np.array(realization[0]) for realization in training_realizations]
    maxTime = max([max(realization) for realization in training_modified_realizations])
    t, phi_estimates, h_arr, CI_below, CI_above = NelsonAalenEstimates(
        training_realizations, maxTime)
  
    t_h = np.linspace(0, maxTime, num=len(h_arr))
    print("Generating spline kernel from Nelson Aalen estimates...")
    mkS = SplineMemoryKernel(
        x_vals=t_h, y_vals=h_arr, timeOffset=0.0)

    return mkS


def chicagoCrimeNonParametric():
    for year in range(2001, 2019):
        print("Fitting chicago crime data for robbery in year{0}".format(year))
        Realizations, maxNumEventsIdx, maxTime = readChicagoCrimeData(
            crime="theft", year=2016)
        Realizations, maxNumEventsIdx, maxTime = generateSyntheticData(
            kernel=GompertzMemoryKernel(gamma=0.3))

        model_tpp = fitModelInhomogenousPoisson(
            Realizations, kernel=RayleighMemoryKernel())

        model_tpp = fitModelHawkes(
            Realizations, kernel=ExponentialPseudoMemoryKernel(beta=1.0))

        # pick, for example, the longest realization that was used for training.

        Realization = np.array(Realizations[maxNumEventsIdx])
        generatePP_plot(Realization[0], model_tpp)



def UnitTestNonParametricSplitPop():
    training_features, training_realizations, training_isExploited, \
    test_features, test_realizations, test_isExploited, \
    validation_features, validation_realizations, validation_isExploited = generateExploitSocialMediaDataset()
    mkList = [WeibullMemoryKernel(0.8),
              PowerLawMemoryKernel(beta=1.0),
              ExponentialPseudoMemoryKernel(beta=0.0112),
              ExponentialPseudoMemoryKernel(beta=0.0112)]
    _, ax = plt.subplots(1, 1, figsize=largeFigSize)
    # this TPP only uses its own events as source.
    sourceNames = ['base', 'github', 'reddit', 'twitter']

    stop_criteria = {'max_iter': 200,
                     'epsilon': 1e-12}

    exploitProcess = SplitPopulationTPP(
        mkList, sourceNames, stop_criteria,
        desc='Split population process with multiple kernels')

    features = exploitProcess.setFeatureVectors(training_features)
    w_tilde = np.array([  3.5783908,  -3.94764495, -1.36925425])
    survival_df = convertRealizationsDataframeSplitPopNPE(training_realizations, features, w_tilde)
    npe_df,obj_values = non_parametric_estimation_split_population(survival_df)
    plt.plot(npe_df['time'].values, npe_df['h'].values)

    realizations = [x for x in training_realizations if len(x[0]) > 1]
    maxTime = 25000
    t, phi_estimates, h_arr, CI_below, CI_above = NelsonAalenEstimates(
        realizations, maxTime)
    plt.xlim(0, maxTime)
    plt.plot(t,-phi_estimates)
    plt.show()
                

def UnitTestNonParametric():
    training_features, training_realizations, training_isExploited, \
    test_features, test_realizations, test_isExploited, \
    validation_features, validation_realizations, validation_isExploited = generateExploitSocialMediaDataset()
    
    def removeSocialMedia(realizations, features):
        social_media_indices = []
        for index, realization in enumerate(realizations):
            if len(realization[1]) > 1 or len(realization[1]) > 1 or len(realization[1]) > 1:
                social_media_indices.append(index)     
        return realizations[~np.array(social_media_indices)], features[~np.array(social_media_indices)]
    
    training_realizations, training_features =removeSocialMedia(training_realizations, training_features) 
    validation_realizations, validation_features = removeSocialMedia(validation_realizations,validation_features )
    test_realizations, test_features = removeSocialMedia(test_realizations,test_features )
    
    training_modified_realizations = [[np.array(realization[0])] for realization in training_realizations]
    validation_modified_realizations = [[np.array(realization[0])] for realization in validation_realizations]
    test_modified_realizations = [[np.array(realization[0])] for realization in test_realizations]

    _, ax = plt.subplots(1, 1, figsize=largeFigSize)
    maxTime = 25000
    t_validation, phi_estimates_val, h_arr_val, CI_below, CI_above = NelsonAalenEstimates(
        validation_modified_realizations, maxTime)
    t_test, phi_estimates_test, h_arr_test, CI_below, CI_above = NelsonAalenEstimates(
        test_modified_realizations, maxTime)
    t_training, phi_estimates_train, h_arr_train, CI_below, CI_above = NelsonAalenEstimates(
        training_modified_realizations, maxTime)
    
    t_h_train = np.linspace(0, maxTime, num=len(h_arr_train))
    t_h_test= np.linspace(0, maxTime, num=len(h_arr_test))
    t_h_validation = np.linspace(0, maxTime, num=len(h_arr_val))

    print("Generating spline kernel from Nelson Aalen estimates...")
    # mk = GompertzMemoryKernel(gamma=0.0004)
    mkS = SplineMemoryKernel(
        x_vals=t_h_train, y_vals=-phi_estimates_train, timeOffset=0.0)
    # print(mkS.psi(0.5))
    # input()
    scenario_name = "base_kernel_exp_6"
    # this TPP only uses its own events as source.
    sourceNames = ['base', 'github', 'reddit', 'twitter']

    stop_criteria = {'max_iter': 50,
                     'epsilon': 1e-9}

    mkList = [mkS,
            PowerLawMemoryKernel(beta=1.0),
            ExponentialPseudoMemoryKernel(beta=1.0),
            ExponentialPseudoMemoryKernel(beta=1.0)]
    exploitProcess = SplitPopulationTPP(
        mkList, sourceNames, stop_criteria,
        desc='Split population process with multiple kernels')
    exploitProcess.alpha = np.ones(4)
    exploitProcess.w_tilde = np.random.rand(3)
    # ############################### Training ##########################################
    exploitProcess.setFeatureVectors(training_features)
    exploitProcess.setupTraining(training_realizations, training_features,scenario_name , validation_MTPPdata=validation_realizations)
    _,plot_list = exploitProcess.train(training_realizations, scenario_name, validation_MTPPdata= validation_realizations)

    new_tpp = PoissonTPP(mkS, ['ownEvents'])
    new_tpp.desc = "Spline kernel based inhomogenous poisson process"
    new_tpp.alpha = 1.0
    _, ax = plt.subplots(1, 1, figsize=normalFigSize)

    # ax.plot(t, -phi_estimates, 'b')

    ax.plot(t_h_train, new_tpp.mk.psi(t_h_train), 'r', label='process with spline kernel')
    # ax.plot(t_h, h_arr, 'g')
    # print(new_tpp.alpha)
    ax.plot(t_validation, -phi_estimates_val, 'r', label='estimates from validation')
    ax.plot(t_test, -phi_estimates_test, 'g', label='estimates from test')
    ax.plot(t_training, -phi_estimates_train, 'y', label='estimates from training')
    ax.legend()
    ax.set_ylim((0,4))

    # ax.plot(t_h[:-1], linear_combination[:-1], 'k-')

    # some test kernels us ed to compare with KM estimate
    plt.legend()

    ax.set_xlabel('t')
    ax.set_ylabel('$\\alpha \\psi(t) $')
    plt.show()
    fig, (ax1) = plt.subplots(2, 2, squeeze=False, figsize=xlargeFigSize)
    


if __name__ == "__main__":
    UnitTestNonParametricSplitPop()
