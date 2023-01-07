#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from point_processes.SplitPopulationTPP import *
from point_processes.SurvivalProcess import *
from point_processes.DiscriminativeSplitPopulationTPP import *
from sklearn.metrics import confusion_matrix
from point_processes.NonParametricEstimator import *
from utils.DataReader import generateSyntheticDataset

normalFigSize = (8, 6)  # (width,height) in inches
largeFigSize = (9, 6)
xlargeFigSize = (18, 12)



# This functions generate the base kernel using Non-parametric estimation
def UnitTestSplitPopNPE():
    training_features, training_realizations, training_isExploited, \
        test_features, test_realizations, test_isExploited, \
        validation_features, validation_realizations, validation_isExploited = generateExploitSocialMediaDataset()

    
    


    # survival_df = convertRealizationsDataframeSplitPopNPE(training_realizations, modified_training_features, w_tilde)
    # npe_df,obj_values = non_parametric_estimation_split_population(survival_df)
    # mkS = SplineMemoryKernel(
    #     x_vals=np.array(npe_df['time'].values), y_vals=np.array(np.cumsum(npe_df['h'].values)), timeOffset=0.0)

    # x = np.linspace(0,25000,1000)
    # y = 1.12019819e-01 * mkS.AntiPhi(x)
    # y_beta = 0.00025052* WeibullMemoryKernel(0.8).phi(x)
    # # y_bathub = WeibullMemoryKernel(gamma=1.0, beta = 1.0).phi(x)
    # plt.plot(x,y,label="Spline, s = 0.5")
    
    # # plt.plot(x,y_beta,label="Weibull")
    # plt.plot(npe_df['time'],1.12019819e-01*np.cumsum(npe_df['h']), label="NPE Estimate")
    # plt.title('alpha*Phi()')
    # plt.legend()
    # plt.show()


    mkList = [WeibullMemoryKernel(0.8),
              PowerLawMemoryKernel(beta=1.0),
              ExponentialPseudoMemoryKernel(beta=0.0112),
              ExponentialPseudoMemoryKernel(beta=0.0112)]
    _, ax = plt.subplots(1, 1, figsize=largeFigSize)

    scenario_name = "splitPopSplineCorrected"
    # this TPP only uses its own events as source.
    sourceNames = ['base', 'github', 'reddit', 'twitter']

    stop_criteria = {'max_iter': 50,
                     'epsilon': 1e-7}

    exploitProcess = SplitPopulationTPPDiscriminative(
        mkList, sourceNames, stop_criteria,
        desc='Split population process with multiple kernels')
    exploitProcess.w_tilde = np.random.rand(3)
    exploitProcess.setFeatureVectors(training_features)
    exploitProcess.setupTraining(training_realizations, training_features,scenario_name , validation_MTPPdata=validation_realizations)
    _,plot_list = exploitProcess.train(training_realizations, scenario_name, validation_MTPPdata= validation_realizations)
   
def remove_social_media_events(realizations):
    realizations_no_soc = []
    for realization in realizations:
        realization_no_soc = [realization[0]]
        # print(realization_no_soc)
        # input()
        realizations_no_soc.append(realization_no_soc)
    
    return realizations_no_soc
    

def UnitTestSplitPopSynthetic():
    stop_criteria = {'max_iter': 800,
                     'epsilon': 1e-20}

    synthetic_realizations, synthetic_features = generateSyntheticDataset()
    synthetic_realizations_no_soc = remove_social_media_events(synthetic_realizations)

    # split-population survival process: exploit
    # Obtained from Xixi
    # exploit_mk = {'base': memory_kernel.WeibullMemoryKernel(0.8), \
    #             'github': memory_kernel.ExponentialMemoryKernel(beta=1.0),\
    #             'reddit': memory_kernel.ExponentialMemoryKernel(beta=1.0), \
    #             'twitter':memory_kernel.PowerLawMemoryKernel(beta=1.0)}

    mkList = [WeibullMemoryKernel(0.8),
              ExponentialPseudoMemoryKernel(beta=1.0),
                ExponentialPseudoMemoryKernel(beta=1.0),
                PowerLawMemoryKernel(beta=1.0)]
  
    _, ax = plt.subplots(1, 1, figsize=largeFigSize)
    
    sourceNames = ['base', 'github', 'reddit', 'twitter']
    exploitProcess = SplitPopulationTPP(
        mkList, sourceNames, stop_criteria,
        desc='Split population process with multiple kernels')

    mkList_no_soc = [WeibullMemoryKernel(0.8)]
    
    sourceNames_no_soc = ['base']
    exploitProcessWithoutSocialMedia = SplitPopulationTPP(
        mkList_no_soc, sourceNames_no_soc, stop_criteria,
        desc='Split population process with multiple kernels')

    
    ##################   Define Survival Process  ################
    mkList = [WeibullMemoryKernel(0.8),
              ExponentialPseudoMemoryKernel(beta=1.0),
                ExponentialPseudoMemoryKernel(beta=1.0),
                PowerLawMemoryKernel(beta=1.0)]
    _, ax = plt.subplots(1, 1, figsize=largeFigSize)
    sourceNames = ['base', 'github', 'reddit', 'twitter']
    survivalProcess = SurvivalProcess(
        mkList, sourceNames, stop_criteria,
        desc='Split population process with multiple kernels')


    mkList_no_soc = [WeibullMemoryKernel(0.8)]
    sourceNames_no_soc = ['base']
    survivalProcessWithoutSocialMedia = SurvivalProcess(
        mkList_no_soc, sourceNames_no_soc, stop_criteria,
        desc='Split population process with multiple kernels')
    # ############################ Training ##########################################
    # ######## +SM: model 6 ###########
    # exploitProcess.w_tilde = np.array([1.38, 2.06, 1.88])
    # scenario_name = "+SM_model_synthetic_newss"
    
    # exploitProcess.setFeatureVectors(synthetic_features, synthetic_features, append=False)
    # exploitProcess.setupTraining(TPPdata=synthetic_realizations,pre_cal_filename= scenario_name , validation_MTPPdata=synthetic_realizations)
    # best_parameter_dict_validation = exploitProcess.train(synthetic_realizations, scenario_name, validation_MTPPdata= synthetic_realizations)
    # best_parameter_dict_validation['scenario'] = scenario_name
    # input()
    
    # # ######## -SM: model 5 ########
    scenario_name = "-SM_model_soc_synthetic"
    exploitProcessWithoutSocialMedia.w_tilde = np.array([1.38, 2.06, 1.88]) 
    exploitProcessWithoutSocialMedia.setFeatureVectors(synthetic_features, synthetic_features, append=False)
    exploitProcessWithoutSocialMedia.setupTraining(synthetic_realizations_no_soc,scenario_name, validation_MTPPdata=synthetic_realizations_no_soc)
    best_parameter_dict_validation = exploitProcessWithoutSocialMedia.train(synthetic_realizations_no_soc, scenario_name, validation_MTPPdata= synthetic_realizations_no_soc)
    best_parameter_dict_validation['scenario'] = scenario_name
    results_df = results_df.append(best_parameter_dict_validation,ignore_index=True)
    
    # # ######## -SM: model 5 for synthetic data ########
    scenario_name = "-SM_model_synthetic_new"
    syntheticExploitProcessWithoutSocialMedia.w_tilde = np.array([1.38, 1.88,1]) 
    syntheticExploitProcessWithoutSocialMedia.setFeatureVectors(synthetic_features, synthetic_features)
    syntheticExploitProcessWithoutSocialMedia.setupTraining(synthetic_realizations_no_soc,scenario_name, validation_MTPPdata=synthetic_realizations_no_soc)
    best_parameter_dict_validation = syntheticExploitProcessWithoutSocialMedia.train(synthetic_realizations_no_soc, scenario_name, validation_MTPPdata= synthetic_realizations_no_soc)
    best_parameter_dict_validation['scenario'] = scenario_name
    results_df = results_df.append(best_parameter_dict_validation,ignore_index=True)
    input()


    # ######## equally susceptible +SM: model 4 ########
    # scenario_name = "equally_susceptible_+SM_model_synthetic"
    # exploitProcess.w_tilde = np.array([1.38, 2.06, 1.88,1])
    # empty_training_feature_vectors = np.zeros_like(training_features)
    # empty_validation_feature_vectors = np.zeros_like(validation_features)
    
    # exploitProcess.setFeatureVectors(empty_training_feature_vectors, empty_validation_feature_vectors)
    # exploitProcess.setupTraining(TPPdata=training_realizations,pre_cal_filename=scenario_name , validation_MTPPdata=validation_realizations)
    # best_parameter_dict_validation = exploitProcess.train(training_realizations, scenario_name, validation_MTPPdata= validation_realizations)
    # best_parameter_dict_validation['scenario'] = scenario_name
    # results_df = results_df.append(best_parameter_dict_validation,ignore_index=True)
    
    # ######## equally susceptible -SM: model 3 ########
    # exploitProcessWithoutSocialMedia.w_tilde = np.array([1.38, 2.06, 1.88,1]) 
    # scenario_name = "equally_susceptible_-SM_model"
    # empty_training_feature_vectors = np.zeros_like(training_features)
    # empty_validation_feature_vectors = np.zeros_like(validation_features)
    # exploitProcessWithoutSocialMedia.setFeatureVectors(empty_training_feature_vectors, empty_validation_feature_vectors)
    # exploitProcessWithoutSocialMedia.setupTraining(training_realizations_no_soc,scenario_name, validation_MTPPdata=validation_realizations_no_soc)
    # best_parameter_dict_validation = exploitProcessWithoutSocialMedia.train(training_realizations_no_soc, scenario_name, validation_MTPPdata= validation_realizations_no_soc)
    # best_parameter_dict_validation['scenario'] = scenario_name
    
    # results_df = results_df.append(best_parameter_dict_validation,ignore_index=True)
    

    ######## all susceptible +SM: model 2 ########
    scenario_name = "all_susceptible_+SM_model_synthetic"
    survivalProcess.w_tilde = np.array([1.38, 2.06, 1.88,1])
    full_training_feature_vectors = np.ones_like(training_features)*np.inf
    full_validation_feature_vectors = np.ones_like(validation_features)*np.inf
    
    survivalProcess.setFeatureVectors(full_training_feature_vectors, full_validation_feature_vectors)
    survivalProcess.setupTraining(TPPdata=training_realizations,pre_cal_filename=scenario_name , validation_MTPPdata=validation_realizations)
    best_parameter_dict_validation = survivalProcess.train(training_realizations, scenario_name, validation_MTPPdata= validation_realizations)
    best_parameter_dict_validation['scenario'] = scenario_name
    results_df = results_df.append(best_parameter_dict_validation,ignore_index=True)
    input()
    # ######## all susceptible -SM: model 1 ########
    scenario_name = "all_susceptible_-SM_model_synthetic"
    survivalProcessWithoutSocialMedia.w_tilde = np.array([1.38, 2.06, 1.88,1])
    full_training_feature_vectors = np.ones_like(training_features)*np.inf
    full_validation_feature_vectors = np.ones_like(validation_features)*np.inf
    
    
    survivalProcessWithoutSocialMedia.setFeatureVectors(full_training_feature_vectors, full_validation_feature_vectors)
    survivalProcessWithoutSocialMedia.setupTraining(training_realizations_no_soc,scenario_name , validation_MTPPdata=validation_realizations_no_soc)
    best_parameter_dict_validation = survivalProcessWithoutSocialMedia.train(training_realizations_no_soc, scenario_name , validation_MTPPdata= validation_realizations_no_soc)
    best_parameter_dict_validation['scenario'] = scenario_name
    results_df = results_df.append(best_parameter_dict_validation,ignore_index=True)
    
    # results_df.to_csv("journal_results.csv")
    ############################### End training ##################################################################

    # exploitProcess.alpha = np.array([1.39461530e-03, 1.33951665e-04 ,2.35407472e-04, 2.85077666e-05])
    # exploitProcess.w_tilde = np.array([ 3.51311777, -1.24805883,  2.26495892])

    # exploitProcessWithoutSocialMedia.alpha = np.array([7.15636894e-01, 1.20748901e-03, 3.46717009e-04, 1.82742722e-04])
    # exploitProcessWithoutSocialMedia.w_tilde = np.array([1.4292924,  -2.21871001, -0.78951763])

    

def UnitTestSplitPop():
    stop_criteria = {'max_iter': 800,
                     'epsilon': 1e-20}

    training_features, training_realizations, training_isExploited, \
        test_features, test_realizations, test_isExploited, \
        validation_features, validation_realizations, validation_isExploited = generateExploitSocialMediaDataset()
    synthetic_realizations, synthetic_features = generateSyntheticDataset()
    
    # features, realizations, isExploited = generateRealWorldExploitDataset()
    
    training_realizations_no_soc = remove_social_media_events(training_realizations)
    validation_realizations_no_soc = remove_social_media_events(validation_realizations)
    test_realizations_no_soc = remove_social_media_events(test_realizations)
    synthetic_realizations_no_soc = remove_social_media_events(synthetic_realizations)

    exponential_beta = 0.5
    power_law_beta = 2.0

    mkList = [WeibullMemoryKernel(0.5),
              ExponentialPseudoMemoryKernel(beta=exponential_beta),
                ExponentialPseudoMemoryKernel(beta=exponential_beta),
                PowerLawMemoryKernel(beta=power_law_beta)]
    mkList_synthetic = [WeibullMemoryKernel(0.8),
              ExponentialPseudoMemoryKernel(beta=1.0),
              ExponentialPseudoMemoryKernel(beta=1.0),
              PowerLawMemoryKernel(beta=1.0)]

    _, ax = plt.subplots(1, 1, figsize=largeFigSize)
    sourceNames = ['base', 'github', 'reddit', 'twitter']
    exploitProcess = SplitPopulationTPP(
        mkList, sourceNames, stop_criteria,
        desc='Split population process with multiple kernels')

    mkList_no_soc = [WeibullMemoryKernel(0.5)]
    mkList_synthetic_no_soc = [WeibullMemoryKernel(0.8)]
    
    sourceNames_no_soc = ['base']
    exploitProcessWithoutSocialMedia = SplitPopulationTPP(
        mkList_no_soc, sourceNames_no_soc, stop_criteria,
        desc='Split population process with multiple kernels')

    ##################   Define Syntehtic Process  ################
    syntheticExploitProcess = SplitPopulationTPP(
        mkList_synthetic, sourceNames, stop_criteria,
        desc='Split population process with multiple kernels')
    syntheticExploitProcessWithoutSocialMedia = SplitPopulationTPP(
        mkList_synthetic_no_soc, sourceNames_no_soc, stop_criteria,
        desc='Split population process with multiple kernels')

    ##############################################################
    

    ##################   Define Survival Process  ################
    mkList = [WeibullMemoryKernel(0.5),
              ExponentialPseudoMemoryKernel(beta=exponential_beta),
                ExponentialPseudoMemoryKernel(beta=exponential_beta),
                PowerLawMemoryKernel(beta=power_law_beta)]
    _, ax = plt.subplots(1, 1, figsize=largeFigSize)
    sourceNames = ['base', 'github', 'reddit', 'twitter']
    survivalProcess = SurvivalProcess(
        mkList, sourceNames, stop_criteria,
        desc='Split population process with multiple kernels')


    mkList_no_soc = [WeibullMemoryKernel(0.5)]
    sourceNames_no_soc = ['base']
    survivalProcessWithoutSocialMedia = SurvivalProcess(
        mkList_no_soc, sourceNames_no_soc, stop_criteria,
        desc='Split population process with multiple kernels')
    ##############################################################
    

    # results_df = pd.DataFrame(columns=['scenario', 'alpha', 'w_tilde', 'neg_log_likelihood_val', 'neg_log_likelihood_train'])
    results_df = pd.read_csv('journal_results.csv')
    ############################# Training ##########################################
    # ######## +SM: model 6 ###########
    # exploitProcess.w_tilde = np.array([1.38, 2.06, 1.88,1])
    # scenario_name = "+SM_model"
    
    # exploitProcess.setFeatureVectors(training_features, validation_features)
    # exploitProcess.setupTraining(TPPdata=training_realizations,pre_cal_filename= scenario_name , validation_MTPPdata=validation_realizations)
    # best_parameter_dict_validation = exploitProcess.train(training_realizations, scenario_name, validation_MTPPdata= validation_realizations)
    # best_parameter_dict_validation['scenario'] = scenario_name
    # results_df = results_df.append(best_parameter_dict_validation,ignore_index=True)
    
    # ######## +SM: model 6 for synthetic data###########
    syntheticExploitProcess.w_tilde = np.array([1.38, 2.06,1,1])
    scenario_name = "+SM_model_synthetic_new"

    syntheticExploitProcess.setFeatureVectors(synthetic_features, synthetic_features)
    syntheticExploitProcess.setupTraining(TPPdata=synthetic_realizations,pre_cal_filename= scenario_name , validation_MTPPdata=synthetic_realizations)
    best_parameter_dict_validation = syntheticExploitProcess.train(synthetic_realizations, scenario_name, validation_MTPPdata= synthetic_realizations)
    best_parameter_dict_validation['scenario'] = scenario_name
    results_df = results_df.append(best_parameter_dict_validation,ignore_index=True)
    input()
    # # ######## -SM: model 5 ########
    # scenario_name = "-SM_model"
    # exploitProcessWithoutSocialMedia.w_tilde = np.array([1.38, 2.06, 1.88,1]) 
    # exploitProcessWithoutSocialMedia.setFeatureVectors(training_features, validation_features)
    # exploitProcessWithoutSocialMedia.setupTraining(training_realizations_no_soc,scenario_name, validation_MTPPdata=validation_realizations_no_soc)
    # best_parameter_dict_validation = exploitProcessWithoutSocialMedia.train(training_realizations_no_soc, scenario_name, validation_MTPPdata= validation_realizations_no_soc)
    # best_parameter_dict_validation['scenario'] = scenario_name
    # results_df = results_df.append(best_parameter_dict_validation,ignore_index=True)
    
    # # ######## -SM: model 5 for synthetic data ########
    scenario_name = "-SM_model_synthetic_new"
    syntheticExploitProcessWithoutSocialMedia.w_tilde = np.array([1.38, 1.88,1]) 
    syntheticExploitProcessWithoutSocialMedia.setFeatureVectors(synthetic_features, synthetic_features)
    syntheticExploitProcessWithoutSocialMedia.setupTraining(synthetic_realizations_no_soc,scenario_name, validation_MTPPdata=synthetic_realizations_no_soc)
    best_parameter_dict_validation = syntheticExploitProcessWithoutSocialMedia.train(synthetic_realizations_no_soc, scenario_name, validation_MTPPdata= synthetic_realizations_no_soc)
    best_parameter_dict_validation['scenario'] = scenario_name
    results_df = results_df.append(best_parameter_dict_validation,ignore_index=True)
    input()


    # ######## equally susceptible +SM: model 4 ########
    # scenario_name = "equally_susceptible_+SM_model"
    # exploitProcess.w_tilde = np.array([1.38, 2.06, 1.88,1])
    # empty_training_feature_vectors = np.zeros_like(training_features)
    # empty_validation_feature_vectors = np.zeros_like(validation_features)
    
    # exploitProcess.setFeatureVectors(empty_training_feature_vectors, empty_validation_feature_vectors)
    # exploitProcess.setupTraining(TPPdata=training_realizations,pre_cal_filename=scenario_name , validation_MTPPdata=validation_realizations)
    # best_parameter_dict_validation = exploitProcess.train(training_realizations, scenario_name, validation_MTPPdata= validation_realizations)
    # best_parameter_dict_validation['scenario'] = scenario_name
    # results_df = results_df.append(best_parameter_dict_validation,ignore_index=True)
    
    # ######## equally susceptible -SM: model 3 ########
    # exploitProcessWithoutSocialMedia.w_tilde = np.array([1.38, 2.06, 1.88,1]) 
    # scenario_name = "equally_susceptible_-SM_model"
    # empty_training_feature_vectors = np.zeros_like(training_features)
    # empty_validation_feature_vectors = np.zeros_like(validation_features)
    # exploitProcessWithoutSocialMedia.setFeatureVectors(empty_training_feature_vectors, empty_validation_feature_vectors)
    # exploitProcessWithoutSocialMedia.setupTraining(training_realizations_no_soc,scenario_name, validation_MTPPdata=validation_realizations_no_soc)
    # best_parameter_dict_validation = exploitProcessWithoutSocialMedia.train(training_realizations_no_soc, scenario_name, validation_MTPPdata= validation_realizations_no_soc)
    # best_parameter_dict_validation['scenario'] = scenario_name
    
    # results_df = results_df.append(best_parameter_dict_validation,ignore_index=True)
    

    ######## all susceptible +SM: model 2 ########
    scenario_name = "all_susceptible_+SM_model"
    survivalProcess.w_tilde = np.array([1.38, 2.06, 1.88,1])
    full_training_feature_vectors = np.ones_like(training_features)*np.inf
    full_validation_feature_vectors = np.ones_like(validation_features)*np.inf
    
    survivalProcess.setFeatureVectors(full_training_feature_vectors, full_validation_feature_vectors)
    survivalProcess.setupTraining(TPPdata=training_realizations,pre_cal_filename=scenario_name , validation_MTPPdata=validation_realizations)
    best_parameter_dict_validation = survivalProcess.train(training_realizations, scenario_name, validation_MTPPdata= validation_realizations)
    best_parameter_dict_validation['scenario'] = scenario_name
    results_df = results_df.append(best_parameter_dict_validation,ignore_index=True)
    input()
    # ######## all susceptible -SM: model 1 ########
    scenario_name = "all_susceptible_-SM_model"
    survivalProcessWithoutSocialMedia.w_tilde = np.array([1.38, 2.06, 1.88,1])
    full_training_feature_vectors = np.ones_like(training_features)*np.inf
    full_validation_feature_vectors = np.ones_like(validation_features)*np.inf
    
    
    survivalProcessWithoutSocialMedia.setFeatureVectors(full_training_feature_vectors, full_validation_feature_vectors)
    survivalProcessWithoutSocialMedia.setupTraining(training_realizations_no_soc,scenario_name , validation_MTPPdata=validation_realizations_no_soc)
    best_parameter_dict_validation = survivalProcessWithoutSocialMedia.train(training_realizations_no_soc, scenario_name , validation_MTPPdata= validation_realizations_no_soc)
    best_parameter_dict_validation['scenario'] = scenario_name
    results_df = results_df.append(best_parameter_dict_validation,ignore_index=True)
    
    # results_df.to_csv("journal_results.csv")
    ############################### End training ##################################################################

    # exploitProcess.alpha = np.array([1.39461530e-03, 1.33951665e-04 ,2.35407472e-04, 2.85077666e-05])
    # exploitProcess.w_tilde = np.array([ 3.51311777, -1.24805883,  2.26495892])

    # exploitProcessWithoutSocialMedia.alpha = np.array([7.15636894e-01, 1.20748901e-03, 3.46717009e-04, 1.82742722e-04])
    # exploitProcessWithoutSocialMedia.w_tilde = np.array([1.4292924,  -2.21871001, -0.78951763])

if __name__ == "__main__":
    UnitTestSplitPopSynthetic()
