from point_processes.SplitPopulationTPP import *
from sklearn.metrics import confusion_matrix
from point_processes.NonParametricEstimator import *
from point_processes.DiscriminativeSplitPopulationTPP import *
import numpy as np
largeFigSize = (12, 9)

def removeNonSocialMedia(realizations, features):
    social_media_indices = []
    for index, realization in enumerate(realizations):
        if len(realization[1]) > 1 or len(realization[1]) > 1 or len(realization[1]) > 1:
            social_media_indices.append(index)     

    return realizations[social_media_indices], features[social_media_indices]

def removeSocialMedia(realizations, features):
    social_media_indices = []
    new_realizations = realizations.copy()
    for index, realization in enumerate(realizations):
        if len(realization[1]) +len(realization[2])+len(realization[3]) >3: 
            # print(realization)
            new_realizations[index][1:] = [np.array([x[-1]]) for x in realization[1:]]   

    return new_realizations, features

def evaluateModel(base_kernel, smk1, smk2, smk3):
    mkList = [base_kernel, smk1, smk2, smk3]
    sourceNames = ['base', 'github', 'reddit', 'twitter']
    training_features, training_realizations, training_isExploited, \
    test_features, test_realizations, test_isExploited, \
    validation_features, validation_realizations, validation_isExploited = generateExploitSocialMediaDataset()

    stop_criteria = {'max_iter': 5000,
                    'epsilon': 1e-5}

    exploitProcess = SplitPopulationTPP(
        mkList, sourceNames, stop_criteria,
        desc='Split population process with multiple kernels')
    
    
    scenario_name = str(base_kernel)+ str(smk1)+ str(smk2)+ str(smk3)
    exploitProcess.w_tilde = np.array([ -1.1403368, 0.416591,-0.7207898,1.0160878,0.85397834,-0.93534726,0.7645714,1.0173987,0.8959401,-1.1393325,1])
    exploitProcess.setFeatureVectors(training_features)
    exploitProcess.setupTraining(training_realizations, training_features,scenario_name , validation_MTPPdata=validation_realizations, verbose=0)
    best_parameter_dict = exploitProcess.train(training_realizations, scenario_name, validation_MTPPdata= validation_realizations, verbose=0)

    return best_parameter_dict

def main():
    # create a custom training functionand 
    # observe which set of kernel choices produces the best result
    training_features, training_realizations, training_isExploited, \
        test_features, test_realizations, test_isExploited, \
        validation_features, validation_realizations, validation_isExploited = generateExploitSocialMediaDataset()

    print("Length of training set: ", len(training_realizations))
    print("Length of validation set: ",len(validation_realizations))
    print("Length of test set: ",len(test_realizations))
    # from sklearn.linear_model import LogisticRegression
    # modified_training_features = np.zeros((len(training_features), 3))
    # for index,feature in enumerate(training_features):
    #     modified_training_features[index] = np.array([feature[0], feature[1], 1])
    # clf = LogisticRegression(random_state=0).fit(modified_training_features, training_isExploited)

    # w_tilde = clf.coef_
    # survival_df = convertRealizationsDataframeSplitPopNPE(training_realizations, modified_training_features, w_tilde)
    # npe_df,obj_values = non_parametric_estimation_split_population(survival_df)
   

    # mkS = SplineMemoryKernel(
    #     x_vals=np.array(npe_df['time'].values), y_vals=np.array(np.cumsum(npe_df['h'].values)), timeOffset=0.0)
    
#     from dask.distributed import Client, progress
#     client = Client()
#     futures = []
#     mkListofLists = [[WeibullMemoryKernel(0.5),ExponentialPseudoMemoryKernel(beta=0.5),ExponentialPseudoMemoryKernel(beta=0.5),PowerLawMemoryKernel(beta=2.0)],
#   [WeibullMemoryKernel(0.5),ExponentialPseudoMemoryKernel(beta=0.6),ExponentialPseudoMemoryKernel(beta=0.5),PowerLawMemoryKernel(beta=2.0)],
#   [WeibullMemoryKernel(0.5),ExponentialPseudoMemoryKernel(beta=0.5),ExponentialPseudoMemoryKernel(beta=0.6),PowerLawMemoryKernel(beta=2.0)],
#   [WeibullMemoryKernel(0.5),ExponentialPseudoMemoryKernel(beta=0.7),ExponentialPseudoMemoryKernel(beta=0.5),PowerLawMemoryKernel(beta=2.0)],
#   [WeibullMemoryKernel(0.5),ExponentialPseudoMemoryKernel(beta=0.8),ExponentialPseudoMemoryKernel(beta=0.5),PowerLawMemoryKernel(beta=2.0)],
#   [WeibullMemoryKernel(0.5),ExponentialPseudoMemoryKernel(beta=0.5),ExponentialPseudoMemoryKernel(beta=0.9),PowerLawMemoryKernel(beta=2.0)],
#   [WeibullMemoryKernel(0.5),ExponentialPseudoMemoryKernel(beta=0.5),ExponentialPseudoMemoryKernel(beta=0.3),PowerLawMemoryKernel(beta=1.0)],
#   [WeibullMemoryKernel(0.5),ExponentialPseudoMemoryKernel(beta=0.5),ExponentialPseudoMemoryKernel(beta=0.4),PowerLawMemoryKernel(beta=1.0)],
#   [WeibullMemoryKernel(0.5),ExponentialPseudoMemoryKernel(beta=0.3),ExponentialPseudoMemoryKernel(beta=0.5),PowerLawMemoryKernel(beta=1.0)],
#   [WeibullMemoryKernel(0.5),ExponentialPseudoMemoryKernel(beta=0.5),ExponentialPseudoMemoryKernel(beta=0.2),PowerLawMemoryKernel(beta=1.0)],
#   [WeibullMemoryKernel(0.5),ExponentialPseudoMemoryKernel(beta=0.5),ExponentialPseudoMemoryKernel(beta=0.1),PowerLawMemoryKernel(beta=1.0)],
#   [WeibullMemoryKernel(1.0),ExponentialPseudoMemoryKernel(beta=0.2),ExponentialPseudoMemoryKernel(beta=0.5),PowerLawMemoryKernel(beta=1.0)]
#     ]

#     # results_df = pd.DataFrame(columns=['base', 'sm1','sm2','sm3', 'training_nll', 'validation_nll'])
#     results_df = pd.read_csv('OptimalKernel_df.csv')
#     for mkList in mkListofLists:
#         futures.append(client.submit(evaluateModel,mkList[0], mkList[1],mkList[2],mkList[3]))
    
#     results = client.gather(futures)
#     for index, result_dict in enumerate(results):
#         results_df = results_df.append({'base':str(mkListofLists[index][0]), 'sm1':str(mkListofLists[index][1]),'sm2':str(mkListofLists[index][2]), 'sm3':str(mkListofLists[index][3])
#         , 'training_nll':result_dict['neg_log_likelihood_train'], 'validation_nll':result_dict['neg_log_likelihood_val']},ignore_index=True)
#     results_df.to_csv('OptimalKernel_df.csv', index=False)
#     print("Completed and saved to file")
#     exit(0)


    exponential_beta = 0.5
    power_law_beta = 2.0
    mkList = [WeibullMemoryKernel(0.5),
    ExponentialPseudoMemoryKernel(beta=exponential_beta),
    ExponentialPseudoMemoryKernel(beta=exponential_beta),
    PowerLawMemoryKernel(beta=power_law_beta)]
    sourceNames = ['base', 'github', 'reddit', 'twitter']

    stop_criteria = {'max_iter': 100,
                     'epsilon': 1e-16}

    exploitProcess = SplitPopulationTPP(
        mkList, sourceNames, stop_criteria,
        desc='Split population process with multiple kernels')
    
    
    scenario_name = "optimal_paper_scenario"

    training_features_empty = [ list(np.zeros(len(training_features[0]))) for x in training_features]
    validation_features_empty = [ list(np.zeros(len(validation_features[0]))) for x in validation_features]
    exploitProcess.w_tilde = np.array([ -1.1403368, 0.416591,-0.7207898,1.0160878,0.85397834,-0.93534726,0.7645714,1.0173987,0.8959401,-1.1393325,1])
    exploitProcess.setFeatureVectors(training_features_empty, validation_features_empty)
    exploitProcess.setupTraining(training_realizations,scenario_name , validation_MTPPdata=validation_realizations)
    best_parameter_dict = exploitProcess.train(training_realizations, scenario_name, validation_MTPPdata= validation_realizations)
    input()

    exploitProcess.w_tilde = np.array([ -1.1403368, 0.416591,-0.7207898,1.0160878,0.85397834,-0.93534726,0.7645714,1.0173987,0.8959401,-1.1393325,1])
    exploitProcess.setFeatureVectors(training_features, validation_features)
    exploitProcess.setupTraining(training_realizations,scenario_name , validation_MTPPdata=validation_realizations)
    best_parameter_dict = exploitProcess.train(training_realizations, scenario_name, validation_MTPPdata= validation_realizations)
    IIDSamples_validation_main = []
    IIDSamples_validation_other = []
    for index, realization in enumerate(training_realizations):
        if len(realization[0]) > 1:
            if realization[0][0] > realization[0][1]:
                continue
            exploitProcess.setSusceptible()
            cum_rc = exploitProcess.cumIntensity(realization[0][-1], realization)
            cum_tc = exploitProcess.cumIntensity(realization[0][0], realization)
            exponential_transform = (1-np.exp(-cum_rc))/(np.exp(-cum_tc) - np.exp(-cum_rc))                
            transformed_time = np.log(exponential_transform)
            IIDSamples_validation_main.extend([transformed_time])
    IIDSamples_validation_main = list(filter((0.0).__ne__, IIDSamples_validation_main))


    for index,realization in enumerate(validation_realizations):
        if len(realization[0]) > 1:
            if realization[0][0] > realization[0][1]:
                continue
            exploitProcess.setSusceptible()
            cum_rc = exploitProcess.cumIntensity(realization[0][-1], realization)
            cum_tc = exploitProcess.cumIntensity(realization[0][0], realization)
            exponential_transform = (1-np.exp(-cum_rc))/(np.exp(-cum_tc) - np.exp(-cum_rc))

            transformed_time = np.log(np.abs(exponential_transform))
            IIDSamples_validation_other.extend([transformed_time])
    IIDSamples_validation_other = list(filter((0.0).__ne__, IIDSamples_validation_other))
    features, realizations, isExploited = generateRealWorldExploitDataset()
    IIDSamples_RW = []
    for index,realization in enumerate(realizations):
        if len(realization[0]) > 1:
            if realization[0][0] > realization[0][1]:
                continue
            exploitProcess.setSusceptible()
            cum_rc = exploitProcess.cumIntensity(realization[0][-1], realization)
            cum_tc = exploitProcess.cumIntensity(realization[0][0], realization)
            exponential_transform = (1-np.exp(-cum_rc))/(np.exp(-cum_tc) - np.exp(-cum_rc))

            transformed_time = np.log(np.abs(exponential_transform))
            IIDSamples_RW.extend([transformed_time])
    IIDSamples_RW = list(filter((0.0).__ne__, IIDSamples_RW))
    

    fig, ax1 = plt.subplots(2, 1, squeeze=False,figsize=largeFigSize)    
    pvalue = KSgoodnessOfFitExp1(sorted(np.random.choice(IIDSamples_validation_main, 100)), ax1[0][0], showConfidenceBands=True, title="Training")
    pvalue = KSgoodnessOfFitExp1(sorted(np.random.choice(IIDSamples_validation_other, 100)), ax1[1][0], showConfidenceBands=True, title="Validation")
    plt.subplots_adjust(hspace=0.4)
    fig.tight_layout()
    plt.savefig('P-P_plot.png', bbox_inches='tight', dpi=200)
    plt.show()
    fig, ax1 = plt.subplots(1, 1, squeeze=False,figsize=largeFigSize) 
    pvalue = KSgoodnessOfFitExp1(sorted(np.random.choice(IIDSamples_RW, 100)), ax1[0][0], showConfidenceBands=True, title="Real World")
    plt.savefig('P-P_plot_rw.png', bbox_inches='tight', dpi=200)
    plt.show()
    
    

    
    # remove all non social media events
    def removeSocialMedia(realizations, features):
        social_media_indices = []
        for index, realization in enumerate(realizations):
            realizations[index] = realizations[index][0:1]
            # print(realizations[index][0:1])
            # input()
            # realizations[index][1] = np.array([realization[1][-1]])
            # realizations[index][2] = np.array([realization[2][-1]])
            # realizations[index][3] = np.array([realization[3][-1]])
        
        return realizations, features
    
    training_realizations_no_soc, training_features =removeSocialMedia(training_realizations, training_features) 
    validation_realizations_no_soc, validation_features = removeSocialMedia(validation_realizations,validation_features )
    test_realizations_no_soc, test_features = removeSocialMedia(test_realizations,test_features)


    mkList = [WeibullMemoryKernel(0.5)]
    # ExponentialPseudoMemoryKernel(beta=0.5),
    # ExponentialPseudoMemoryKernel(beta=0.5),
    # PowerLawMemoryKernel(beta=2.0)]
    # this TPP only uses its own events as source.
    sourceNames = ['base']

    stop_criteria = {'max_iter': 100,
                     'epsilon': 1e-16}
    scenario_name = "optimal_paper_scenario_-sm2"
    exploitProcessWithoutSocialMedia = SplitPopulationTPP(
        mkList, sourceNames, stop_criteria,
        desc='Split population process with multiple kernels')

    exploitProcessWithoutSocialMedia.w_tilde = np.array([ -1.1403368, 0.416591,-0.7207898,1.0160878,0.85397834,-0.93534726,0.7645714,1.0173987,0.8959401,-1.1393325,1])
    exploitProcessWithoutSocialMedia.setFeatureVectors(training_features, validation_features)
    exploitProcessWithoutSocialMedia.setupTraining(training_realizations_no_soc,scenario_name , validation_MTPPdata=validation_realizations_no_soc)
    best_parameter_dict = exploitProcessWithoutSocialMedia.train(training_realizations_no_soc, scenario_name, validation_MTPPdata= validation_realizations_no_soc)
    

    

if __name__ == "__main__":
    main()