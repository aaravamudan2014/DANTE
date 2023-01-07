from point_processes.SplitPopulationTPP import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve

from point_processes.NonParametricEstimator import *
from point_processes.IntervalTest import *
from point_processes.DiscriminativeSplitPopulationTPP import *
from utils.DataReader import generateSyntheticDataset
import numpy as np
import collections
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn import metrics

global test_features,test_realizations,test_isExploited


synthetic_realizations, synthetic_features = generateSyntheticDataset()
sim_data_json = np.load('temporary_simulation_results_synthetic.npy', allow_pickle = True).item()

def removeNegativeSocialMedia(realizations):
    realizations_new = []
    for realization in realizations:
        github  = [x for x in realization[1] if x > 0]
        reddit = [x for x in realization[2] if x > 0]
        twitter = [x for x in realization[3] if x > 0]
        realizations_new.append(np.array([realization[0], github, reddit, twitter]))
    
    return realizations_new


# exponential_beta = 0.5
# power_law_beta = 2.0

exponential_beta = 1.0
power_law_beta = 1.0


mkList = [WeibullMemoryKernel(0.8),
            ExponentialPseudoMemoryKernel(beta=exponential_beta),
            ExponentialPseudoMemoryKernel(beta=exponential_beta),
            PowerLawMemoryKernel(beta=power_law_beta)]
sourceNames = ['base', 'github', 'reddit', 'twitter']

stop_criteria = {'max_iter': 100,
                    'epsilon': 1e-16}
features_rw, realizations_rw, isExploited_rw = generateRealWorldExploitDataset()
    
exploitProcess = SplitPopulationTPP(
    mkList, sourceNames, stop_criteria,
    desc='Split population process with multiple kernels')
# this TPP only uses its own events as source.
sourceNames = ['base', 'github', 'reddit', 'twitter']

stop_criteria = {'max_ iter': 20,
                    'epsilon': 1e-15}
mkList = [WeibullMemoryKernel(0.8)]
sourceNames = ['base']

exploitProcessWithoutSocialMedia = SplitPopulationTPP(
    mkList, sourceNames, stop_criteria,
    desc='Split population process with multiple kernels')

equallySusceptibleProcessWithoutSocialMedia = SplitPopulationTPP(
    mkList, sourceNames, stop_criteria,
    desc='Split population process with multiple kernels')



allSusceptibleProcessWithoutSocialMedia = SplitPopulationTPP(
    mkList, sourceNames, stop_criteria,
    desc='Split population process with multiple kernels')


# exploitProcess.alpha = np.array([0.01031176,  0.00280867, 0.00856644, 0.00512576])
# exploitProcess.w_tilde = np.array( [2.25440783 , 1.94604405,  2.20794262, -3.86097509])

# exploitProcessWithoutSocialMedia.alpha = np.array([0.02120183])
# exploitProcessWithoutSocialMedia.w_tilde = np.array( [2.40631924,  2.02749456,  2.31609842, -4.0723442 ])


# equallySusceptibleProcessWithoutSocialMedia.alpha = np.array([0.00248325])
# equallySusceptibleProcessWithoutSocialMedia.w_tilde = np.array( [ 0.,          0.,          0.,         -0.76393143])

# allSusceptibleProcessWithoutSocialMedia.alpha = np.array([0.02551898])
# allSusceptibleProcessWithoutSocialMedia.w_tilde = np.array( [np.inf, np.inf, np.inf, np.inf])


exploit_para = {'base'   : 0.01,  'github' : 0.02, \
                'reddit' : 0.005, 'twitter': 0.01}

exploitProcess.alpha = np.array([0.010,  0.002, 0.005, 0.001])
exploitProcess.w_tilde =  np.array([ 3.00674769, 0.12732851, -1.45712857])

exploitProcessWithoutSocialMedia.alpha = np.array([0.03721553])
exploitProcessWithoutSocialMedia.w_tilde = np.array( [3.07842726,  0.11725162, -1.36191113 ])


equallySusceptibleProcessWithoutSocialMedia.alpha = np.array([0.00248325])
equallySusceptibleProcessWithoutSocialMedia.w_tilde = np.array( [ 0.,          0.,             -0.76393143])

allSusceptibleProcessWithoutSocialMedia.alpha = np.array([0.02551898])
allSusceptibleProcessWithoutSocialMedia.w_tilde = np.array( [np.inf, np.inf,  np.inf])



IIDSamples_training = []
IIDSamples_validation = []
exploitProcess.setFeatureVectors(synthetic_features, append=False)
for realization in synthetic_realizations:
    if len(realization[0]) > 1:
        if realization[0][0] > 0:
            exploitProcess.setSusceptible()
            IIDSamples_training.extend(
                exploitProcess.transformEventTimes(realization))
IIDSamples_synthetic = list(filter((0.0).__ne__, IIDSamples_training))



fig, (ax1) = plt.subplots(2, 1, squeeze=False,)    
pvalue = KSgoodnessOfFitExp1(sorted(np.random.choice(IIDSamples_synthetic, 100)), ax1[0][0], showConfidenceBands=True, title="P-P plot for Training Dataset")

# plt.show()

def getTerms(process, realization,tc, delta_t, feature_vector):
    if len(realization[0]) == 1 :
        gt = 0
    elif len(realization[0]) == 2 :
        if realization[0][0] <= tc + delta_t and realization[0][0] > tc:
            gt = 1
        elif realization[0][0] >= tc + delta_t:
            gt = 0
        elif realization[0][0] <= tc:
            gt = None
    Psi_vector_tc = np.zeros(len(process.alpha), dtype=np.float128)
    Psi_vector_delta_t = np.zeros(len(process.alpha),dtype=np.float128)
    Psi_vector_tc[0] = process.mk[0].psi(tc)
    Psi_vector_delta_t[0] = process.mk[0].psi(tc + delta_t)
    for source_index in range(1, len(process.alpha)):
        source_events = realization[source_index][:-1]
        for se in source_events:
            if se < tc:
                Psi_vector_tc[source_index] += process.mk[source_index].psi(
                    tc - se)
                Psi_vector_delta_t[source_index] += process.mk[source_index].psi(
                    tc + delta_t - se)

    return gt, np.nan_to_num(Psi_vector_tc), np.nan_to_num(Psi_vector_delta_t)



def trainLogisticRegressionCell(tc, delta_t, training_realizations,validation_realizations, 
    training_features,validation_features):
    import numpy as np
  
    # RW_feature_vector_logistic = []
    # RW_outputs = []
    # RW_feature_vector_logistic_no_soc = []
    # RW_outputs_no_soc = []
    
    rw_features_with_bias,_ = exploitProcess.setFeatureVectors(features_rw)
    training_features_with_bias,validation_features_with_bias = exploitProcess.setFeatureVectors(training_features,validation_features, append=False)
    # validation_features_with_bias = exploitProcess.setFeatureVectors(validation_features)
 
    synthetic_features_updated = np.array(synthetic_features)[sim_data_json[tc/24.0,delta_t/24.0][2]]
    synthetic_realizations_updated = np.array(synthetic_realizations)[sim_data_json[tc/24.0,delta_t/24.0][2]]
    # synthetic_isExploited_updated = np.array(synthetic_isExploited)[sim_data_json[tc/24.0,delta_t/24.0][2]]
    synthetic_features_with_bias,_ = exploitProcess.setFeatureVectors(synthetic_features_updated, append=False)

    def obtain_lr_features(realizations, features_with_bias):

        feature_vector_logistic_no_soc = []
        feature_vector_logistic_soc = []
        feature_vector_logistic_features = []
        feature_vector_logistic_equally_susceptible_no_soc = []
        feature_vector_logistic_all_susceptible_no_soc = []
         
        
        outputs_features  = []
        outputs_no_soc = []  
        output_equally_susceptible_no_soc = []
        output_all_susceptible_no_soc = []
        outputs_soc = []

        for index, realization in enumerate(realizations):
            gt_s, Psi_vector_tc, Psi_vector_delta_t = getTerms(exploitProcess,realization,tc, delta_t, features_with_bias[index])
            if gt_s is None:
                continue
            prior_term_1  = 1/(1 + np.exp(-np.dot(features_with_bias[index], exploitProcess.w_tilde)))
            social_term_1 =  np.exp(-np.dot(exploitProcess.alpha,Psi_vector_tc))
            social_term_2 =  np.exp(-np.dot(exploitProcess.alpha,Psi_vector_delta_t))
            if np.isnan(social_term_1):
                social_term_1 = 0
            if np.isnan(social_term_2):
                social_term_2 = 0
            
            feature_vector_logistic_soc.append(np.array([prior_term_1, social_term_1, social_term_2]))
            outputs_soc.append(gt_s)
            feature_vector_logistic_features.append(features_with_bias[index])
            outputs_features.append(gt_s)
            
            gt, Psi_vector_tc, Psi_vector_delta_t = getTerms(exploitProcessWithoutSocialMedia,realization,tc, delta_t, features_with_bias[index])
            prior_term_1  = 1/(1 + np.exp(-np.dot(features_with_bias[index], exploitProcessWithoutSocialMedia.w_tilde)))
            social_term_1 = np.exp(-np.dot(exploitProcessWithoutSocialMedia.alpha,Psi_vector_tc))
            social_term_2 =  np.exp(-np.dot(exploitProcessWithoutSocialMedia.alpha,Psi_vector_delta_t))
            if np.isnan(social_term_1):
                social_term_1 = 0
            if np.isnan(social_term_2):
                social_term_2 = 0

            feature_vector_logistic_no_soc.append(np.array([prior_term_1,social_term_1,social_term_2]))
            outputs_no_soc.append(gt)

            gt, Psi_vector_tc, Psi_vector_delta_t = getTerms(equallySusceptibleProcessWithoutSocialMedia,realization,tc, delta_t, features_with_bias[index])
            prior_term_1  = 1/(1 + np.exp(-np.dot(features_with_bias[index], equallySusceptibleProcessWithoutSocialMedia.w_tilde)))
            social_term_1 = np.exp(-np.dot(equallySusceptibleProcessWithoutSocialMedia.alpha,Psi_vector_tc))
            social_term_2 =  np.exp(-np.dot(equallySusceptibleProcessWithoutSocialMedia.alpha,Psi_vector_delta_t))
            if np.isnan(social_term_1):
                social_term_1 = 0
            if np.isnan(social_term_2):
                social_term_2 = 0

            feature_vector_logistic_equally_susceptible_no_soc.append(np.array([prior_term_1,social_term_1,social_term_2]))
            output_equally_susceptible_no_soc.append(gt)

            gt, Psi_vector_tc, Psi_vector_delta_t = getTerms(allSusceptibleProcessWithoutSocialMedia,realization,tc, delta_t, features_with_bias[index])
            prior_term_1  = 1/(1 + np.exp(-np.dot(features_with_bias[index], allSusceptibleProcessWithoutSocialMedia.w_tilde)))
            social_term_1 = np.exp(-np.dot(allSusceptibleProcessWithoutSocialMedia.alpha,Psi_vector_tc))
            social_term_2 =  np.exp(-np.dot(allSusceptibleProcessWithoutSocialMedia.alpha,Psi_vector_delta_t))
            if np.isnan(social_term_1):
                social_term_1 = 0
            if np.isnan(social_term_2):
                social_term_2 = 0

            feature_vector_logistic_all_susceptible_no_soc.append(np.array([prior_term_1,social_term_1,social_term_2]))
            output_all_susceptible_no_soc.append(gt)

        feature_vector_logistic_soc = np.array(feature_vector_logistic_soc)
        feature_vector_logistic_features = np.array(feature_vector_logistic_features)
        feature_vector_logistic_no_soc = np.array(feature_vector_logistic_no_soc)
        feature_vector_logistic_equally_susceptible_no_soc = np.array(feature_vector_logistic_equally_susceptible_no_soc)
        feature_vector_logistic_all_susceptible_no_soc = np.array(feature_vector_logistic_all_susceptible_no_soc)

        outputs_soc = np.array(outputs_soc)
        outputs_features = np.array(outputs_features)
        outputs_no_soc = np.array(outputs_no_soc)
        output_equally_susceptible_no_soc = np.array(output_equally_susceptible_no_soc)
        output_all_susceptible_no_soc = np.array(output_all_susceptible_no_soc)


        return  feature_vector_logistic_soc,outputs_soc,feature_vector_logistic_features ,outputs_features, feature_vector_logistic_no_soc, outputs_no_soc, \
        feature_vector_logistic_equally_susceptible_no_soc, output_equally_susceptible_no_soc, feature_vector_logistic_all_susceptible_no_soc, output_all_susceptible_no_soc


    def getProbs(feature_vectors):
        prob_list = []
        for item in feature_vectors:
            prior = item[0]
            s_tc = item[1]
            s_tc_delta_t = item[2]
            prob = (prior*(s_tc-s_tc_delta_t))/(1-prior + prior*s_tc)
            prob_list.append(prob)
        
        prob_array = np.array(prob_list)
        return prob_array
            
    synthetic_feature_vector_logistic_soc,synthetic_outputs_soc,synthetic_feature_vector_logistic_features ,synthetic_outputs_features,\
    synthetic_feature_vector_logistic_no_soc, synthetic_outputs_no_soc, \
    synthetic_feature_vector_logistic_equally_susceptible_no_soc, synthetic_output_equally_susceptible_no_soc, \
    synthetic_feature_vector_logistic_all_susceptible_no_soc, synthetic_output_all_susceptible_no_soc =\
                obtain_lr_features(synthetic_realizations_updated, synthetic_features_with_bias)
    
    


    print("Data statistics")
    synthetic_counter = collections.Counter(synthetic_outputs_no_soc)
    print(synthetic_counter[0])
    print(synthetic_counter[1])

    
    no_soc_prob_list = getProbs(synthetic_feature_vector_logistic_no_soc)
    # soc_prob_list = getProbs(synthetic_feature_vector_logistic_soc)
    
    
    
    # soc_precision, soc_recall, _ = precision_recall_curve(synthetic_outputs_soc, soc_prob_list)
    soc_precision, soc_recall, _ = precision_recall_curve(sim_data_json[tc/24,delta_t/24][1],sim_data_json[(tc/24,delta_t/24)][0])
    
    no_soc_precision, no_soc_recall, _ = precision_recall_curve(synthetic_outputs_no_soc, no_soc_prob_list)

    
    
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    fig, ax = plt.subplots()
    ax.axis("square")
    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,1.1)

    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('F1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))    
        lines.append(l)
    labels.append('iso-f1 curves')
    plt.title("$t_c$ = {} mo, $\Delta t$ = {} mo".format(int(tc/(24*30)), int(delta_t/(24*30))))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    
    
    from sklearn.metrics import PrecisionRecallDisplay
    pr_display = PrecisionRecallDisplay(precision = no_soc_precision, recall = no_soc_recall).plot(ax =ax,label="-SM model",color='red')
    pr_display = PrecisionRecallDisplay(precision = soc_precision, recall = soc_recall).plot(ax =ax,label="+SM model",color='green')
    
    plt.legend(loc='upper right')
    plt.savefig('paper_results/PRC_tc:'+str(tc/24.0) + 'delta_t:'+str(delta_t/24.0)+'.png', dpi=600)
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    fig, ax = plt.subplots()
    ax.axis("square")
    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,1.1)

    return None

def genPvector():
    
    training_features, training_realizations, training_isExploited, \
        test_features, test_realizations, test_isExploited, \
        validation_features, validation_realizations, validation_isExploited = generateExploitSocialMediaDataset()

    features_rw, realizations_rw, isExploited_rw = generateRealWorldExploitDataset()

    company_names = ['microsoft', 'apple', 'linux', 'cisco', 'oracle','ibm','google', 'adobe', 'debian',]
    master_df = pd.read_csv('../data/MasterDataset.csv')
    df_train = pd.read_csv('../data/point_process_training.csv')
    df_test = pd.read_csv('../data/point_process_test.csv')
    df_validation = pd.read_csv('../data/point_process_validation.csv')
    req_indices_all_validation = [[] for x in range(len(company_names))]
    req_indices_all_test = [[] for x in range(len(company_names))]
    
    # for index, cve_id in enumerate(df_validation['cve_id'].values):
    #     desc = master_df[master_df['cve_id'] == cve_id]['mitre_desc'].values[0]
    #     for inner_index, company in enumerate(company_names):
    #         if company in desc.lower():
    #             req_indices_all_validation[inner_index].append(index)

    for index, cve_id in enumerate(df_test['cve_id'].values):
        desc = master_df[master_df['cve_id'] == cve_id]['mitre_desc'].values[0]
        for inner_index, company in enumerate(company_names):
            if company in desc.lower():
                req_indices_all_test[inner_index].append(index)
    import matplotlib.pyplot as plt
    # for index,company in enumerate(company_names):
    #     req_realizations_test = test_realizations[req_indices_all_test[index]]
    #     exploit_times = []
    #     for realization in req_realizations_test:
    #         if len(realization[0]) > 1:
    #             exploit_times.append(realization[0][0]/(24.0))
    #     print(company)
    #     print("Total Realizations: ", len(req_realizations_test))
    #     print("Total Exploited: ", len(exploit_times))
    #     plt.hist(exploit_times, bins=100)
    #     plt.show()
        
            
    # tc_list = [10.0,25.0,25.0,25.0,25.0,25.0,0.0,0.0,0.0]
    # delta_list = [360.0,270.0,300.0,360.0,270.0,270.0,120.0,270.0,10.0]
    # # tc_list *= 24.0
    # # delta_list *= 24.0


    # for index,company in enumerate(company_names):
    #     req_realizations_test = test_realizations[req_indices_all_test[index]]
    #     req_features_test = test_features[req_indices_all_test[index]]
    #     test_features_with_bias, _ = exploitProcess.setFeatureVectors(req_features_test)

    #     actual_exploited = 0
    #     pVector = []
    #     for inner_index, realization in enumerate(req_realizations_test):
    #         gt_s, Psi_vector_tc, Psi_vector_delta_t  = getTerms(exploitProcess, realization,tc_list[index]*24.0, delta_list[index]*24.0, test_features_with_bias[inner_index])
    #         if gt_s is None:
    #             continue
    #         actual_exploited += gt_s
    #         prior_term  = 1/(1 + np.exp(-np.dot(test_features_with_bias[inner_index], exploitProcess.w_tilde)))
    #         social_term_1 =  np.exp(-np.dot(exploitProcess.alpha,Psi_vector_tc))
    #         social_term_2 =  np.exp(-np.dot(exploitProcess.alpha,Psi_vector_delta_t))
    #         pred_prob = (prior_term*(social_term_1-social_term_2))/(1-prior_term + prior_term*social_term_1)
    #         # print(Psi_vector_tc)
    #         # print(Psi_vector_delta_t)
    #         # print(pred_prob)
    #         # input()
    #         pVector.append(pred_prob)
    
    #     print(actual_exploited)
    #     print(company)
    #     probIntervalsSumBernoulli(np.array(pVector), company)
    #     input()

    # getTerms(exploitProcess, realization,tc, delta_t, feature_vector)







def main():
    # genPvector()
    
    # tc_list = [0, 30,60,90,120, 150, 180, 210, 240, 270, 300]
    # delta_list = [30, 90, 180,270, 360]

    tc_list = np.array([5/24, 10/24, 20/24])
    delta_list = np.array([5/24, 10/24, 20/24])


    prec_list = []
    rec_list = []
    f1_list = []
    for tc in tc_list:
        for delta_t in delta_list:
            trainLogisticRegressionCell(tc*24.0, delta_t*24.0, synthetic_realizations,synthetic_realizations, synthetic_features, synthetic_features )
            # results_df = pd.DataFrame({'Precision': prec, 'Recall': rec})
            # results_df.to_csv('paper_results/results_tc_'+str(tc)+'_delta_t_'+str(delta_t)+'.csv', index=False)
    

    
if __name__ == "__main__":
    main()