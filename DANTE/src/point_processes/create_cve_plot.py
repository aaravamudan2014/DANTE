from point_processes.SplitPopulationTPP import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve

from point_processes.NonParametricEstimator import *
from point_processes.IntervalTest import *
from point_processes.DiscriminativeSplitPopulationTPP import *
import numpy as np
import collections
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import json 

def func(tc,delta_t):
        exponential_beta = 0.5
        power_law_beta = 2.0

        mkList = [WeibullMemoryKernel(0.5),
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
        mkList = [WeibullMemoryKernel(0.5)]
        sourceNames = ['base']
        exploitProcessWithoutSocialMedia = SplitPopulationTPP(
        mkList, sourceNames, stop_criteria,
        desc='Split population process with multiple kernels')


        exploitProcess.alpha = np.array([0.0081207,  0.00214808, 0.00998513, 0.00398643])
        exploitProcess.w_tilde = np.array( [3.33554311,  2.95909785 , 3.27399972, -4.67334175])

        training_features, training_realizations, training_isExploited, \
                test_features, test_realizations, test_isExploited, \
                validation_features, validation_realizations, validation_isExploited = generateExploitSocialMediaDataset()

        tc = 30*24
        delta_t = 30*24
        test_features_with_bias,_ = exploitProcess.setFeatureVectors(test_features)
        

        def getTerms(process, realization,tc, delta_t, feature_vector):
                if len(realization[0]) == 1 :
                        if realization[0][0] > tc:
                                gt = 0
                        else:
                                gt = None
                
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

        indexes_to_include = []

        for index, realization in enumerate(test_realizations):
                if  len(realization[0]) == 1:
                        indexes_to_include.append(index)
                elif len(realization[0]) == 2:
                        if realization[0][0] > 0:
                                indexes_to_include.append(index)
                        

        for index, realization in enumerate(test_realizations):
                gt, Psi_vector_tc, Psi_vector_delta_t = getTerms(exploitProcess,realization,tc, delta_t, test_features_with_bias[index])
                if gt is None:
                        continue

                indexes_to_include.append(index)

        indexes_to_include = np.unique(indexes_to_include)
        exploit_times = []

        sim_data_json = np.load('../data/temporary_simulation_results.json.npy', allow_pickle = True).item()


        soc_prob_precision, soc_prob_recall, _ = precision_recall_curve(sim_data_json[int(0/24),int(delta_t/24)][1],sim_data_json[(int(0/24),int(delta_t/24))][0] )

        fig, ax = plt.subplots()
        ax.axis("square")
        ax.set_xlim(-0.1,1.1)
        ax.set_ylim(-0.1,1.1)
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
                plt.annotate('F1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))    
                lines.append(l)
        labels.append('iso-f1 curves')
        # ax.axis("equal")
        plt.title("$t_c$ = {} mo, $\Delta t$ = {} mo".format(int(tc/(24*30)), int(delta_t/(24*30))))
        plt.xlabel("Recall")
        plt.ylabel("Precision")


        from sklearn.metrics import PrecisionRecallDisplay
        pr_display = PrecisionRecallDisplay(precision = soc_prob_precision, recall = soc_prob_recall).plot(ax =ax,label="SM Simulation model",color='green')
        
        # plt.show()

        print(len(sim_data_json[int(0/24),int(delta_t/24)][2]))
        print(len(test_realizations[sim_data_json[int(0/24),int(delta_t/24)][2]]))


def main():
        # genPvector()
        # tc_list = [0,30,60,90, 120, 150, 180, 210, 240, 270, 300]
        # delta_list = [30, 90, 180, 270, 360]

        tc_list = [ 0,30,60,90,120, 150, 180, 210, 240, 270, 300]
        # tc_list = [0]
        # delta_list = [180]
        delta_list = [30, 90, 180,270, 360 ]
        for tc in tc_list:
                for delta_t in delta_list:
                        func(tc,delta_t)

if __name__ == "__main__":
    main()