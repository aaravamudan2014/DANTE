# ========modification data format for Akshay ==============================
import json
import numpy as np 
from point_processes.SplitPopulationTPP import *
from sklearn.metrics import confusion_matrix
from point_processes.NonParametricEstimator import *
from point_processes.DiscriminativeSplitPopulationTPP import *
import numpy as np


data = 'simulation.json'
with open(data, 'r') as f:
    TPPdata = json.load(f)
# akshay: list of list data format
newTPPdata = []
newFeatureVector = np.zeros((len(TPPdata), 3))
newSusceptibleLabel = np.zeros(len(TPPdata))
ep_sourceNames = ['exploit','social_media_1', 'social_media_2']
for c, realization in enumerate(TPPdata):
    new_realization = []
    Tc = realization['right_censored_time']
    for s in ep_sourceNames:
        if s in realization:
            new_realization.append(realization[s] + [Tc])
        else:
            new_realization.append([Tc]) 
    # print(realization)
    # input()   
    newTPPdata.append(new_realization)        
    newFeatureVector[c,:] = realization['feature_vector'] 
    newSusceptibleLabel[c] = realization['susceptible']
np.save('simulated_realization.npy', newTPPdata)    
np.save('feature_vector.npy', newFeatureVector)   
# np.save('gt_y.npy', newSusceptibleLabel)     
# np.save('gt_w_tilde.npy', w_tilde)   
# np.save('gt_exploit_alpha.npy', mp.para.loc['exploit', :])

mkList = [WeibullMemoryKernel(0.8), ExponentialPseudoMemoryKernel(beta=1.0),ExponentialPseudoMemoryKernel(beta=1.0)]
stop_criteria = {'max_iter': 50,
                    'epsilon': 1e-7}
exploitProcess = SplitPopulationTPP(
    mkList, ep_sourceNames, stop_criteria,
    desc='Split population process with multiple kernels')
scenario_name = "test_6"
exploitProcess.w_tilde = np.array([ -1.1403368, 0.8959401,1])
exploitProcess.setFeatureVectors(newFeatureVector, append=False)
exploitProcess.setupTraining(newTPPdata, newFeatureVector,scenario_name , validation_MTPPdata=newTPPdata, verbose=1, append=False)
best_parameter_dict = exploitProcess.train(newTPPdata, scenario_name, validation_MTPPdata= newTPPdata, verbose=1)
IIDSamples_validation_main = []
IIDSamples_validation_other = []
exploitProcess.setFeatureVectors(newFeatureVector, append=False)
for index, realization in enumerate(newTPPdata):
    if len(realization[0]) > 1:
        # if realization[0][0] > 1:
        if realization[0][0] > realization[0][1]:
            continue
        exploitProcess.setSusceptible()
        cum_rc = exploitProcess.cumIntensity(realization[0][-1], realization)
        cum_tc = exploitProcess.cumIntensity(realization[0][0], realization)
        exponential_transform = (1-np.exp(-cum_rc))/(np.exp(-cum_tc) - np.exp(-cum_rc))
        transformed_time = np.log(exponential_transform)
        # print(cum_rc)
        # print(cum_tc)
        # input()
        # transformed_time = exploitProcess.transformEventTimes(realization)
        # prior = 1/(1+np.exp(-np.dot(exploitProcess.w_tilde, np.append(training_features[index],1))))
        # log_prior = np.log(np.ones(len(transformed_times)) * prior)
        IIDSamples_validation_main.extend([transformed_time])
IIDSamples_validation_main = list(filter((0.0).__ne__, IIDSamples_validation_main))
fig, (ax1) = plt.subplots(2, 1, squeeze=False,)    
pvalue = KSgoodnessOfFitExp1(sorted(IIDSamples_validation_main), ax1[0][0], showConfidenceBands=True, title="P-P plot for Training Dataset")
plt.show()
