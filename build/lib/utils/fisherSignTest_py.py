#
# Sample0: 1D numpy.ndarray of N elements that only take values 0,1 or True, False. 
# Sample1: 1D numpy.ndarray of N elements that only take values 0,1 or True, False. 

# OUTPUTS
# pvalue: float in [0,1]; observed significance level (p-value) of the test.
# idxBest: integer in {0,1}; indicates the most accurate model of the two.

# NOTES
# 1) If both sample arrays are identical, idxBest defaults to 0.

import scipy.stats
import numpy as np
from sklearn.metrics import confusion_matrix



def FisherSignTest(Sample0, Sample1):
    n0 = sum((Sample0 == 1) & (Sample1 == 0))
    n1 = sum((Sample0 == 0) & (Sample1 == 1))
    w = max(n0, n1)
    Nw = n0 + n1
    idxBest = 0 if n0 >= n1 else 1
    pvalue = scipy.stats.binom.sf(w-1, Nw, 0.5)
    return pvalue, idxBest


def main():
    gt = np.array(np.load('ground_truth.npy', allow_pickle=True))
    soc = np.array(np.load('model_with_soc.npy', allow_pickle=True))
    no_soc = np.array(np.load('model_without_soc.npy', allow_pickle=True))
    no_weights = np.array(np.load('model_without_weights.npy', allow_pickle=True))
    # print(np.sum(gt == soc))

    print("Correct for no social media model",np.sum(gt == soc))
    print("Correct for social media model",np.sum(gt == soc))
    print("Correct for no weights model",np.sum(gt == no_weights))


    print(FisherSignTest(soc, no_soc))
    # print(FisherSignTest(soc, gt))
    print(confusion_matrix(no_soc, gt))
    print(confusion_matrix(soc, gt))
    print(confusion_matrix(no_weights, gt))
    
    # print(len((soc == gt)))
    # print(sum((gt != soc) & (gt == no_soc)))
    # print(sum((gt == soc) & (gt != no_soc)))
    

    

if __name__ == "__main__":
    main()