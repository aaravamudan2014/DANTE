import numpy as np

# Receives a vector of probabilities of ``success'' of independent Bernoullis,
# computes the PMF of their sum and returns probability intervals for this PMF.
# Notes: By construction,
# - the intervals are nested
# - for the stated probability, the interval returned is of shortest length
# - for a given probability, the interval returned may be only one possible interval out of many
# Author: gca, Sep 2020.
def probIntervalsSumBernoulli(pVector, keyword=None):
    numBernoulliDistributions = len(pVector)
    pmfSum = np.array([1.0])
    for n in range(numBernoulliDistributions):
        pmf = np.array([1.0 - pVector[n], pVector[n]])
        pmfSum = np.convolve(pmfSum, pmf)
    print('PMF of sum of Bernoullis:\t', end='')
    print(pmfSum)
    idx = np.argsort(pmfSum)[::-1] # get the respective indices from sorting pmfSum in descending order
    cumProb = np.cumsum(pmfSum[idx]) # compute cumulative probability of sorted pmfSum
    orgIdx = np.arange(len(pmfSum))
    import sys
    original_stdout = sys.stdout
    if keyword is None:
        filename = 'output.txt'
    else:
        filename = str(keyword)+"_sum_bernoulli_output.txt"
    with open(filename, 'w') as f:
        sys.stdout = f
        for n in np.arange(len(pmfSum)):
            idxTmp =  np.arange(n+1)
            print('Probability={:.06f}\t'.format(cumProb[n]), end='') 
            print(np.sort(orgIdx[idx[idxTmp]]))
        sys.stdout = original_stdout
    return


def main():
    # Unit test
    pVector = np.random.random(5)
    probIntervalsSumBernoulli(pVector)


if __name__=="__main__":
    main()