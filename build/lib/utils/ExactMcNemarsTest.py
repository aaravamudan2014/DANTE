import numpy as np



import scipy.stats

# ExactMcNemarsTest
#
# Implements the p-value (observed significance level) of the exact version of 
# McNemar's Test for comparing two binomial proportions in terms of matched
# pairs.
#
# SYNTAX
# pvalue, idxBest = ExactMcNemarsTest(MatchedPairs)
#
# INPUTS
# MatchedPairs: 2D numpy.ndarray of N>=1 rows and 2 columns that only takes 
#               values 0,1 or True, False. Each row contains the classification 
#               outcome (correct/incorrect prediction) of 2 classification 
#               models for a specific sample.
#
# OUTPUTS
# pvalue:  float in [0,1]; p-value (observed significance level) of the test.
# idxBest: integer in {0,1}; indicates the classification model (column index
#          of MatchedPairs) with the most number of 1/True's between the two.
#          The test's alternative hypothesis is that the model with index
#          idxBest is better performing than the other model.
#
# NOTES
# 1) Beware: no validity checking is performed on the entries of MatchedPairs.
# 2) Assume that MatchedPairs[:,0] and MatchedPairs[:,1] are paired, i.i.d 
#    samples generated from model0 and model1 respectively. The null hypothesis
#    of the test is that the probabilities of observing a discordant pair (1,0)
#    or (0,1) are equal, i.e., the two models are indistingushable is terms
#    of clasisfication performance.
#    Without loss of generality, assume that the number of 
#    1/True's is higher for model0. Then, the alternative hypothesis states
#    that the probability of observing (1,0) pairs is higher than observing
#    (0,1) pairs.
#    An alpha-level significance test would reject the null hypothesis if
#    p-value <= alpha and would conclude that model0 is preferable to model1.
# 3) If both columns of MatchedPairs are identical, idxBest defaults to 0.
# 4) If the user selects a significance level alpha, she would reject the
#    null hypothesis, if pvalue <= alpha, and would conclude that the model
#    with index idxBest is the best performing of the two.
#
# DEPENDENCIES
#  import numpy as np
#  import scipy.stats
# 
# AUTHOR
#  Georgios C. Anagnostopoulos, June 2020
#
def ExactMcNemarsTest(MatchedPairs):
    n0 = sum((MatchedPairs[:,0] == 1) & (MatchedPairs[:,1] == 0))
    n1 = sum((MatchedPairs[:,0] == 0) & (MatchedPairs[:,1] == 1))
    w = max(n0, n1)
    Nw = n0 + n1
    idxBest = 0 if n0 >= n1 else 1
    pvalue = scipy.stats.binom.sf(w-1, Nw, 0.5)
    return pvalue, idxBest









import scipy.optimize
import functools
from numba import jit


# logBinomialCDF
#
# Returns the natural-log CDF value for the Binomial distribution.
#
# SYNTAX
#  logCDF = logBinomialCDF(n, N, p)
#
# INPUTS
#  n: integer; number of "successes" in N independent Bernoulli(p) trials.  
#  N: integer >= 1; number of independent trials.
#  p: float in [0,1]; probability of "success".
#
# OUTPUTS
# logCDF:  float <= 0; the natural-log of the CDF at n. 
#
# NOTES
# 1) Beware: No checks are made whether N >= 1 or whether 0 <= p <= 1.
# 2) The function will return the correct value for n < 0 (=numpy.NINF)
#    and for n >= N (= 0).
# 3) This function had to be re-implemented, as, at the time of writing,
#    numba does not allow calling scipy functions in nopython mode.
# 4) scipy.stats.binomial.cdf() uses the incomplete beta function approach
#    to compute the CDF value and, towards this, it leverages the CEPHES 
#    mathematical function library. On the other hand, logBinomialCDF()
#    uses a recursive approach based on the Binomial's PMF, as well as
#    log-probabilities, and, although not yielding obvious wrong values, 
#    appears to be numerically less robust, especially when N is large 
#    and n is very close to N. This phenomenon could be alleviated, if
#    one could robustly compute differences of very small probabilities
#    from 1.
#
# DEPENDENCIES
#  import numpy as np
# 
# AUTHOR
#  Georgios C. Anagnostopoulos, July 2020
#
@jit(nopython=True)
def logBinomialCDF(n, N, p):
    if n < 0:
        logCDF = np.NINF
    elif n >= N:
        logCDF = 0.0
    else:
        if p == 0.0:
            logCDF = 0.0
        elif p == 1.0:
            logCDF = np.NINF
        else:
            # At this point: 0 <= n < N and 0 < p < 1
            logPMF = N * np.log1p(-p)
            logCDF = logPMF
            for k in range(1, n+1):
                logPMF += np.log(float(N - k + 1) * p / (k * (1.0 - p)))
                logCDF = np.logaddexp(logCDF, logPMF)
            
    return logCDF if logCDF <= 0.0 else 0.0


# negNullPowerFunctionSuissaShuster
#
# Returns the negative of the null-power function of the Suissa-Shoster
# statistical test. This function is very specific to the computing the 
# p-value of the aforementioned test. 
#
# SYNTAX
#  negLogValue = negNullPowerFunctionSuissaShuster(p, z, N)
#
# INPUTS
#  p: float in [0,1].  
#  z: float >= 0.
#  N: integer >= 1.
#
# OUTPUTS
# negLogValue:  float >= 0. 
#
# NOTES
# 1) Beware: No valid range checks are made on its input arguments.
# 2) This function is described in Samy Suissa and Jonathan J. 
#    Shuster. The 2x2 Matched-Pairs Trial: Exact Unconditional Design
#    and Analysis. Biometrics, 47(2), 361-372, June 1991.
#
# DEPENDENCIES
#  import numpy as np
#  logBinomialCDF()
# 
# AUTHOR
#  Georgios C. Anagnostopoulos, July 2020
#
@jit(nopython=True)
def negNullPowerFunctionSuissaShuster(p, z, N):
    def inz(_n, _z):
        return np.ceil( (_z * np.sqrt(_n) + _n) / 2.0 )
    
    if z == 0.0:
        logValue = 0.0
    else:
        if p == 0.0:
            logValue = np.NINF
        else:
            k = np.ceil(z * z)
            logConst = np.log(1.0 - p) - np.log(p)
    
            logCoeff = N * np.log(p)
            logTerm = logCoeff + logBinomialCDF(N - inz(N, z), N, 0.5)
            logValue = logTerm
            for n in range(N, k, -1): # n = N, ..., k+1    
                logCoeff += np.log(n) - np.log(N - n + 1) + logConst
                logTerm = logCoeff + logBinomialCDF(n - 1 - inz(n - 1, z), n - 1, 0.5)
                logValue = np.logaddexp(logValue, logTerm)
        
    return -logValue


# SuissaShusterTest
#
# Implements the p-value (observed significance level) of the Suissa-Shuster 
# Test for comparing two binomial proportions in terms of matched pairs.
#
# SYNTAX
#  pvalue, idxBest = SuissaShusterTest(MatchedPairs)
#
# INPUTS
# MatchedPairs: 2D numpy.ndarray of N>=1 rows and 2 columns that only takes 
#               values 0,1 or True, False. Each row contains the classification 
#               outcome (correct/incorrect prediction) of 2 classification 
#               models for a specific sample.
#
# OUTPUTS
# pvalue:  float in [0,1]; p-value (observed significance level) of the test.
# idxBest: integer in {0,1}; indicates the classification model (column index
#          of MatchedPairs) with the most number of 1/True's between the two.
#          The test's alternative hypothesis is that the model with index
#          idxBest is better performing than the other model.
#
# NOTES
# 1) Beware: no validity checking is performed on the entries of MatchedPairs.
# 2) Assume that MatchedPairs[:,0] and MatchedPairs[:,1] are paired, i.i.d 
#    samples generated from model0 and model1 respectively. The null hypothesis
#    of the test is that the probabilities of observing a discordant pair (1,0)
#    or (0,1) are equal, i.e., the two models are indistingushable is terms
#    of clasisfication performance.
#    Without loss of generality, assume that the number of 
#    1/True's is higher for model0. Then, the alternative hypothesis states
#    that the probability of observing (1,0) pairs is higher than observing
#    (0,1) pairs.
#    An alpha-level significance test would reject the null hypothesis if
#    p-value <= alpha and would conclude that model0 is preferable to model1.
# 3) If both columns of MatchedPairs are identical, idxBest defaults to 0.
# 4) If the user selects a significance level alpha, she would reject the
#    null hypothesis, if pvalue <= alpha, and would conclude that the model
#    with index idxBest is the best performing of the two.
# 5) The Suissa-Shuster Test is an unconditional test; it takes into account 
#    both concordant and discordant pairs of observations. It is described in
#    Samy Suissa and Jonathan J. Shuster. The 2x2 Matched-Pairs Trial: Exact 
#    Unconditional Design and Analysis. Biometrics, 47(2), 361-372, June 1991. 
#
# DEPENDENCIES
#  import numpy as np
#  import scipy.stats
#  import scipy.optimize
#  import functools
#  negNullPowerFunctionSuissaShuster()
# 
# AUTHOR
#  Georgios C. Anagnostopoulos, July 2020
#
def SuissaShusterTest(MatchedPairs):
    N = len(MatchedPairs)
    ND0obs = sum((MatchedPairs[:,0] == 1) & (MatchedPairs[:,1] == 0))
    ND1obs = sum((MatchedPairs[:,0] == 0) & (MatchedPairs[:,1] == 1))
    NDobs = ND0obs + ND1obs
    if ND0obs >= ND1obs:
        idxBest = 0
        y = ND0obs; x = ND1obs
    else:
        idxBest = 1
        y = ND1obs; x = ND0obs
    z = float(y - x) / np.sqrt(x+y)
    nnpf = functools.partial(negNullPowerFunctionSuissaShuster, z=z, N=N)
    minRes = scipy.optimize.minimize_scalar(nnpf, bounds=(0.0, 1.0), method='bounded')
    pvalue = np.exp(- minRes.fun)
    return pvalue, idxBest


# HolmBonferroniProcedure
#
# The function implements the Holm-Bonferroni (a.k.a. Holm's Step-Down) 
# procedure for multiple comparisons of classification models.
#
# SYNTAX
#  pseudoPvalues, idxBest, idxCol = SuissaShusterTest(MatchedTuples, pValueFunc)
#
# INPUTS
# MatchedTuples: 2D numpy.ndarray of N>=1 rows and M columns that only takes 
#                values 0,1 or True, False. Each row contains the classification 
#                outcome (correct/incorrect prediction) of M>=2 classification 
#                models for a specific sample. There are a total of N samples.
# pValueFunc:    callable; pointer to a function that computes the p-value
#                of a matched pair test that compares two classification models
#                and that has a signature of 
#                pvalue, idxbest = pValueFunc(MatchedPairs).
#                SuissaShusterTest() is such an example function. 
#
# OUTPUTS
# pseudoPvalues: 1D numpy.ndarray of M-1 floats >= 0 that can be compared to a
#                significance level alpha; see notes below for more details.
# idxBest:       integer in {0,M-1}; indicates the classification model 
#                (column index of MatchedTuples) with the most number of 
#                1/True's among all models. In case of tied columns,
#                the smallest column index is returned.
# idxCol:        1D numpy.ndarray of M-1 integers in {0, .., M-1} but excluding
#                idxBest. Shows which value in pseudoPvalues corresponds to 
#                which model being compared to idxBest.
#
# NOTES
# 1) The procedure intends to control the Family-wise Error Rate (FWER), i.e.,
#    to strictly bound the probability of a Type I error, of the family of 
#    hypothesis tests considered.
#
# DEPENDENCIES
#  import numpy as np
# 
# AUTHOR
#  Georgios C. Anagnostopoulos, July 2020
#
def HolmBonferroniProcedure(MatchedTuples, pValueFunc):
    N, M = MatchedTuples.shape
    idxBest = np.argmax(np.sum(MatchedTuples, axis=0))
    pValues = np.empty(M-1)
    pseudoPvalues = np.empty(M-1)
    
    # compute p-values for all tests (idxBest vs the rest)
    columnList = [col for col in range(M) if col != idxBest]
    i = 0
    for idx in columnList:
        MatchedPairs = MatchedTuples[:, [idxBest, idx]]
        pvalue, _ = pValueFunc(MatchedPairs)
        pValues[i] = pvalue
        i += 1
        
    # sort p-values in ascending order
    sortedPvalues = np.sort(pValues)
    sortexIdx = np.argsort(pValues)
    # order column indices in the same order as pValues
    idxCol = np.array(columnList)
    idxCol = idxCol[sortexIdx]
    # computed pseudo-p-values
    idx = np.arange(0, M-1)
    pseudoPvalues = (M - idx) * sortedPvalues    
    
    return pseudoPvalues, idxBest, idxCol
    
	
# Demonstration of Holm-Bonferroni multiple comparison procedure
# using McNemar's Exact Test for pair-wise comparisons.
if __name__ == "__main__":
	
	# Below, rows (columns) represent samples (models).
	# A 1 (0) reflects that the corresponding model (in)correctly classified the sample. 
	MatchedTuples = np.array([[0, 1, 0, 0],
    	                      [0, 0, 0, 1],
        	                  [1, 1, 0, 0],
            	              [0, 1, 0, 0],
                	          [0, 1, 0, 1],
                    	      [0, 1, 0, 0],
                        	  [0, 1, 0, 0]
                         	])

	pseudoPvalues, idxBest, idxCol = HolmBonferroniProcedure(MatchedTuples, ExactMcNemarsTest)

	alpha = 0.2 # multiple comparison test's significance level
	rejected = idxCol[pseudoPvalues <= alpha] 
	if rejected.size == 0:
    	# no null hypothesis was rejected
    	print('The best model (column #{:0}) does not significantly outperform the other models.'.format(idxBest))
	elif rejected.size == MatchedTuples.shape[1]:
    	print('The best model (column #{:0}) significantly outperforms all models.'.format(idxBest))
	else:
    	print('The best model (column #{:0}) significantly outperforms the following models:'.format(idxBest))
    	print(rejected)
    	print('but does not significantly outperform the following models:')
    	print(idxCol[pseudoPvalues > alpha])
