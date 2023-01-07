import numpy as np
import scipy.stats 

##################################################
# computeMeanFromErrorProb()
# 
# Computes the means of two uni-variate normal distributions N(mu,1) for class 1 and N(-mu,1) for class 2,
# which form a mixture, so that a given optimal misclassification probability is achieved.
#
# INPUTS
# classPrior1: scalar in (0,1); prior probability for class 1. 
# misclassificationProbability: scalar in (0,0.5); Bayes-optimal probability of misclassification. 
#                               Must be less than min{classPrior1, 1-classPrior1}. 
# OUTPUTS
# mu: non-negative scalar; the (negative) mean of class 1 (class 2).
#
# DEPENDENCIES
# packages: numpy as np, scipy.stats
#
# AUTHOR: Georgios C. Anagnostopoulos, December 2019
# 
##################################################
def computeMeanFromErrorProb(classPrior1, misclassificationProbability):
    classPrior2 = 1.0 - classPrior1
    
    if misclassificationProbability > min([classPrior1, classPrior2]):
        raise ValueError('misclassificationProbability is too high for the given prior probability!')
    
    logRatioPriors = np.log(classPrior2 / classPrior1)
    standardNormal = scipy.stats.norm()
    
    # Newton's method
    numerator = np.inf
    mu = np.sqrt(np.abs(logRatioPriors)) + 0.0001
    while np.log10(np.abs(numerator)) > -10:
        xo = 0.5 * logRatioPriors / mu
        numerator = classPrior1 * standardNormal.cdf(xo-mu) + classPrior2 * (1.0 - standardNormal.cdf(xo+mu)) - misclassificationProbability
        denominator = - ( classPrior1 * standardNormal.pdf(xo-mu) * (1.0 + xo/mu) + classPrior2 * standardNormal.pdf(xo+mu) * (1.0 - xo/mu) )
        mu = mu - numerator/denominator
        
    return mu


##################################################
# gen2IsotropicGaussiansSamples()
#
# Draws labeled samples from a mixture of two multi-variate Normal distributions with the following characteristics:
# - both distributions have the identity matrix as a covariance matrix
# - the distribution of class 1 has mean vector [mu, 0, 0, ...]
# - the distribution of class 2 has mean vector [-mu, 0, 0, ...]
# - the scalar mean parameter mu is computed for given class prior probabilities and optimal misclassification rate (probability) 
# Finally, it also returns coefficients for the optimal label prediction rule. 
#
# INPUTS 
# numSamples: positive integer; number of i.i.d. samples to draw.
# sampleDimensionality: positive integer; dimensionality of samples (excluding label).
# classPrior1: scalar in (0,1); prior probability for class 1. 
# misclassificationProbability: scalar in (0,0.5); Bayes-optimal probability of misclassification. 
#                               Must be less than min{classPrior1, 1-classPrior1}. 
#
# OUTPUTS
# X: numpy array of real values and of shape (numSamples, sumpleDimensionality). 
#    Contains the i.i.d. samples in rows.
# y: numpy array of shape (numSamples,) with values 1 (for class 1) or -1 (for class 2).
#    Contains the corresponding class labels.
# w_tilde: numpy array of real values and of shape (sumpleDimensionality+1,). Contains the
#          coefficients of the hyperplane which achieves an error of misclassificationProbability.
#
# DEPENDENCIES
# packages: numpy as np
# functions: computeMeanFromErrorProb()
#
# NOTES
# 1) Due to the problem's setting, w_tilde's non-zero elements are only its first and last elements.
# 2) Use of w_tilde: if x is an (input) sample, and x_tilde is x appended by a 1, then the sign of the dot product
#    between w_tilde and x_tilde conveys the optimal predicted label of x.  
# 
# AUTHOR: Georgios C. Anagnostopoulos, December 2019
# 
##################################################
def gen2IsotropicGaussiansSamples(numSamples, sampleDimensionality, classPrior1, misclassificationProbability):
    classPrior2 = 1.0 - classPrior1
    
    # Compute appropriate means to achieve the given misclassification probability 
    scalarMean1 = computeMeanFromErrorProb(classPrior1, misclassificationProbability)
    Mean1 = np.zeros(sampleDimensionality)
    Mean1[0] = scalarMean1
    Mean2 = - Mean1
    
    # Construct optimal augemented weight vector
    w_tilde = Mean1 - Mean2
    w_tilde = np.array(np.append(w_tilde, np.log(classPrior1/classPrior2)))
    
    
    # Determine number of samples to be drawn from each class
    u = np.random.uniform(0.0, 1.0, numSamples)
    numSamplesClass1 = np.sum(u <= classPrior1)
    numSamplesClass2 = numSamples - numSamplesClass1
    
    # Draw samples
    CovMx = np.eye(sampleDimensionality)
    X1 = np.array(np.random.multivariate_normal(Mean1, CovMx, numSamplesClass1))
    y1 = np.ones(numSamplesClass1)
    X2 = np.array(np.random.multivariate_normal(Mean2, CovMx, numSamplesClass2))
    y2 = - np.ones(numSamplesClass2)
    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2))
    
    return X, y, w_tilde


# Unit Testing
if __name__ == "__main__":


    # original samples used for example
    # numSamples = 5000
    # sampleDimensionality = 2
    # classPrior1 = 0.6
    # misclassificationProbability = 0.1

    numSamples = 10000
    sampleDimensionality = 2
    classPrior1 = 0.85
    misclassificationProbability = 0.1

    X, y, w_tilde = gen2IsotropicGaussiansSamples(
        numSamples, sampleDimensionality, classPrior1, misclassificationProbability)
    
    allOnes = np.ones((numSamples,1))
    X_tilde = np.concatenate((X, allOnes), axis=1)
    print(w_tilde)
    y_pred = np.sign(np.dot(X_tilde, w_tilde))
    errorRate = np.sum(y_pred != y) / numSamples
    print('gen2IsotropicGaussiansSamples(): errorRate={:1} should approximately match misclassificationProbability={:2} for {:3} samples'.format(errorRate, misclassificationProbability, numSamples))
