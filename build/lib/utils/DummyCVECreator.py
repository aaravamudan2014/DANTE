import numpy as np
from datetime import datetime, timedelta
from random import expovariate
######################################################################################################
# DummyCVECreator.py
# contains all the relevant function to create dummy CVEs.
#  create_split_pop_feature_vectors()   :
#   createSocialMediaData()             :
#
######################################################################################################


# Function to generate feature vectors along with truth labels for associated exploited values.
# this is just to set up stuff for the toy problem
# cannot be used for real world examples because feature vectors will be obtained
# from another mechanism : deep NLP architecture


def create_split_pop_feature_vectors(cve_container, exploitModeller):
    numSamples = cve_container.getNumCVEs()
    sampleDimensionality = 2
    classPrior1 = 0.6
    misclassificationProbability = 0.1

    X, y, w_tilde = gen2IsotropicGaussiansSamples(
        numSamples, sampleDimensionality, classPrior1, misclassificationProbability)

    allOnes = np.ones((numSamples, 1))
    X_tilde = np.concatenate((X, allOnes), axis=1)

    y_pred = np.sign(np.dot(X_tilde, w_tilde))
    errorRate = np.sum(y_pred != y) / numSamples

    print('gen2IsotropicGaussiansSamples(): errorRate={:1} should approximately match misclassificationProbability={:2} for {:3} samples'.format(
        errorRate, misclassificationProbability, numSamples))

    exploitModeller.setGroundTruthOptimalWeights(w_tilde)

    for x, cve, y_i in zip(X_tilde, cve_container.getCVEList(), y):
        cve.setFeatureVector(x)
        cve.setGroundTruthLabel(y_i)

    print('All Feature vectors and associated truth values have been updated...')

    return cve_container, exploitModeller


def createSocialMediaData(poisson_param, startDate, numEvents):
    simulated_time = 0
    simulated_events = []
    events_generated = 0
    while events_generated < numEvents:
        ev = expovariate(
            poisson_param)
        simulated_events.append(
            startDate + timedelta(seconds=ev+simulated_time))
        simulated_time += ev
        events_generated += 1
    return simulated_events
