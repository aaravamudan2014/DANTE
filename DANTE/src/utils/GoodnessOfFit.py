'''KS test related functions'''
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
import matplotlib.cm as cm

##################################################
# kstwo_isv()
#
# Compute the critical value of the Kolmogorov-Smirnov (KS) test statistic
# given the significance level and number of samples N for the KS test that
# that uses a two-sided alternative hypothesis. This is provided by the inverse
# tail probability ("survival function") of the KS test statistic's null
# hypothesis distribution.
#
# INPUTS
# significanceLevel: scalar in [0,1]; significance level ("alpha value") of the test.
# N:                 positive integer; number of i.i.d. samples used for the test.
# mode:              string equal to 'approx' or 'asymp'. Indicates whether to use the
#                    approximated or asymptotic (for very large N) null hypothesis distribution.
# OUTPUTS
# criticalValue: non-negative scalar; critical value of the two-sided KS test statistic.
#                Conceptually, if SF() is the tail probability ("survival function") of
#                the two-sided KS test under the null hypothesis (sample was drawn from
#                a model distribution), then criticalValue = ISF(significanceLevel),
#                where ISF() is the inverse function of SF().
#
# DEPENDENCIES
# numpy, scipy.stats.kstwobign, scipy.stats.ksone
#
# NOTES
# 1) The KS test statistic for the relevant test equals the maximum absolute distance
#    between the empirical CDF of the data and the model distribution. The test statistic
#    is independent of the model distribution.
# 2) The 'approx' mode seems to be preferably used for small number of samples.
#
# USAGE EXAMPLE:
# Assume that Samples is a numpy array containing a collection of N scalar i.i.d. samples drawn
# from an unknown distribution. We would like to test whether the elements of Samples were drawn
# i.i.d. from a model distribution, say, Exp(1). Assume that D is the value of the KS test
# statistic.
#
# Dcritical = kstwo_isv(alpha, N)
#
# If D >= Dcritical, then one would conclude that the alternative hypothesis is true at significance
# level alpha.
#
# AUTHOR: Georgios C. Anagnostopoulos, April 2019
#
##################################################


def kstwo_isv(significanceLevel, N, mode='approx'):
    '''Critical value of KS test'''
    if mode == 'asymp':
        criticalValue = scipy.stats.kstwobign.isf(
            significanceLevel) / np.sqrt(N)
    if mode == 'approx':
        if N > 2666 or significanceLevel > 0.80 - 0.0003 * N:
            criticalValue = scipy.stats.kstwobign.isf(
                significanceLevel) / np.sqrt(N)
        else:
            criticalValue = scipy.stats.ksone.isf(significanceLevel / 2.0, N)
    return criticalValue


##################################################
# KSgoodnessOfFitExp1()
#
# Provides the p-value of the two-sided Kolmogorov-Smirnov test
# that tests whether a collection of N i.i.d. samples follows an
# exponential distribution of rate 1.0, denoted as Exp(1).
# Optionally, it generates a probability-probability (PP) plot
# of the empirical CDF of the samples against the CDF of Exp(1)
# with 90%, 95% and 99% confidence bands (simultaneous/uniform
# confidence intervals).
#
# INPUTS
# IIDSamples: numpy array of floats; contains i.i.d. samples.
# ax:         matplotlib axis object; used to generate the PP plot, unless =None.
#
# OUTPUTS
# pvalue:     float in [0,1]; p-value of the KS test.
#
# DEPENDENCIES
# numpy.sort, numpy.arange, numpy.clip
# scipy.stats.kstest, scipy.stats.expon
# matplotlib
# kstwo_isv
#
# NOTES
# 1) For two-sided KS test, the null hypothesis states that the given samples
#    were sampled from an Exp(1) distribution, while the alternative states the
#    opposite.
# 2) If one desires to use this test at significance level alpha, one would reject
#    the null hypothesis if pvalue <= alpha.
# 3) For goodness-of-fit purposes, a large value of pvalue (close to 1.0) indicates
#    a good fit of the samples to a Exp(1) distribution.
# 4) The plotted data will lie in a (1.0-pvalue)*100% confidence band.
# 5) An Anderson-Darling test testing non-conformity of the data to Exp(1) would have
#    more statistical power for small number of samples than the two-sided KS test
#    considered here. However, for large samples the two tests become asymptotically
#    equivalent. Also, SciPy does not feature code that yields p-values for such a test.
# 6) Semantics for goodness of fit: if the p-value is larger than a specified
#    significance level alpha, then we conclude that "we do not have enough evidence
#    to hypothesize that the given data were not sampled from an Exp(1) at significance
#    level alpha." Otherwise, we conclude that "at a significance level alpha, the data
#    imply that they were not sampled from an Exp(1)." The latter conclusion points to
#    the fact that Exp(1) is not a good model for the data.
# 7) A P-P plot is a scatter plot of the pairs (F_N(x_(n)), F(x_(n))) = (n/N, F(x_(n)))
#    for n = 1, 2, ..., N, where F() is the model CDF (here, Exp(1)), F_N() is the
#    empirical CDF of the samples, x_(n) is the n-th sorted (in ascending order) sample
#    (i.e. the n-th empirical quantile) and N is the number of samples.
#
# AUTHOR: Georgios C. Anagnostopoulos, April 2019 (slightly modified January 2020).
#
##################################################
def KSgoodnessOfFitExp1(IIDSamples, ax=None, showConfidenceBands=True, title=""):
    '''p-value of two-sided KS test and optional P-P plot'''
    N = len(IIDSamples)
    # Compute two-sided KS test statistic D and the test's p-value
    # for an exponential distribution with rate 1.0 as the model distribution.
    # D below is the KS test satistic.
    _, pvalue = scipy.stats.kstest(
        np.array(IIDSamples), scipy.stats.expon.cdf, (0.0, 1.0), N, alternative='two-sided', mode='approx')

    if ax is not None:
        # sort samples in ascending order
        sortedIIDSamples = np.sort(IIDSamples)

        n = np.arange(1.0, N + 1.0, 1.0)
        ECDFvalues = n / N
        # print(sortedIIDSamples)
        # input()
        CDFExp1values = scipy.stats.expon.cdf(
            sortedIIDSamples, 0.0, 1)  # cdf of Exp(1)

        p = np.arange(0.0, 1.01, 0.01)

        if showConfidenceBands is True:
            # 90% confidence bands
            epsilon10 = kstwo_isv(0.1, N, 'approx')
            lower_band10 = np.clip(p - epsilon10, 0.0, None)
            upper_band10 = np.clip(p + epsilon10, None, 1.0)
            # 95% confidence bands
            epsilon05 = kstwo_isv(0.05, N, 'approx')
            lower_band05 = np.clip(p - epsilon05, 0.0, None)
            upper_band05 = np.clip(p + epsilon05, None, 1.0)
            # 99% confidence bands
            epsilon01 = kstwo_isv(0.01, N, 'approx')
            lower_band01 = np.clip(p - epsilon01, 0.0, None)
            upper_band01 = np.clip(p + epsilon01, None, 1.0)

        ax.plot(p, p, 'k--', CDFExp1values, ECDFvalues, 'r.')
        if showConfidenceBands is True:
            ax.plot(p, lower_band10, 'lightgray', p, upper_band10, 'lightgray',
                    p, lower_band05, 'lightgray', p, upper_band05, 'lightgray',
                    p, lower_band01, 'lightgray', p, upper_band01, 'lightgray')

        ax.set_xlabel('Exp(1) CDF')
        ax.set_ylabel('Empirical CDF')
        ax.set_title(
            'P-P Plot for {:s} dataset (KS p-value={:.02f})'.format(title, pvalue))
        ax.axis('square')

    return pvalue


def KSgoodnessOfFitExp1MV(IIDSamples, ax=None, showConfidenceBands=True):
    '''p-value of two-sided KS test and optional P-P plot'''
    N = [0] * len(IIDSamples)

    for i in range(len(IIDSamples)):
        N[i] = len(IIDSamples[i])
    color_palette = ['rs--', '-g', 'rs:', 'go -']
    # Compute two-sided KS test statistic D and the test's p-value
    # for an exponential distribution with rate 1.0 as the model distribution.
    # D below is the KS test satistic.
    pvalue = [[]] * len(IIDSamples)

    for i in range(len(IIDSamples)):
        _, pvalue[i] = scipy.stats.kstest(
            IIDSamples[i], scipy.stats.expon.cdf, (0.0, 1.0), N[i], alternative='two-sided', mode='approx')

    if ax is not None:
        # sort samples in ascending order
        sortedIIDSamples = [np.sort(l) for l in IIDSamples]
        n = [[]] * len(IIDSamples)
        ECDFvalues = [[]] * len(IIDSamples)
        for i in range(len(IIDSamples)):
            n[i] = np.arange(1.0, N[i] + 1.0, 1.0)
            ECDFvalues[i] = n[i] / N[i]

        # CDFExp1values = [[]]*len(IIDSamples)
        CDFExp1values = [scipy.stats.expon.cdf(
            l, 0.0, 1.0) for l in sortedIIDSamples]  # cdf of Exp(1)

        # print(ECDFvalues)
        # input()
        p = np.arange(0.0, 1.01, 0.01)

        lower_band10 = [0] * len(IIDSamples)
        upper_band10 = [0] * len(IIDSamples)

        lower_band05 = [0] * len(IIDSamples)
        upper_band05 = [0] * len(IIDSamples)

        lower_band01 = [0] * len(IIDSamples)
        upper_band01 = [0] * len(IIDSamples)

        if showConfidenceBands is True:
            # 90% confidence bands
            for j in range(len(IIDSamples)):
                epsilon10 = kstwo_isv(0.1, N[j], 'approx')
                lower_band10[j] = np.clip(p - epsilon10, 0.0, None)
                upper_band10[j] = np.clip(p + epsilon10, None, 1.0)
                # 95% confidence bands
                epsilon05 = kstwo_isv(0.05, N[j], 'approx')
                lower_band05[j] = np.clip(p - epsilon05, 0.0, None)
                upper_band05[j] = np.clip(p + epsilon05, None, 1.0)
                # 99% confidence bands
                epsilon01 = kstwo_isv(0.01, N[j], 'approx')
                lower_band01[j] = np.clip(p - epsilon01, 0.0, None)
                upper_band01[j] = np.clip(p + epsilon01, None, 1.0)

        colors = ['black', 'green', 'red', 'yellow']

        labels = ['Process 1', 'Process 2', 'Process 3', 'Process 4']

        for j in range(len(IIDSamples)):
            # ax.plot(p, p, color_palette[j], CDFExp1values[j],
            #         ECDFvalues[j], 'r.')
            # z = np.array([np.random.random()*365]*len(IIDSamples[j]))
            ax[j].plot(CDFExp1values[j],
                       ECDFvalues[j], c=colors[j], label=labels[j])

        if showConfidenceBands is True:
            for j in range(len(IIDSamples)):
                ax[j].plot(p, lower_band10[j], 'black', p, upper_band10[j], 'black',
                           p, lower_band05[j], 'black', p, upper_band05[j], 'black',
                           p, lower_band01[j], 'black', p, upper_band01[j], 'black')

        for j in range(len(IIDSamples)):
            ax[j].set(xlabel='Exp(1) CDF', ylabel='Empirical CDF')
            ax[j].set_title(
                'P-P Plot (KS p-value={:.02f})'.format(pvalue[j]))
            ax[j].axis('square')
            ax[j].label_outer()
            ax[j].legend()
    return pvalue
