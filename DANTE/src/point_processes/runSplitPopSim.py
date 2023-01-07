import numpy as np

from point_processes.SplitPopulationTPP import *
from sklearn.metrics import confusion_matrix


def main():
    # First kernel is base rate, the rest is for the external source.
    mk = ExponentialPseudoMemoryKernel(beta=1.0)
    mkList = [ConstantMemoryKernel(), mk]

    # this TPP only uses its own events as source.
    sourceNames = ['base', 'soc_media_1']
    # sourceNames = ['base']

    stop_criteria = {'max_iter': 600,
                     'epsilon': 1e-4}
    exploitProcess = SplitPopulationTPP(
        mkList, sourceNames, stop_criteria,
        desc='Split population process with multiple kernels')

    numRealizations = 200
    Realizations = []
    maxNumEvents = 0
    maxNumEventsIdx = 0
    total_susceptible = 0
    sourceNamesSocialMedia = ['socialMediaProcess']
    socialMediaProcess = HawkesTPP(
        mk, sourceNamesSocialMedia, stop_criteria,
        desc='Hawkes TPP with exponetial(beta=1) kernel')
    socialMediaProcess.mu = 1.0
    alphas =  [0.05]
    socialMediaProcess.alpha = pd.Series(
            data=np.array(alphas), index=["socialMediaProcess"])
    
    numSamples = 2000
    sampleDimensionality = 2
    classPrior1 = 0.5
    misclassificationProbability = 0.1
    X, y, w_tilde = gen2IsotropicGaussiansSamples(
        numSamples, sampleDimensionality, classPrior1, misclassificationProbability)
    numProcesses = 2
    for r in range(0, numSamples):
        # Exponential(100)-distributed right-censoring time
        T = scipy.stats.expon.rvs(loc=0.0, scale=600.0)
        processList = [exploitProcess, socialMediaProcess]
        Realization, isSusceptible = simulation_split_population(
            processList, T, [[]] * numProcesses, w_tilde, X[r], resume=False,
            resume_after_split_pop=False)

        print(
            "Right Censoring Time for the current realization {0}, realization: {1}".format(T, r))

        if (isSusceptible == 1):
            total_susceptible += 1
        Realizations.append(Realization)
        # number of realizations of social media
        numEvents = len(Realization[0])
        if numEvents > maxNumEvents:
            maxNumEvents = numEvents
            maxNumEventsIdx = r

    print("Total susceptible among simulated population: ", total_susceptible)

    # X = np.load('../data/feature_vector.npy', allow_pickle=True)
    # w_tilde = np.load('../data/gt_w_tilde.npy', allow_pickle=True)
    # alpha = np.load('../data/gt_alpha.npy', allow_pickle=True)

    exploitProcess.setFeatureVectors(X)
    # exploitProcess.feature_vectors = X
    exploitProcess.w_tilde = w_tilde
    exploitProcess.setupTraining(Realizations, X,"test" , Realizations)
    _, plot_list = exploitProcess.train(Realizations,"test" , Realizations)

    IIDSamples = []

    for realization in Realizations:
        if len(realization[0]) > 1:
            if realization[0][0] > 0:
                exploitProcess.setSusceptible()
                IIDSamples.extend(
                    exploitProcess.transformEventTimes(realization))

    IIDSamples = list(filter((0.0).__ne__, IIDSamples))

    # print(IIDSamples)
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=largeFigSize)
    pvalue = KSgoodnessOfFitExp1(
        sorted(IIDSamples), ax, showConfidenceBands=True, title="Training")

    fig.savefig('GOFplot.png', format='png', dpi=600)
    plt.show()


if __name__ == "__main__":
    main()
