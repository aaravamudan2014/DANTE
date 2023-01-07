#!/usr/bin/python
'''Temporal Point Process (TPP) sub-classes'''
import json
import pandas as pd
import numpy as np
import scipy.stats
# import logging
import os
import h5py
from matplotlib import pyplot as plt

from point_processes.TemporalPointProcess import TemporalPointProcess as TPP, TrainingStatus
from utils.MemoryKernel import ExponentialPseudoMemoryKernel
from utils.Simulation import *
from core.DataStream import DataStream
import tables


class HawkesTPP(TPP):
    '''Hawkes Temporal Point Process'''

    # Instance variables
    #
    #   sourceNames:           list of strings; names of the TPP's sources; inherited from TPP base class.
    #   ownName:               string, name of the current TPP, must be one of the sourceNames; inherited form TPP base class
    #   desc:                  string; short textual description of the TPP; inherited from TPP base class.
    #   _logger:               logger object; for logging; inherited from TPP base class.
    #   _setupTrainingDone:    boolean; informs whether the setupTraining(); inherited from TPP base class.
    #   mu:                    float; TPP parameter: base intensity
    #   alpha                  numpy array; TPP parameter.
    #   mk                     object of a MemoryKernel subclass; TPP parameter.
    #   stop_criteria:         dict: stop criteria for learning ; TPP parameter
    #   __totalTrainingEvents: non-negative integer; a quantity required during training.
    #   __sumOfPsis:           non-negative real scalar; a quantity required during training.

    # mk:          a specific MemoryKernel object with fixed parameters.
    # sourceNames: list of strings with the source names. The list's first element is the name of the TPP's own source.
    def __init__(self, mk, sourceNames, stop_criteria, desc='Hawkes TPP', logger=None):
        """[summary]

        Args:
            mk ([type]): [description]
            sourceNames ([type]): [description]
            stop_criteria ([type]): [description]
            desc (str, optional): [description]. Defaults to 'Hawkes TPP'.
            logger ([type], optional): [description]. Defaults to None.
        """
        self.dim = len(sourceNames)
        self.desc = desc
        self.sourceNames = sourceNames
        self.ownName = sourceNames[0]
        self.stop_criteria = stop_criteria
        self.mk = mk
        self.pre_cal_path = '../temp_storage/'
        self.pre_cal_name = '{}-pre-cal.h5'.format(self.ownName)
        self.initParams()

        super().__init__(sourceNames, desc, logger)

    # t: float or 1-dimensional numpy.ndarray of reals; relative event time(s). Typically, larger than 0.
    # realization: a single realization of model including all sources events

    def intensity(self, t, realization):
        """[summary]

        Args:
            t ([type]): [description]
            realization ([type]): [description]

        Returns:
            [type]: [description]
        """
        # realization = ['self', 'external']
        # calculate history excitation
        hist_excitation = 0   # History events excitation

        for s in range(len(self.sourceNames)):
            #  gather all history relevant to the given point process
            s_hist = [h for h in realization[s] if h < t]
            # iterate over each event in this specific history
            for e in s_hist:
                hist_excitation = hist_excitation + \
                    self.alpha.loc[self.sourceNames[s]] * self.mk.phi(t-e)
        # intensity value = base intensity + hist_excitaion

        # realization = [[0.2, 0.3,0.4], [0.4]]
        # intensity  = self.mu + self.alpha.phi(t-e)

        value = self.mu + hist_excitation

        return value

    # t: float or 1-dimensional numpy.ndarray of reals; relative event time(s). Typically, larger than 0.
    # TPPdata: a list containing the process' own (possibly, empty) realization.

    def cumIntensity(self, t, TPPdata):
        """[summary]

        Args:
            t ([type]): [description]
            TPPdata ([type]): [description]

        Returns:
            [type]: [description]
        """
        Realization = TPPdata[0]
        tN = Realization[-1] if len(Realization) > 0 else 0.0
        value = self.alpha * (self.mk.psi(t) - self.mk.psi(tN))
        return np.sum(value)

    def loglikelihood(self, TPPdata):
        """[summary]

        Args:
            TPPdata ([type]): [description]

        Returns:
            [type]: [description]
        """
        pre_cal_file = os.path.join(self.pre_cal_path, self.pre_cal_name)

        activatedTerm = 0
        survivalTerm = 0

        for realization in TPPdata:

            # load pre calculated file
            rk = realization['cascade_id']
            pre_cal_df = pd.read_hdf(pre_cal_file, rk)
            # locate the event df
            event_df = pre_cal_df.iloc[:, :-1]
            # number of own source events
            own_events_num = event_df.shape[1]

            if own_events_num > 0:

                for own_event_ind in range(own_events_num):
                    # calculate the intensity at own event time
                    activatedTerm = activatedTerm + np.log((self.mu +
                                                            np.dot(self.alpha, event_df.loc[:, own_event_ind])))

            T = realization['right_censored_time']
            # survival terms
            survival_df = pre_cal_df.loc[:, 'survival']

            # calculate the log-likelihood for survival part
            survivalTerm = survivalTerm + \
                np.dot(self.alpha, survival_df) + (T * self.mu)

        loglikelihood = activatedTerm - survivalTerm

        return loglikelihood

    def preProcessingData(self, ori_data):
        """[summary]

        Args:
            ori_data ([type]): [description]

        Returns:
            [type]: [description]
        """
        new_data = []
        
        for k in range(len(ori_data)):
            d = ori_data[k]
            # create the object
            obj = {}
            # data for each dim
            if len(d[0]) < 1:
                continue
            for i in range(len(d)):
                event_time = d[i][:-1]                
                obj[self.sourceNames[i]] = list(event_time)
            # right censoring time
            obj['right_censored_time'] = d[0][-1]
            # assign cascade id to n-th simulation
            obj['cascade_id'] = 'c{}'.format(k)
            # append current cascade data
            # if len(obj['exploit']) > 0:
            #     print(obj)
            #     input()
            new_data.append(obj)
        return new_data

    def transformEventTimes(self, TPPdata):
        """[summary]

        Args:
            TPPdata ([type]): [description]

        Returns:
            [type]: [description]
        """
        TPPdata = self.preProcessingData(TPPdata)

        rescaled_df = pd.DataFrame(columns=['cve', 'rescaled_time'])
        # print(self.sourceNames)
        # print("here")
        # input()
        for realization in TPPdata:

            activated_sources = [s for s in list(realization.keys())
                                 if s in self.sourceNames]
            temp = None
            if self.ownName in activated_sources:
                own_events = sorted(realization[self.ownName])
                own_events_num = len(own_events)

                # initialization
                rescaled_value_array = np.zeros(own_events_num)

                for own_event_ind in range(own_events_num):
                    tk_1 = own_events[own_event_ind -
                                      1] if own_event_ind >= 1 else 0
                    tk = own_events[own_event_ind]

                    rescaled_value = 0
                    for source in activated_sources:

                        events_1 = [c for c in realization[source] if c < tk_1]
                        for event in events_1:
                            rescaled_value = rescaled_value + \
                                self.alpha.loc[source] *\
                                (self.mk.psi(tk-event) - self.mk.psi(tk_1-event))

                        events_2 = [c for c in realization[source]
                                    if c >= tk_1 and c < tk]
                        for event in events_2:
                            rescaled_value = rescaled_value + \
                                self.alpha.loc[source]*self.mk.psi(tk-event)

                    rescaled_value = rescaled_value + (tk-tk_1) * self.mu
                    rescaled_value_array[own_event_ind] = rescaled_value
                    temp = pd.DataFrame(
                        rescaled_value_array, columns=['rescaled_time'])
                    temp['cve'] = realization['cascade_id']
                if temp is not None:
                    rescaled_df = pd.concat([rescaled_df, temp])

        return rescaled_df['rescaled_time'].values

    def training_pre_calculation(self, TPPdata):
        """[summary]

        Args:
            TPPdata ([type]): [description]
        """
        # create the h5py file
        pre_cal = h5py.File(self.pre_cal_path + self.pre_cal_name, 'w')
        pre_cal.close()
        # print(TPPD)
        print("Pre-processing the cascade data for training")
        for index, realization in enumerate(TPPdata):
            try:
                # realization group name
                print(realization['cascade_id'])
                rk = 'cascade_id_{}'.format(realization['cascade_id'])
                # create a group for each cascade
                # pre_cal.create_group(rk)

                activated_sources = [s for s in list(realization.keys())
                                    if s in self.sourceNames]

                if self.ownName in activated_sources:
                    own_events = sorted(realization[self.ownName])
                    own_events_num = len(own_events)

                    # Calculate Recurssive Matrix for user == userId
                    event_df = pd.DataFrame(data=np.zeros((self.dim, own_events_num)),
                                            index=self.sourceNames)

                    if self.mk.desc.startswith('ExponentialPseudo'):
                        for source in activated_sources:
                            for own_event_ind in range(own_events_num):
                                ts = own_events[own_event_ind -
                                                1] if own_event_ind >= 1 else 0
                                te = own_events[own_event_ind]

                                newEvents = [
                                    c for c in realization[source] if c < te and c >= ts]

                                newArr = 0
                                for event in newEvents:
                                    newArr = newArr + self.mk.phi(te - event)

                                event_df.loc[source, own_event_ind] = self.mk.phi(te-ts) * \
                                    event_df.loc[source, own_event_ind-1] + newArr \
                                    if own_event_ind > 1 else newArr

                    else:
                        for source in activated_sources:
                            for own_event_ind in range(own_events_num):
                                cal_source_events = [c for c in realization[source]
                                                    if c < own_events[own_event_ind]]

                                for event in cal_source_events:
                                    event_df.loc[source, own_event_ind] = \
                                        event_df.loc[source, own_event_ind] +\
                                        self.mk.phi(
                                            own_events[own_event_ind] - event)

                else:
                    event_df = pd.DataFrame(index=self.sourceNames)

                # survival terms
                T = realization['right_censored_time']
                # survival terms
                survival_df = pd.Series(data=np.zeros(
                    self.dim), index=self.sourceNames, name='survival')
                for source in activated_sources:
                    for event in realization[source]:
                        if T > event:
                            survival_df.loc[source] = survival_df.loc[source] + \
                                self.mk.psi(T-event)

                # combine
                pre_cal_df = pd.concat([event_df, survival_df], axis=1, sort=False)
                # retreive cacsade ID
                # pre_cal.close()
                # save into group with cascade id
                # pre_cal = h5py.File(self.pre_cal_name, 'r')
                # print(pre_cal_df)
                print(index, end='\r')
                if len(pre_cal_df.columns):
                    pre_cal_df.to_hdf(self.pre_cal_path + self.pre_cal_name, key=rk,
                                    format='table', append=True)
                # if realization['cascade_id'] == 'c0':
                #     break
            except:
                pass
        # pre_cal.close()

        # ===============================================================================

    def learning_EM_update(self, TPPdata):
        """[summary]

        Args:
            TPPdata ([type]): [description]

        Returns:
            [type]: [description]
        """
        # query from class
        epsilon = self.stop_criteria['epsilon']
        max_iter = self.stop_criteria['max_iter']

        epoch_iter = 0
        obj_diff = epsilon + 1

        # save objective value in each iteration (optional)
        epoch_obj_value = []

        # keep updating until having met stopping criteria
        while obj_diff > epsilon and (epoch_iter < max_iter):

            mu_next_numerator = 0
            mu_next_denominator = 0
            alpha_next_numerator = pd.Series(
                data=np.zeros(self.dim), index=self.sourceNames)
            alpha_next_denominator = pd.Series(
                data=np.zeros(self.dim), index=self.sourceNames)

            obj_value = 0
            total_skipped = 0
            # loop through cascades
            for index, realization in enumerate(TPPdata):
                loglikelihood_activated_term = 0
                loglikelihood_survival_term = 0
                # load pre calculated file
                rk = 'cascade_id_{}'.format(realization['cascade_id'])
                try:
                    pre_cal_df = pd.read_hdf(
                        self.pre_cal_path + self.pre_cal_name, key=rk, mode='r')
                    # locate the event df
                    event_df = pre_cal_df.iloc[:, :-1]
                
                except (KeyError, AttributeError) as e:
                    total_skipped += 1                    
                    continue
                
                # activated source
                activated_sources = [s for s in list(realization.keys())
                                     if s in self.sourceNames]

                own_events_num = event_df.shape[1]
                if own_events_num > 0:

                    active_df = pd.DataFrame(data=np.zeros((self.dim+1, own_events_num)),
                                             index=self.sourceNames + ['base'])

                    for own_event_ind in range(own_events_num):
                        # calculate the intensity at own event time
                        intensity_at_event = (self.mu +
                                              np.dot(self.alpha, event_df.loc[:, own_event_ind]))

                        active_df.loc['base',
                                      own_event_ind] = self.mu / intensity_at_event
                        for source in activated_sources:
                            active_df.loc[source, own_event_ind] = \
                                (self.alpha.loc[source] * event_df.loc[source, own_event_ind])\
                                / intensity_at_event

                        # calculate loglikehood for activation term at each event
                        loglikelihood_activated_term = loglikelihood_activated_term + \
                            np.log(intensity_at_event)
                else:

                    active_df = pd.DataFrame(data=np.zeros((self.dim+1, 0)),
                                             index=self.sourceNames + ['base'])

                T = realization['right_censored_time']
                # survival terms
                survival_df = pre_cal_df.loc[:, 'survival']

                # calculate the log-likelihood for survival part
                loglikelihood_survival_term = loglikelihood_survival_term + \
                    np.dot(self.alpha, survival_df) + (T * self.mu)

                # loglikelihood
                loglikelihood = loglikelihood_activated_term - loglikelihood_survival_term

                # obj
                if 'log-likelihood-weight' in realization.keys():
                    obj_value = obj_value + \
                        np.power(loglikelihood,
                                 realization['log-likelihood-weight'])
                else:
                    obj_value = obj_value + loglikelihood

                # updates parameters
                active_df_sum = active_df.sum(axis=1)

                mu_next_numerator = mu_next_numerator + \
                    active_df_sum.loc['base']
                mu_next_denominator = mu_next_denominator + T

                alpha_next_numerator = alpha_next_numerator + \
                    active_df_sum[:-1]
                alpha_next_denominator = alpha_next_denominator + survival_df

                
            # updates parameters
            mu_next = mu_next_numerator / mu_next_denominator
            alpha_next = alpha_next_numerator / alpha_next_denominator
            print(mu_next)
            print(alpha_next)
            # objective value in current iteration
            epoch_obj_value.append(obj_value)

            # add iteration count
            epoch_iter = epoch_iter + 1

            obj_diff = abs(epoch_obj_value[-1] - epoch_obj_value[-2]) \
                if (epoch_iter > 1) else obj_diff
            print("iterations for the hawkes process: ", epoch_iter)
            # updata model parameters
            self.mu = mu_next.copy()
            self.alpha = alpha_next.copy()
        print(self.mu)
        print(self.alpha)
        print("Total realizations skipped: ", total_skipped)
        return TrainingStatus.CONVERGED

    # Setup TPP model for training.
    # Typically used for pre-computing quantities that are necessary for training.
    # This method is expected to be called prior to a call to the train() method.
    # TPPdata: List of S lists, where each inner list contains R realizations.
    #          Each realization contains at least a relative right-censoring time as its last event time.

    def setupTraining(self, TPPdata):
        """[summary]

        Args:
            TPPdata ([type]): [description]
        """
        # extract data associated with itself.
        self.__totalTrainingEvents = 0
        self.__sumOfPsis = 0.0

        pre_cal_file = os.path.join(self.pre_cal_path, self.pre_cal_name)
        # if not, create the pre calculation file
        if not os.path.isfile(pre_cal_file):
            self._setupTrainingDone = True  # indicate that setupTraining() has been ran already
        self._logger.info('PoissonTPP.setupTraining() finished.')
        self.TPPdata = TPPdata.copy()
        return self

    # Train the TPP model.
    # Returns an element of the TrainingStatus enumeration.
    def train(self, TPPdata,overwrite_precal=False):
        """[summary]

        Args:
            overwrite_precal (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """

        TPPdata = self.preProcessingData(TPPdata)
        # print(self.TPPdata[0])
        # print(self)
        # print(self.sourceNames)
        # input()
        # check if pre calculation file exist
        pre_cal_file = os.path.join(self.pre_cal_path, self.pre_cal_name)
        # if not, create the pre calculation file
        if not os.path.isfile(pre_cal_file) or overwrite_precal:
            print("creating  file")
            self.training_pre_calculation(TPPdata)
            self._setupTrainingDone = True
        else:
            print('The pre_cal file for this particular instance is already available, will be using', pre_cal_file,
                  '.. If you wish to create a new precal file, change the name of the process or pass the overwrite flag to the given function')

        # start training
        return self.learning_EM_update(TPPdata)

    # Is used by an Ogata's thinning algorithm to simulate the process.

    def intensityUB(self, t, rightCensoringTime, TPPdata):
        """[summary]

        Args:
            t ([type]): [description]
            rightCensoringTime ([type]): [description]
            TPPdata ([type]): [description]

        Returns:
            [type]: [description]
        """
        # TPPdata = self.preProcessingData(TPPdata)
        # calculate history excitation
        hist_excitation = 0   # History events excitation
        for s in range(len(TPPdata)):
            s_hist = [h for h in TPPdata[s] if h < t]
            for e in s_hist:
                hist_excitation = hist_excitation + \
                    self.alpha.loc[self.sourceNames[s]] * self.mk.phiUB(e, t)
        # intensity value = base intensity + hist_excitaion
        value = self.mu + hist_excitation
        return value
    # TODO: replace the following function with ogata's thinning for a single process
    # Simulate the TPP via the inverse time transformation approach
    # Returns a realization, which includes the relative right-censoring time (rightCensoringTime; see below)
    # as its last event time.
    # rightCensoringTime: strictly positive float; represents the relative censoring time to be used.
    # TPPdata: List of S (possibly, empty) realizations.
    # resume:  boolean; if True, assumes that the TPP's own realization includes a relative
    #          right-censoring time and removes it in order to resume the simulation from the last recorded
    #          event time.

    def simulate(self, rightCensoringTime, TPPdata, resume):
        """[summary]

        Args:
            rightCensoringTime ([type]): [description]
            TPPdata ([type]): [description]
            resume ([type]): [description]

        Returns:
            [type]: [description]
        """
        # calling the simulation for a multivariate ogata's thinning algorithm for a single process
        realizations = simulation([self], rightCensoringTime=rightCensoringTime,
                                  MTPPdata=TPPdata, resume=resume)
        # realizations is a set of realizations, one for each sourceName
        return realizations

    # Returns the TTP's model parameters as a tuple.
    def getParams(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.alpha, self.mk

    # Sets a TPP's parameters.
    # params: a tuple of parameters
    def setParams(self, params):
        """[summary]

        Args:
            params ([type]): [description]
        """
        self.alpha = params[0]
        self.mk = params[1]

    # Initializes a TPP's parameters
    def initParams(self):
        """[summary]
        """
        self.mu = 0.5  # assing a dummy positive value
        self.alpha = pd.Series(
            data=0.5*np.ones(self.dim), index=self.sourceNames)
        
        # self.mu = 0.000520209624023545
        # self.alpha = pd.Series(data = np.array([0.602281,0.061859,0.003505]), index=self.sourceNames)

    def getSourceNames(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.sourceNames

################################################################################
#
# U N I T   T E S T I N G
#
################################################################################


# Some global settings for figure sizes
normalFigSize = (14, 6)  # (width,height) in inches
largeFigSize = (12, 9)
xlargeFigSize = (18, 12)


def main():

    # Define the ground-truth TPP as a Hawkes TPP with base intensity(mu = 0.4)
    # and a HawkesPseudoMemoryKernel(beta = 1) memory kernel.
    mk = ExponentialPseudoMemoryKernel(beta=1.0)

    sourceNames = ['u0']  # this TPP only uses its own events as source.
    stop_criteria = {'max_iter': 50,
                     'epsilon': 1e-12}

    # class
    hawkesTPP = HawkesTPP(
        mk, sourceNames, stop_criteria,
        desc='Hawkes TPP with exponetial(beta=1) kernel')

    # hardcoding this for the unit test, we know the HawkessTPP is for a single process
    numSources = len(sourceNames)
    numRealizations = 50
    Realizations = []
    maxNumEvents = 0
    maxNumEventsIdx = 0
    for r in range(0, numRealizations):
        # Exponential(100)-distributed right-censoring time
        T = scipy.stats.expon.rvs(loc=0.0, scale=50.0)
        Realization = hawkesTPP.simulate(
            T, [np.array([])] * numSources, resume=False)
        print("Right Censoring Time for the current realization {0}".format(T))

        Realizations.append(Realization)
        numEvents = len(Realization[0])
        if numEvents > maxNumEvents:
            maxNumEvents = numEvents
            maxNumEventsIdx = r

    _, ax = plt.subplots(1, 1, figsize=normalFigSize)
    # save simulated values back into the DataStream object
    # TODO: pass in dataStream object instead of TPPdata into training functin

    training_features, training_realizations, training_isExploited, \
        test_features, test_realizations, test_isExploited, \
        validation_features, validation_realizations, validation_isExploited = generateExploitSocialMediaDataset()
    
    modified_training_realizations = []
    modified_validation_realizations = []
    
    for index, realization in enumerate(training_realizations):
        modified_training_realizations.append(np.array(realization[1:]))
    maxLenIndex = 0
    maxLen = 0
    for index, realization in enumerate(training_realizations):
        modified_validation_realizations.append(np.array(realization[1:]))
        if len(realization[1]) >maxLen:
            maxLen = len(realization[1])
            maxLenIndex = index
    
    
    hawkesTPP.setupTraining(Realizations)
    hawkesTPP.train(Realizations)

    print(len(modified_validation_realizations[maxLenIndex]))
    hawkesTPP.GOFks([modified_validation_realizations[maxLenIndex]],
                    ax=ax, showConfidenceBands=True)
    plt.show()
    


if __name__ == '__main__':
    main()
