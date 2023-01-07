'''
Simulation.py
This file contains the simulation algorithm using a thinning process, 
this is mainly used for multivariate processes, or hawkes process
    Author: Akshay Aravamudan, January 2020
'''


import numpy as np
from point_processes.SplitPopulationTPP import *
from scipy.special import logit

# Simulate the MTPP via ogata's thinning algorithm
# Returns a List of Realization which contains only one event, which includes the relative right-censoring time (rightCensoringTime; see below)
# as its last event time.
# As is the nature of the thinning algorithm, not all generated points get accepted (as opposed to the inversion method)
# So this simulation function will run until a point is accepted
#
# rightCensoringTime: strictly positive float; represents the relative censoring time to be used.
# MTPPdata: List of S (possibly, empty) realizations.
# resume:  boolean; if True, assumes that the TPP's own realization includes a relative
#          right-censoring time and removes it in order to resume the simulation from the last recorded
#          event time.


def simulation(listOfProcesses, rightCensoringTime, MTPPdata, resume=False, resume_after_split_pop=True, isRightCensoringTimeAttached = False):
    """[summary]

    Args:
        listOfProcesses ([type]): [description]
        rightCensoringTime ([type]): [description]
        MTPPdata ([type]): [description]
        resume (bool, optional): [description]. Defaults to False.
        resume_after_split_pop (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    # RealizationList is a list of numpy array of timestamps
    RealizationList = MTPPdata

    # RealizationList = [l for l in MTPPdata]

    # SimulatedRealization contains the realization of the relevant streams excluding the rightCensoringTime
    # if the resume flag is indicated to be true
    if not isRightCensoringTimeAttached:
        SimulatedRealization = RealizationList[:-1] if (
            resume == True and len(RealizationList) > 0) else RealizationList
    else:
        SimulatedRealization = RealizationList if (
            resume == True and len(RealizationList) > 0) else RealizationList
    
    
    # time of last event of the realization
    if max(SimulatedRealization) is not None:
        if len(max(SimulatedRealization)) > 0:
            t = np.max(SimulatedRealization)
        else:
            t = 0.0
    else:
        t = 0.0
    if isinstance(t, list):
        t = t[0]
    number_of_fetches = 0
    while t < rightCensoringTime:
        # print("stuck", t)
        #  upper_bound_intensity is \lambda^*
        upper_bound_intensity = 0.0
        # print(listOfProcesses)
        # print(SimulatedRealization)
        # input()
        for index, process in enumerate(listOfProcesses):
            # print("Simulated realization: ", SimulatedRealization)
            ModifiedRealization = SimulatedRealization.copy()
            # print(ModifiedRealization)

            if not index == 0:
                ModifiedRealization[index], ModifiedRealization[0] = ModifiedRealization[0], ModifiedRealization[index]
                # ModifiedRealization = ModifiedRealization[:-1]
            # print(ModifiedRealization)
            # print(index)
            # input()
            upper_bound_intensity += process.intensityUB(
                t, rightCensoringTime, ModifiedRealization)
        # print(upper_bound_intensity)

        u1, u2 = np.random.uniform(0.0, 1.0, 2)
        w = -np.log(u1)/upper_bound_intensity

        t_n = t + w
        # sum of all intensities
        l90uambda_cap = 0.0

        for index, process in enumerate(listOfProcesses):
            ModifiedRealization = SimulatedRealization.copy()
            if not index == 0:
                ModifiedRealization[index], ModifiedRealization[0] = ModifiedRealization[0], ModifiedRealization[index]
                # ModifiedRealization = ModifiedRealization[:-1]
            lambda_cap += process.intensity(t_n, ModifiedRealization)
        if u2 * upper_bound_intensity <= lambda_cap:
            # point is accepted
            t = t_n
            intensity_array = [0]*len(listOfProcesses)

            for i in range(len(listOfProcesses)):
                ModifiedRealization = SimulatedRealization.copy()
                processSourceNames = listOfProcesses[i].getSourceNames()
                # ensure that it is getting only the ones that it needs for calclaulating the intensity function
                if not i == 0:
                    ModifiedRealization[index], ModifiedRealization[0] = ModifiedRealization[0], ModifiedRealization[index]
                    # ModifiedRealization = ModifiedRealization[:-1]
                intensity_array[i] =\
                    listOfProcesses[i].intensity(t, ModifiedRealization)

            cumSumIntensityArray = np.cumsum(intensity_array)

            if t > rightCensoringTime:
                for streamRealization in SimulatedRealization:
                    streamRealization = np.append(streamRealization,rightCensoringTime)
            elif lambda_cap > 0:
                affected_node_id = np.argmax(
                    u2 <= cumSumIntensityArray/lambda_cap)
                SimulatedRealization[affected_node_id]  = np.append(SimulatedRealization[affected_node_id],t)
                if (not resume_after_split_pop) and affected_node_id == 0:

                    for streamRealization in SimulatedRealization:
                        streamRealization = np.append(streamRealization,rightCensoringTime)
                    break
            else:
                for streamRealization in SimulatedRealization:
                    streamRealization.append(rightCensoringTime)
                t = rightCensoringTime
        
    return SimulatedRealization.copy()


'''
The following function simulates multiple processes wherein one of the processes is a
split population process. This means that after the split population process has a realization, simulation of other 
relevant processes will also stop.
'''


def simulation_split_population(listOfProcesses, rightCensoringTime, MTPPdata, resume=False, resume_after_split_pop=True, isRightCensoringTimeAttached = False): 
    """[summary]

    Args:
        listOfProcesses ([type]): [description]
        rightCensoringTime ([type]): [description]
        MTPPdata ([type]): [description]
        resume (bool, optional): [description]. Defaults to False.
        resume_after_split_pop (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    # RealizationList is a list of numpy array of timestamps
    RealizationList = MTPPdata

    # RealizationList = [l for l in MTPPdata]

    # SimulatedRealization contains the realization of the relevant streams excluding the rightCensoringTime
    # if the resume flag is indicated to be true
    if not isRightCensoringTimeAttached:
        SimulatedRealization = RealizationList[:-1] if (
            resume == True and len(RealizationList) > 0) else RealizationList
    else:
        SimulatedRealization = RealizationList if (
            resume == True and len(RealizationList) > 0) else RealizationList
    
    
    # time of last event of the realization
    if max(SimulatedRealization) is not None:
        if len(max(SimulatedRealization)) > 0:
            t = np.max(SimulatedRealization)
        else:
            t = 0.0
    else:
        t = 0.0
    if isinstance(t, list):
        t = t[0]
    number_of_fetches = 0
    while t < rightCensoringTime:
        # print("stuck", t)
        #  upper_bound_intensity is \lambda^*
        upper_bound_intensity = 0.0
        # print(listOfProcesses)
        # print(SimulatedRealization)
        # input()
        for index, process in enumerate(listOfProcesses):
            # print("Simulated realization: ", SimulatedRealization)
            ModifiedRealization = SimulatedRealization.copy()
            # print(ModifiedRealization)

            if not index == 0:
                ModifiedRealization[index], ModifiedRealization[0] = ModifiedRealization[0], ModifiedRealization[index]
                # ModifiedRealization = ModifiedRealization[:-1]
            # print(ModifiedRealization)
            # print(index)
            # input()
            upper_bound_intensity += process.intensityUB(
                t, rightCensoringTime, ModifiedRealization)
        # print(upper_bound_intensity)

        u1, u2 = np.random.uniform(0.0, 1.0, 2)
        w = -np.log(u1)/upper_bound_intensity

        t_n = t + w
        # sum of all intensities
        l90uambda_cap = 0.0

        for index, process in enumerate(listOfProcesses):
            ModifiedRealization = SimulatedRealization.copy()
            if not index == 0:
                ModifiedRealization[index], ModifiedRealization[0] = ModifiedRealization[0], ModifiedRealization[index]
                # ModifiedRealization = ModifiedRealization[:-1]
            lambda_cap += process.intensity(t_n, ModifiedRealization)
        if u2 * upper_bound_intensity <= lambda_cap:
            # point is accepted
            t = t_n
            intensity_array = [0]*len(listOfProcesses)

            for i in range(len(listOfProcesses)):
                ModifiedRealization = SimulatedRealization.copy()
                processSourceNames = listOfProcesses[i].getSourceNames()
                # ensure that it is getting only the ones that it needs for calclaulating the intensity function
                if not i == 0:
                    ModifiedRealization[index], ModifiedRealization[0] = ModifiedRealization[0], ModifiedRealization[index]
                    # ModifiedRealization = ModifiedRealization[:-1]
                intensity_array[i] =\
                    listOfProcesses[i].intensity(t, ModifiedRealization)

            cumSumIntensityArray = np.cumsum(intensity_array)

            if t > rightCensoringTime:
                for streamRealization in SimulatedRealization:
                    streamRealization = np.append(streamRealization,rightCensoringTime)
            elif lambda_cap > 0:
                affected_node_id = np.argmax(
                    u2 <= cumSumIntensityArray/lambda_cap)
                SimulatedRealization[affected_node_id]  = np.append(SimulatedRealization[affected_node_id],t)
                if (not resume_after_split_pop) and affected_node_id == 0:

                    for streamRealization in SimulatedRealization:
                        streamRealization = np.append(streamRealization,rightCensoringTime)
                    break
            else:
                for streamRealization in SimulatedRealization:
                    streamRealization.append(rightCensoringTime)
                t = rightCensoringTime
        
    return SimulatedRealization.copy()



    

def acceleratedSimulation(listOfProcesses, rightCensoringTime, MTPPdata, resume=False, resume_after_split_pop=True, isRightCensoringTimeAttached = False):
    """[summary]

    Args:
        listOfProcesses ([type]): [description]
        rightCensoringTime ([type]): [description]
        MTPPdata ([type]): [description]
        resume (bool, optional): [description]. Defaults to False.
        resume_after_split_pop (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    # RealizationList is a list of numpy array of timestamps
    RealizationList = MTPPdata

    # RealizationList = [l for l in MTPPdata]

    # SimulatedRealization contains the realization of the relevant streams excluding the rightCensoringTime
    # if the resume flag is indicated to be true
    if not isRightCensoringTimeAttached:
        SimulatedRealization = RealizationList[:-1] if (
            resume == True and len(RealizationList) > 0) else RealizationList
    else:
        SimulatedRealization = RealizationList if (
            resume == True and len(RealizationList) > 0) else RealizationList
    
    
    # time of last event of the realization
    if max(SimulatedRealization) is not None:
        if len(max(SimulatedRealization)) > 0:
            t = np.max(SimulatedRealization)
        else:
            t = 0.0
    else:
        t = 0.0
    if isinstance(t, list):
        t = t[0]
    number_of_fetches = 0
    while t < rightCensoringTime:
        # print("stuck", t)
        #  upper_bound_intensity is \lambda^*
        upper_boupythnd_intensity = 0.0
        # print(listOfProcesses)
        # print(SimulatedRealization)
        # input()
        for index, process in enumerate(listOfProcesses):
            # print("Simulated realization: ", SimulatedRealization)
            ModifiedRealization = SimulatedRealization.copy()
            # print(ModifiedRealization)

            if not index == 0:
                ModifiedRealization[index], ModifiedRealization[0] = ModifiedRealization[0], ModifiedRealization[index]
                # ModifiedRealization = ModifiedRealization[:-1]
            # print(ModifiedRealization)
            # print(index)
            # input()
            upper_bound_intensity += process.intensityUB(
                t, rightCensoringTime, ModifiedRealization)
        # print(upper_bound_intensity)

        u1, u2 = np.random.uniform(0.0, 1.0, 2)
        w = -np.log(u1)/upper_bound_intensity

        t_n = t + w
        # sum of all intensities
        lambda_cap = 0.0

        for index, process in enumerate(listOfProcesses):
            ModifiedRealization = SimulatedRealization.copy()
            if not index == 0:
                ModifiedRealization[index], ModifiedRealization[0] = ModifiedRealization[0], ModifiedRealization[index]
                # ModifiedRealization = ModifiedRealization[:-1]
            lambda_cap += process.intensity(t_n, ModifiedRealization)
        if u2 * upper_bound_intensity <= lambda_cap:
            # point is accepted
            t = t_n
            intensity_array = [0]*len(listOfProcesses)

            for i in range(len(listOfProcesses)):
                ModifiedRealization = SimulatedRealization.copy()
                processSourceNames = listOfProcesses[i].getSourceNames()
                # ensure that it is getting only the ones that it needs for calclaulating the intensity function
                if not i == 0:
                    ModifiedRealization[index], ModifiedRealization[0] = ModifiedRealization[0], ModifiedRealization[index]
                    # ModifiedRealization = ModifiedRealization[:-1]
                intensity_array[i] =\
                    listOfProcesses[i].intensity(t, ModifiedRealization)

            cumSumIntensityArray = np.cumsum(intensity_array)

            if t > rightCensoringTime:
                for streamRealization in SimulatedRealization:
                    streamRealization.append(rightCensoringTime)
            elif lambda_cap > 0:
                affected_node_id = np.argmax(
                    u2 <= cumSumIntensityArray/lambda_cap)
                SimulatedRealization[affected_node_id].append(t)
                if (not resume_after_split_pop) and affected_node_id == 0:

                    for streamRealization in SimulatedRealization:
                        streamRealization.append(rightCensoringTime)
                    break
            else:
                for streamRealization in SimulatedRealization:
                    streamRealization.append(rightCensoringTime)
                t = rightCensoringTime
        
    return SimulatedRealization.copy()




# simulation of a single realization in a split population setting
def simulation_split_population_mv(MSPP,realization,rightCensoringTime, susceptible_labels, start_time):
    ## right_censoring_time = realization['right_censoring_time']
    s = start_time
    delta = 0.00006  # s+
    check_val = False
    
    while s < rightCensoringTime:
        intensity_vector = MSPP.intensity(s+delta, realization, susceptible_labels)
        lambda_bar = sum(intensity_vector) # upper bound intensity
        if lambda_bar == 0:
            return realization
        
        # homogeneous poisson simulation with intensity lambda_bar
        s = s + (-np.log(random.uniform(0, 1))) / lambda_bar

        # accepting with probability  sum(curr_intensity.values()) / lambda_bar
        next_intensity = MSPP.intensity(s, realization, susceptible_labels)
        D = random.uniform(0, 1) # uniformly generated from [0,1)
        
        # acceptance criteria
        if D * lambda_bar <= sum(next_intensity):
            
            check_val = True
            # accept this point and assign to a dimension
            intensity_cumsum = np.cumsum(next_intensity)
            
            # assign_to = intensity_cumsum.get_loc(min(i for i in intensity_cumsum if i > D*lambda_bar))
            
            intensity_cumsum_masked = intensity_cumsum[intensity_cumsum >= D*lambda_bar]
            
            min_cumsum_acceptance = min(intensity_cumsum_masked)
            
            # the source is the first occurence of the previously calculate minimum cumulative sum 
            assign_source = np.argwhere(intensity_cumsum==min_cumsum_acceptance)[0][0]
            
            realization['timestamps'] = np.append(realization['timestamps'], s)
            realization['timestamp_ids'] = np.append(realization['timestamp_ids'], assign_source)

    # if the last event out of time range, exclude that event
    if check_val and (realization['timestamps'][-1] > rightCensoringTime):
        realization['timestamps'] = realization['timestamps'][:-1]
        realization['timestamp_ids'] = realization['timestamp_ids'][:-1]
       
    return realization