import random
import time
import timeit
import optuna

import dask
from dask.distributed import Client, LocalCluster
# from dask_jobqueue import PBSCluster
from distributed import LocalCluster
from optuna.visualization import plot_optimization_history
import matplotlib.pyplot as plt

def outer_function(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2

def serialoptimization(num, function):
    """Runs Optuna optimization in series

    :param num:
        Int number of trials to run for optimization
    :param function:
        function to be optimized
    :return:
        An output is returned for each trial showing the trial number,
        optimization value [(x-2)**2] for that trial, value of parameter x,
        and the best trial number so far along with its optimization value.
        The time taken for the process to complete is also printed.
    """
    start = timeit.default_timer()
    for i in range(num):
        study.optimize(function, n_trials=1)

    stop = timeit.default_timer()
    print('Time before dask: ', stop - start)
    
def paralleloptimization(num, function):
    """Runs Optuna optimization in parallel

    :param num:
        Int number of trials to run for optimization
    :param function:
        function to be optimized
    :return:
        An output is returned for each trial showing the trial number,
        optimization value [(x-2)**2] for that trial, value of parameter x,
        and the best trial number so far along with its optimization value.
        The time taken for the process to complete is also printed.
    """
    start = timeit.default_timer()
    lazy_results = []
    for i in range(num):
        lazy_result = dask.delayed(study.optimize)(function, n_trials=1)
        lazy_results.append(lazy_result)

    dask.compute(*lazy_results)
    stop = timeit.default_timer()
    print('Time after dask: ', stop - start)
    return study

if __name__ == '__main__':
    """
    Creates an Optuna study named "distributed-example", runs the study with 
    the aim of minimizing the applied function, and stores information from each
    optimization in a database
    
    A local cluster is initiated on Dask with default settings for the given machine
    based on its number of cores. A client is created based on this cluster to allow
    multiple workers at a time when dask is called (parallelization).
    """
    study = optuna.create_study(
         direction="minimize", storage='sqlite:///optuna_study.db'
    )
    cluster: LocalCluster = LocalCluster()
    print("Cluster works")
    client = Client(cluster)  # Connect this local process to remote

    print(cluster.scheduler)


    num_simulations = 50
    
    
    # Serialized Function
    # serialoptimization(num_simulations, outer_function)

    # parallelized for loop
    study = paralleloptimization(num_simulations, outer_function)
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()