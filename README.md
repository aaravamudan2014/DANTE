# split-population-survival-exploits

This repository holds the code for the AAAI paper ""

All documentation and infromation regarding posting things will be available on docs/build/html . The docs were created by sphinx and alabaster. 

For any question and/or concerns, contact Akshay Aravamudan (aaravamudan2014@my.fit.edu)

In order to run the code using existing datasets, please follow the following steps

1. If you are working off of an environment, please install all the required packages from `requirement.txt`
2. Install the `DANTE` package via `pip install .`
3. Once installed, you can edit the configuration file in `DANTE/DANTE/src/point_process/MVSPP_config.py` to select the prepared dataset, model & training hyperparameters.
4. Depending on the mode selected in `MVSPP_config.py`, you can run `DANTE/DANTE/src/point_process/MultiVariateSurvivalSplitPopulation.py` to train and test the model. 


In order to run the code using newer datasets, please follow the following steps.
1. The data should be an `npy` file which contains a dictionary of the following format

```


```
