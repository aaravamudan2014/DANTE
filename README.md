# split-population-survival-exploits

This repository holds the code for the AAAI 2023 paper "ANytime USer Engagement Prediction in Information Cascades for Arbitrary Observation Periods". All documentation and infromation regarding posting things will be available on docs/build/html . The docs were created by sphinx and alabaster. Finally this repository is still in the final stages of editing, so there might be certain aspects that may not coopperate. We appreciate yor patience. This README will be updated when the code has been rigourously tested. 

For any question and/or concerns, contact [https://aaravamudan2014.github.io/Akshay-Aravamudan/](Akshay Aravamudan) (aaravamudan2014@my.fit.edu

In order to run the code using existing datasets, please follow the following steps

1. If you are working off of an environment, please install all the required packages from `requirement.txt`
2. Install the `DANTE` package via `pip install .`
3. Once installed, you can edit the configuration file in `DANTE/DANTE/src/point_process/MVSPP_config.py` to select the prepared dataset, model & training hyperparameters.
4. Depending on the mode selected in `MVSPP_config.py`, you can run `DANTE/DANTE/src/point_process/MultiVariateSurvivalSplitPopulation.py` to train and test the model. 


In order to run the code using newer datasets, please follow the following steps.
1. The data should be an `npy` file which contains a dictionary of the following format

```
{'Cascade ID #1': {'timestamps':[0.0,..., *right censoring time*], 'timestamp_ids':[1,5,3,6,7]},
 'Cascade ID #2': ...
 .
 .
 .
 }
```

It is much more preferrable to have the relative right cnoesring time as opposed to an absolute time. It is also suggested that the first item in the timestamps list be a zero to indicate the kickoff event. Finally, the user ids in the `timestamp_ids` list must start from zero. This is to ensure connections the to the feature vectors. 


To construct the feature vectors, you can either leave them to be zero or instead if there is an adjacency graph representation of the relationships, you can use that to extract features.

File naming conventions:
- for realizations, '*dataset_name*_realizations.npy'
- for features, '*dataset_name*_user_features.npy'
