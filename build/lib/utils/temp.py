import numpy as np
scenario ='twitter_link'
training_dataset = np.load("../data/KDD_data/training_" +scenario+".npy", allow_pickle=True).item()
test_dataset = np.load("../data/KDD_data/test_" +scenario+".npy", allow_pickle=True).item()
validation_dataset = np.load("../data/KDD_data/validation_" +scenario+".npy", allow_pickle=True).item()
del test_dataset['rightCensoring']
for key, value in test_dataset.items():
    training_dataset[str(int(key) + 456)] = value

print(training_dataset.keys())
np.save('../data/KDD_data/twitter_link_realizations.npy', training_dataset)