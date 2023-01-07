import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
train_realizations = np.load("../data/KDD_data/training_lastfm.npy",allow_pickle=True).item()

hist = []
del train_realizations["rightCensoring"]
for realization_index in train_realizations.keys():
    timestamps = train_realizations[realization_index]['timestamps']
    # diff = (timestamps[1] -timestamps[0])/np.timedelta64(1, 'h')
    diff = (timestamps[1] - timestamps[0]).total_seconds()
    hist.append(diff)

plt.hist(hist, bins = 1000)
plt.show()

