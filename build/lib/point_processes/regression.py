import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, PoissonRegressor

def getFeatures_start_size(realizations,features,  scenario, start_size_list):
    realizations = realizations.copy()
    if scenario == 'github' or scenario == 'twitter':
        rc_df = pd.DataFrame({'year': [2018], 'month': [3], 'day': [31]})
        rightCensoringTime = pd.to_datetime(rc_df, unit='ns')
    elif scenario == 'lastfm' or scenario == 'irvine':
        rc = realizations['rightCensoring']
        rightCensoringTime = pd.to_datetime(rc, unit='ns')
        del realizations['rightCensoring']
    elif scenario  == 'twitter_link':
        rc = realizations['rightCensoring']
        rightCensoringTime = rc
        del realizations['rightCensoring']
        
    X = np.zeros((len(start_size_list), len(realizations),features.shape[1]+2))
    Y = np.zeros((len(start_size_list), len(realizations)))
    exclude_indices = [[] for x in range(len(start_size_list))]
    total_predictions = np.zeros(len(start_size_list))
    
    timestamp_list = []
    node_ids_list = []
    
    for index, realization_id in enumerate(realizations):
        information_cascade = realizations[realization_id]
        timestamps = list(information_cascade['timestamps'])
        node_ids = list(information_cascade['timestamp_ids'])
        if scenario == 'github' or scenario == 'twitter':
            relative_rc_time = ((rightCensoringTime - timestamps[0])/np.timedelta64(1, 'h')).values[0]
        elif scenario == 'lastfm' or scenario == 'irvine':
            relative_rc_time = ((rightCensoringTime - timestamps[0])/np.timedelta64(1, 'h'))
        elif scenario == 'twitter_link':
            relative_rc_time = ((rightCensoringTime - timestamps[0])/3600)

        if scenario != 'twitter_link':
            timestamps = [((x-timestamps[0])/np.timedelta64(1,'h')) for x in timestamps]
        else: 
            timestamps = [((x-timestamps[0])/3600) for x in timestamps]
        
        timestamps.append(relative_rc_time)
        node_ids.append(-1)
        timestamps = np.array(timestamps)
        node_ids = np.array(node_ids)
        
        timestamp_list.append(np.array(timestamps))
        node_ids_list.append(np.array(node_ids))
        for start_size_index, start_size in enumerate(start_size_list):
            if len(np.unique(timestamp_list[index])) - 1 <= start_size :
                exclude_indices[start_size_index].append(index)
                continue 
            total_predictions[start_size_index] += 1
            t_c = sorted(np.unique((timestamp_list[index])))[start_size]
            delta_t = timestamp_list[index][-1]

            timestamps = timestamp_list[index]
            node_ids = np.array(node_ids_list[index])
            (req_indices ,)= np.where(timestamps < t_c)
            prediction_timestamps = timestamps[req_indices]
            prediction_node_ids = node_ids[ req_indices]
            # print(prediction_timestamps)
            # print(timestamp_list[i])
            # input()
            
            remaining_user_ids = np.arange(len(features))
            remaining_user_ids = np.array([x for x in remaining_user_ids if x not in prediction_node_ids])
            
            # mean time between messages
            mean_diff = np.average(np.diff(prediction_timestamps))

            # cumulative popularity
            cum_popularity = len(prediction_timestamps)

            # incremental popularity every x hours
            
            # number of followers of early adopters
            early_adopter_count = 5
            # average of each of the features
            features_avg = np.zeros(features.shape[1])
            features_avg = np.average(features[prediction_node_ids[:early_adopter_count]], axis=0)
            X[start_size_index][index] = np.append(features_avg, [mean_diff,cum_popularity])
            
            Y[start_size_index][index] = len(timestamps) - 1
            
            
        if index % 100 == 0:
            print("Processed ", index, " Out of ", len(realizations))

    return X, Y, exclude_indices, total_predictions


def getFeatures_tc(realizations,features,  scenario, t_c_list):
    realizations = realizations.copy()
    if scenario == 'github' or scenario == 'twitter':
        rc_df = pd.DataFrame({'year': [2018], 'month': [3], 'day': [31]})
        rightCensoringTime = pd.to_datetime(rc_df, unit='ns')
    elif scenario == 'lastfm' or scenario == 'irvine':
        rc = realizations['rightCensoring']
        rightCensoringTime = pd.to_datetime(rc, unit='ns')
        del realizations['rightCensoring']
    elif scenario  == 'twitter_link':
        rc = realizations['rightCensoring']
        rightCensoringTime = rc
        del realizations['rightCensoring']
        
    X = np.zeros((len(t_c_list), len(realizations),features.shape[1]+2))
    Y = np.zeros((len(t_c_list), len(realizations)))
    exclude_indices = [[] for x in range(len(t_c_list))]
    total_predictions = np.zeros(len(t_c_list))
    
    timestamp_list = []
    node_ids_list = []
    
    for index, realization_id in enumerate(realizations):
        information_cascade = realizations[realization_id]
        timestamps = list(information_cascade['timestamps'])
        node_ids = list(information_cascade['timestamp_ids'])
        if scenario == 'github' or scenario == 'twitter':
            relative_rc_time = ((rightCensoringTime - timestamps[0])/np.timedelta64(1, 'h')).values[0]
        elif scenario == 'lastfm' or scenario == 'irvine':
            relative_rc_time = ((rightCensoringTime - timestamps[0])/np.timedelta64(1, 'h'))
        elif scenario == 'twitter_link':
            relative_rc_time = ((rightCensoringTime - timestamps[0])/3600)

        if scenario != 'twitter_link':
            timestamps = [((x-timestamps[0])/np.timedelta64(1,'h')) for x in timestamps]
        else: 
            timestamps = [((x-timestamps[0])/3600) for x in timestamps]
        
        timestamps.append(relative_rc_time)
        node_ids.append(-1)
        timestamps = np.array(timestamps)
        node_ids = np.array(node_ids)
        
        timestamp_list.append(np.array(timestamps))
        node_ids_list.append(np.array(node_ids))
        for t_c_index, t_c in enumerate(t_c_list):
            (req_indices ,)= np.where(np.array(timestamps)[:-1] < t_c)
            prediction_timestamps = timestamps[req_indices]
            prediction_node_ids = node_ids[ req_indices]
            if len(prediction_timestamps) == 1:
                exclude_indices[t_c_index].append(index) 
                continue
            total_predictions[t_c_index] += 1
            remaining_user_ids = np.arange(len(features))
            remaining_user_ids = np.array([x for x in remaining_user_ids if x not in prediction_node_ids])
        
            # mean time between messages
            mean_diff = np.average(np.diff(prediction_timestamps))

            # cumulative popularity
            cum_popularity = len(prediction_timestamps)

            # incremental popularity every x hours
            
            # number of followers of early adopters
            early_adopter_count = 5
            # average of each of the features
            features_avg = np.zeros(features.shape[1])
            features_avg = np.average(features[prediction_node_ids[:early_adopter_count]], axis=0)
            X[t_c_index][index] = np.append(features_avg, [mean_diff,cum_popularity])
            # print(X_train[t_c_index][index])
            # input()
            Y[t_c_index][index] = len(timestamps[:-1])
            
        if index % 100 == 0:
            print("Processed ", index, " Out of ", len(realizations))
    return X, Y, exclude_indices, total_predictions


def train_on_features_tc(train_realizations,test_realizations, features, scenario, t_c_list, delta_t_list):
    X_train, Y_train, exclude_indices_train , total_predictions_train   = getFeatures_tc(train_realizations,features,  scenario, t_c_list)
    X_test, Y_test, exclude_indices_test , total_predictions_test   = getFeatures_tc(test_realizations,features,  scenario, t_c_list)
    msle = np.zeros(len(t_c_list))
    for t_c_index, t_c in enumerate(t_c_list):
        train_mask = np.ones(len(X_train[t_c_index]), dtype=bool)
        train_mask[exclude_indices_train[t_c_index]] = False
        X_input_tc = X_train[t_c_index][train_mask]
        Y_input_tc = Y_train[t_c_index][train_mask]
        
        test_mask = np.ones(len(X_test[t_c_index]), dtype=bool)
        test_mask[exclude_indices_test[t_c_index]] = False
        X_test_tc = X_test[t_c_index][test_mask]
        Y_test_tc = Y_test[t_c_index][test_mask]
        

        reg = LinearRegression().fit(X_input_tc, np.log(Y_input_tc))
        pred = reg.predict(X_test_tc)
        pred = np.clip(pred, 0, None)
        msle[t_c_index] = np.sum(np.abs(pred - np.log(Y_test_tc))**2)/total_predictions_test[t_c_index]
    print("MSLE for tc: ", t_c_list," is ", msle)
    
def train_on_features_start_size(train_realizations,test_realizations, features, scenario, start_size_list, delta_t_list):
    X_train, Y_train, exclude_indices_train , total_predictions_train   = getFeatures_start_size(train_realizations,features,  scenario, start_size_list)
    X_test, Y_test, exclude_indices_test , total_predictions_test   = getFeatures_start_size(test_realizations,features,  scenario, start_size_list)
    msle = np.zeros(len(start_size_list))
    for start_size_index, start_size in enumerate(start_size_list):
        train_mask = np.ones(len(X_train[start_size_index]), dtype=bool)
        train_mask[exclude_indices_train[start_size_index]] = False
        X_input_tc = X_train[start_size_index][train_mask]
        Y_input_tc = Y_train[start_size_index][train_mask]
        
        test_mask = np.ones(len(X_test[start_size_index]), dtype=bool)
        test_mask[exclude_indices_test[start_size_index]] = False
        X_test_tc = X_test[start_size_index][test_mask]
        Y_test_tc = Y_test[start_size_index][test_mask]
        

        reg = LinearRegression().fit(X_input_tc, np.log(Y_input_tc))
        pred = reg.predict(X_test_tc)
        msle[start_size_index] = np.sum((pred - np.log(Y_test_tc))**2)/total_predictions_test[start_size_index]
    print("MSLE for start sizes: ", start_size_list," is ", msle)


def main():
    scenario = "irvine"
    training_filename = '../data/KDD_data/training_'+str(scenario)+'.npy'
    test_filename = '../data/KDD_data/test_'+str(scenario)+'.npy'
    
    training_features_filename = '../data/KDD_data/'+str(scenario)+'_user_features.npy'
    
    training_realizations = np.load(training_filename, allow_pickle=True).item()
    test_realizations = np.load(test_filename, allow_pickle=True).item()
    
    if scenario == "github" or scenario == "twitter" or scenario == "twitter_link":
        training_features = np.load(training_features_filename, allow_pickle=True)
    else:
        num_users = 1000 if scenario == "lastfm" else 893
        training_features = np.zeros((num_users,1))
    train_on_features_start_size(training_realizations,test_realizations,training_features, scenario, start_size_list = [5,7,10, 20], delta_t_list =None)    
    input()
    train_on_features_tc(training_realizations,test_realizations,training_features, scenario, t_c_list = [12,24,48, 96, 120, 500], delta_t_list =None)

if __name__ == "__main__":
    main()