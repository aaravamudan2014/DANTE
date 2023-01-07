import MVSPP_config
import numpy as np
import pandas as pd
def calculate_bce_loss(node_id,x_i_bias,w_s, alpha_i, t_c=10000, delta_t=20000, threshold=0.5):
    bce = 0.0
    gt = []
    probs = []
    scenario = MVSPP_config.scenario
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
    
    if scenario == 'github' or scenario == 'twitter':
        rc_df = pd.DataFrame({'year': [2018], 'month': [3], 'day': [31]})
        rightCensoringTime = pd.to_datetime(rc_df, unit='ns')
    elif scenario == 'lastfm' or scenario == 'irvine':
        rc = training_realizations['rightCensoring']
        rightCensoringTime = pd.to_datetime(rc, unit='ns')
        del training_realizations['rightCensoring']
    elif scenario  == 'twitter_link':
        rc = training_realizations['rightCensoring']
        rightCensoringTime = rc
        del training_realizations['rightCensoring']

    
    for realization_id in training_realizations.keys():
        node_ids = training_realizations[realization_id]['timestamp_ids']
        timestamps = training_realizations[realization_id]['timestamps']
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
        
        if node_id not in node_ids:
            gt.append(0)
        else:
            node_timestamp = timestamps[list(node_ids).index(node_id)]

            if node_timestamp > t_c and node_timestamp < t_c + delta_t:
                gt.append(1)
            else:
                gt.append(0)
        
        req_indices = np.where(np.array(timestamps) < t_c)[0]
        prediction_timestamps = np.array(timestamps)[req_indices]
        prediction_node_ids = np.array(node_ids)[req_indices]

        pi_x_w = 1.0/(1.0 + np.exp(-np.dot(x_i_bias, w_s)))
        psi_tc = 0.0
        psi_tc_delta_t = 0.0
        
        for event, event_node_id in zip(prediction_timestamps, prediction_node_ids):
            psi_tc += alpha_i[event_node_id] * MVSPP_config.memory_kernel.psi(t_c - event)
            psi_tc_delta_t += alpha_i[event_node_id] * MVSPP_config.memory_kernel.psi(t_c + delta_t - event)
            
        prob = pi_x_w * (np.exp(-psi_tc) - np.exp(-psi_tc_delta_t))/\
            (1.0 - pi_x_w + pi_x_w*np.exp(-psi_tc))
        probs.append(prob)
        


    preds = np.zeros(len(probs))
    for index,prob in enumerate(probs):
        if prob > 0.5:
            preds[index] = 1
    
    
    bce = sum([-(x*np.log(y + 1.e-20)  + (1-x)*np.log(1- y + 1.e-20)) for x,y in zip(gt, probs)])

    print("bce: ", bce)

    return bce
