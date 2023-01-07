
import sys
import pandas as pd
import json
import csv
import random
import os
import pickle
from datetime import datetime, timedelta
from core.DataStream import DataStream
from point_processes.utilities import json2ListOfList
import math
from utils.DummyCVECreator import createSocialMediaData
import numpy as np
import csv
from datetime import datetime
import collections
from sklearn.model_selection import train_test_split
import time
import datetime as dt
from sknetwork.embedding import SVD
import matplotlib.pyplot as plt




def GenerateDatasetFromTopoLSTMPaper(dataset_name,num_nodes):
    csv.field_size_limit(sys.maxsize)
    filename_qualifier = "../data/" + dataset_name + "/"

    # load graph data
    graph_filename = filename_qualifier + "graph.txt"
    
    # get a subset of the top outgoing nodes
    node_list = []
    
    # create a graph of outgoing nodes, this is in order to subset the node list 
    graph_outgoing_dict = {}
    if os.path.isfile(filename_qualifier + 'graph_outgoing_dict_'+str(num_nodes)+'.pkl') :
        with open(filename_qualifier + 'graph_outgoing_dict_'+str(num_nodes)+'.pkl', 'rb') as f:
            graph_outgoing_dict = pickle.load(f)
    else:
        with open(graph_filename) as f:
            lines = f.readlines()
            row_id = 0
            for row in lines:
                if row_id == 0:
                    row_id += 1
                    continue
                print("processed ", row_id, " edge out of ")
                nodes = row.strip().split(' ')
                if int(nodes[0]) in graph_outgoing_dict.keys():
                    graph_outgoing_dict[int(nodes[0])] += 1 
                else:
                    graph_outgoing_dict[int(nodes[0])] = 1
                row_id += 1

        with open(filename_qualifier + 'graph_outgoing_dict_'+str(num_nodes)+'.pkl', 'wb') as f:
            pickle.dump(graph_outgoing_dict, f)

    
    for w in sorted(graph_outgoing_dict, key=graph_outgoing_dict.get, reverse=True):
        node_list.append(w)
        if len(node_list) == num_nodes:
            break
    
    def convert_to_datetime(epoch_time):
        new_time = datetime.datetime.fromtimestamp(int(epoch_time)).strftime('%Y-%m-%d %H:%M:%S')
        return new_time
    
    # obtain a mapping of indexes to the updated graph
    dataset_dict_map = {}
    for i in range(len(node_list)):
        dataset_dict_map[int(node_list[i])] = i


    # get a new adjacency matrix for the nodes in the subset
    updated_adj_matrix = np.zeros((num_nodes, num_nodes))

    if os.path.isfile(filename_qualifier + 'graph_adj_matrix_'+ str(num_nodes) +'.pkl') :
        with open(filename_qualifier + 'graph_adj_matrix_'+ str(num_nodes) +'.pkl', 'rb') as f:
            updated_adj_matrix = pickle.load(f)
    else:
        with open(graph_filename) as f:
            lines = f.readlines()
            row_id = 0
            for row in lines:
                if row_id == 0:
                    row_id += 1
                    continue
                print("processed ", row_id, " edge out of ")
                nodes = row.strip().split(' ')
                if all(node in node_list for node in [int(nodes[0]),int(nodes[1])]):
                    updated_adj_matrix[ dataset_dict_map[int(nodes[0])], dataset_dict_map[int(nodes[1])] ] += 1 
                row_id += 1

        with open(filename_qualifier + 'graph_adj_matrix_'+ str(num_nodes) +'.pkl', 'wb') as f:
            pickle.dump(updated_adj_matrix, f)
    
    # convert adjacency matrix to user features via dimensionality reduction
    
    svd = SVD(n_components=4)
    embedding = svd.fit_transform(updated_adj_matrix)
    if not os.path.exists('../data/dataset_name/'):
        os.makedirs('../data/dataset_name/')
    np.save('../data/KDD_data/'+dataset_name+'_user_features.npy', embedding)

    import datetime
    # generate training and test realizations
    dataset_filename = filename_qualifier + "test.txt"
    test_realizations = {}
    rightCensoringTime = 0.0
    with open(dataset_filename) as f:
        csv_reader = csv.reader(f, delimiter='\n')
        for row_id, row in enumerate(csv_reader):
            realization = {}
            print("processed ", row_id, " realizations out of " )
            events = row[0].split(' ')
            root_node = events[0]
            
            new_events = [(x,y) for x,y in zip(events[1:][::2],events[1:][1::2]) if int(x) in node_list]
            if len(new_events) <= 1:
                continue

            nodes = [dataset_dict_map[int(x)] for x,y in new_events]
            timestamps = [y for x,y in new_events]
            timestamps = [int(float(x))-int(float(timestamps[0])) for x in timestamps]
            timestamps = [datetime.timedelta(seconds=x).total_seconds()/3600 for x in timestamps]
            #print(timestamps,new_events)
            #input()
            realization['timestamps'] = timestamps
            if max(timestamps) > rightCensoringTime:
                rightCensoringTime = max(timestamps)
            
            realization['timestamp_ids'] = nodes
            test_realizations[row_id] = realization
    
    # generate training and test realizations
    dataset_filename = filename_qualifier + "train.txt"
    training_realizations = {}
    validation_realizations = {}
    
    with open(dataset_filename) as f:
        csv_reader = csv.reader(f, delimiter='\n')
        for row_id, row in enumerate(csv_reader):
            realization = {}
            print("processed ", row_id, " realizations out of " )
            events = row[0].split(' ')
            root_node = events[0]
            
            new_events = [(x,y) for x,y in zip(events[1:][::2],events[1:][1::2]) if int(x) in node_list]
            if len(new_events) <= 1:
                continue
            nodes = [dataset_dict_map[int(x)] for x,y in new_events]
            timestamps = [y for x,y in new_events]
            timestamps = [int(float(x))-int(float(timestamps[0])) for x in timestamps]
            timestamps = [ datetime.timedelta(seconds=x).total_seconds()/3600 for x in timestamps]
            
            if max(timestamps) > rightCensoringTime:
                rightCensoringTime = max(timestamps)
            
            realization['timestamps'] = timestamps
            realization['timestamp_ids'] = nodes

            set_assignment = np.random.binomial(size=1, n=1,p=0.8)
            if set_assignment == 1:
                training_realizations[len(training_realizations)] = realization
            else:
                validation_realizations[len(validation_realizations)] = realization
    
    test_realizations['rightCensoring'] = rightCensoringTime
    training_realizations['rightCensoring'] = rightCensoringTime
    validation_realizations['rightCensoring'] = rightCensoringTime
    
    with open("../data/KDD_data/training_"+dataset_name+".npy", mode="wb") as f:
        np.save(f,training_realizations)
    
    with open("../data/KDD_data/test_"+dataset_name+".npy", mode="wb") as f:
        np.save(f,test_realizations)
    
    with open("../data/KDD_data/validation_"+dataset_name+".npy", mode="wb") as f:
        np.save(f,validation_realizations)
    
    print("Files have been saved")

    

def generateDatasetsForForestPaper(start_size, scenario,num_nodes):
    def generateForestDataset(scenario):
        training_dataset = np.load("../data/KDD_data/training_" +scenario+".npy", allow_pickle=True).item()
        try:
            del training_dataset['rightCensoring']
        except KeyError:
            pass
        if not os.path.exists("../data/KDD_data/"+scenario+"/"+str(start_size)):
            os.makedirs("../data/KDD_data/"+scenario+"/"+str(start_size))

        with open("../data/KDD_data/"+scenario+"/"+str(start_size)+"/cascade.txt", "w") as f:     
            for realization_index in training_dataset.keys():
                # print(training_dataset[realization_index])
                timestamps = training_dataset[realization_index]['timestamps']
                if len(np.unique(timestamps)) <= start_size:
                    continue
                node_ids = training_dataset[realization_index]['timestamp_ids']
                realization_str = " "
                for timestamp, node_id in zip(timestamps, node_ids):
                    if scenario == "irvine" or scenario == "lastfm":
                        epoch = (timestamp -timestamps[0])/np.timedelta64(1, 'h')
                    else:
                        epoch = (timestamp -timestamps[0])
                    realization_str +=  str(node_id) + "," + str(epoch)+" "
                realization_str += "\n"
                f.write(realization_str)



        
        test_dataset = np.load("../data/KDD_data/test_" +scenario+".npy", allow_pickle=True).item()
        try:
            del test_dataset['rightCensoring']
        except KeyError:
            pass
        num_test_samples = 0
        with open("../data/KDD_data/"+scenario+"/"+str(start_size)+"/cascadetest.txt", "w") as f:     
            for realization_index in test_dataset.keys():
                timestamps = test_dataset[realization_index]['timestamps']
                if len(np.unique(timestamps)) <= start_size:
                    continue
                num_test_samples += 1
                node_ids = test_dataset[realization_index]['timestamp_ids']
                realization_str = " "
                for timestamp, node_id in zip(timestamps, node_ids):
                    if scenario == "irvine" or scenario == "lastfm":
                        epoch = (timestamp -timestamps[0])/np.timedelta64(1, 'h')
                    else:
                        epoch = (timestamp -timestamps[0])
                    realization_str += str(node_id) + "," + str(epoch)+" "
                realization_str += "\n"
                f.write(realization_str)
        

        validation_dataset = np.load("../data/KDD_data/validation_" +scenario+".npy", allow_pickle=True).item()
        try:
            del validation_dataset['rightCensoring']
        except KeyError:
            pass
        #num_test_samples = 0
        with open("../data/KDD_data/"+scenario+"/"+str(start_size)+"/cascadevalid.txt", "w") as f:     
            for realization_index in validation_dataset.keys():
                timestamps = validation_dataset[realization_index]['timestamps']
                if len(np.unique(timestamps)) - 1 <= start_size:
                    continue
                #num_test_samples += 1
                node_ids = validation_dataset[realization_index]['timestamp_ids']
                realization_str = " "
                for timestamp, node_id in zip(timestamps, node_ids):
                    if scenario == "irvine" or scenario == "lastfm":
                        epoch = (timestamp -timestamps[0])/np.timedelta64(1, 'h')
                    else:
                        epoch = (timestamp -timestamps[0])
                    realization_str += str(node_id) + "," + str(epoch)+" "
                realization_str += "\n"
                f.write(realization_str)
        print("total number of test samples: ", num_test_samples)

        try:
            with open("../data/" + scenario + "/" + 'graph_adj_matrix_'+ str(num_nodes) +'.pkl', 'rb') as f:
                updated_adj_matrix = pickle.load(f)
        except:
            updated_adj_matrix = np.zeros((num_nodes, num_nodes))
        edge_list = []
        if np.sum(updated_adj_matrix) != 0:
            with open("../data/KDD_data/"+scenario+"/"+str(start_size)+"/edges.txt", "w") as f:
                for i in range(len(updated_adj_matrix)):
                    for j in range(len(updated_adj_matrix)):
                        if updated_adj_matrix[i][j]>0:
                            edge_str = str(i)+","+str(j)+"\n"
                            f.write(edge_str)

            
    generateForestDataset(scenario=scenario)


def PoissonBinomialPMF(pVector, confidence, events):
    PMF = np.array([0.0, 1.0])
    for p in pVector:
        seq = np.array([1.0-p, p])
        PMF = np.convolve(PMF, seq)
    plt.stem(PMF)
    print(events)
    # plt.xlim([0,200])

    # sort PMF
    sortedPMF = -np.sort(-PMF)
    sortedIndexPMF = np.argsort(-PMF)
    cumSumPMF = np.cumsum(sortedPMF)
    endIndex = np.argmax(confidence <= cumSumPMF)
    # find cumsum
    # add all indexes before 
    ret_list = []
    for i in range(endIndex):
        ret_list.append(sortedIndexPMF[i])
    ret_list = np.sort(ret_list)
    plt.axvline(x = ret_list[0], color='r', linestyle = "--",label = "95 % CI", zorder=2)
    plt.axvline(x = ret_list[-1], color='r' , linestyle = "--",zorder=2 )
    plt.axvline(x = events, color='k', linewidth = 7,  label = "Actual Future Infections", zorder = 1)
    
    plt.legend(loc = 'best')
    
    return ret_list
            
def generateDatasetsForRegression_start_size(scenario, num_nodes, start_size=5):
    
    training_dataset = np.load("../data/KDD_data/training_" +scenario+".npy", allow_pickle=True).item()
    try:
        del training_dataset['rightCensoring']
    except KeyError:
        pass
    test_dataset = np.load("../data/KDD_data/test_" +scenario+".npy", allow_pickle=True).item()
    try:
        del test_dataset['rightCensoring']
    except KeyError:
        pass
    validation_dataset = np.load("../data/KDD_data/validation_" +scenario+".npy", allow_pickle=True).item()
    try:
        del validation_dataset['rightCensoring']
    except KeyError:
        pass
    

    if not os.path.exists("../data/KDD_data/"+scenario):
        os.makedirs("../data/KDD_data/"+scenario)

    filename_qualifier = "../data/KDD_data/" + scenario + "/"
    try:
        with open(filename_qualifier + 'graph_adj_matrix_'+ str(num_nodes) +'.pkl', 'rb') as f:
                updated_adj_matrix = pickle.load(f)
    except:
        updated_adj_matrix = np.zeros((num_nodes, num_nodes))


    follower_counts = np.zeros(len(updated_adj_matrix))

    for node in range(len(updated_adj_matrix)):
        follower_counts[node] = np.sum(updated_adj_matrix[:,node])


    #{'cid':, 'post_time_day':, hw:[]}
    training_list_json = []
    for realization_index in training_dataset.keys():
        timestamps = training_dataset[realization_index]['timestamps']
        node_ids = training_dataset[realization_index]['timestamp_ids']
        if len(timestamps) <= start_size:
            continue
        sample_dict = {}
        sample_dict['cid'] = str(realization_index)
        
        realization_str = " "
        epoch_list = []

        
        for timestamp, node_id in zip(timestamps, node_ids):
            if scenario == "irvine" or scenario == "lastfm":
                epoch = (timestamp -timestamps[0])/np.timedelta64(1, 'h')
            else:
                epoch = (timestamp -timestamps[0])
            
            epoch_list.append(epoch)
        sample_dict['hw'] = [(x,y) for x,y in zip(epoch_list,follower_counts)]
        training_list_json.append(sample_dict)

    #json_str = json.dumps(training_dict_json)
    json.dump(training_list_json, open(filename_qualifier+'train.json', 'w'))
            
    test_list_json = []
    total_count = 0
    for realization_index in test_dataset.keys():
        timestamps = test_dataset[realization_index]['timestamps']
        node_ids = test_dataset[realization_index]['timestamp_ids']
        total_count += 1
        sample_dict = {}
        sample_dict['cid'] = str(realization_index)
        
        realization_str = " "
        epoch_list = []
        for timestamp, node_id in zip(timestamps, node_ids):
            if scenario == "irvine" or scenario == "lastfm":
                epoch = (timestamp -timestamps[0])/np.timedelta64(1, 'h')
            else:
                epoch = (timestamp -timestamps[0])
            
            epoch_list.append(epoch)
        sample_dict['hw'] = [(x,y) for x,y in zip(epoch_list,follower_counts)]
        test_list_json.append(sample_dict)

    #json_str = json.dumps(test_dict_json)
    json.dump(test_list_json, open(filename_qualifier+'test.json', 'w'))

    val_list_json = []
    val_count = 0
    for realization_index in validation_dataset.keys():
        timestamps = test_dataset[realization_index]['timestamps']
        node_ids = test_dataset[realization_index]['timestamp_ids']
        total_count += 1
        sample_dict = {}
        sample_dict['cid'] = str(realization_index)
        
        realization_str = " "
        epoch_list = []
        for timestamp, node_id in zip(timestamps, node_ids):
            if scenario == "irvine" or scenario == "lastfm":
                epoch = (timestamp -timestamps[0])/np.timedelta64(1, 'h')
            else:
                epoch = (timestamp -timestamps[0])
            
            epoch_list.append(epoch)
        sample_dict['hw'] = [(x,y) for x,y in zip(epoch_list,follower_counts)]
        val_list_json.append(sample_dict)

    #json_str = json.dumps(test_dict_json)
    json.dump(val_list_json, open(filename_qualifier+'val.json', 'w'))



def generateStartSizes(scenario):
    training_dataset = np.load("../data/KDD_data/training_" +scenario+".npy", allow_pickle=True).item()
    try:
        del training_dataset['rightCensoring']
    except KeyError:
        pass
    validation_dataset = np.load("../data/KDD_data/validation_" +scenario+".npy", allow_pickle=True).item()
    try:
        del validation_dataset['rightCensoring']
    except KeyError:
        pass
    test_dataset = np.load("../data/KDD_data/test_" +scenario+".npy", allow_pickle=True).item()
    try:
        del test_dataset['rightCensoring']
    except KeyError:
        pass
    
    dataset =test_dataset
    cascade_sizes = []
    cascade_times = []
    for realization_index in dataset.keys():
        
        timestamps = dataset[realization_index]['timestamps']


        node_ids = dataset[realization_index]['timestamp_ids']
        timestamp_new = []
        if scenario == "irvine" or scenario == "lastfm":
            for timestamp in timestamps:
                timestamp_new.append((timestamp -timestamps[0])/np.timedelta64(1, 'h'))
            
        else:
            for timestamp in timestamps:
                timestamp_new.append((timestamp -timestamps[0]))
        timestamps = timestamp_new
        cascade_times.extend(timestamps)
        
        cascade_sizes.append(len(timestamps))
        
    plt.hist(cascade_times, bins = 100)
    plt.show()

def generateNeuralDiffusionPaperDataset(num_nodes = 200):
    irvineDatasetFilename = '../data/KDD_data/irvine/cascade.txt'
    lastfmDatasetFilename = '../data/KDD_data/lastfm/cascade.txt'
    information_cascades = {}
    print("Starting data processing")
    with open(irvineDatasetFilename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        cascade_id = 0
        # length  = len(list(csv_reader))
        # print([row for row in csv_reader])
        maxTime = None
        minNodeID =  np.inf
        maxNodeID = -np.inf

        node_counts = {}
        for row in csv_reader:
            events = row
            events = [x.split(',') for x in events][:-1]
            
            node_ids = [int(x[0].split('_')[-1])- 1 for x in events]

            for node_id in node_ids:
                if node_id not in node_counts:
                    node_counts[node_id] = 1
                else:
                    node_counts[node_id] += 1
        selected_nodes = []
        for k in sorted(node_counts, key=node_counts.get, reverse=True):
            selected_nodes.append(k)
        selected_nodes = np.array(selected_nodes)

    with open(lastfmDatasetFilename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        cascade_id = 0
        # length  = len(list(csv_reader))
        # print([row for row in csv_reader])
        maxTime = None
        minNodeID =  np.inf
        maxNodeID = -np.inf

        for index, row in enumerate(csv_reader):
            print("processed ", index, " realizations out of " )
            events = row
            events = np.array([x.split(',') for x in events][:-1])
            node_ids = np.array([int(x[0].split('_')[-1])- 1 for x in events])
            
            req_indices = []
            for inner_index, node_id in enumerate(node_ids):
                if node_id in selected_nodes:
                    req_indices.append(inner_index)

            events = events[req_indices]
            node_ids = node_ids[req_indices]
            if len(events) <= 1:
                continue
            if minNodeID > min(node_ids):
                minNodeID = min(node_ids)
            if maxNodeID < max(node_ids):
                maxNodeID = max(node_ids)
            timestamps = [pd.Timestamp(datetime.fromtimestamp(int(x[1]))) for x in events]
            if maxTime is None: 
                maxTime = max(timestamps)
            elif maxTime < max(timestamps):
                maxTime = max(timestamps)
            cascade = {}
            cascade['timestamps'] = timestamps
            cascade['timestamp_ids'] = node_ids
            information_cascades[str(cascade_id)] = cascade
            cascade_id += 1
        information_cascades['rightCensoring'] = pd.Timestamp(maxTime)

    print("done")
    with open('../data/KDD_data/irvine_realizations.npy', 'wb') as f:
        np.save(f, np.array(information_cascades))

def generateSplitsForKDDPaper():
    # take the realizations and split them, retaining node_ids and everything else.
    def split_dataset(filename, scenario):
        # load dataset
        realizations = np.load(filename, allow_pickle = True).item()
        modified_realizations = realizations.copy()
        if scenario == "irvine" or scenario == "lastfm":
            del modified_realizations['rightCensoring']
        index_array = [*modified_realizations]
        train_index, test_index, _, _ = train_test_split(index_array,index_array,test_size=0.40, random_state = 55)
        test_index, val_index, _, _ = train_test_split(test_index,test_index,test_size=0.5, random_state = 55)
        
        training_realizations = {}
        for index in train_index:
            training_realizations[index] = realizations[index]
        validation_realizations = {}
        for index in val_index:
            validation_realizations[index] = realizations[index]
        test_realizations = {}
        for index in test_index:
            test_realizations[index] = realizations[index]
        
        if scenario == "irvine" or scenario == "lastfm":
            training_realizations["rightCensoring"] = realizations["rightCensoring"]
            test_realizations["rightCensoring"] = realizations["rightCensoring"]
            validation_realizations["rightCensoring"] = realizations["rightCensoring"]
        

        with open("../data/KDD_data/training_"+scenario+".npy", mode="wb") as f:
            np.save(f,training_realizations)
        with open("../data/KDD_data/test_"+scenario+".npy", mode="wb") as f:
            np.save(f,test_realizations)
        with open("../data/KDD_data/validation_"+scenario+".npy", mode="wb") as f:
            np.save(f,validation_realizations)
        
        print(len(train_index))
        print(len(test_index))
        print(len(val_index))
        input()
    #split_dataset(filename='../data/KDD_data/twitter_realizations.npy', scenario="twitter")
    # github dataset
    #split_dataset(filename='../data/KDD_data/github_realizations.npy', scenario="github")
    # irvine dataset
    split_dataset(filename='../data/KDD_data/irvine_realizations.npy', scenario="irvine")
    # lastfm dataset
    split_dataset(filename='../data/KDD_data/lastfm_realizations.npy', scenario="lastfm")
    pass

        
def generateCasFlowDataset(dataset_name, t_c_list):
    def generateDataset(scenario, t_c):
        training_dataset = np.load("../data/KDD_data/training_" +scenario+".npy", allow_pickle=True).item()
        try:
            del training_dataset['rightCensoring']
        except KeyError:
            pass
        if not os.path.exists("../data/KDD_data/"+scenario+"/"+str(t_c)):
            os.makedirs("../data/KDD_data/"+scenario+"/"+str(t_c))

        with open("../data/KDD_data/"+scenario+"/"+str(t_c)+"/train.txt", "w") as f:     
            for realization_index in training_dataset.keys():
                # print(training_dataset[realization_index])
                timestamps = training_dataset[realization_index]['timestamps']
                node_ids = training_dataset[realization_index]['timestamp_ids']
                #realization_str = str(realization_index) +"\t" + str(node_ids[0]) + "\t" + str(0) + "\t" + str(len(timestamps))
                realization_str = str(realization_index)
                
                timestamp_index = 0
                incremental_pop = 0
                for timestamp, node_id in zip(timestamps, node_ids):
                    if scenario == "irvine" or scenario == "lastfm":
                        epoch = int((timestamp -timestamps[0])/np.timedelta64(1, 'h'))
                        if epoch > t_c:
                            incremental_pop += 1
                    else:
                        epoch = int(3600*(timestamp -timestamps[0]))
                        if epoch/3600 > t_c:
                            incremental_pop += 1
                    if timestamp_index == 0:
                        realization_str +=  "\t"+ str(node_id) + ":" + str(epoch)+" "
                        timestamp_index +=1
                    else:
                        realization_str +=  "\t"+ str(node_ids[0])+","+str(node_id) + ":" + str(epoch)
                realization_str += "\t"+str(incremental_pop)+"\n"
                f.write(realization_str)

        test_dataset = np.load("../data/KDD_data/test_" +scenario+".npy", allow_pickle=True).item()
        try:
            del test_dataset['rightCensoring']
        except KeyError:
            pass
        num_test_samples = 0
        with open("../data/KDD_data/"+scenario+"/"+str(t_c)+"/test.txt", "w") as f:     
            for realization_index in test_dataset.keys():
                timestamps = test_dataset[realization_index]['timestamps']
                num_test_samples += 1
                node_ids = test_dataset[realization_index]['timestamp_ids']
                #realization_str = str(realization_index) +"\t" + str(node_ids[0]) + "\t" + str(0) + "\t" + str(len(timestamps))
                realization_str = str(realization_index)
                
                timestamp_index = 0
                incremental_pop = 0
                for timestamp, node_id in zip(timestamps, node_ids):
                    if scenario == "irvine" or scenario == "lastfm":
                        epoch = int((timestamp -timestamps[0])/np.timedelta64(1, 'h'))
                        if epoch > t_c:
                            incremental_pop += 1
                    else:
                        epoch = int(3600*(timestamp -timestamps[0]))
                        if epoch/3600 > t_c:
                            incremental_pop += 1
                    if timestamp_index == 0:
                        realization_str +=  "\t"+ str(node_id) + ":" + str(epoch)+" "
                        timestamp_index +=1
                    else:
                        realization_str +=  "\t"+ str(node_ids[0])+","+str(node_id) + ":" + str(epoch)
                realization_str += "\t"+str(incremental_pop)+"\n"
                f.write(realization_str)


        validation_dataset = np.load("../data/KDD_data/validation_" +scenario+".npy", allow_pickle=True).item()
        try:
            del validation_dataset['rightCensoring']
        except KeyError:
            pass

        with open("../data/KDD_data/"+scenario+"/"+str(t_c)+"/val.txt", "w") as f:     
            for realization_index in validation_dataset.keys():
                timestamps = validation_dataset[realization_index]['timestamps']
                node_ids = validation_dataset[realization_index]['timestamp_ids']
                #realization_str = str(realization_index) +"\t" + str(node_ids[0]) + "\t" + str(0) + "\t" + str(len(timestamps))
                realization_str = str(realization_index)
                timestamp_index = 0
                incremental_pop = 0
                for timestamp, node_id in zip(timestamps, node_ids):
                    if scenario == "irvine" or scenario == "lastfm":
                        epoch = int((timestamp -timestamps[0])/np.timedelta64(1, 'h'))
                        if epoch > t_c:
                            incremental_pop += 1
                    else:
                        epoch = int(3600*(timestamp -timestamps[0]))
                        if epoch/3600 > t_c:
                            incremental_pop += 1
                    if timestamp_index == 0:
                        realization_str +=  "\t"+ str(node_id) + ":" + str(epoch)+" "
                        timestamp_index +=1
                    else:
                        realization_str +=  "\t"+ str(node_ids[0])+","+str(node_id) + ":" + str(epoch)
                realization_str += "\t"+str(incremental_pop)+"\n"
                f.write(realization_str)

        print("total number of test samples: ", num_test_samples)
    for t_c in t_c_list:
        generateDataset(dataset_name, t_c)

def generateDatasetsForRegression_t_c(scenario, num_nodes):
    
    training_dataset = np.load("../data/KDD_data/training_" +scenario+".npy", allow_pickle=True).item()
    try:
        del training_dataset['rightCensoring']
    except KeyError:
        pass
    test_dataset = np.load("../data/KDD_data/test_" +scenario+".npy", allow_pickle=True).item()
    try:
        del test_dataset['rightCensoring']
    except KeyError:
        pass
    valdiation_dataset = np.load("../data/KDD_data/validation_" +scenario+".npy", allow_pickle=True).item()
    try:
        del valdiation_dataset['rightCensoring']
    except KeyError:
        pass
    
    

    if not os.path.exists("../data/KDD_data/"+scenario):
        os.makedirs("../data/KDD_data/"+scenario)

    filename_qualifier = "../data/KDD_data/" + scenario + "/"
    try:
        with open(filename_qualifier + 'graph_adj_matrix_'+ str(num_nodes) +'.pkl', 'rb') as f:
                updated_adj_matrix = pickle.load(f)
    except:
        updated_adj_matrix = np.zeros((num_nodes, num_nodes))


    follower_counts = np.zeros(len(updated_adj_matrix))

    for node in range(len(updated_adj_matrix)):
        follower_counts[node] = np.sum(updated_adj_matrix[:,node])


    #{'cid':, 'post_time_day':, hw:[]}
    training_list_json = []
    for realization_index in training_dataset.keys():
        timestamps = training_dataset[realization_index]['timestamps']
        node_ids = training_dataset[realization_index]['timestamp_ids']
        sample_dict = {}
        sample_dict['cid'] = str(realization_index)
        
        realization_str = " "
        epoch_list = []

        
        for timestamp, node_id in zip(timestamps, node_ids):
            if scenario == "irvine" or scenario == "lastfm":
                epoch = (timestamp -timestamps[0])/np.timedelta64(1, 'h')
            else:
                epoch = (timestamp -timestamps[0])
            
            epoch_list.append(epoch)
        sample_dict['hw'] = [(x,y) for x,y in zip(epoch_list,follower_counts[node_ids])]
        training_list_json.append(sample_dict)

    #json_str = json.dumps(training_dict_json)
    json.dump(training_list_json, open(filename_qualifier+'train.json', 'w'))
            
    test_list_json = []
    total_count = 0
    for realization_index in test_dataset.keys():
        timestamps = test_dataset[realization_index]['timestamps']
        node_ids = test_dataset[realization_index]['timestamp_ids']
        total_count += 1
        sample_dict = {}
        sample_dict['cid'] = str(realization_index)
        
        realization_str = " "
        epoch_list = []
        for timestamp, node_id in zip(timestamps, node_ids):
            if scenario == "irvine" or scenario == "lastfm":
                epoch = (timestamp -timestamps[0])/np.timedelta64(1, 'h')
            else:
                epoch = (timestamp -timestamps[0])
            
            epoch_list.append(epoch)
        sample_dict['hw'] = [(x,y) for x,y in zip(epoch_list,follower_counts[node_ids])]
        test_list_json.append(sample_dict)

    #json_str = json.dumps(test_dict_json)
    json.dump(test_list_json, open(filename_qualifier+'test.json', 'w'))


    val_list_json = []
    total_count = 0
    for realization_index in valdiation_dataset.keys():
        timestamps = valdiation_dataset[realization_index]['timestamps']
        node_ids = valdiation_dataset[realization_index]['timestamp_ids']
        total_count += 1
        sample_dict = {}
        sample_dict['cid'] = str(realization_index)
        
        realization_str = " "
        epoch_list = []
        for timestamp, node_id in zip(timestamps, node_ids):
            if scenario == "irvine" or scenario == "lastfm":
                epoch = (timestamp -timestamps[0])/np.timedelta64(1, 'h')
            else:
                epoch = (timestamp -timestamps[0])
            
            epoch_list.append(epoch)

        sample_dict['hw'] = [(x,y) for x,y in zip(epoch_list,follower_counts[node_ids])]
        val_list_json.append(sample_dict)

    #json_str = json.dumps(test_dict_json)
    json.dump(val_list_json, open(filename_qualifier+'val.json', 'w'))
    
def createEBMaseptideData(scenario, num_nodes):
    training_dataset = np.load("../data/KDD_data/training_" +scenario+".npy", allow_pickle=True).item()
    try:
        del training_dataset['rightCensoring']
    except KeyError:
        pass
    test_dataset = np.load("../data/KDD_data/test_" +scenario+".npy", allow_pickle=True).item()
    try:
        del test_dataset['rightCensoring']
    except KeyError:
        pass
    valdiation_dataset = np.load("../data/KDD_data/validation_" +scenario+".npy", allow_pickle=True).item()
    try:
        del valdiation_dataset['rightCensoring']
    except KeyError:
        pass
    
    if not os.path.exists("../data/KDD_data/"+scenario):
        os.makedirs("../data/KDD_data/"+scenario)

    filename_qualifier = "../data/KDD_data/" + scenario + "/"
    try:
        with open("../data/" + scenario + "/" + 'graph_adj_matrix_'+ str(num_nodes) +'.pkl', 'rb') as f:
                updated_adj_matrix = pickle.load(f)
                
    except:
        updated_adj_matrix = np.zeros((num_nodes, num_nodes))


    follower_counts = np.zeros(len(updated_adj_matrix))

    for node in range(len(updated_adj_matrix)):
        follower_counts[node] = np.sum(updated_adj_matrix[:,node])

    train_list = {}
    for realization_index in training_dataset.keys():
        timestamps = training_dataset[realization_index]['timestamps']
        node_ids = training_dataset[realization_index]['timestamp_ids']
        sample_dict = {}
        epoch_list = []
        for timestamp, node_id in zip(timestamps, node_ids):
            if scenario == "irvine" or scenario == "lastfm":
                epoch = (timestamp -timestamps[0])/np.timedelta64(1, 'h')
            else:
                epoch = (timestamp -timestamps[0])
            
            epoch_list.append(epoch)
        train_list[realization_index] = [list(epoch_list),list(follower_counts[node_ids]), str(realization_index)]
        

    #json_str = json.dumps(test_dict_json)
    json.dump(train_list, open(filename_qualifier+'EBMaseptide_train.json', 'w'))


    test_list = {}
    for realization_index in test_dataset.keys():
        timestamps = test_dataset[realization_index]['timestamps']
        node_ids = test_dataset[realization_index]['timestamp_ids']
        sample_dict = {}
        epoch_list = []
        for timestamp, node_id in zip(timestamps, node_ids):
            if scenario == "irvine" or scenario == "lastfm":
                epoch = (timestamp -timestamps[0])/np.timedelta64(1, 'h')
            else:
                epoch = (timestamp -timestamps[0])
            
            epoch_list.append(epoch)
        test_list[realization_index] = [list(epoch_list),list(follower_counts[node_ids]), str(realization_index)]
    

    #json_str = json.dumps(test_dict_json)
    json.dump(test_list, open(filename_qualifier+'EBMaseptide_test.json', 'w'))
    
    print("total test set size for ", scenario, " :", len(test_list))

    

def main():

    #generateStartSizes("digg")
    #input()
    # generateNeuralDiffusionPaperDataset()
    # generateSplitsForKDDPaper()
    # input()
    start_sizes = [5,7,10,12]
    scenario_names = ["digg", "memes", "synthetic", "lastfm", "irvine", "twitter"]
    
    scenario_names = [ "memes"]

    for scenario in scenario_names:
        if scenario == "lastfm":
            num_nodes = 1000
        elif scenario == "synthetic":
            num_nodes = 100
        elif scenario == "irvine":
            num_nodes = 893
        else:
            num_nodes = 200
        
        
        #GenerateDatasetFromTopoLSTMPaper(scenario, num_nodes=num_nodes)
        #generateDatasetsForRegression_t_c(scenario=scenario,num_nodes=num_nodes)
        #createEBMaseptideData(scenario=scenario,num_nodes=num_nodes)
        #generateDatasetsForRegression_start_size(scenario=scenario,num_nodes=num_nodes)
        
        #generateCasFlowDataset(scenario, t_c_list = [1*24,2*24,30*24,60*24])
        for start_size in start_sizes:
            generateDatasetsForForestPaper(start_size=start_size, scenario=scenario, num_nodes=num_nodes)
    
            
if __name__ == "__main__":
    main()




def trash():
    validation_dataset = np.load("../data/KDD_data/validation_" +scenario+".npy", allow_pickle=True).item()
    try:
        del validation_dataset['rightCensoring']
    except KeyError:
        pass
    with open("../data/KDD_data/cascadevalid.txt", "w") as f:     
        for realization_index in validation_dataset.keys():
            timestamps = validation_dataset[realization_index]['timestamps']
            # if len(timestamps) == 1:
            #     continue
            node_ids = validation_dataset[realization_index]['timestamp_ids']
            realization_str = " "
            for timestamp, node_id in zip(timestamps, node_ids):
                # epoch = (timestamp - dt.datetime(1970,1,1)).total_seconds()
                # epoch = (timestamp -timestamps[0])/3600
                
                epoch = (timestamp -timestamps[0])/np.timedelta64(1, 'h')
                realization_str += str(node_id) + "," + str(epoch)+" "
            realization_str += "\n"
            f.write(realization_str)