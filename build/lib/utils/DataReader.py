''' DataReader.py
    version: 1.2
    contains all functionality to read from external datasets (datasets not generated synthetically).
    as a results, this file contains some dataset specific code that might lack generality.

    (i) it right now has functionality to read Mitre data, CVE data from json file.
    (ii) It can read chicago crime data extracted as a csv from Big query.

      Edited by:                                              '''

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

def generateDatasetForDeepHawkesPaper():
    def generateDeepHawkesPaper(scenario):
        realizations = np.load("../data/KDD_data/"+scenario+"_realizations.npy", allow_pickle=True).item()
        
        training_dataset = np.load("../data/KDD_data/training_" +scenario+".npy", allow_pickle=True).item()
        test_dataset = np.load("../data/KDD_data/test_" +scenario+".npy", allow_pickle=True).item()
        validation_dataset = np.load("../data/KDD_data/validation_" +scenario+".npy", allow_pickle=True).item()
        
        try:
            del realizations['rightCensoring']
        except KeyError:
            pass
        training_keys = training_dataset.keys()
        validation_keys = validation_dataset.keys()
        test_keys = test_dataset.keys()
        cascade_types = {} # 1 for training, 2 for validation and 3 for test
        # print(training_keys)
        # print(validation_keys)
        # print(test_keys)
        # input()
        with open('../data/KDD_data/dataset_'+str(scenario)+'.txt', 'w')as f:
            for index, realization_index in enumerate(realizations.keys()):
                timestamps = realizations[realization_index]['timestamps'][:-1]
                node_ids = realizations[realization_index]['timestamp_ids'][:-1]
                # if realization_index in test_keys:
                print(realization_index)
                if str(realization_index) in training_keys:
                    cascade_types[index] = 1
                elif str(realization_index) in validation_keys:
                    cascade_types[index] = 2
                elif str(realization_index) in test_keys:
                    cascade_types[index] = 3
                else:
                    raise Exception("nowhere to be found")
                # if scenario == 'github' or scenario == 'twitter':
                # epoch_0 = (timestamps[0] -timestamps[0])/np.timedelta64(1, 'h')
                epoch_0 = (timestamps[0] -timestamps[0])/3600
                
                # else:
                #     epoch_0 = (timestamps[0] - dt.datetime(1970,1,1)).total_seconds()
                realization_str = str(index) + "\t" + str(node_ids[0]) + "\t" + str(epoch_0) + "\t" + str(len(timestamps)-1) + "\t"
                for timestamp, node_id in zip(timestamps[1:], node_ids[1:]):
                    # if scenario == 'github' or scenario == 'twitter':
                    # epoch = (timestamp -timestamps[0])/np.timedelta64(1, 'h')
                    epoch = (timestamp -timestamps[0])/3600
                    
                    # else:
                    #     epoch = (timestamp - dt.datetime(1970,1,1)).total_seconds()
                    
                    realization_str += str(node_ids[0])+"/"+str(node_id)+":"+str(epoch) + " "
                f.write(realization_str+"\n")
        with open('../data/KDD_data/'+str(scenario)+'_cascade_types.npy','w') as f:
            np.save('../data/KDD_data/'+str(scenario)+'_cascade_types.npy', cascade_types)
    # generateDeepHawkesPaper('twitter')
    # generateDeepHawkesPaper('lastfm')
    generateDeepHawkesPaper('twitter_link')
    

        
def generateDatasetsForDeepDiffusePaper(start_size = 5):
    def generateDeepDiffuseDataset(scenario):
        training_dataset = np.load("../data/KDD_data/training_" +scenario+".npy", allow_pickle=True).item()
        try:
            del training_dataset['rightCensoring']
        except KeyError:
            pass
        with open("../data/KDD_data/train.txt", "w") as f:
            for index, realization_index in enumerate(training_dataset.keys()):
                timestamps = training_dataset[realization_index]['timestamps']
                # if len(timestamps) == 1:
                #     continue
                node_ids = training_dataset[realization_index]['timestamp_ids']
                realization_str = str(index) + " "
                for timestamp, node_id in zip(timestamps, node_ids):
                    epoch = (timestamp -timestamps[0])/np.timedelta64(1, 'h')
                    # epoch = (timestamp - dt.datetime(1970,1,1)).total_seconds()
                    
                    realization_str +=  str(node_id) + " " + str(epoch)+" "
                realization_str += "\n"
                f.write(realization_str)
        test_dataset = np.load("../data/KDD_data/test_" +scenario+".npy", allow_pickle=True).item()
        try:
            del test_dataset['rightCensoring']
        except KeyError:
            pass
        with open("../data/KDD_data/test.txt", "w") as f:
            for index, realization_index in enumerate(test_dataset.keys()):
                timestamps = test_dataset[realization_index]['timestamps']
                if len(timestamps) <= start_size:
                    continue
                node_ids = test_dataset[realization_index]['timestamp_ids']
                realization_str = str(index + len(training_dataset.keys())) + " "
                for timestamp, node_id in zip(timestamps, node_ids):
                    epoch = (timestamp -timestamps[0])/np.timedelta64(1, 'h')
                    # epoch = (timestamp - dt.datetime(1970,1,1)).total_seconds()

                    realization_str +=  str(node_id) + " " + str(epoch)+" "
                realization_str += "\n"
                f.write(realization_str)
    generateDeepDiffuseDataset("lastfm")    


def generateDatasetsForForestPaper(start_size, scenario):
    def generateForestDataset(scenario):
        training_dataset = np.load("../data/KDD_data/training_" +scenario+".npy", allow_pickle=True).item()
        try:
            del training_dataset['rightCensoring']
        except KeyError:
            pass
        with open("../data/KDD_data/cascade.txt", "w") as f:     
            for realization_index in training_dataset.keys():
                # print(training_dataset[realization_index])
                timestamps = training_dataset[realization_index]['timestamps']
                # if len(timestamps) == 1:
                #     continue
                node_ids = training_dataset[realization_index]['timestamp_ids']
                realization_str = " "
                for timestamp, node_id in zip(timestamps, node_ids):
                    epoch = (timestamp -timestamps[0])/np.timedelta64(1, 'h')
                    # epoch = (timestamp -timestamps[0])/3600
                    
                    # epoch = (timestamp - dt.datetime(1970,1,1)).total_seconds()
                    
                    realization_str +=  str(node_id) + "," + str(epoch)+" "
                realization_str += "\n"
                f.write(realization_str)
        test_dataset = np.load("../data/KDD_data/test_" +scenario+".npy", allow_pickle=True).item()
        try:
            del test_dataset['rightCensoring']
        except KeyError:
            pass
        with open("../data/KDD_data/cascadetest.txt", "w") as f:     
            for realization_index in test_dataset.keys():
                # print(training_dataset[realization_index])
                timestamps = test_dataset[realization_index]['timestamps']
                if len(timestamps) <= start_size:
                    continue
                node_ids = test_dataset[realization_index]['timestamp_ids']
                realization_str = " "
                for timestamp, node_id in zip(timestamps, node_ids):
                    # epoch = (timestamp - dt.datetime(1970,1,1)).total_seconds()
                    epoch = (timestamp -timestamps[0])/np.timedelta64(1, 'h')
                    # epoch = (timestamp -timestamps[0])/3600
                    
                    realization_str += str(node_id) + "," + str(epoch)+" "
                # print(realization_str)
                # input()
                realization_str += "\n"
                f.write(realization_str)
        
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
            
    generateForestDataset(scenario=scenario)



def generateSplitsForKDDPaper():
    # take the realizations and split them, retaining node_ids and everything else.
    def split_dataset(filename, scenario):
        # load dataset
        realizations = np.load(filename, allow_pickle = True).item()
        modified_realizations = realizations.copy()
        if scenario == "irvine" or scenario == "lastfm":
            del modified_realizations['rightCensoring']
        index_array = [*modified_realizations]
        train_index, test_index, _, _ = train_test_split(index_array,index_array,test_size=0.20, random_state = 55)
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
    split_dataset(filename='../data/KDD_data/twitter_realizations.npy', scenario="twitter")
    # github dataset
    split_dataset(filename='../data/KDD_data/github_realizations.npy', scenario="github")
    # irvine dataset
    split_dataset(filename='../data/KDD_data/irvine_realizations.npy', scenario="irvine")
    # lastfm dataset
    split_dataset(filename='../data/KDD_data/lastfm_realizations.npy', scenario="lastfm")
    pass

def convertTwitterLinkSharingDataset():
    # generate node list
    with open('../data/KDD_data/twitter_link_sharing/seen_nodes.txt') as f:
        content = f.readlines()
        content = [int(x.strip()) for x in content]
        unique_ids =range(len(np.unique(content)))
        map_dict = dict(zip(content,unique_ids))
    
    incoming_count = np.zeros(len(unique_ids))
    outgoing_count = np.zeros(len(unique_ids))
    # generate features from graph
    with open('../data/KDD_data/twitter_link_sharing/graph.txt') as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        for edge_str in content:
            edge = edge_str.split(' ')
            try:
                outgoing_count[map_dict[int(edge[0])]] +=1
            except:
                pass
            try:
                incoming_count[map_dict[int(edge[1])]] +=1
            except:
                pass

    feature_vectors = np.zeros((len(unique_ids),2))
    feature_vectors[:, 0] = incoming_count
    feature_vectors[:, 1] = outgoing_count
    
    # generate training realizations
    training_realizations = {}
    maxTime = None
    with open('../data/KDD_data/twitter_link_sharing/train.txt') as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        for index, realization in enumerate(content):
            cascade = {}
            vals = realization.split(' ')[1:]
            node_ids = vals[::2]
            node_ids = [map_dict[int(x)] for x in node_ids]
            timestamps = list(map(int, vals[1::2]))
            seen_nodes = []
            req_indices = []
            for inner_index,node_id in enumerate(node_ids):
                if node_id not in seen_nodes:
                    seen_nodes.append(node_id)
                    req_indices.append(inner_index)
            
            node_ids = np.array(node_ids)[req_indices]
            timestamps = np.array(timestamps)[req_indices] 
            timestamps = timestamps-min(timestamps)       
            if maxTime is None: 
                maxTime = max(timestamps)
            elif maxTime < max(timestamps):
                maxTime = max(timestamps)
            
            cascade['timestamp_ids'] = node_ids
            cascade['timestamps'] = timestamps

            training_realizations[str(index)] = cascade
        training_realizations['rightCensoring'] = maxTime
    
    # generate test realizations
    test_realizations = {}
    maxTime = None
    with open('../data/KDD_data/twitter_link_sharing/test.txt') as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        for index, realization in enumerate(content):
            cascade = {}
            vals = realization.split(' ')[1:]
            node_ids = vals[::2]
            node_ids = [map_dict[int(x)] for x in node_ids]
            timestamps = list(map(int, vals[1::2]))
            seen_nodes = []
            req_indices = []
            for inner_index,node_id in enumerate(node_ids):
                if node_id not in seen_nodes:
                    seen_nodes.append(node_id)
                    req_indices.append(inner_index)
            
            node_ids = np.array(node_ids)[req_indices]
            timestamps = np.array(timestamps)[req_indices] 

            timestamps = timestamps-min(timestamps)    

            if maxTime is None: 
                maxTime = max(timestamps)
            elif maxTime < max(timestamps):
                maxTime = max(timestamps)
            
            cascade['timestamp_ids'] = node_ids
            cascade['timestamps'] = timestamps

            test_realizations[str(index + 456)] = cascade
        test_realizations['rightCensoring'] = maxTime


    np.save('../data/KDD_data/training_twitter_link.npy', training_realizations)
    np.save('../data/KDD_data/test_twitter_link.npy', test_realizations)
    np.save('../data/KDD_data/validation_twitter_link.npy', test_realizations)
    
    np.save('../data/KDD_data/twitter_link_user_features.npy', feature_vectors)
     


def convertRetweetStatusToUserID():
    twitter_events = pd.read_csv('../data/cp3_twitter_all.csv')
    unique_twitter_users = twitter_events['user'].unique()
    # replace all source with user ids
    ids = twitter_events[twitter_events['type'] == 'retweet']['id'].values
    user = twitter_events[twitter_events['type'] == 'retweet']['user'].values
    source = twitter_events[twitter_events['type'] == 'retweet']['source'].values
    event_type = twitter_events[twitter_events['type'] == 'retweet']['type'].values
    updated_source = source.copy()
    for index, source_status_id in enumerate(source):
        print(index, "out of ", len(user), end='\r')
        if len(source_status_id) > 0:
            if event_type[index] == 'retweet':
                location = np.where(ids == source_status_id)[0]
                if len(location)  > 0:
                    status_index = np.where(ids ==location[0])
                    updated_source[index] = user[status_index]

    twitter_events.loc[twitter_events['type'] == 'retweet']['source'] = updated_source
    twitter_events.to_csv('../data/cp3_twitter_all.csv',index=False )
    print("saved")
    input()
# function to create graphs out of various dataset for comparison
def constructGraphs():
    def saveTwitterGraph():
        twitter_events = pd.read_csv('../data/cp3_twitter_all.csv')
        unique_twitter_users = twitter_events['user'].unique()
        # convertRetweetStatusToUserID()
        graph_dict = {}
        for index, user in enumerate(unique_twitter_users):
            twitter_user_df = twitter_events[twitter_events['user']== user]
            source_ids = twitter_user_df['source'].values
            ctr = collections.Counter(source_ids)
            graph_dict[user] = ctr
            print(index, "out of ", len(unique_twitter_users), end='\r')

        with open('../data/cp3_twitter_graph.npy', mode='wb') as f:
            np.save(f, graph_dict)
    def saveGithubGraph():
        github_events = pd.read_csv('../data/cp3_github_all.csv')
        unique_github_users = github_events['user'].unique()
        unique_github_cves = github_events['cve'].unique()
        for cve in unique_github_cves:
            cve_df = github_events[github_events['cve'] == cve]
            
        graph_dict = {}
    saveTwitterGraph()

        
        
    

def generateNeuralDiffusionPaperDataset():
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
        for row in csv_reader:
            print("processed ", cascade_id, " realizations out of " )
            events = row
            events = [x.split(',') for x in events][:-1]
            node_ids = [int(x[0])- 1 for x in events]
            if minNodeID > min(node_ids):
                minNodeID = min(node_ids)
            if maxNodeID < max(node_ids):
                maxNodeID = max(node_ids)
            timestamps = [pd.Timestamp(datetime.fromtimestamp(int(x[1]))) for x in events]
            # print(timestamps)
            # input()
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

    with open('../data/KDD_data/irvine_realizations.npy', 'wb') as f:
        np.save(f, np.array(information_cascades))



def generateSplitPopulationSocialMediaDataset():
    github_events = pd.read_csv('../data/cp3_github_all.csv')
    github_events['time'] = pd.to_datetime(github_events['time'])

    reddit_events = pd.read_csv('../data/cp3_reddit_all.csv')
    reddit_events['time'] = pd.to_datetime(reddit_events['time'])
    
    twitter_events = pd.read_csv('../data/cp3_twitter_all.csv')
    twitter_events['time'] = pd.to_datetime(twitter_events['time'])
    
    
    # generating cascades for each social media event
    github_cves = github_events['cve'].unique()
    reddit_cves = reddit_events['cve'].unique()
    twitter_cves = twitter_events['cve'].unique()

    intersection_cves = np.intersect1d(github_cves, reddit_cves)
    intersection_cves = np.intersect1d(intersection_cves, twitter_cves)

    union_cves = np.union1d(github_cves, reddit_cves)    
    union_cves = np.union1d(union_cves,twitter_cves)
    github_users_df = pd.read_csv('../data/cp3_github_users.csv')   
    reddit_unique_subreddits = reddit_events['subreddit'].unique()
    
    github_unique_users = github_events['user'].unique()
    reddit_unique_users = reddit_events['user'].unique()
    

    # print("Social media platform data extracted..")
    # found_users_github = []
    # # extract github user features
    # for index, user in enumerate(github_unique_users):
    #     # print(user)
    #     user_df = github_users_df[github_users_df['user']==user]
    #     if len(user_df) > 0:
    #         found_users_github.append(user)
    # github_events = github_events[github_events['user'].isin(found_users_github)]
    # github_user_dict = dict(zip(found_users_github, range(len(found_users_github))))
    # github_events = github_events.replace({'user':github_user_dict})
    # github_users_df = github_users_df.replace({'user':github_user_dict})
    # github_cves = github_events['cve'].unique()
    # print("Github users dehashed to integer identities..")


    # extract twitter user features
    
    # twitter_events = twitter_events[twitter_events['user'].isin(twitter_unique_users)]
    # twitter_user_dict = dict(zip(twitter_unique_users, range(len(twitter_unique_users))))
    # twitter_events = twitter_events.replace({'user':twitter_user_dict})
    # reddit_events = reddit_events[reddit_events['user'].isin(reddit_unique_users)]
    # reddit_user_dict = dict(zip(reddit_unique_users, range(len(reddit_unique_users))))
    # reddit_events = reddit_events.replace({'user':reddit_user_dict})
    # reddit_cves = reddit_events['cve'].unique()
    # print("Reddit users dehashed to integer identities..")


    # look for users who are in more than x = 20 cascades
    users_appearing_multiple_times = twitter_events.groupby("user").cve.apply(lambda x: len(x) > 5)
    users_appearing_multiple_times = users_appearing_multiple_times[users_appearing_multiple_times== True].index.values
    req_cves = twitter_events[twitter_events['user'].isin(users_appearing_multiple_times)]
    hist_values = req_cves.groupby('cve')['user'].count()
    req_cves = hist_values[hist_values  > 1].index
    twitter_events = twitter_events[twitter_events['cve'].isin(req_cves)]
    twitter_events = twitter_events[twitter_events['user'].isin(users_appearing_multiple_times)]
    twitter_events = twitter_events.dropna(subset=['user_followers_count'])
    twitter_unique_users = twitter_events['user'].unique()
    twitter_cves = twitter_events['cve'].unique()
    twitter_user_dict = dict(zip(twitter_unique_users, range(len(twitter_unique_users))))
    twitter_events = twitter_events.replace({'user':twitter_user_dict})
    
    print("Twitter users dehashed to integer identities..")

    information_cascades = {}
    for index, cve in enumerate(twitter_cves):
        # each information cascade consists of three lists: 
        # 1. Sorted timestamps list 
        # 2. ID list that maps to sorted timestamps list
        # 3. parent ID list that maps to both ID and sorted timestamps list.
        # for a given index, all of these values must be consistent, that means all three lists must have same size

        # github events
        # github_timestamps = []
        # github_timestamp_ids = []
        # github_timestamp_parent_ids = []
        # # extract all github events for this cve
        # github_events_cve_df = github_events[github_events['cve'] == cve]
        # github_events_cve_df = github_events_cve_df.sort_values(by=['time'])
        # github_timestamps.extend(github_events_cve_df['time'].values)
        # github_timestamp_ids.extend(github_events_cve_df['user'].values)

        # unique_ids = []
        # unique_timestamps = []
        # for i, val in enumerate(github_timestamp_ids):
        #     if val not in unique_ids:
        #         unique_ids.append(val)
        #         unique_timestamps.append(github_timestamps[i])

        # timestamps = np.array(unique_timestamps)
        # node_ids = np.array(unique_ids)

        # assert len(github_timestamp_ids) == len(github_timestamp_ids), "Inconsistency in event information"

        # extract all twitter events for this cve
        twitter_timestamps = []
        twitter_timestamp_ids = []
        twitter_timestamp_parent_ids = []
        twitter_event_types = []
        twitter_events_cve_df = twitter_events[twitter_events['cve'] == cve]
        twitter_events_cve_df = twitter_events_cve_df.sort_values(by=['time'])
        twitter_timestamps.extend(twitter_events_cve_df['time'].values)
        twitter_timestamp_ids.extend(twitter_events_cve_df['user'].values)
        twitter_timestamp_parent_ids.extend(twitter_events_cve_df['source'].values)
        twitter_event_types.extend(twitter_events_cve_df['type'].values)

        unique_ids = []
        unique_timestamps = []
        for i, val in enumerate(twitter_timestamp_ids):
            if val not in unique_ids:
                unique_ids.append(val)
                unique_timestamps.append(twitter_timestamps[i])

        timestamps  = unique_timestamps
        node_ids = unique_ids

        # # extract all reddit events for this cve
        # reddit_timestamps = []
        # reddit_timestamp_ids = []
        # reddit_events_cve_df = reddit_events[reddit_events['cve'] == cve]
        # reddit_timestamps.extend(reddit_events_cve_df['time'].values)
        # reddit_timestamp_ids.extend(reddit_events_cve_df['user'].values)
        

        if len(timestamps) == 1:
            continue
        if index % 100 == 0:
            print("CVEs processed: ", index, "out of ",len(twitter_cves), end = '\r')
        cascade = {}
        cascade['timestamps'] = timestamps
        cascade['timestamp_ids'] = node_ids
        information_cascades[str(cve)] = cascade

    with open('../data/KDD_data/twitter_realizations.npy', 'wb') as f:
        np.save(f, np.array(information_cascades))
    
    
    # generate user features for each platform

    
    print('Generating Twitter user features...')
    twitter_user_features = np.zeros((len(twitter_unique_users),4))
    for user in range(len(twitter_unique_users)):
        user_df = twitter_events[twitter_events['user'] == user]
        num_followers = user_df['user_followers_count'].values[0]
        num_friends = user_df['user_friends_count'].values[0]
        num_statuses = user_df['user_statuses_count'].values[0]
        verified= user_df['user_verified'].values[0]
        if verified == True:
            verified = 1.0
        else:
            verified = 0.0
 
        user_feature_vector = np.array([num_followers, num_friends,num_statuses, verified])
        print(user_feature_vector)
        twitter_user_features[user] = user_feature_vector
        if user % 100 == 0:
            print("Completed ", user, " users out of ", len(twitter_unique_users), end = '\r')
    
    with open('../data/KDD_data/twitter_user_features.npy', 'wb') as f:
        np.save(f, np.array(twitter_user_features))
    
    # print('Generating Github user features...')
    # github_user_features = np.zeros((len(found_users_github),5))
    # for user in range(len(found_users_github)):
    #     user_df = github_users_df[github_users_df['user'] == user]
    #     num_followers = user_df['followers'].values[0]
    #     num_following = user_df['following'].values[0]
    #     public_repos = user_df['public_repos'].values[0]
    #     site_admin = user_df['site_admin'].values[0]
    #     if user_df['type'].values[0] == 'User':
    #         user_type = 1 
    #     else:
    #         user_type = -1
    #     user_feature_vector = np.array([num_followers, num_following,public_repos, site_admin, user_type])
    #     github_user_features[user] = user_feature_vector
    #     if index % 100 == 0:
    #         print("Completed ", index, " users out of ", len(found_users_github), end = '\r')
    
    # with open('../data/KDD_data/github_user_features.npy', 'wb') as f:
    #     np.save(f, np.array(github_user_features))
    
 
    
    # reddit_user_features = []
    # print("Generating Reddit user feature vectors...")
    # for index, user in enumerate(reddit_users):
    #     subreddits = reddit_events[reddit_events['user']==user]['subreddit'].unique()
    #     user_feature_vector = np.zeros(len(reddit_unique_subreddits))
    #     for subreddit in subreddits:
    #         subreddit_index = np.where(reddit_unique_subreddits == subreddit)
    #         user_feature_vector[subreddit_index]  = 1.0
    #     reddit_user_features.append(user_feature_vector)
    #     if index % 100 == 0:
    #         print("Completed ", index, " users out of ", len(reddit_users), end = '\r')
        
    # with open('../data/KDD_data/reddit_user_features.npy', 'wb') as f:
    #     np.save(f, np.array(reddit_user_features))


def generatePointProcessDataset():
    NN_training_cves = pd.read_csv('../data/cve_train.csv')['cve_id'].values
    NN_test_cves = pd.read_csv('../data/cve_test.csv')['cve_id'].values
    NN_validation_cves = pd.read_csv('../data/cve_validation.csv')['cve_id'].values
    
    cp3_soc_df = pd.read_csv('../data/cp3_social_media.csv')


    training_soc = cp3_soc_df[cp3_soc_df['cve'].isin(NN_training_cves)]
    test_soc = cp3_soc_df[cp3_soc_df['cve'].isin(NN_test_cves)]
    validation_soc = cp3_soc_df[cp3_soc_df['cve'].isin(NN_validation_cves)]
    print("Training: ", len(training_soc['cve'].unique()))
    print("Test: ", len(test_soc['cve'].unique()))
    print("Validation: ", len(validation_soc['cve'].unique()))
    
    point_process_training_df = pd.DataFrame({'cve_id':training_soc['cve'].unique()})
    point_process_test_df = pd.DataFrame({'cve_id':test_soc['cve'].unique()})
    point_process_validation_df = pd.DataFrame({'cve_id':validation_soc['cve'].unique()})
    
    point_process_training_df.to_csv('../data/point_process_training.csv')
    point_process_test_df.to_csv('../data/point_process_test.csv')
    point_process_validation_df.to_csv('../data/point_process_validation.csv')
    
    print('Point process dataset generated')


def readNVDDataFromList(mitre_dataframe):
    severity = None
    obtainUserPrivelege = None
    obtainAllPrivelege = None
    impactScore = None
    exploitabilityScore = None
    impactScore = None
    exploitabilityScore = None
    vectorString = None
    accessVector = None
    accessComplexity = None
    authentication = None
    confidentialityImpact = None
    integrityImpact = None
    availabilityImpact = None
    baseScoreV2 = None
    baseScoreV3 = None
    vendors = None
    products = None
    publishedDate = None
    lastModifiedDate = None
    exploitIDs = []
    num_exploits = 0
    num_non_exploits = 0
    CVE_dataframe = pd.DataFrame(
        columns=['CVE', 'mitre_description', 'isExploited'])
    isExploited = []

    unique_cves = mitre_dataframe['cve_id'].unique()
    print(len(unique_cves))
    # unique_cves = unique_cves.sort()
    print("sorted list")
    cve_exploit_dict = {}
    cve_exploit_id_dict = {}
    start_year = 1999
    year_end = 2020
    i = 0

    post_processing_files = ['modified', 'recent']
    #  extra two iterations for going through `modified` and `recent` files
    while start_year <= year_end + 2:
        print("\nyear:{0} ", format(start_year), end="\r")
        fileName = '../../data/nvdcve-1.1-'
        if start_year <= 2002:
            fileName = fileName + '2002.json'
        else:
            fileName = fileName + \
                str(start_year) + '.json'
        if start_year == 1999:
            start_year += 3
        start_year += 1

        if start_year > year_end:
            print(start_year)
            print(year_end)
            fileName = '../../data/nvdcve-1.1-' + \
                str(post_processing_files[start_year-year_end]) + '.json'
        # if i == 1000:
        #     break
        with open(fileName) as json_file:
            data = json.load(json_file)
            cnt = 0

            for cve in data['CVE_Items']:
                cve_number_obtained = cve['cve']['CVE_data_meta']['ID']
                if cve_number_obtained in unique_cves:
                    exploit_found = False
                    if cve_number_obtained in cve_exploit_dict.keys():
                        print("\nDuplicate found")

                    print('\nCVEs processed: {0}'.format(i), end="\r")
                    i += 1
                    # if i == 1000:
                    #     break
                    try:
                        publishedDate = cve['publishedDate']
                        # numPublishedDate = 1
                    except:
                        pass
                    try:
                        cpe_config = cve['configurations']
                        # numCPEInfo = 1
                    except:
                        pass
                    try:
                        references = cve['cve']['references']['reference_data']
                        exploitIDs = []
                        for ref in references:
                            if ref['refsource'] == 'EXPLOIT-DB':
                                exploitIDs.append(str(ref['name']))
                                exploit_found = True
                    except:
                        pass
                    if exploit_found:
                        cve_exploit_dict[cve_number_obtained] = 1
                        cve_exploit_id_dict[cve_number_obtained] = exploitIDs
                    else:
                        cve_exploit_dict[cve_number_obtained] = 0
                        cve_exploit_id_dict[cve_number_obtained] = []

    mitre_dataframe['isExploited'] = mitre_dataframe['cve_id'].map(
        cve_exploit_dict)

    mitre_dataframe['exploitID'] = mitre_dataframe['cve_id'].map(
        cve_exploit_id_dict)

    return mitre_dataframe


def readNVDData(FileName, numCVEs):
    severity = None
    obtainUserPrivelege = None
    obtainAllPrivelege = None
    impactScore = None
    exploitabilityScore = None
    impactScore = None
    exploitabilityScore = None
    vectorString = None
    accessVector = None
    accessComplexity = None
    authentication = None
    confidentialityImpact = None
    integrityImpact = None
    availabilityImpact = None
    baseScoreV2 = None
    baseScoreV3 = None
    vendors = None
    products = None
    publishedDate = None
    lastModifiedDate = None
    exploitIDs = []
    num_exploits = 0
    num_non_exploits = 0
    CVE_dataframe = pd.DataFrame(columns=['CVE', 'publishedDate', 'exploitID'])

    with open(FileName) as json_file:
        data = json.load(json_file)
        cnt = 0
        exploit_found = False
        for cve in data['CVE_Items']:
            cve_number_obtained = cve['cve']['CVE_data_meta']['ID']

            try:
                publishedDate = cve['publishedDate']
                # numPublishedDate = 1
            except:
                pass
            try:
                cpe_config = cve['configurations']
                # numCPEInfo = 1
            except:
                pass
            try:
                references = cve['cve']['references']['reference_data']
                exploitIDs = []
                for ref in references:
                    if ref['refsource'] == 'EXPLOIT-DB':
                        exploitIDs.append(ref['name'])
                        exploit_found = True
            except:
                pass
            CVE_dataframe = CVE_dataframe.append({'CVE': cve_number_obtained,
                                                  'publishedDate': publishedDate,
                                                  'exploitID': exploitIDs},
                                                 ignore_index=True)
            cnt += 1
            if cnt >= numCVEs:
                break

    if len(CVE_dataframe) < numCVEs:
        print("Lesser than required CVE's are found")
    return CVE_dataframe


# # read the exploit dates from the associated Bid
def readExploits(FileName, CVE_dataframe):
    unique_cves = CVE_dataframe['cve_id'].unique()
    cve_exploits = CVE_dataframe['exploitID']

    unique_exploits_ids = []

    for index, row in CVE_dataframe[CVE_dataframe['isExploited'] == 1].iterrows():
        unique_exploits_ids.extend(row['exploitID'])

    # making sure all unique values are appended to the list
    unique_exploits_ids = list(set(unique_exploits_ids))

    line_count = 0
    exploit_date_dictionary = {}

    with open(FileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        exploit_ids_found = 0
        for row in csv_reader:
            if line_count > 0:
                if row[0] in unique_exploits_ids:
                    exploit_date_dictionary[row[0]] = row[3]
                    exploit_ids_found += 1
            if exploit_ids_found >= len(unique_exploits_ids):
                break
            line_count += 1
    exploit_dates = []
    earliest_exploit_dates = []
    exploit_date_indexes = exploit_date_dictionary.keys()
    for index, row in CVE_dataframe.iterrows():
        exploit_date_cve_list = []
        exploit_ids = row['exploitID']
        if isinstance(exploit_ids, list) and len(exploit_ids) > 0:
            for exploit_id in exploit_ids:
                if exploit_id in exploit_date_indexes:
                    exploit_date_cve_list.append(pd.to_datetime(
                        exploit_date_dictionary[exploit_id]))

            exploit_dates.append(exploit_date_cve_list)
            exploit_date_cve_list.sort()
            if len(exploit_date_cve_list) > 0:
                earliest_exploit_dates.append(exploit_date_cve_list[0])
            else:
                earliest_exploit_dates.append(np.nan)
        else:
            exploit_dates.append([])
            earliest_exploit_dates.append(np.nan)

    CVE_dataframe['exploitDate'] = exploit_dates

    exploit_dates.sort()
    CVE_dataframe['earliestExploit'] = earliest_exploit_dates

    print("Exploit dates has been recorded to the dataframe...")

    return CVE_dataframe


# meant to read mitre publish dates from the file
def readMitreDate(Filename, CVE_Dataframe):
    mitre_dataframe = pd.read_csv(Filename)

    mitre_dates = []
    for index, row in CVE_Dataframe.iterrows():
        # print(mitre_dataframe['cve_id'])
        date = mitre_dataframe.loc[mitre_dataframe['cve_id']
                                   == row['CVE']]['mitre_date'].values[0]
        mitre_dates.append(date)
    CVE_Dataframe['mitre_dates'] = pd.to_datetime(mitre_dates)

    return CVE_Dataframe

# function to return teh CVEs associated with top vendors as determined by https://www.cvedetails.com/top-50-vendors.php
def getTopVendorCVEs():
    vendor_list = ['microsoft', 'oracle','ibm','google','apple','cisco', 'adobe', 'debian','redhat', 'linux']
    master_df = pd.read_csv('../data/cp3_cve_vendor_pair.csv')
    for vendor in vendor_list:
        vendor_df = master_df[master_df['vendor'].str.lower() == vendor]
        print(vendor_df)
        input()

def createDummyDataSplitPopulation(numCVEs, NVDFileName, MitreFileName, ExploitFileName):

    # create dummy cve-id
    # Get CVE names and dates from NVD
    CVE_Dataframe = readNVDData(NVDFileName, numCVEs)

    # Get start dates from mitre
    CVE_Dataframe = readMitreDate(MitreFileName, CVE_Dataframe)

    # create dummy exploit dates
    # get exploits from file and update the CVE_Dataframe

    CVE_Dataframe = readExploits(ExploitFileName, CVE_Dataframe)

    cve_list = []
    CVE_Dataframe['publishedDate'] = pd.to_datetime(
        CVE_Dataframe['publishedDate'], infer_datetime_format=True)

    i = 0

    data_list = []
    # streams to be included: exploit, nvd, twitter, reddit, github, news
    exploit_list = []
    nvd_list = []
    twitter_list = []
    reddit_list = []
    github_list = []
    news_list = []

    # for each cve create an entry
    for index, row in CVE_Dataframe.iterrows():

        exploit_list_cve = []
        nvd_list_cve = []
        twitter_list_cve = []
        reddit_list_cve = []
        github_list_cve = []
        news_list_cve = []
        # first create the event dataframes

        mitre_timestamp = row['mitre_dates']
        nvd_event_timestamp = row['publishedDate']

        # pick the earliest date
        exploit_dates = row['exploitDate']
        exploit_dates = exploit_dates.sort()

        if exploit_dates is None:
            exploit_dates = []
        # create social media cascades
        twitter_events = createSocialMediaData(
            poisson_param=0.5, startDate=row['mitre_dates'], numEvents=300)
        reddit_events = createSocialMediaData(
            poisson_param=0.25, startDate=row['mitre_dates'], numEvents=150)
        github_events = createSocialMediaData(
            poisson_param=0.05, startDate=row['mitre_dates'], numEvents=50)

        i += 1
        time_list = [mitre_timestamp] + exploit_dates +\
            twitter_events + reddit_events + github_events
        absoluteRightCensoringTime = max(time_list)

        if len(exploit_dates) > 0:
            exploit_event_timestamp = exploit_dates[0]
            exploit_list_cve.append(
                (exploit_event_timestamp-mitre_timestamp).seconds)

        exploit_list_cve.append(
            (absoluteRightCensoringTime-mitre_timestamp).seconds)

        nvd_list_cve.append((nvd_event_timestamp-mitre_timestamp).seconds)
        nvd_list_cve.append(
            (absoluteRightCensoringTime-mitre_timestamp).seconds)

        twitter_list_cve.extend(
            [(l-mitre_timestamp).seconds for l in twitter_events])
        twitter_list_cve.append(
            (absoluteRightCensoringTime-mitre_timestamp).seconds)

        reddit_list_cve.extend(
            [(l-mitre_timestamp).seconds for l in reddit_events])
        reddit_list_cve.append(
            (absoluteRightCensoringTime-mitre_timestamp).seconds)

        github_list_cve.extend(
            [(l-mitre_timestamp).seconds for l in github_events])
        github_list_cve.append(
            (absoluteRightCensoringTime-mitre_timestamp).seconds)

        exploit_list.append(np.array(exploit_list_cve))
        nvd_list.append(np.array(nvd_list_cve))
        twitter_list.append(np.array(twitter_list_cve))
        reddit_list.append(np.array(reddit_list_cve))
        github_list.append(np.array(github_list_cve))

    print("CVE Objects created: ", i)
    dataStream = DataStream()
    dataStream.initializeFromSimulatedSamples(
        [exploit_list, nvd_list, twitter_list, reddit_list, github_list], sourceNames=['exploit', 'nvd', 'twitter', 'reddit', 'github'])

    # save CVE list onto file
    # cve_container.save("../../data/testSaveFile.obj")
    # retreive it from file

    # check all of the values exist

    return dataStream


def createMasterDataset(Filename):
    # read csv file from mitre
    mitre_dataframe = pd.read_csv('../../data/mitre_cve_list.csv')

    # search for all exploit ids and nvd published dates
    mitre_dataframe = readNVDDataFromList(mitre_dataframe)

    # correlate all exploit ids to exploit dates

    ExploitFileName = "../../data/files_exploits.csv"

    mitre_dataframe = readExploits(ExploitFileName, mitre_dataframe)

    return mitre_dataframe


def readChicagoCrimeData(crime=None, year='*'):
    #########################################################################################################################
    # readChicagoCrimeData(): This function will read chicago crime data from the data folder and produce the realizations
    #                           in a format friendly to the point process class.
    # return:                   Realizations:      List of list of list structure. (repository standard for realizations)
    #                           maxNumEventsIdx:   Index id of realization with maximum elements
    #                           maxTime:           The right censoring time (realization time of the final event in chosen crime and year)
    # Arguments:
    # crime:  The type of crime whose realizations can be found. options are ['robbery','gambling','waepons']
    # year:   The year from which to extract data, default is '*' which refers to all the years in the dataset (2001-2019)
    #########################################################################################################################
    # This function is designed to retrieve all the realizations associated with the following data
    # filename = '../data/chicago_crime_robbery.csv'
    if crime is not None:
        filename = '../data/chicago_crime_'+crime+'.csv'

    df = pd.read_csv(filename)
    # select year if specified
    if year != '*':
        df = df[df['year'] == year]

    # get all the police beats in town
    unique_beats = df['beat'].unique()
    df['date'] = pd.to_datetime(df['date'])

    # convert the time to seconds and make it relative time,
    # relative to the lowest time in the entire dataset.
    minTime = df['date'].min()
    df['date'] = (df['date'] - minTime).dt.total_seconds()
    df['date'] = df['date'] / df['date'].max()
    Realizations = []

    maxNumEvents = 0
    maxTime = 0
    minMaxTime = np.inf
    r = 0
    maxNumEventsIdx = 0

    # Iterate through every beat in the dataset.
    # Each police beat will produce a single realization.
    # This additionally also allows us to treat every point in the
    # realization as coming from the same set of beat cops; indicating
    # similar characteristics when it comes to response to a crime.
    for beat in unique_beats:
        beat_df = df[df['beat'] == beat]
        # beat_df['date'] /= df['date'].max()
        realization = beat_df['date'].tolist()

        # checking to make sure the realization is not empty (It doesnt only have the right censoring time)
        if len(realization) > 1:
            # It is a list of lists structure to conform to the
            # structure primarily used by this reposistory. Each entry in
            # the list is associated with a single user, since we have just one user
            # for chicago data, realization is a list containing a single list.
            cascade = [realization]
            Realizations.append(cascade)
            if beat_df['date'].max() > maxTime:
                maxTime = beat_df['date'].max()
            if maxTime < minMaxTime:
                minMaxTime = maxTime
            # removing the first realization which occurs at t_1 = 0.0
            numEvents = len(realization)
            if numEvents > maxNumEvents:
                maxNumEvents = numEvents
                maxNumEventsIdx = r
            r += 1

    # Add the right-censoring time to each realization
    for realization in Realizations:
        realization[0].append(maxTime)

    print("Total number of realizations: ", len(Realizations))
    print("All Realizations of the chicago crime dataset have been generated...")
    print("The maximum time of all realizations are ", maxTime)

    return Realizations, maxNumEventsIdx, maxTime


def readExploitSocialMediaData():

    rt_df = pd.DataFrame({'year': [2018], 'month': [3], 'day': [31]})
    rightCensoringTime = pd.to_datetime(rt_df, unit='ns')

    # cp3_cve_df = pd.read_csv("../data/cp3_cve_info.csv")
    # cp3_cve_df['exploitDate'] = pd.to_datetime(
    #     cp3_cve_df['exploitDate'], format='%Y-%m-%d')
    # cp3_cve_df['publishedDate'] = pd.to_datetime(
    #     cp3_cve_df['publishedDate'], format='%Y-%m-%d')

    cp3_social_media_df = pd.read_csv("../data/cp3_social_media.csv")
    cp3_social_media_df['time'] = pd.to_datetime(
        cp3_social_media_df['time'], format='%Y-%m-%d %H:%M:%S')

    MasterDataset = pd.read_csv("../data/MasterDataset.csv")
    MasterDataset['mitre_date'] = pd.to_datetime(
        MasterDataset['mitre_date'], format='%Y-%m-%d')
    # MasterDataset['exploitDate'] = datetime.strptime(str(MasterDataset['exploitDate']),'%Y-%m-%d')
    MasterDataset['exploitDate'] = MasterDataset['exploitDate'].replace(
        '[]', pd.NaT)

    training_cves = pd.read_csv('../data/point_process_training.csv')['cve_id'].values
    test_cves = pd.read_csv('../data/point_process_test.csv')['cve_id'].values
    validation_cves = pd.read_csv('../data/point_process_validation.csv')['cve_id'].values

    df_train = pd.read_csv('../data/cve_train.csv')
    df_test = pd.read_csv('../data/cve_test.csv')
    df_validation = pd.read_csv('../data/cve_validation.csv')
    val_subset_indices = []
    train_subset_indices = []
    test_subset_indices = []
    
    for index,x in enumerate(df_validation['cve_id'].values):
        if x in validation_cves:
            val_subset_indices.append(index)

    for index,x in enumerate(df_train['cve_id'].values):
        if x in training_cves:
            train_subset_indices.append(index)

    for index,x in enumerate(df_test['cve_id'].values):
        if x in test_cves:
            test_subset_indices.append(index)
    
    unique_training_cves = df_train['cve_id'].values[train_subset_indices]
    unique_test_cves = df_test['cve_id'].values[test_subset_indices]
    unique_validation_cves = df_validation['cve_id'].values[val_subset_indices]
    np.save('training_cves_exploit_process.npy',unique_training_cves)
    np.save('test_cves_exploit_process.npy',unique_test_cves)
    np.save('validation_cves_exploit_process.npy',unique_validation_cves)

    
    
    def generateRealizations(cve_dataset,dataset):
        Realizations = []
        isExploited = np.zeros(len(cve_dataset))
        # print(len(cve_dataset))
        # input()
        for index, cve in enumerate(cve_dataset):
            realization = []
            print(index)
            # get exploit and publish date
            exploit_date = MasterDataset[MasterDataset['cve_id']
                                         == cve]['exploitDate']
            published_date = MasterDataset[MasterDataset['cve_id']
                                           == cve]['mitre_date']
            relativeRightCensoringTime = (
                rightCensoringTime - published_date.values[0]).dt.total_seconds().values[0] / (60.0*60.0)

            cve_soc_events = cp3_social_media_df[cp3_social_media_df['cve'] == cve]
            if pd.isna(exploit_date.values[0]):
                exploit_realization = np.array([relativeRightCensoringTime])
                isExploited[index] = 0.0
            else:
                exploit_date = [pd.to_datetime(exp_date.partition('(')[2].partition(')')[
                                               0].replace('\'', '')) for exp_date in exploit_date.values]
                if (min(exploit_date)-published_date.values[0]).total_seconds()/(60.0*60.0) > relativeRightCensoringTime:
                    exploit_realization = np.array([relativeRightCensoringTime])
                else:
                    exploit_realization = np.array(
                        [(min(exploit_date)-published_date.values[0]).total_seconds()/(60.0*60.0), relativeRightCensoringTime])
                    isExploited[index] = 1.0

            reddit_realization = []
            github_realization = []
            twitter_realization = []

            if len(cve_soc_events) > 1:
                for index, row in cve_soc_events.iterrows():
                    platform = row['platform']
                    if platform == 'reddit':
                        reddit_realization.append(
                            (row['time']-published_date.values[0]).total_seconds()/(60.0*60.0))
                    elif platform == 'twitter':
                        twitter_realization.append(
                            (row['time']-published_date.values[0]).total_seconds()/(60.0*60.0))
                    elif platform == 'github':
                        github_realization.append(
                            (row['time']-published_date.values[0]).total_seconds()/(60.0*60.0))
                    else:
                        print(platform)
                        raise Exception("Invalid platform encountered in data")
            reddit_realization.append(relativeRightCensoringTime)
            twitter_realization.append(relativeRightCensoringTime)
            github_realization.append(relativeRightCensoringTime)

            realization = [exploit_realization, sorted(np.array(github_realization)), sorted(np.array(
                reddit_realization)), sorted(np.array(twitter_realization))]
            Realizations.append(realization)
        
        np.save('../data/'+dataset+'_isExploited.npy',isExploited)
        np.save('../data/'+dataset+'_realizations.npy', Realizations)

        return Realizations

    training_realizations = generateRealizations(unique_training_cves, 'training')
    test_realizations = generateRealizations(unique_test_cves, 'test')
    validation_realizations = generateRealizations(unique_validation_cves, 'validation')
    print("Done generating realizations")

    return None


def readRealWorldExploitSocialMediaData():

    rt_df = pd.DataFrame({'year': [2018], 'month': [3], 'day': [31]})
    rightCensoringTime = pd.to_datetime(rt_df, unit='ns')

    cp3_cve_df = pd.read_csv("../data/cp3_cve_info.csv")
    cp3_cve_df['exploitDate'] = pd.to_datetime(
        cp3_cve_df['exploitDate'], format='%Y-%m-%d')
    cp3_cve_df['publishedDate'] = pd.to_datetime(
        cp3_cve_df['publishedDate'], format='%Y-%m-%d')

    cp3_social_media_df = pd.read_csv("../data/cp3_social_media.csv")
    cp3_social_media_df['time'] = pd.to_datetime(
        cp3_social_media_df['time'], format='%Y-%m-%d %H:%M:%S')

    MasterDataset = pd.read_csv("../data/MasterDataset.csv")
    MasterDataset['mitre_date'] = pd.to_datetime(
        MasterDataset['mitre_date'],unit='ns')

    MasterDataset['earliestRealWorldExploits'] = pd.to_datetime(
        MasterDataset['earliestRealWorldExploits'])
    cve_df = MasterDataset[MasterDataset['isRealWorldExploited'] > 0]
    cve_df  = cve_df[~cve_df['mitre_date'].isnull()]

    unique_cves = cve_df[cve_df['mitre_date'] <= rightCensoringTime.values[0]]['cve_id'].values

    # Only use the real world exploits for which we have social media data
    unique_cves = [x for x in unique_cves if int(x[4:8])>= 2016]
    print(unique_cves)
    print(len(unique_cves))

    social_media_cves = cp3_social_media_df['cve'].unique()

    CVE_classifier_training_cve = pd.read_csv('../data/cve_train.csv')['cve_id'].values
    unique_cves = [x for x in unique_cves if x not in CVE_classifier_training_cve] 

    print(unique_cves)
    print(len(unique_cves))
    print("social media out of them: ", len([x for x in unique_cves if x in social_media_cves]))

    rw_df = cve_df[cve_df['mitre_date'] <= rightCensoringTime.values[0]]
    rw_df = cve_df[cve_df['cve_id'].isin(unique_cves)]
    print(rw_df)
    rw_df.to_csv('../data/cve_rw_df.csv', index=False)
    input()
    def generateRealizations(cve_dataset):
        Realizations = []
        isExploited = np.zeros(len(cve_dataset))
        # print(len(cve_dataset))
        # input()
        for index, cve in enumerate(cve_dataset):
            realization = []
            print(index)

            # get exploit and publish date
            exploit_date = MasterDataset[MasterDataset['cve_id']
                                         == cve]['earliestRealWorldExploits']
            published_date = MasterDataset[MasterDataset['cve_id']
                                           == cve]['mitre_date']
            relativeRightCensoringTime = (
                rightCensoringTime - published_date.values[0]).dt.total_seconds().values[0] / (60.0*60.0)

            cve_soc_events = cp3_social_media_df[cp3_social_media_df['cve'] == cve]
            if pd.isna(exploit_date.values[0]):
                exploit_realization = np.array([relativeRightCensoringTime])
                isExploited[index] = 0.0
            else:

                exploit_realization = np.array(
                    [((exploit_date - published_date).dt.total_seconds()/(60.0*60.0)).values[0], relativeRightCensoringTime])
                isExploited[index] = 1.0

            reddit_realization = []
            github_realization = []
            twitter_realization = []

            if len(cve_soc_events) > 1:
                for index, row in cve_soc_events.iterrows():
                    platform = row['platform']
                    if platform == 'reddit':
                        reddit_realization.append(
                            (row['time']-published_date.values[0]).total_seconds()/(60.0*60.0))
                    elif platform == 'twitter':
                        twitter_realization.append(
                            (row['time']-published_date.values[0]).total_seconds()/(60.0*60.0))
                    elif platform == 'github':
                        github_realization.append(
                            (row['time']-published_date.values[0]).total_seconds()/(60.0*60.0))
                    else:
                        print(platform)
                        raise Exception("Invalid platform encountered in data")
            reddit_realization.append(relativeRightCensoringTime)
            twitter_realization.append(relativeRightCensoringTime)
            github_realization.append(relativeRightCensoringTime)

            realization = [exploit_realization, np.array(github_realization), np.array(
                reddit_realization), np.array(twitter_realization)]
            Realizations.append(realization)
        # np.save('../data/validation_isExploited.npy',isExploited)
        np.save('../data/real_world_realizations.npy', Realizations)

        return Realizations


    realizations = generateRealizations(unique_cves)
    print("Done")

    return None


def generateRealWorldExploitDataset():
    features = np.load('../data/real_world_feature_vectors.npy', allow_pickle=True)
    realizations = np.load('../data/real_world_realizations.npy', allow_pickle=True)

    assert len(features) ==len(realizations), "Number of realizations and feature vectors are inconsistent, they must be equal"
    isExploited = np.zeros(len(realizations))


    return features, realizations, isExploited

def generateSyntheticDataset():
    with open('../data/syn_test_data.json') as f:
        data_json = json.load(f)
        realizations, features = json2ListOfList(data_json, sourceNames=['exploit','github', 'reddit', 'twitter'])
 
    return  np.array(realizations), np.array(features)




def generateExploitSocialMediaDataset():
    
    training_features = np.load(
        '../data/training_feature_vectors.npy', allow_pickle=True)
    test_features = np.load(
        '../data/test_feature_vectors.npy', allow_pickle=True)
    validation_features = np.load(
        '../data/validation_feature_vectors.npy', allow_pickle=True)

    training_realizations = np.load(
        '../data/training_realizations.npy', allow_pickle=True)
    test_realizations = np.load(
        '../data/test_realizations.npy', allow_pickle=True)
    validation_realizations = np.load(
        '../data/validation_realizations.npy', allow_pickle=True)

    training_isExploited = np.load(
        '../data/training_isExploited.npy', allow_pickle=True)
    test_isExploited = np.load(
        '../data/test_isExploited.npy', allow_pickle=True)
    validation_isExploited = np.load(
        '../data/validation_isExploited.npy', allow_pickle=True)

    def getValidIndices(realizations):
        valid_indices = []
        for index, realization in enumerate(realizations):
            if not (len(realization[0]) > 1 and realization[0][0] < 0):
                valid_indices.append(index)
        return np.array(valid_indices)

    # get rid of realization where exploit times are negative (zero day exploits)

    training_valid_indices = getValidIndices(training_realizations)
    test_valid_indices = getValidIndices(test_realizations)
    validation_valid_indices = getValidIndices(validation_realizations)
    
    training_features = training_features[training_valid_indices]
    training_realizations = training_realizations[training_valid_indices]
    training_isExploited = training_isExploited[training_valid_indices]

    test_features = test_features[test_valid_indices]
    test_realizations = test_realizations[test_valid_indices]
    test_isExploited = test_isExploited[test_valid_indices]

    validation_features = validation_features[validation_valid_indices]
    validation_realizations = validation_realizations[validation_valid_indices]
    validation_isExploited = validation_isExploited[validation_valid_indices]

    return training_features, training_realizations, training_isExploited, test_features, test_realizations, test_isExploited,  validation_features, validation_realizations, validation_isExploited


def generateRealWorldData():
    master_df = pd.read_csv('../data/MasterDataset.csv')
    unique_cves = master_df['cve_id'].values
    import xml.etree.ElementTree as ET
    tree = ET.parse('../data/symantec.xml')
    root = tree.getroot()
    from xml.dom import minidom
    xmldoc = minidom.parse('../data/symantec.xml')
    itemlist = xmldoc.getElementsByTagName('item')
    real_world_dict = {}
    real_world_data_dict = {}
    real_world_earliest_exploit = {}
    cnt = 0
    total_found = 0
    master_df['isRealWorldExploited'] = np.zeros(len(master_df))
    hasPoc = 0
    from datetime import datetime
    for item in itemlist:
        print(cnt)
        cnt += 1
        title = item.getElementsByTagName("title")[0].firstChild.data
        if "CVE" in title:
            for cve in unique_cves:
                if cve in title:
                    # print(master_df[master_df['cve_id'] == cve]
                    #       ['isExploited'].values[0])
                    pubDate = item.getElementsByTagName(
                        "pubDate")[0].firstChild
                    if pubDate is not None:
                        pubDate = pubDate.data
                        # print(pubDate.data)
                        if master_df[master_df['cve_id'] == cve]['isExploited'].values[0] == 1.0:
                            hasPoc += 1
                        if cve in real_world_dict:
                            real_world_dict[cve] += 1.0
                            real_world_data_dict[cve].append(
                                pubDate)
                            arr = real_world_data_dict[cve]
                            arr = [
                                x.replace('Nov', 'November') for x in arr]
                            arr = [
                                x.replace('Oct', 'October') for x in arr]
                            arr = [
                                x.replace('Sept', 'September') for x in arr]
                            arr = [
                                x.replace('Dec', 'December') for x in arr]
                            real_world_earliest_exploit[cve] = min(
                                [datetime.strptime(date, '%B %d, %Y') for date in arr])
                        else:
                            real_world_dict[cve] = 1.0
                            real_world_data_dict[cve] = [pubDate]
                            real_world_earliest_exploit[cve] = pubDate
                        total_found += 1
                        break

    print("Total CVE for which real world exploits exist: ", total_found)
    print("Total CVE which also contain PoC exploits", hasPoc)
    master_df['isRealWorldExploited'] = master_df['cve_id'].map(
        real_world_dict)
    master_df['realWorldExploits'] = master_df['cve_id'].map(
        real_world_data_dict)
    master_df['earliestRealWorldExploits'] = master_df['cve_id'].map(
        real_world_earliest_exploit)
    master_df.to_csv('../data/MasterDataset.csv')


def generateMetasploitRWData():
    metasploit_df = pd.read_csv("../data/Metasploit_RW.csv")

    metasploit_df = metasploit_df[metasploit_df['Module Type']
                                  != 'Post-Exploitation']
    master_df = pd.read_csv("../data/MasterDataset.csv")
    for index, row in metasploit_df.iterrows():
        print(index, "of", len(metasploit_df))
        if (not str(row['CVE']) == "nan"):
            cve_id_list = row['CVE'].split(',')
            cve_id_list = ['CVE-' + str(cve) for cve in cve_id_list]
            disclosure_date = row['Disclosure Date']

            for cve_id in cve_id_list:
                master_df.loc[master_df['cve_id'] ==
                              cve_id]['isExploited'] = master_df[master_df['cve_id'] == cve_id]['isExploited'] + 1
                currentEarliestDate = pd.to_datetime(
                    master_df[master_df['cve_id'] == cve_id]['earliestRealWorldExploits'])

                exploit_list = master_df[master_df['cve_id']
                                         == cve_id]['realWorldExploits']
                if np.sum(exploit_list.isna()) or len(exploit_list) == 0:
                    exploit_list = []
                else:
                    import ast
                    exploit_list = ast.literal_eval(exploit_list.values[0])
                exploit_list.append(disclosure_date)
                exploit_list = [str(date).replace(',', '').replace(' ', '/')
                                for date in exploit_list]

                # exploit_list = [datetime.strptime(
                #     str(date), '%B/%d/%Y') for date in exploit_list]
                exploit_list_new = exploit_list.copy()
                for index, date in enumerate(exploit_list):
                    if not str(date) == "nan":
                        try:
                            exploit_list_new[index] = datetime.strptime(
                                str(date), '%B/%d/%Y')
                        except:
                            exploit_list_new[index] = datetime.strptime(
                                str(date), '%b/%d/%Y')
                if 'nan' in exploit_list:
                    exploit_list.remove('nan')

                if len(exploit_list) > 0:
                    master_df.loc[master_df['cve_id'] == cve_id, 'isRealWorldExploited'] = len(
                        exploit_list)
                    master_df.loc[master_df['cve_id'] ==
                                  cve_id, 'realWorldExploits'] = str(exploit_list)
                    master_df.loc[master_df['cve_id'] == cve_id, 'earliestRealWorldExploits'] = min(
                        exploit_list)
    master_df.to_csv('../data/MasterDataset.csv', index=False)
    print(currentEarliestDate)


    
    
    
    

def main():
    # dataStream = createDummyDataSplitPopulation(numCVEs=50,
    #                                             NVDFileName="../../data/nvdcve-1.1-2005.json",
    #                                             MitreFileName="../../data/mitre_cve_list.csv",
    #                                             ExploitFileName="../../data/files_exploits.csv")

    # masterDf = createMasterDataset("CVE_NLP_dataset.csv")
    # masterDf.to_csv("../../data/MasterDataset.csv")

    # master_df = pd.read_csv("../data/MasterDataset.csv")
    # Realizations = readExploitSocialMediaData()
    # generateRealWorldData()
    # generateMetasploitRWData()
    # readRealWorldExploitSocialMediaData()
    # readExploitSocialMediaData() 
    # generatePointProcessDataset()
    # getTopVendorCVEs()
    # generateSplitPopulationSocialMediaDataset()
    # generateNeuralDiffusionPaperDataset()
    # constructGraphs()
    # generateSplitsForKDDPaper()
    # generateDatasetsForForestPaper(start_size=20, scenario="github")
    # generateDatasetsForDeepDiffusePaper()
    # generateDatasetForDeepHawkesPaper()
    # readExploitSocialMediaData()
    # convertTwitterLinkSharingDataset()
    generateSyntheticDataset()
    

if __name__ == "__main__":
    main()