 def generate_baseline_results(self, realizations, 
                                    feature_vectors, 
                                    dataset_name, 
                                    filename, 
                                    tc_list, 
                                    delta_list,
                                    training_index_set_filename,
                                    training_realizations):
        """[summary]

        Args:
            realizations ([type]): [description]
            feature_vectors ([type]): [description
            """
        assert filename is not None, "filename of file containing index set is missing"
        realization_index_set = np.load(
            filename, allow_pickle=True).item()


        if training_index_set_filename is not None:
            training_index_set = np.load(
                training_index_set_filename, allow_pickle=True).item()

        # prin      realization_index_set)
        # input()
        cnt = 0
        model_predictions = np.array([])
        new_index_dict = {}


        # generate sub samples tuple dictioanry
        for key, value in realization_index_set.items():
            (tc, delta_t) = key
            if tc in tc_list and delta_t in delta_list:
                new_index_dict[key] = value

        # print(len(new_index_dict))
        # input()
        ground_truth = np.array([])

        all_tc = []
        all_delta = []
        for key, value in new_index_dict.items():
            print("Finished {0} tuples out of {1} tuples".format(
                cnt, len(new_index_dict)))
            cnt += 1
            # if cnt == 5:
            #     break
            (tc, delta_t) = key
            all_tc.append(tc)
            all_delta.append(delta_t)

        all_tc = sorted(list(set(all_tc)))
        all_delta = sorted(list(set(all_delta)))

        matrix = np.ones((len(all_tc), len(all_delta)))


        fp_matrix = np.empty((len(all_tc), len(all_delta)))
        fp_matrix[:] = np.nan

        fn_matrix = np.empty((len(all_tc), len(all_delta)))
        fn_matrix[:] = np.nan
        

        cnt = 0
        for key, value in new_index_dict.items():
            print("Finished {0} tuples out of {1} tuples".format(
                cnt, len(new_index_dict)))
            (tc, delta_t) = key
            cnt += 1

            if tc in all_tc and delta_t in all_delta:
                selected_training_realizations = training_realizations[training_index_set[(tc,delta_t)]]
                tc *= 24.0
                delta_t *= 24.0
                realization_indices = [x for x in list(value)]

                selected_realizations = realizations[realization_indices]
                
                i=0
                illegal_indices = []
                for index, x in enumerate(selected_realizations):
                    if len(x[0]) == 2:
                        if x[0][0] <= tc or x[0][0] < 0:
                            illegal_indices.append(index)
                    i +=1
                feature_vectors = [x for i,x in enumerate(feature_vectors) if i not in illegal_indices]
                selected_realizations = [x for i,x in enumerate(selected_realizations) if i not in illegal_indices]
      
                # naive model
                num_exploited = 0
                gt_E = np.zeros(len(selected_realizations))
                es_E = None
                

                for index,realization in enumerate(selected_realizations):
                    if len(realization[0]) > 1 and realization[0][0] < tc + delta_t:
                        gt_E[index] = 1
                
                
                for index,realization in enumerate(selected_training_realizations):
                    if len(realization[0]) > 1:
                        num_exploited +=1
                
                if len(selected_training_realizations) - num_exploited > num_exploited:
                    es_E = np.zeros(len(selected_realizations))
                else:
                    es_E = np.ones(len(selected_realizations))


                results_soc = pd.DataFrame(data={'true_E': gt_E, 'estimated_E': es_E, 'probability': np.ones(len(selected_realizations))})

                cm = confusion_matrix(results_soc.true_E,
                                      results_soc.estimated_E)
                model_predictions = np.append(
                    model_predictions, np.array(results_soc.estimated_E))
                ground_truth = np.append(
                    ground_truth, np.array(results_soc.true_E))
                Accuracy = sum(results_soc.true_E ==
                               results_soc.estimated_E) / len(results_soc)
                matrix[all_tc.index(
                    tc/24.0)][all_delta.index(delta_t/24.0)] = Accuracy

                # print(np.sum(results_soc.estimated_E))
                # print(len(results_soc.estimated_E)-np.sum(results_soc.estimated_E))
                # input()
                if len(cm) > 1:
                    print(cm)
                    print(Accuracy)

                    if tc/24.0 == 21.0 and delta_t/24.0 == 31.0:
                        print(tc/24)
                        print(delta_t/24)
                        print(np.sum(gt_E))
                        # print(realization_indices)
                        input()

                    # false positive rate
                    if cm[0][0] + cm[0][1] > 0: 
                        fp_matrix[all_tc.index(tc/24.0)][all_delta.index(delta_t/24.0)] = cm[0][1]/(cm[0][1] + cm[0][0])
                    
                    # miss rate
                    if cm[1][0] + cm[1][1] > 0:
                        fn_matrix[all_tc.index(
                            tc/24.0)][all_delta.index(delta_t/24.0)] = cm[1][0]/(cm[1][0] + cm[1][1])

            # print(Accuracy)



        import itertools
        plt.imshow(matrix, interpolation='none', cmap='jet',
                   aspect='auto', origin='lower')
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, "{:,}".format(round(matrix[i, j], 2)),
                    horizontalalignment="center",
                    color="black") 
        plt.xticks(np.arange(0, len(delta_list)),
                   np.array(delta_list), rotation=45)
        plt.yticks(np.arange(0, len(tc_list)), np.array(tc_list))
        plt.xlabel("Prediction window, $\delta_{t}$ (days)")
        plt.ylabel("Prediction start time, $t_{c}$ (days)")
        plt.title("Accuracy on test dataset: naive model")
        plt.colorbar()
        plt.savefig("../data/prediction_results/Prediction_accuracy_naive_"+dataset_name+".png", dpi=600)
        with open("../data/prediction_results/Prediction_accuracy_naive_"+dataset_name+".npy", "wb") as f:
            np.save(f,matrix)

        plt.show()

        plt.imshow(fn_matrix, interpolation='none',
                   cmap='jet', aspect='auto', origin='lower')
        for i, j in itertools.product(range(fn_matrix.shape[0]), range(fn_matrix.shape[1])):
            plt.text(j, i, "{:,}".format(round(fn_matrix[i, j], 2)),
                    horizontalalignment="center",
                    color="black") 
        
        plt.xticks(np.arange(0, len(delta_list)),
                   np.array(delta_list), rotation=45)
        plt.yticks(np.arange(0, len(tc_list)), np.array(tc_list))
        plt.xlabel("Prediction window, $\delta_{t}$ (days)")
        plt.ylabel("Prediction start time, $t_{c}$ (days)")
        plt.title("miss rate on validation dataset: naive model")
        plt.clim(0.0,1.0)
        plt.colorbar()
        plt.savefig("../data/prediction_results/Prediction_fn_rate_naive_"+dataset_name+".png", dpi=600)
        with open("../data/prediction_results/Prediction_fn_rate_naive_"+dataset_name+".npy", "wb") as f:
            np.save(f,fn_matrix)
        plt.show()

        plt.imshow(fp_matrix, interpolation='none',
                   cmap='jet', aspect='auto', origin='lower')
        for i, j in itertools.product(range(fp_matrix.shape[0]), range(fp_matrix.shape[1])):
            plt.text(j, i, "{:,}".format(round(fp_matrix[i, j], 2)),
                    horizontalalignment="center",
                    color="black") 
        plt.xticks(np.arange(0, len(delta_list)),
                   np.array(delta_list), rotation=45)
        plt.yticks(np.arange(0, len(tc_list)), np.array(tc_list))
        plt.xlabel("Prediction window, $\delta_{t}$ (days)")
        plt.ylabel("Prediction start time, $t_{c}$ (days)")
        plt.title("False alarm rate on validation dataset: naive model")
        plt.clim(0.0,1.0)
        plt.colorbar()
        plt.savefig("../data/prediction_results/Prediction_fp_rate_naive_"+dataset_name+".png", dpi=600)
        with open("../data/prediction_results/Prediction_fp_rate_naive_"+dataset_name+".npy", "wb") as f:
            np.save(f,fp_matrix)
        plt.show()





    def predict_test_data(self, 
                            realizations, 
                            feature_vectors, 
                            filename, 
                            scenario_name, 
                            tc_list, 
                            delta_list,
                            consider_social_media_realizations_only):
        """[summary]

        Args:
            realizations ([type]): [description]
            feature_vectors ([type]): [description]
        """

        assert filename is not None, "Filename of file containing index test set is missing"
        realization_index_set = np.load(
            filename, allow_pickle=True).item()
            

        cnt = 0
        model_predictions = np.array([])
        new_index_dict = {}
        roc_curve_points = []

        # generate sub samples tuple dictioanry
        for key, value in realization_index_set.items():
            (tc, delta_t) = key
            if tc in tc_list and delta_t in delta_list:
                new_index_dict[key] = value

        ground_truth = np.array([])

        all_tc = []
        all_delta = []
        for key, value in new_index_dict.items():
            print("Finished {0} tuples out of {1} tuples".format(
                cnt, len(new_index_dict)))
            cnt += 1
            # if cnt == 5:
            #     break
            (tc, delta_t) = key
            all_tc.append(tc)
            all_delta.append(delta_t)

        all_tc = sorted(list(set(all_tc)))
        all_delta = sorted(list(set(all_delta)))

        matrix = np.ones((len(all_tc), len(all_delta)))


        fp_matrix = np.empty((len(all_tc), len(all_delta)))
        fp_matrix[:] = np.nan

        fn_matrix = np.empty((len(all_tc), len(all_delta)))
        fn_matrix[:] = np.nan
        

        cnt = 0
        for key, value in new_index_dict.items():
            print("Finished {0} tuples out of {1} tuples".format(
                cnt, len(new_index_dict)))
            (tc, delta_t) = key
            cnt += 1

            if tc in all_tc and delta_t in all_delta:
                tc *= 24.0
                delta_t *= 24.0
                # print(realizations[0])
                # print(feature_vectors[0])
                # for index,feature in enumerate(feature_vectors):
                #     if feature[0].round(4) == 0.9834 and feature[1].round(4) == 0.0166:
                #         print(realizations[index])
                
                realization_indices = [x for x in list(value)]

                selected_realizations = realizations[realization_indices]
                selected_feature_vectors = feature_vectors[realization_indices]
                
                if consider_social_media_realizations_only:
                    social_media_indices = []
                    for index, realization in enumerate(selected_realizations):
                        if len(realization[1]) > 1 or len(realization[2]) > 1 or len(realization[3]) > 1:
                            social_media_indices.append(index)

                    selected_realizations = selected_realizations[social_media_indices]
                    selected_feature_vectors = selected_feature_vectors[social_media_indices]

                # selected_realizations = []
                # selected_feature_vectors = []
                # for index, realization in enumerate(realizations):
                #     if index in realization_indices:
                #         selected_realizations.append(realization)
                #         selected_feature_vectors.append(feature_vectors[index])
                
                results_soc, roc_list = self.predict(tc, delta_t, selected_realizations, selected_feature_vectors)
                roc_curve_points.append(roc_list)
                results_soc = results_soc.dropna()
  
                cm = confusion_matrix(results_soc.true_E,
                                      results_soc.estimated_E)
                model_predictions = np.append(
                    model_predictions, np.array(results_soc.estimated_E))
                ground_truth = np.append(
                    ground_truth, np.array(results_soc.true_E))
                Accuracy = sum(results_soc.true_E ==
                               results_soc.estimated_E) / len(results_soc)
                matrix[all_tc.index(
                    tc/24.0)][all_delta.index(delta_t/24.0)] = Accuracy

                # print(np.sum(results_soc.estimated_E))
                # print(len(results_soc.estimated_E)-np.sum(results_soc.estimated_E))
                # input()
                if len(cm) > 1:
                    print(cm)
                    print(Accuracy)


                    # atleast one non-exploit
                    # recall
                    if cm[0][0] + cm[0][1] > 0: 
                        fp_matrix[all_tc.index(tc/24.0)][all_delta.index(delta_t/24.0)] = cm[1][1]/(cm[1][1] + cm[1][0])
                    
                    # precision
                    if cm[1][0] + cm[1][1] > 0:
                        fn_matrix[all_tc.index(
                            tc/24.0)][all_delta.index(delta_t/24.0)] = cm[1][1]/(cm[1][1] + cm[0][1])

        if not os.path.exists("../data/prediction_results"+scenario_name):
            os.makedirs("../data/prediction_results"+scenario_name)
    


        plt.imshow(matrix, interpolation='none', cmap='jet',
                   aspect='auto', origin='lower')
        plt.xticks(np.arange(0, len(delta_list)),
                   np.array(delta_list), rotation=45)
        plt.yticks(np.arange(0, len(tc_list)), np.array(tc_list))
        plt.xlabel("Prediction window, $\delta_{t}$ (days)")
        plt.ylabel("Prediction start time, $t_{c}$ (days)")
        plt.title("Accuracy on "+ scenario_name + "dataset")
        plt.colorbar()
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, "{:,}".format(round(matrix[i, j], 2)),
                    horizontalalignment="center",
                    color="black") 
        plt.savefig("../data/prediction_results"+scenario_name+"/Prediction_accuracy_"+scenario_name+".png", dpi=600)
        with open("../data/prediction_results"+scenario_name+"/Prediction_accuracy_"+scenario_name+".npy", "wb") as f:
            np.save(f,matrix)
        plt.figure()

        plt.imshow(fn_matrix, interpolation='none',
                   cmap='jet', aspect='auto', origin='lower')
        plt.xticks(np.arange(0, len(delta_list)),
                   np.array(delta_list), rotation=45)
        plt.yticks(np.arange(0, len(tc_list)), np.array(tc_list))
        plt.xlabel("Prediction window, $\delta_{t}$ (days)")
        plt.ylabel("Prediction start time, $t_{c}$ (days)")
        plt.title("Precision on " + scenario_name + " dataset")
        plt.clim(0.0,1.0)
        plt.colorbar()
        for i, j in itertools.product(range(fn_matrix.shape[0]), range(fn_matrix.shape[1])):
            plt.text(j, i, "{:,}".format(round(fn_matrix[i, j], 2)),
                    horizontalalignment="center",
                    color="black")
        plt.savefig("../data/prediction_results"+scenario_name+"/Prediction_fn_rate_"+scenario_name+".png", dpi=600)
        
        with open("../data/prediction_results"+scenario_name+"/Prediction_fn_rate_"+scenario_name+".npy", "wb") as f:
            np.save(f,fn_matrix)
        plt.figure()

        plt.imshow(fp_matrix, interpolation='none',
                   cmap='jet', aspect='auto', origin='lower')
        plt.xticks(np.arange(0, len(delta_list)),
                   np.array(delta_list), rotation=45)
        plt.yticks(np.arange(0, len(tc_list)), np.array(tc_list))
        plt.xlabel("Prediction window, $\delta_{t}$ (days)")
        plt.ylabel("Prediction start time, $t_{c}$ (days)")
        plt.title("Recall on " + scenario_name + " dataset")
        plt.clim(0.0,1.0)
        plt.colorbar()
        for i, j in itertools.product(range(fp_matrix.shape[0]), range(fp_matrix.shape[1])):
            plt.text(j, i, "{:,}".format(round(fp_matrix[i, j], 2)),
                    horizontalalignment="center",
                    color="black") 
        
        plt.savefig("../data/prediction_results"+scenario_name+"/Prediction_fp_rate_"+scenario_name+".png", dpi=600)
        with open("../data/prediction_results"+scenario_name+"/Prediction_fn_rate_"+scenario_name+".npy", "wb") as f:
            np.save(f,fp_matrix)
        plt.figure()

        return roc_curve_points
def predict_single(self, tc, delta_t, realization, 
    feature_vector, theta, use_gt_prior):
        """[summary]

        Args:
            tc ([type]): [description]
            delta_t ([type]): [description]
            realization ([type]): [description]

        Raises:
            Exception: [description]

        Returns:
            [type]: [description]
        """
        if realization[0][-1] >= tc:
            # cut
            new_realization = [sorted([x for x in source_realization if x <= tc])
                               for source_realization in realization]
        else:
            new_realization = [sorted(x[:-1]) for x in realization]

        if len(new_realization[0]) == 0:
            if len(realization[0]) == 1 :
                gt = 0
            elif len(realization[0]) == 2 :
                if realization[0][0] < tc + delta_t:
                    gt = 1
                else:
                    gt = 0

            Psi_vector_tc = np.zeros(len(self.sourceNames))
            Psi_vector_delta_t = np.zeros(len(self.sourceNames))
            Psi_vector_tc[0] = self.mk[0].psi(tc)
            Psi_vector_delta_t[0] = self.mk[0].psi(tc + delta_t)
            for source_index in range(1, len(self.sourceNames)):
                source_events = new_realization[source_index]
                for se in source_events:
                    Psi_vector_tc[source_index] += self.mk[source_index].psi(
                        tc - se)
                    Psi_vector_delta_t[source_index] += self.mk[source_index].psi(
                        tc + delta_t - se)

            survive_tc = np.exp(-np.dot(self.alpha, Psi_vector_tc))
            survive_delta_t = np.exp(-np.dot(self.alpha, Psi_vector_delta_t))
            survive_prior = np.exp(-np.dot(self.w_tilde, feature_vector))
        
            if use_gt_prior:
                if len(realization[0]) > 1:
                    survive_prior = 0.0
                else:
                    survive_prior = np.inf
            if survive_prior != np.inf:
                prob = (survive_tc - survive_delta_t) / \
                    (survive_tc + survive_prior)
                # prob = (1/(1+ np.exp(-np.dot(self.w_tilde, feature_vector))))*(survive_tc - survive_delta_t)
            else:
                prob = 0.0
            es = 1 if prob > theta else 0
        else:
            #  it contains an exploit
            gt = np.nan
            es = np.nan
            prob = np.nan
            # raise Exception("Such a predictive scenario should never be encountered")

        return gt, es, prob

    def predict(self, tc, delta_t, MTPPData, feature_vectors):
        """[summary]

        Args:
            tc ([type]): [description]
            delta_t ([type]): [description]
            MTPPData ([type]): [description]
            feature_vectors ([type]): [description]

        Returns:
            [type]: [description]
        """
        # create ground truth labels
        gt_E = []
        es_E = []
        Prob = []

        
        theta_list = np.arange(0.0, 1.0, 0.001)
        best_results = None
        far = []
        hr = []
        precision = []
        recall = []
        print(len(theta_list))
        gt_E = []
        es_E = []
        Prob = []
        for idx, realization in enumerate(MTPPData):
            gt, es, prob = self.predict_single(
                tc, delta_t, realization,feature_vectors[idx], 
                0.5, False)
            gt_E.append(gt)
            es_E.append(es)
            Prob.append(prob)
        results = pd.DataFrame(
            data={'true_E': gt_E, 'estimated_E': es_E, 'probability': Prob})
        results = results.dropna()
        prob_list = results.probability
        for index,theta in enumerate(theta_list):
            print(index, end='\r')
        
            es = [1 if prob >= theta else 0 for prob in prob_list]

            cm = confusion_matrix(results.true_E,
                                        es)

            if len(cm) > 1:
                far_tuple = cm[0][1]/(cm[0][1] + cm[0][0])
                mr_tuple = cm[1][0]/(cm[1][0] + cm[1][1])
                precision_tuple = cm[1][1]/(cm[1][1] + cm[0][1])
                recall_tuple = cm[1][1]/(cm[1][1] + cm[1][0])
                precision.append(precision_tuple)
                recall.append(recall_tuple)

         best_theta = 0.15 
        es = [1 if prob >= best_theta else 0 for prob in prob_list]
        results = pd.DataFrame(
            data={'true_E': results.true_E, 'estimated_E': es, 'probability': prob_list})
        # results = results.dropna()
        
        cm = confusion_matrix(results.true_E,
                                    es)
        return results, [(x,y) for x,y in zip(recall, precision)]
