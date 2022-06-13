
# author Linna Fan

# judge the new devices according to confidence score 
from __future__ import absolute_import, division, print_function
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf 
import numpy as np 
import pandas as pd 
from cclp.data_utils.compute_utils import get_max_probs, get_threshold, get_delta_s, get_new_type_num_from_cluster,compute_probs_dist_through_logits,compute_KS, plot_cdf, relabel_data
from configs import local_dirs_to_data
import json
# import os
import math 
from cclp.routines import train
import time
from configs.iot.cfg_iot import out_path

theta_percentile = 0.99
theta1 = 0.1
theta2 = 9
s0 = 5
instances_culmulated = 24 * 2 * 2 # 2days, if we don't need set the culmulation time of new type devices instance, we can set it to -1
# instances_culmulated = -1
JUDGE_LEN = 3 # N_min
REJUDGE_LEN = 2 # JUDGE_LEN + REJUDGE_LEN  = N_max
store_analysis_iot_niot = True
from cclp.data_utils.log_utils import init_log
from configs.iot.cfg_iot import out_path



time0 = time.time()
class New_devices(object):
    def __init__ (self,sessionNModelFlags, trainerFlags, logger):
        self.logger = logger
        self.sessionNModelFlags = sessionNModelFlags
        self.trainerFlags = trainerFlags
        self.use_cnn_layer_for_cluster = sessionNModelFlags['use_cnn_layer_for_cluster']
        self._l2_reg_classif = sessionNModelFlags["l2_reg_classif"]
        self.new_devices_list = sessionNModelFlags['new_devices_list']
        self.max_iters = sessionNModelFlags['max_iters']
        self.new_devices_postfix = '_'.join(str(item) for item in self.new_devices_list)
        self.seed = sessionNModelFlags['seed']
        self.pathToDataFolder = local_dirs_to_data.iot
        self.record_data_index = sessionNModelFlags['record_data_index']
        self.merge_cnn_layer_method = sessionNModelFlags['merge_cnn_layer_methods']
        # self.main_model_path = './output/iot/trainTf/'
        self.main_model_path = out_path + 'trainTf/'
        self.old_devices_data = pd.read_csv(self.pathToDataFolder +self.new_devices_postfix +'/old_devices_train_data.csv'.format(self.new_devices_postfix))
        
        self.column_names = list(self.old_devices_data.columns)
        self.num_classes = int(np.max(self.old_devices_data['label'])) # only iot
        self.old_devices_data.drop(self.old_devices_data[self.old_devices_data['label']==np.max(self.old_devices_data['label'])].index,inplace=True)       
        self.drop_label = self.num_classes
        self.val_data = self.get_val_data_old_new()
        
        self.new_devices_test_data_pd = pd.read_csv(self.pathToDataFolder+self.new_devices_postfix+'/new_devices_test_data.csv') 
        
        
        # self.val_data.drop(self.val_data[self.val_data['label']==self.drop_label].index,inplace=True)
        self.val_batch_size = sessionNModelFlags["eval_batch_size"]
        self.val_new_devices_flag = None
        self.seed = sessionNModelFlags["seed"]
        self.inf_output_dir = out_path + "test_new_devices_multi_instance"
        if not os.path.exists(self.inf_output_dir):
            os.makedirs(self.inf_output_dir)
        self.new_old_label_dict = self.get_new_old_label_dict()
        ##self._find_theta()

        # self.observe_max_probs_of_devices()
        self.filtered_iot_data_new = self._judge_devices(2)
        self.threshold = -1
        self.new_devices_num = self.get_new_type_num(1) # get the new type number, 1 is best(our method)
        if self.new_devices_num == -1:
            return
        self.old_label_new_label_dict = {}
        self.adjust_model(2) # retrain the model

    # get data of old test and new train
    def get_val_data_old_new(self):
        
        test_old = pd.read_csv(self.pathToDataFolder+self.new_devices_postfix+'/old_devices_test_data.csv')
        train_new = pd.read_csv(self.pathToDataFolder+self.new_devices_postfix+'/new_devices_train_data.csv')
        
        data = pd.concat([test_old,train_new],axis=0)
        return data

    def get_new_old_label_dict(self):        
        path = self.pathToDataFolder + self.new_devices_postfix + '/old_new_device_idx_dict.json'
        
        new_old_label_dict = {}
        with open(path) as f:
            old_new_label_dict = json.load(f)
        for key,value in old_new_label_dict.items():
            new_old_label_dict[int(value)] = int(key)
        return new_old_label_dict

    def observe_max_probs_of_devices(self):
        from cclp.data_utils import compute_utils
        threshold = 0.8
        self.val_samples = self.val_data[:,1:]
        path = self.main_model_path + self.new_devices_postfix
        with open(path + '/model_name','r') as f:
            model_name = f.readline().strip('\n')

        with tf.Session() as sess:
            saver = tf.compat.v1.train.import_meta_graph(self.main_model_path + self.new_devices_postfix + '/' + model_name + '.meta')
            saver.restore(sess, tf.train.latest_checkpoint(self.main_model_path + self.new_devices_postfix))
            graph = tf.compat.v1.get_default_graph()
            # need input and pred value
            # model = graph.get_operation_by_name('model.forward_')
            input_placeholder = sess.graph.get_tensor_by_name('eval_in:0')
            prediction = sess.graph.get_tensor_by_name('compute_logits_name_scope/fully_connected/BiasAdd:0')
            eval_pred_logits = None # list of embeddings for each val batch: [ [batchSize, 10], .... [bs, 10] ]       
                      
            for i in range(0, len(self.val_samples), self.val_batch_size):         
                pred_logits = sess.run(prediction,feed_dict = {input_placeholder:self.val_samples[i:i+self.val_batch_size]})
                val_new_device_flags = compute_utils.get_new_devices_flag(pred_logits,self.val_labels[i:i+self.val_batch_size],threshold)
                if (eval_pred_logits is None):
                    eval_pred_logits = pred_logits                   
                    self.val_new_devices_flag = val_new_device_flags         
                else:    
                    eval_pred_logits = np.concatenate([eval_pred_logits,pred_logits],axis = 0)
                    self.val_new_devices_flag = np.concatenate([self.val_new_devices_flag,val_new_device_flags],axis = 0)
            
        eval_prob_distri = compute_utils.eval_probability_distribution(eval_pred_logits)
        compute_utils.observe_prob_distri(eval_prob_distri,self.val_labels,self.new_devices_list,self.prob_fig,self.old_new_device_label_dict)
        print("finish getting prob distribution of new devices") 


    # find theta according to old data and store it to a json file named theta.json
    def _find_theta(self):
        label_max_prob_dict = {label:[] for label in range(self.num_classes)}
        with tf.Session() as sess:
            model_folder_str = '_'.join(str(item) for item in self.new_devices_list)
            f_model_name = open(self.main_model_path + 'model_name','r')
            model_name = f_model_name.readline().rstrip('\n')
            f_model_name.close()
            # saver = tf.compat.v1.train.import_meta_graph(self.main_model_path +'/'+ model_folder_str + 'model-2000-2020.08.28.-13.27.19-2000.meta')
            saver = tf.compat.v1.train.import_meta_graph(self.main_model_path + model_folder_str + '/' + model_name + '.meta')
            saver.restore(sess,tf.train.latest_checkpoint(self.main_model_path + model_folder_str))
            graph = tf.compat.v1.get_default_graph()
            input_placeholder = sess.graph.get_tensor_by_name('eval_in:0')
            prediction = sess.graph.get_tensor_by_name('compute_logits_name_scope/fully_connected/BiasAdd:0')
            eval_pred_logits = None
            all_c_data = self.old_devices_data
            for label in range(self.num_classes):
                data_c = all_c_data[all_c_data['label']==label]
                data_c_X = data_c.values[:,1:]
                pred_logits_c = sess.run(prediction,feed_dict={input_placeholder:data_c_X})
                max_probs_c = get_max_probs(pred_logits_c)
                label_max_prob_dict[label] = max_probs_c
        threshold = get_threshold(label_max_prob_dict)
        self.threshold = round(threshold,2) - 0.01
        inf_json = {'threshold':self.threshold}
        if not os.path.exists(self.inf_output_dir):
            os.makedirs(self.inf_output_dir)
        f = open(self.inf_output_dir + '/threshold.json','w')
        json.dump(inf_json,f)
        f.close()

    def get_known_devices_max_probs(self):
        known_probs = []
        with tf.Session() as sess:
            model_folder_str = '_'.join(str(item) for item in self.new_devices_list)
            f_model_name = open(self.main_model_path + '/' + model_folder_str + '/model_name','r')
            model_name = f_model_name.readline().rstrip('\n')
            f_model_name.close()
            # saver = tf.compat.v1.train.import_meta_graph(self.main_model_path +'/'+ model_folder_str + 'model-2000-2020.08.28.-13.27.19-2000.meta')
            saver = tf.compat.v1.train.import_meta_graph(self.main_model_path + model_folder_str + '/' + model_name + '.meta')
            saver.restore(sess,tf.train.latest_checkpoint(self.main_model_path + model_folder_str))
            graph = tf.compat.v1.get_default_graph()
            input_placeholder = sess.graph.get_tensor_by_name('eval_in:0')
            prediction = sess.graph.get_tensor_by_name('compute_logits_name_scope/fully_connected/BiasAdd:0')
            eval_pred_logits = None
            all_c_data = self.old_devices_data
            for label in range(self.num_classes):
                data_c = all_c_data[all_c_data['label']==label]
                data_c_X = data_c.values[:,1:]
                pred_logits_c = sess.run(prediction,feed_dict={input_placeholder:data_c_X})
                max_probs_c = get_max_probs(pred_logits_c)
                known_probs.append(max_probs_c)
            known_probs = [item for sublist in known_probs for item in sublist]
            return known_probs

    # return new devices data, minus label stands for new devices, merged_data:pd
    def get_new_devices_data(self,merged_data):
        label_set = np.unique(merged_data['label'])
        result = None
        for label in label_set:
            if label < 0:
                data_c = merged_data[merged_data['label']==label]
                if result is None:
                    result = data_c
                else:
                    result = pd.concat([result,data_c],axis=0)
        return result

    # rearrange labels of old_data and new_data, assign non-iot a max label, if self.old_label_new_label_dict is blank, fill it
    def rearrange_labels(self,old_data_train,new_data_train,old_data_test,new_data_test):
        # self.new_devices_num, self.old_label_new_label_dict       
        new_labels = np.unique(new_data_train['label'])
        old_labels = np.unique(old_data_train['label'])
        label_non_iot_cur = np.max(old_labels)
        label_non_iot_new = len(new_labels) + np.max(old_labels)
        self.old_label_new_label_dict[label_non_iot_cur] = label_non_iot_new
        old_data_train['label'] = old_data_train['label'].replace(label_non_iot_cur,label_non_iot_new)
        old_data_test['label'] = old_data_test['label'].replace(label_non_iot_cur,label_non_iot_new)
        blank_list = list(range(int(label_non_iot_cur),int(label_non_iot_new)))
        for blank, new_device_label in zip(blank_list,new_labels):
            self.old_label_new_label_dict[new_device_label] = blank
            new_data_train['label'] = new_data_train['label'].replace(new_device_label,blank)
            new_data_test['label'] = new_data_test['label'].replace(new_device_label,blank)
        data_train = pd.concat([old_data_train,new_data_train],axis=0)
        data_test = pd.concat([old_data_test,new_data_test],axis=0)
        return data_train, data_test

    def get_new_type_num(self,type=1):
        
        path = self.main_model_path +'/'+ self.new_devices_postfix + '/sse_cluster_number.jpg'
        # for label in self.filtered_iot_data_new.keys():
        #     if label < 0:
        #         if data is None:
        #             data = self.filtered_iot_data_new[label]
        #         else:
        #             data = np.concatenate([data,self.filtered_iot_data_new[label]],axis=0)
        if self.use_cnn_layer_for_cluster is True:
            k_cnn_layers,relabeled_data_cnn_layers = get_new_type_num_from_cluster(self.filtered_iot_data_new,type,path,self.filtered_cnn_emb,self.merge_cnn_layer_method,self.use_cnn_layer_for_cluster,self.logger)
            # write to txt
            types = k_cnn_layers.keys()
            for type in types:
                if type == 'c3_1':
                    k = k_cnn_layers[type]
                    relabeled_data = relabeled_data_cnn_layers[type]
                    self.logger.info('--------------using cnn layers for cluster,new_devices:{}, cnn_type:{}--------------'.format(self.new_devices_list,type))
                    if relabeled_data is None:
                        self.logger.info('the cluster num is wrong, wrong cluster num:{}'.format(k))
                        k_cnn_layers[type] = -1
                    else:
                        self.logger.info('optimal cluster num:{}'.format(k))
                        relabeled_data.to_csv(self.pathToDataFolder + self.new_devices_postfix + '/relabeled_filtered_new_devices_train_data({}).csv'.format(type), index=False, header = self.column_names)
            return k_cnn_layers

        else:
            k,relabeled_data = get_new_type_num_from_cluster(self.filtered_iot_data_new,type,path,None,self.merge_cnn_layer_method,self.use_cnn_layer_for_cluster,self.logger)
            if relabeled_data is None:
                self.logger.info('the cluster num is wrong, wrong cluster num for {}:{}({})'.format(self.new_devices_list,k,self.merge_cnn_layer_method))
                return -1
            else:
                relabeled_data.to_csv(self.pathToDataFolder + self.new_devices_postfix + '/relabeled_filtered_new_devices_train_data({}).csv'.format(self.merge_cnn_layer_method), index=False, header = self.column_names)              
                self.logger.info('optimal cluster num for {}:{}'.format(self.new_devices_list,k))
                return k

    # judge instance of new devices and old devices according to theta and formula defined by me
    def _judge_devices(self,type=1):
        
        #---------------------------------identify iot and non-iot------------------------------------
        label_set = np.unique(self.val_data['label'])
        label_niot = max(label_set)
        label_iot = [label for label in label_set if label != label_niot]
        if self.record_data_index == True:
            new_devices_data_before_normalization = pd.read_csv(self.pathToDataFolder+self.new_devices_postfix + '/new_devices_with_index_before_normalization.csv')
            new_devices_data_before_normalization['iot'] = 2
        tf.reset_default_graph()
        with tf.Session() as sess:
            model_folder_str = '_'.join(str(item) for item in self.new_devices_list)
            f_model_name = open(self.main_model_path + '/'+ model_folder_str + '/model_name','r')
            model_name = f_model_name.readline().rstrip('')
            f_model_name.close()
            saver = tf.compat.v1.train.import_meta_graph(self.main_model_path + model_folder_str + '/' + model_name + '.meta')
            saver.restore(sess,tf.train.latest_checkpoint(self.main_model_path + model_folder_str))
            graph = tf.compat.v1.get_default_graph()
            input_placeholder = sess.graph.get_tensor_by_name('eval_in:0')
            prediction = sess.graph.get_tensor_by_name('compute_logits_name_scope/fully_connected/BiasAdd:0')
            prediction_iot_niot = sess.graph.get_tensor_by_name('compute_logits_name_scope/fully_connected_1/BiasAdd:0')
            # use_cnn_layer_for_cluster (1st round modification))
            if self.use_cnn_layer_for_cluster == True:
                c1_1 = sess.graph.get_tensor_by_name('emb_z_name_scope/cnn2/c1_1/BiasAdd:0')
                c1_2 = sess.graph.get_tensor_by_name('emb_z_name_scope/cnn2/c1_2/BiasAdd:0')
                c2_1 = sess.graph.get_tensor_by_name('emb_z_name_scope/cnn2/c2_1/BiasAdd:0')
                c2_2 = sess.graph.get_tensor_by_name('emb_z_name_scope/cnn2/c2_2/BiasAdd:0')
                c3_1 = sess.graph.get_tensor_by_name('emb_z_name_scope/cnn2/c3_1/BiasAdd:0')
                c3_2 = sess.graph.get_tensor_by_name('emb_z_name_scope/cnn2/c3_2/BiasAdd:0')
                flatten = sess.graph.get_tensor_by_name('emb_z_name_scope/cnn2/flatten/Reshape:0')
            
            # -----------------------------decide iot or non-iot----------------------------------------------------------------
            filtered_iot_data = {label:{'filtered_data':[],'whole_num':0} for label in label_iot} # filter iot instances and store its labels
            
            
            whole_num = 0
            identification_num = 0
            self.logger.info("======================{}=======================".format(self.new_devices_postfix))
            self.logger.info("use_cnn_layer_for_cluster:{}".format(self.use_cnn_layer_for_cluster))
            for label in label_iot:
                data_c_ = self.val_data[self.val_data['label']==label]
                data_c = self.val_data[self.val_data['label']==label].values
                cols_num = self.val_data.shape[1]
                if self.record_data_index == True:
                    cols_num -= 1
                data_c_X = data_c[:,1:cols_num] # 
                              
                pred_iot_niot_logits = sess.run(prediction_iot_niot,feed_dict={input_placeholder:data_c_X})
                prob_iot_niot = compute_probs_dist_through_logits(pred_iot_niot_logits)
                prob_iot_niot_np = np.array(prob_iot_niot)
                iot_indicator = prob_iot_niot_np[:,0]>= self.sessionNModelFlags['threshold']
                
                # record wrong iot/non-iot in data before normalizaiton
                # if store_analysis_iot_niot==True and label in wrong_iot_niot_labels:
                #     data_c_['iot'] = iot_indicator
                #     path_ = self.pathToDataFolder + self.new_devices_postfix + '/wrong_iot_niot_devices_{}.csv'.format(label)
                #     data_c_.to_csv(path_, index=False)

                # record the wrong in new_devices_data_before_normalization
                if store_analysis_iot_niot==True and self.record_data_index==True:
                   data_c_['iot'] = iot_indicator
                   for idx,row in data_c_.iterrows():
                       index_i = int(row['index'])
                       iot_indicator_i = int(row['iot'])
                       new_devices_data_before_normalization.loc[new_devices_data_before_normalization['index']==index_i,'iot'] = iot_indicator_i
                    
                # del index column 
                data_c_X_filtered = data_c_X[iot_indicator]
                filtered_iot_data[label]['filtered_data'] = data_c_X_filtered
                filtered_iot_data[label]['whole_num'] = data_c_X.shape[0]
                self.logger.info('label {}, iot identification accuracy:{}'.format(label,data_c_X_filtered.shape[0]/data_c_X.shape[0]))  
                
                whole_num += data_c_X.shape[0]
                identification_num += data_c_X_filtered.shape[0]   


            if self.record_data_index == True: 
                new_devices_data_before_normalization.to_csv(self.pathToDataFolder + self.new_devices_postfix + '/new_devices_with_index_before_normalization_indicator.csv',index=False)   
            # compute overall accuracy
            iot_niot_accuracy_whole = identification_num/whole_num
            self.logger.info('whole iot/non-iot identification accuracy:{}'.format(iot_niot_accuracy_whole))
            time1 = time.time()
            self.logger.info('iot identification time dur:{}'.format(time1-time0))

            # ---------------------------- also filter new_devices_test_data--------------------------------------, change
            filtered_iot_test_data = {label:{'filtered_data':[],'whole_num':0} for label in np.unique(self.new_devices_test_data_pd['label'])}
            whole_num = 0
            identification_num = 0
            for label in np.unique(self.new_devices_test_data_pd['label']):
                data_c_ = self.new_devices_test_data_pd[self.new_devices_test_data_pd['label']==label]
                data_c = self.new_devices_test_data_pd[self.new_devices_test_data_pd['label']==label].values
                cols_num = self.val_data.shape[1]
                data_c_X = data_c[:,1:cols_num] 
                              
                pred_iot_niot_logits = sess.run(prediction_iot_niot,feed_dict={input_placeholder:data_c_X})
                prob_iot_niot = compute_probs_dist_through_logits(pred_iot_niot_logits)
                prob_iot_niot_np = np.array(prob_iot_niot)
                iot_indicator = prob_iot_niot_np[:,0]>= self.sessionNModelFlags['threshold']
                
                # del index column 
                data_c_X_filtered = data_c_X[iot_indicator]
                filtered_iot_test_data[label]['filtered_data'] = data_c_X_filtered
                filtered_iot_test_data[label]['whole_num'] = data_c_X.shape[0]
                self.logger.info('test label {}, iot identification accuracy:{}'.format(label,data_c_X_filtered.shape[0]/data_c_X.shape[0]))  
                
                whole_num += data_c_X.shape[0]
                identification_num += data_c_X_filtered.shape[0]   

            # # compute overall accuracy
            # iot_niot_accuracy_whole = identification_num/whole_num
            # self.logger.info('test whole iot/non-iot identification accuracy:{}'.format(iot_niot_accuracy_whole))
            
            # -----------------------------find new iot devices---------------------------------------------------  
            
            label_result_dict = {label:{'res':[],'len':[]} for label in label_set}
            filtered_cnn_emb = {label:{'c1_1':[],'c1_2':[],'c2_1':[],'c2_2':[],'c3_1':[],'c3_2':[],'flatten':[]} for label in label_iot}
            # filtered_cnn_emb = {label:{'c3_1':[]} for label in label_iot}
            if type == 1: # decide according to threshold
                f = open(self.inf_output_dir + '/threshold.json','r')
                threshold_dict = json.load(f)
                f.close()
                
                new_devices_str = '_'.join(str(item) for item in self.new_devices_list)
                threshold = threshold_dict['threshold'] 
                self.logger.info('------------decide new devices or old devices from filted iot devices-------------')       
                for label in filtered_iot_data.keys():
                    data_c_X = filtered_iot_data[label]['filtered_data']
                    num_c = len(data_c_X)
                    start = 0
                    i = 0
                    score = s0
                    seq = []
                    # current_data_Xs = None
                    while(i < num_c):
                        instance = data_c_X[i].reshape([1,-1])
                        # if currrent_data_Xs is None:
                        #     current_data_Xs = instance
                        # else:
                        #     current_data_Xs = np.concatenate([current_data_Xs,instance],axis = 0)
                        pred_logits = sess.run(prediction,feed_dict={input_placeholder:instance})
                        max_prob = get_max_probs(pred_logits)[0]
                        if max_prob > threshold:
                            seq.append(1) # 1 stands for falling into B
                        else:
                            seq.append(0)
                        score += get_delta_s(seq)
                        if score > theta2:
                            label_result_dict[label]['res'].append(0) # 0 stands for old devices
                            label_result_dict[label]['len'].append(len(seq))
                            seq = []
                            start = i+1
                            score = s0
                        elif score < theta1:
                            label_result_dict[label]['res'].append(1) # 0 stands for old devices
                            label_result_dict[label]['len'].append(len(seq))
                            seq = []
                            start = i+1
                            score = s0
                        i += 1
            elif type == 2: # judge according to KS test, put the result into label_result_dict, store filtered data into filter_iot_data_new
                known_probs = self.get_known_devices_max_probs()
                #--------------------------------plot CDF------------------------------------------------
                max_probs_filtered_iot = {label:[] for label in filtered_iot_data.keys()}
                for label in filtered_iot_data.keys():
                    data_c_X = filtered_iot_data[label]['filtered_data']
                    pred_logits = sess.run(prediction,feed_dict={input_placeholder:data_c_X})
                    max_prob = get_max_probs(pred_logits)
                    max_probs_filtered_iot[label] += max_prob
                plot_cdf(known_probs,max_probs_filtered_iot,self.new_old_label_dict, self.main_model_path + self.new_devices_postfix + '/cdf.pdf')
                # ----------------------------------------------------------------------------------------------
                filtered_iot_data_new = {label:[] for label in filtered_iot_data.keys()}
                self.logger.info('------------decide new devices or old devices from filted iot devices-------------')  
                
                
                self.logger.info('JUDGE_LEN:'+str(JUDGE_LEN)+'')
                self.logger.info('REJUDGE_LEN:'+str(REJUDGE_LEN)+'')
                
                for label in filtered_iot_data.keys():
                    data_c_X = filtered_iot_data[label]['filtered_data']
                    num_c = len(data_c_X)    
                    cur_probs = []
                    rejudge = 0    
                    current_data_Xs = None              
                    for instance in data_c_X:
                        instance = instance.reshape([1,-1])
                        if current_data_Xs is None:
                            current_data_Xs = instance
                        else:
                            current_data_Xs = np.concatenate([current_data_Xs,instance],axis = 0)
                        
                        pred_logits = sess.run(prediction,feed_dict={input_placeholder:instance})
                        max_prob = get_max_probs(pred_logits)[0]
                        cur_probs.append(round(max_prob,4))
                        if len(cur_probs) < JUDGE_LEN:
                            continue
                        [d_statistic,critical] = compute_KS(cur_probs,known_probs,type=0.01)
                        if d_statistic > critical:
                            # label_result_dict[label]['res'].append(1)
                            # label_result_dict[label]['len'].append(len(cur_probs))
                            if rejudge < REJUDGE_LEN:
                                rejudge += 1
                                continue
                            else: # new types of devices
                                label_result_dict[label]['res'].append(1)
                                label_result_dict[label]['len'].append(len(cur_probs))
                                cur_probs = []
                                rejudge = 0
                                # use_cnn_layer_for_cluster
                                if self.use_cnn_layer_for_cluster is True:
                                    c1_1_t,c1_2_t,c2_1_t,c2_2_t,c3_1_t,c3_2_t,flatten_t = sess.run([c1_1,c1_2,c2_1,c2_2,c3_1,c3_2,flatten],feed_dict={input_placeholder:current_data_Xs})                                   
                                    # c3_1_t = sess.run([c3_1],feed_dict={input_placeholder:current_data_Xs})                                   
                                    filtered_cnn_emb[label]['c1_1'] += list(c1_1_t)
                                    filtered_cnn_emb[label]['c1_2'] += list(c1_2_t)
                                    filtered_cnn_emb[label]['c2_1'] += list(c2_1_t)
                                    filtered_cnn_emb[label]['c2_2'] += list(c2_2_t)
                                    filtered_cnn_emb[label]['c3_1'] += list(c3_1_t)
                                    filtered_cnn_emb[label]['c3_2'] += list(c3_2_t)
                                    filtered_cnn_emb[label]['flatten'] += list(flatten_t)
                                # original features used for cluster
                                if len(filtered_iot_data_new[label])==0:
                                    filtered_iot_data_new[label] = list(current_data_Xs)
                                    current_data_Xs = None
                                else:
                                    filtered_iot_data_new[label] += list(current_data_Xs)
                                    current_data_Xs = None
                        else:
                            label_result_dict[label]['res'].append(0)
                            label_result_dict[label]['len'].append(len(cur_probs))
                            cur_probs = []
                            rejudge = 0
                            current_data_Xs = None

                #-----------------------------------------filter new_devices_test_data------------------------------------------------------,change
                label_result_dict_test = {label:{'res':[],'len':[]} for label in np.unique(self.new_devices_test_data_pd['label'])}
                filtered_iot_data_new_test = {label:[] for label in filtered_iot_test_data.keys()}
                for label in filtered_iot_test_data.keys():
                    data_c_X = filtered_iot_test_data[label]['filtered_data']
                    num_c = len(data_c_X)    
                    cur_probs = []
                    rejudge = 0    
                    current_data_Xs = None              
                    for instance in data_c_X:
                        instance = instance.reshape([1,-1])
                        if current_data_Xs is None:
                            current_data_Xs = instance
                        else:
                            current_data_Xs = np.concatenate([current_data_Xs,instance],axis = 0)
                        
                        pred_logits = sess.run(prediction,feed_dict={input_placeholder:instance})
                        max_prob = get_max_probs(pred_logits)[0]
                        cur_probs.append(round(max_prob,4))
                        if len(cur_probs) < JUDGE_LEN:
                            continue
                        [d_statistic,critical] = compute_KS(cur_probs,known_probs,type=0.01)
                        if d_statistic > critical:
                            # label_result_dict[label]['res'].append(1)
                            # label_result_dict[label]['len'].append(len(cur_probs))
                            if rejudge < REJUDGE_LEN:
                                rejudge += 1
                                continue
                            else: # new types of devices
                                label_result_dict_test[label]['res'].append(1)
                                label_result_dict_test[label]['len'].append(len(cur_probs))
                                cur_probs = []
                                rejudge = 0                  
                                # original features used for cluster
                                if len(filtered_iot_data_new_test[label])==0:
                                    filtered_iot_data_new_test[label] = list(current_data_Xs)
                                    current_data_Xs = None
                                else:
                                    filtered_iot_data_new_test[label] += list(current_data_Xs)
                                    current_data_Xs = None
                        else:
                            label_result_dict_test[label]['res'].append(0)
                            label_result_dict_test[label]['len'].append(len(cur_probs))
                            cur_probs = []
                            rejudge = 0
                            current_data_Xs = None
                # ---------------------------------------------------------------------------------------------------------------------------     
                       
            # save filtered_iot_data_new
            labels = []
            data_X = []
            for label in filtered_iot_data_new.keys():
                if label < 0:
                    if instances_culmulated > 0:
                        data_X += filtered_iot_data_new[label][:instances_culmulated]
                        num = len(filtered_iot_data_new[label][:instances_culmulated])
                        filtered_iot_data_new[label] = filtered_iot_data_new[label][:instances_culmulated]
                    else:
                        data_X += filtered_iot_data_new[label]
                        num = len(filtered_iot_data_new[label])
                    labels += [label] * num
            labels = np.array(labels).reshape([-1,1])
            data_X = np.array(data_X)
            data = np.concatenate([labels,data_X],axis=1)
            data = pd.DataFrame(data)
            data.to_csv(self.pathToDataFolder +self.new_devices_postfix+'/filtered_new_devices_train_data.csv',index=False,header = self.column_names)
            # --------------------------------------------save filtered_iot_data_test_new--------------------------------------------------,change
            labels = []
            data_X = []
            for label in filtered_iot_data_new_test.keys():
                if label < 0:
                    
                    data_X += filtered_iot_data_new[label]
                    num = len(filtered_iot_data_new[label])
                    labels += [label] * num
            labels = np.array(labels).reshape([-1,1])
            data_X = np.array(data_X)
            data = np.concatenate([labels,data_X],axis=1)
            data = pd.DataFrame(data)
            data.to_csv(self.pathToDataFolder +self.new_devices_postfix+'/filtered_new_devices_test_data.csv',index=False,header = self.column_names)


            # ----------------------------------------------save filtered cnn_layer_emb------------------------------------------------------
            if self.use_cnn_layer_for_cluster is True:
                for label in filtered_iot_data_new.keys():
                    if label < 0:
                        if instances_culmulated > 0:
                            for label_ in ['c1_1','c1_2','c2_1','c2_2','c3_1','c3_2','flatten']:
                            # for label_ in ['c3_1']:
                                filtered_cnn_emb[label][label_] = filtered_cnn_emb[label][label_][:instances_culmulated]


                self.filtered_cnn_emb = filtered_cnn_emb

        # judge accuracy according to label_result_dict
        true = []
        pred = []
        max_len = -1
        all_len = 0
        instance_num = 0
        self.logger.info("----------------------------new devices accuracy------------------------------")
        for label,items in label_result_dict.items():
            pred_results = items['res']
            pred_instance_lens = items['len']
            all_c = len(pred_results)
            true_c = 0
            for pred_res,pred_instance_len in zip(pred_results,pred_instance_lens):
                pred.append(pred_res)
                max_len = max(max_len,pred_instance_len)
                instance_num += 1
                all_len += pred_instance_len
                if label >= 0:
                    true.append(0)
                else:
                    true.append(1)
                if (label>=0 and pred_res==0) or (label<0 and pred_res==1):
                    true_c += 1
            if all_c!=0:
                accuracy_c = round(true_c/all_c,4)
                self.logger.info('accuracy of label {}:{}'.format(label,accuracy_c))
        true_np = np.array(true)
        pred_np = np.array(pred)
        accuracy_all = round(np.sum(true_np==pred_np)/len(true),4)
        self.logger.info('overall accuracy:{}'.format(accuracy_all))
        # compute the max len and average len of instances used to discriminate new devices or old devices
        self.logger.info('max len of instances for discrimination:{},average:{},judge_num:{}'.format(max_len,all_len/instance_num,instance_num))
        time2 = time.time()
        self.logger.info('new device identification time dur:{}'.format(time2-time1))
        return filtered_iot_data_new

    # adjust model according to self.new_devices_num, type==1:retrain; type==2:transfer learning; type==3:distill learning
    def adjust_model(self,type=1):
        if type == 1:# all labeled data to retrain
            # data: train:old_devices_train_data + new_devices_train_data; test:old_devices_test_data + new_devices_test_data
            

            train_data = pd.read_csv(self.pathToDataFolder + self.new_devices_postfix + '/old_devices_train_data.csv')
            train_data_ = pd.read_csv(self.pathToDataFolder + self.new_devices_postfix + '/filtered_new_devices_train_data.csv')
            test_data = pd.read_csv(self.pathToDataFolder + self.new_devices_postfix + '/old_devices_test_data.csv')
            test_data_ = pd.read_csv(self.pathToDataFolder + self.new_devices_postfix + '/new_devices_test_data.csv')
            train_data,test_data = self.rearrange_labels(train_data,train_data_,test_data,test_data_)
            train_data = train_data.values 
            test_data = test_data.values
        
            #-----------------------------------hyper parameters---------------------------------------------
            EPOCHS = 2000
            BATCH_SIZE = 128
            iot_niot_threshold = 0.7
            tf.reset_default_graph()
            
            weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=True, seed=self.seed, dtype=tf.float32)#An initializer that generates tensors with unit variance.
            weights_regularizer = tf.contrib.layers.l2_regularizer  
            model_folder_str = '_'.join(str(item) for item in self.new_devices_list)
            f_model_name = open(self.main_model_path + 'model_name','r')
            model_name = f_model_name.readline().rstrip('\n')
            f_model_name.close()
            saver = tf.compat.v1.train.import_meta_graph(self.main_model_path + model_folder_str + '/' + model_name + '.meta')
            graph = tf.compat.v1.get_default_graph()     
            with tf.Session() as sess:
                saver.restore(sess,tf.train.latest_checkpoint(self.main_model_path + model_folder_str))          
                input_placeholder_x = sess.graph.get_tensor_by_name('eval_in:0')
                input_placeholder_y = tf.placeholder("int32",[None],'ph_y')
                # input_placeholder_y_specific = tf.placeholder("int32",[None,self._num_classes-1+self.new_devices_num],'ph_y_specific')
                # input_placeholder_y_iot_niot = tf.placeholder("int32",[None,2],'ph_y_iot_niot')
                t_learning_rate = tf.Variable(0.01,dtype = tf.float32,name = 'lr')
                emb_z = sess.graph.get_tensor_by_name('emb_z_name_scope/cnn2/flatten/Reshape:0')
                # reconstruct last layer
                with tf.variable_scope('new_last_layer'):
                    logits_1 = tf.contrib.layers.fully_connected(
                                    emb_z,
                                    self.num_classes+self.new_devices_num,
                                    weights_initializer=weights_initializer,
                                    activation_fn=None,
                                    # scope = 'new_last_layer/fully_connected',
                                    weights_regularizer=tf.contrib.layers.l2_regularizer(self._l2_reg_classif))
                    print('name of logits_1:{}'.format(logits_1.name)) 
                    logits_2 = tf.contrib.layers.fully_connected(
                                    emb_z,
                                    2,
                                    weights_initializer=weights_initializer,
                                    activation_fn=None,
                                    # scope = 'new_last_layer/fully_connected2',
                                    weights_regularizer=tf.contrib.layers.l2_regularizer(self._l2_reg_classif))
                    print('name of logits_2:{}'.format(logits_2.name))    
                predictions_1 = tf.nn.softmax(logits_1)
                predictions_2 = tf.nn.softmax(logits_2)
                # test
                # vars = tf.global_variables()
                # print(vars)

                # loss
                label_non_iot = logits_1.get_shape()[-1]
                condition = tf.equal(input_placeholder_y,label_non_iot) #col-3=27-3=24,condition=1 stands for NoT(lbls==24)
                tf_ones = tf.ones_like(input_placeholder_y) #each element of lbls is an scaler
                tf_zeros = tf.zeros_like(input_placeholder_y)
                expand_ = tf.where(condition,tf_ones,tf_zeros) # 1 stands for NoT
                onehot_labels = tf.one_hot(input_placeholder_y, logits_1.get_shape()[-1]) #change scaler to onehot code, true onehot labels
                onehot_expand = tf.one_hot(expand_,2)  #IoT--NoT, true onehot iot/non-iot label
                # weights = generate_weights(onehot_labels=onehot_labels,num_classes=logits.get_shape()[-1])
                weights = 1
                loss_logit_weighted_1 = tf.losses.softmax_cross_entropy(
                                        onehot_labels = onehot_labels,
                                        logits = logits_1[:,0:self.num_classes+self.new_devices_num],
                                        scope = 'loss_logit_weighted_1_new',
                                        weights = weights)
                # weights2 = generate_weights(onehot_labels=onehot_expand,num_classes=2)
                weight2 = 1
                loss_logit_weighted_2 = tf.losses.softmax_cross_entropy(
                                        onehot_labels = onehot_expand,
                                        logits = logits_2,
                                        scope = 'loss_logit_weighted_2',
                                        weights = weight2)   
                loss_total_weighted = tf.losses.get_total_loss(add_regularization_losses=True) 
              

                # accuracy (predictions_1,predictions_2)
                condition = tf.greater(predictions_2[:,0],iot_niot_threshold)
                label_non_iot_f = tf.cast(label_non_iot,tf.int64)
                niot_labels = tf.ones_like(predictions_2[:,0],dtype=tf.int64)*label_non_iot_f
                predictions = tf.where(condition,tf.argmax(predictions_1,1),niot_labels)
                correct_prediction = tf.equal(tf.argmax(onehot_labels,1),predictions)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
                lr_adjust = tf.assign(t_learning_rate,t_learning_rate/1.5)
                
                vars = tf.global_variables()
                uninitialized_vars = []
                for var in tf.all_variables():
                    try:
                        sess.run(var)
                    except tf.errors.FailedPreconditionError:
                        uninitialized_vars.append(var)
                var_to_init = [var for var in vars if 'new_last_layer' in var.name]
                train_step = tf.train.AdamOptimizer(t_learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-07,name='new_last_layer/new_adam').minimize(loss_total_weighted,var_list=vars)
                # train_step = tf.train.AdamOptimizer(t_learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-07,name='new_last_layer/new_adam').minimize(loss_total_weighted)
                # train_step = sess.graph.get_operation_by_name('emb_z_name_scope/c1_1/weights/Adam')
                
                uninitialized_vars = []
                for var in tf.all_variables():
                    try:
                        sess.run(var)
                    except tf.errors.FailedPreconditionError:
                        uninitialized_vars.append(var)
                

                # tf.initialize_variables(uninitialized_vars)
                sess.run(tf.variables_initializer(uninitialized_vars))
                # sess.run(learning_rate)
                for epoch in range(EPOCHS):
                    print('epoch:{}'.format(epoch))
                    for i in range(0,train_data.shape[0],BATCH_SIZE):
                        train_X_batch = train_data[i:i+BATCH_SIZE,1:]
                        train_y_batch = train_data[i:i+BATCH_SIZE,0]
                        sess.run(train_step,feed_dict={input_placeholder_x:train_X_batch,input_placeholder_y:train_y_batch})
                    if epoch % 20 == 0:
                        print("epoch:{}".format(epoch))
                        train_accuracy = sess.run(accuracy,feed_dict={input_placeholder_x:train_X_batch,input_placeholder_y:train_y_batch})
                        print("accuracy of training data is:{}".format(train_accuracy))
                        test_accuracy = sess.run(accuracy,feed_dict={input_placeholder_x:test_data[:,1:],input_placeholder_y:test_data[:,0]})
                        print("accuracy of testing data is:{}".format(test_accuracy))
                        learning_rate = sess.run(t_learning_rate)
                        print('learning rate:{}'.format(learning_rate))
                        sess.run(lr_adjust)
                        prediction1 = sess.run(predictions_1,feed_dict={input_placeholder_x:train_X_batch,input_placeholder_y:train_y_batch})
        elif type == 2:# semi-supervised model retrain and not recover parameters
            tl = True # transfer learning flag          
            # self.sessionNModelFlags['max_iters'] = 2000
            sessionNModelFlags = self.sessionNModelFlags
            sessionNModelFlags['session_type'] = 'train'
            trainerFlags = self.trainerFlags
            # trainerFlags['lr_expon_decay_factor'] = 0.8
            # trainerFlags['lr_expon_decay_steps'] = 400
            sessionNModelFlags.print_params() #put session, model parameters into sessionNModelFlags class
            trainerFlags.print_params() # put training parameters into trainerFlags class
            
            if self.use_cnn_layer_for_cluster is True:
                for type,k in self.new_devices_num.items():
                    if (type == 'c3_1') and (k == len(self.new_devices_list)):
                        train.train(sessionNModelFlags=sessionNModelFlags, trainerFlags=trainerFlags,retrain=True,tl=tl,cnn_type=type) # core
                        # also compare ordinary retraining
                        train.train(sessionNModelFlags=sessionNModelFlags, trainerFlags=trainerFlags,retrain=True,tl=False,cnn_type=type) # core
                
            else:
                train.train(sessionNModelFlags=sessionNModelFlags, trainerFlags=trainerFlags,retrain=True,tl=tl,cnn_type=sessionNModelFlags['merge_cnn_layer_methods'],logger=self.logger) # core
                # also compare ordinary retraining
                train.train(sessionNModelFlags=sessionNModelFlags, trainerFlags=trainerFlags,retrain=True,tl=False,cnn_type=sessionNModelFlags['merge_cnn_layer_methods'],logger=self.logger) # core
       



            

        
            




