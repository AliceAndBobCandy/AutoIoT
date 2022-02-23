#!/usr/bin/env python
# changed by Linna Fan: data sample idea: split data into train and test, train data's Y_generated == 1, which can be resampled. then in sample process, val data can only be 
# get through data which Y_generated == 0, after that, train data can be got through left data.

# Copyright (c) 2018, Konstantinos Kamnitsas
# This program is free software; you can redistribute and/or modify
# it under the terms of the Apache License, Version 2.0. See the 
# accompanying LICENSE file or read the terms at:
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import division
from __future__ import print_function
from asyncio.log import logger
# from imblearn.over_sampling import SMOTE


import logging
LOG = logging.getLogger('main')

import os
import sys # for checking python version
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
#from tflearn.data_utils import shuffle, to_categorical
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler,MinMaxScaler,PowerTransformer,QuantileTransformer,RobustScaler
from sklearn.utils import resample
from sklearn.utils import shuffle

from cclp.data_utils.datasets import DataSet
from cclp.data_utils.misc_utils import makeArrayOneHot, sample_uniformly, sample_by_class
from cclp.data_utils.compute_utils import train_test_split,rearrange_labels

import gzip
import time
import pickle
import copy
import json
from sklearn.impute import SimpleImputer
# from imblearn.pipeline import make_pipeline
# from imblearn.combine import SMOTEENN

from configs import local_dirs_to_data


# constants
TRAIN_DB = 'train'
TEST_DB = 'test'



class DataManager(object):
    def __init__(self):
        self._currentBatch = None
        self.datasetsDict = OrderedDict()
        
    def _timeReading(self, readerFunction, filepath):
        LOG.info("Loading data from: "+str(filepath))
        startLoadTime = time.clock()
        dataRead = readerFunction(filepath)
        endLoadTime = time.clock()
        totalLoadTime = endLoadTime - startLoadTime
        LOG.info("[TIMING] totalLoadTime = "+str(totalLoadTime)) # svhn: train - 1.2 sec, test - 0.5, extra - 9.5 secs.
        return dataRead
        
    # Not used because we use tf functionality to do batching.
    def getCurrentBatch(self):
        return self._currentBatch
    
    def getNextBatch(self, batchSize, trainValTestStr):
        # x_value: the images for batch. [batchSize, r, c, rgb] or [batchSize, r*c*rgb] if previously asked to reshape dataset. Eg for MLP.
        # y_value: the labels for batch. Can be chosen to come as one_hot (shape [batchSize, 10]) or not ([batchSize, ]). See DataSet() for choice. 
        x_value, y_value = self.datasetsDict[trainValTestStr].next_batch(batchSize)
        self._currentBatch = (x_value, y_value)
        return self._currentBatch
    
    def getShapeOfAnImage(self, trainValTestStr):
        return self.datasetsDict[trainValTestStr].getShapeOfAnImage()
    def getShapeOfASample(self, trainValTestStr):
        return self.datasetsDict[trainValTestStr].getShapeOfASample()
        
    def print_characteristics_of_db(self, pause=True):
        LOG.debug("\n")
        LOG.debug("========= PRINTING CHARACTERISTICS OF THE DATABASE ==============")
        for data_str in self.datasetsDict.keys():
            
            if self._dataset_name == 'tmc' or (self._retrain==True):
                imgs, lbls, Y_generated = self.datasetsDict[data_str].get_samples_and_labels()
            else:
                imgs, lbls = self.datasetsDict[data_str].get_samples_and_labels()
            # LOG.debug("data upsampling has finished, generated ",np.sum(Y_generated),"new data!")
            
            LOG.debug("["+data_str+"]-images, Shape: "+str(imgs.shape)+"| Type: "+str(imgs.dtype)+"| Min: "+str(np.min(imgs))+"| Max: "+str(np.max(imgs)))
            LOG.debug("["+data_str+"]-labels, Shape: "+str(lbls.shape)+"| Type: "+str(lbls.dtype)+"| Min: "+str(np.min(lbls))+"| Max: "+str(np.max(lbls)))
            count_class_occ = [ np.sum(lbls == class_i) for class_i in range(int(np.max(lbls))+1) ]
            LOG.debug("Class occurrences: "+str(count_class_occ))
        if pause:
            try:
                user_input = raw_input("Input a key to continue...")
            except:
                LOG.warn("Tried to wait and ask for user input to continue, but something went wrong. Probably nohup? Continuing...")
        LOG.debug("==========================================================================\n")
    
    
    def normalize_datasets(self, norm_type):
        train_db_normalized_already = False
        for data_str in self.datasetsDict.keys():
            stats_to_use = None
            if data_str == TRAIN_DB:
                train_db_normalized_already = True # To check that train is normed before test.
            elif data_str == TEST_DB:
                assert train_db_normalized_already
                stats_to_use = self.datasetsDict[TRAIN_DB].get_db_stats()

            self.datasetsDict[data_str].normalize( norm_type, stats_to_use )
        
    
    def sample_folds(self,
                     val_on_test,
                     num_val_samples,
                     num_lbl_samples,
                     num_unlbl_samples,
                     unlbl_overlap_val,
                     unlbl_overlap_lbl,
                     seed=None,
                     retrain=False):
        # Mode 1:
        # |---validation---||---supervision---|
        #                   |------- unsupervised -------|
        # Mode 2:
        # |---validation---||---supervision---|
        # |------------------------- unsupervised -------|
        
        LOG.info("==== Sampling the folds for validation/labelled/unlabelled training ====")
        rng = np.random.RandomState(seed=seed)
        num_classes = self.datasetsDict[TRAIN_DB].num_classes
        dataset_for_lbl = self.datasetsDict[TRAIN_DB]
        # Sample VALIDATION data.
        dataset_for_val = self.datasetsDict[TEST_DB] if val_on_test else self.datasetsDict[TRAIN_DB]
        assert num_val_samples >= -1
        available_indices_val = np.asarray(range( len(dataset_for_val.samples)), dtype="int32")
        
        selected_for_val = available_indices_val[np.invert(np.isin(available_indices_val,np.where(dataset_for_lbl.Y_generated==1)))]
        #select samples for validation, compute average num_val_samples_per_class, if some class do not have so many, more samples will choosen in other classes.
        val_available_indices_lbl_list_by_class = []
        for c in range(num_classes):
            indicator_ = dataset_for_val.labels[selected_for_val] == c ##changed by fln
            val_indices_of_lbl_samples_class_c = selected_for_val[ indicator_ ]
            val_available_indices_lbl_list_by_class.append(val_indices_of_lbl_samples_class_c)
        
        # changed by fln
        # val_num_lbl_samples_per_c = num_val_samples // num_classes if num_lbl_samples != -1 else -1 # -1 will use all data for each class.
        # val_selected_for_lbl_list_by_class = sample_by_class( val_available_indices_lbl_list_by_class, val_num_lbl_samples_per_c, rng, upsampling=True)
        if retrain == False:
            val_num_lbl_samples_per_c = num_val_samples // num_classes if num_lbl_samples != -1 else -1 # -1 will use all data for each class.
            val_selected_for_lbl_list_by_class = sample_by_class( val_available_indices_lbl_list_by_class, val_num_lbl_samples_per_c, rng, upsampling=True)
        else:
            val_num_lbl_samples_per_c = -1
            val_selected_for_lbl_list_by_class = val_available_indices_lbl_list_by_class

        # selected_for_lbl_list_by_class: Will be a list of arrays. Each array, the indices of selected samples for c.
        
        selected_for_val = [item for sublist in val_selected_for_lbl_list_by_class for item in sublist] #? flatten the by_class list of sublists.
        # val_Y_generated = dataset_for_val.Y_generated[selected_for_val] # all is zero
        
        val_samples = dataset_for_val.samples[ selected_for_val ]
        # cols_ = val_samples.shape[1]
        # val_samples = val_samples[:,0:cols_]
        val_labels = dataset_for_val.labels[ selected_for_val ]
        # store val_samples and val_labels to csv
        val_labels_ = val_labels.reshape([-1,1])        
        val_data_ = np.concatenate([val_labels_,val_samples],axis=1)
        cols_all_ = len(self._column_names)
        if val_data_.shape[1] != cols_all_:
            val_data_ = val_data_[:,0:-1]
        val_data_pd = pd.DataFrame(val_data_,columns = self._column_names)
        if self._retrain == False and self._new_devices_list is not None:
            val_data_pd.to_csv(self.new_devices_dir + '/old_devices_test_data.csv',index=False,header = self._column_names)
            print('validation data has been stored')
        # Debugging info for unlabelled.
        LOG.debug("[Validation] fold is test data: "+str(val_on_test)+"| Shape: Samples: "+str(val_samples.shape)+"| Labels: "+str(val_labels.shape))
        LOG.debug("[Validation] fold Class occurrences: "+str( [ np.sum(val_labels == class_i) for class_i in range(num_classes) ] ))
        
        # Sample LABELLED training data.
        # Labelled data used for supervision makes no sense to overlap with validation data in any setting.
        
        available_indices_lbl = np.asarray(range(len(dataset_for_lbl.samples)), dtype="int32" )
        
        if not val_on_test: # If we are validating on subset of train data, exclude val from training.
            LOG.info("[Labelled] fold will exclude samples selected for validation.")
            available_indices_lbl = available_indices_lbl[np.invert(np.isin(available_indices_lbl,selected_for_val))]
        # Get available samples per class because we will sample per class.
        indices_of_all_lbl_samples = np.asarray(range(len(dataset_for_lbl.samples)), dtype="int32") # Array, to do advanced indexing 3 lines below.
        # ------------store train_data----------------
        train_data_X = dataset_for_lbl.samples[available_indices_lbl]
        train_data_Y = dataset_for_lbl.labels[available_indices_lbl].reshape([-1,1])
        train_data_ = np.concatenate([train_data_Y,train_data_X],axis=1)
        if train_data_.shape[1] != cols_all_:
            train_data_ = train_data_[:,0:-1]
        train_data_pd_ = pd.DataFrame(train_data_,columns = self._column_names)
        if self._retrain == False and self._new_devices_list is not None:
            train_data_pd_.to_csv(self.new_devices_dir + '/old_devices_train_data.csv',index=False,header = self._column_names)
            print("train data has been stored")
        #---------------------------------------------
        available_indices_lbl_list_by_class = []
        for c in range(num_classes):
            indices_of_lbl_samples_class_c = indices_of_all_lbl_samples[ dataset_for_lbl.labels == c ] # all data_c 
            available_indices_lbl_c = indices_of_lbl_samples_class_c[ np.isin(indices_of_lbl_samples_class_c, available_indices_lbl) ] # val data has been excluded
            available_indices_lbl_list_by_class.append(available_indices_lbl_c)
                
        # Sample per class for labeled data
        if (hasattr(self,'_percent_lbl_samples') and self._percent_lbl_samples>0):
            if retrain == False:
                selected_for_lbl_list_by_class = sample_by_class( available_indices_lbl_list_by_class, 0, rng, percent_lbl_samples=self._percent_lbl_samples )
            else: # all data for new devices have labels
                # get labels of new devices, they are values of new_arrange_label_dict
                f = open('./data/iot/'+self._new_devices_postfix + '/new_rearange_label_dict.json','r')
                old_new_label_dict_ = json.load(f)
                f.close()
                new_devices_labels_ = old_new_label_dict_.values()
                selected_for_lbl_list_by_class = sample_by_class( available_indices_lbl_list_by_class, 0, rng, percent_lbl_samples=self._percent_lbl_samples,new_devices_labels=new_devices_labels_ )
        else:
            num_lbl_samples_per_c = num_lbl_samples // num_classes if num_lbl_samples != -1 else -1 # -1 will use all data for each class.
            # selected_for_lbl_list_by_class: Will be a list of arrays. Each array, the indices of selected samples for c.
            selected_for_lbl_list_by_class = sample_by_class( available_indices_lbl_list_by_class, num_lbl_samples_per_c, rng )
        selected_for_lbl = [item for sublist in selected_for_lbl_list_by_class for item in sublist] #? flatten the by_class list of sublists.
        
        train_samples_lbl_list_by_class = [] # Will be a list of arrays. Each of the c arrays has lbled samples of class c, to train on.
        for c in range(num_classes):
            train_samples_lbl_list_by_class.append( dataset_for_lbl.samples[ selected_for_lbl_list_by_class[c] ] )
            LOG.debug("[Labelled] fold for Class ["+str(c)+"] has Shape: Samples: " + str(train_samples_lbl_list_by_class[c].shape) )
        
        # Sample UNLABELLED training data.
        dataset_for_unlbl = self.datasetsDict[TRAIN_DB]
        available_indices_unlbl = np.asarray(range( len(dataset_for_unlbl.samples)), dtype="int32" )
        
        if not val_on_test and not unlbl_overlap_val: # If validating on train, and unlabelled should not to overlap, exclude val.
            LOG.info("[Unlabelled] fold will exclude samples selected for validation.")
            available_indices_unlbl = available_indices_unlbl[ np.invert( np.isin(available_indices_unlbl, selected_for_val) ) ]
        if not unlbl_overlap_lbl:
            LOG.info("[Unlabelled] fold will exclude samples selected as labelled.")
            available_indices_unlbl = available_indices_unlbl[ np.invert( np.isin(available_indices_unlbl, selected_for_lbl) ) ]
        selected_for_unlbl = sample_uniformly(available_indices_unlbl, num_unlbl_samples, rng)
        
        train_samples_unlbl = dataset_for_unlbl.samples[ selected_for_unlbl ]
        if train_samples_unlbl.shape[1] != cols_all_-1:
            train_samples_unlbl = val_data_[:,0:-1]
        # Debugging info for unlabelled.
        DEBUG_train_labels_unlbl = dataset_for_unlbl.labels[ selected_for_unlbl ]
        LOG.debug("[Unlabelled] fold has Shape: Samples: "+str(train_samples_unlbl.shape)+"| Labels: "+str(DEBUG_train_labels_unlbl.shape))
        LOG.debug("[Unlabelled] fold Class occurrences: "+str( [ np.sum(DEBUG_train_labels_unlbl == class_i) for class_i in range(num_classes) ] ))
        
        LOG.info("==== Done sampling the folds ====")
        return train_samples_lbl_list_by_class, train_samples_unlbl, val_samples, val_labels

# added code
class IotManager(DataManager):
    def __init__(self, pathToDataFolder=None, boolOneHot=False, dtypeStrX="float", reshape=False, upsampling = False, epsilon=1e-3, num_least=100,seed=None,disturbe = True,prob = 0.5,percent_lbl_samples=0, new_devices_list = None,merged_devices_file = None,train_ratio = 0.7,retrain=False, dataset_name = 'tmc',record_data_index=False,niot_label=None,upsample_niot = -1,percent_lbl_samples_small=-1,lbl_samples_small=-1,cnn_type=None,logger=None):
        DataManager.__init__(self)
        if pathToDataFolder is None:
            pathToDataFolder = local_dirs_to_data.iot
        if retrain == False:
            if dataset_name == 'tmc':
                pathToTrainData = pathToDataFolder + "unsw.csv"    #"yourthings"
                # pathToTrainData = pathToDataFolder + "whole_unsw_preprocess.csv"    #"yourthings"
            elif dataset_name == 'yourthings':
                pathToTrainData = pathToDataFolder + "yourthings.csv"    #"yourthings"
            else:
                print('****************** dataset_name wrong **********************')
                return
        self.logger = logger
        self._column_names = list(pd.read_csv(pathToDataFolder + 'unsw.csv').columns)
        self._retrain = retrain
        self._dataset_name = dataset_name
        self._num_least = num_least 
        self._train_ratio = train_ratio 
        self._upsampling = upsampling
        self._upsample_niot = upsample_niot
        self._disturbe = disturbe
        self._percent_lbl_samples_small = percent_lbl_samples_small
        self._lbl_samples_small = lbl_samples_small
        self._epsilon = epsilon
        self._seed = seed
        self._prob = prob
        self._percent_lbl_samples = percent_lbl_samples
        self._new_devices_list = new_devices_list
        self._niot_label = niot_label
        self._pathToDataFolder = pathToDataFolder
        self._old_new_device_idx_dict = {}
        self._record_data_index = record_data_index
        self._cnn_type = cnn_type
        if self._new_devices_list is not None:
            self._new_devices_postfix = '_'.join(str(item) for item in self._new_devices_list)
            self.new_devices_dir = self._pathToDataFolder + '_'.join(str(item) for item in self._new_devices_list)
        
        self._merged_devices_file = merged_devices_file
        self._scaler_ins = QuantileTransformer()
        self._dtypeStrX = dtypeStrX
        self._new_devices_data_without_normalized = None      
        
        if hasattr(self,'new_devices_dir') and not os.path.exists(self.new_devices_dir):
            os.makedirs(self.new_devices_dir)

        if retrain == False:
            (npArrTrainX, npArrTrainY,Y_generated) = self._readNpArrXYFromDisk( pathToTrainData, boolOneHot, dtypeStrX)
        else:
            (npArrTrainX, npArrTrainY,Y_generated) = self._readNpArrXYFromDisk_retrain(dtypeStrX)
        
        if (self._dataset_name == 'tmc') or (self._retrain==True):
            self.datasetsDict[TRAIN_DB] = DataSet(npArrTrainX, npArrTrainY, reshape=reshape,Y_generated=Y_generated)
        else:
            self.datasetsDict[TRAIN_DB] = DataSet(npArrTrainX, npArrTrainY, reshape=reshape,Y_generated=None)
        
        LOG.debug("[SHAPE] npArrTrainX.shape = "+str(npArrTrainX.shape))
        LOG.debug("[SHAPE] npArrTrainY.shape = "+str(npArrTrainY.shape))
        
        if self._new_devices_list is not None and retrain == False:
            self._normalize_and_store_merged_data(new_devices_list) # return pandas

        if retrain == False:
            # store scaler to self._pathToDataFolder + '/quantileTransformer_scaler.pkl
            pickle.dump(self._scaler_ins,open(self._pathToDataFolder+'/quantileTransformer_scaler.pkl','wb'))
            # store old_devices_data to csv , only original data from (npArrTrainX, npArrTrainY,Y_generated)
            # new_devices_str = '-'.join(str(item) for item in new_devices_list)
            # data_x_origin = npArrTrainX[Y_generated==0]
            # data_y_origin = npArrTrainY[Y_generated==0].reshape([-1,1])
            # data_old = np.concatenate([data_y_origin,data_x_origin],axis=1)
            # data_old_pd = pd.DataFrame(data_old)
            # data_old_pd.to_csv(self._pathToDataFolder+'/old_devices_data_exclude_{}.csv'.format(new_devices_str),index=False,header=list(self._column_names))

        # (npArrTestX, npArrTestY) = self._readNpArrXYFromDisk( pathToTestImages, pathToTestLabels, boolOneHot, dtypeStrX )
        # self.datasetsDict[TEST_DB] = DataSet(npArrTestX, npArrTestY, reshape=reshape)
        # LOG.debug("[SHAPE] npArrTestX.shape = "+str(npArrTestX.shape))
        # LOG.debug("[SHAPE] npArrTestY.shape = "+str(npArrTestY.shape))
    
    # read data and do upsampling
    def _readNpArrXYFromDisk_retrain(self,dtypeStrX):
        if self._cnn_type is None:
            train_data_ = pd.read_csv(self._pathToDataFolder + self._new_devices_postfix + '/relabeled_filtered_new_devices_train_data.csv')
        else:
            train_data_ = pd.read_csv(self._pathToDataFolder + self._new_devices_postfix + '/relabeled_filtered_new_devices_train_data({}).csv'.format(self._cnn_type))
        train_data = pd.read_csv(self._pathToDataFolder + self._new_devices_postfix + '/old_devices_train_data.csv')       
        test_data = pd.read_csv(self._pathToDataFolder + self._new_devices_postfix + '/old_devices_test_data.csv')
        test_data_ = pd.read_csv(self._pathToDataFolder + self._new_devices_postfix + '/filtered_new_devices_test_data.csv')

        train_data,test_data = rearrange_labels(train_data,train_data_,test_data,test_data_,self._pathToDataFolder + self._new_devices_postfix)
        
        train_data['generated_data'] = 1
        test_data['generated_data'] = 0
        train_num_labels = np.unique(train_data['label'])
        print(f'train data unique labels:{len(train_num_labels)}')


        raw_data = pd.concat([train_data,test_data],axis=0)
        print(raw_data.shape)
        if(self._upsampling==True):
            raw_data_ = self._upsamplingF_retrain(raw_data)   
            raw_data = raw_data_            
        raw_data = raw_data.values
        cols_ = raw_data.shape[1]           
        if(self._upsampling==True) or (self._retrain==True):
            X = raw_data[:,1:cols_-1]
            Y_generated = raw_data[:,cols_-1]
        else:
            X = raw_data[:,1:]
            Y_generated = None
        Y = raw_data[:,0]
        X = X.reshape(X.shape[0],-1)
        return (X,Y,Y_generated)      
    
    # the generated_data col has been filled, only sample generated_data == 1 data
    def _upsamplingF_retrain(self,data):
        data_resampled_all = None
        labels_set = np.unique(data['label'])
        for label in labels_set:
            data_train_c_indicator = (data['label'] == label) & (data['generated_data'] == 1)
            data_train_c = data[data_train_c_indicator]
            if len(data_train_c) < self._lbl_samples_small and len(data_train_c) > 0:
                data_resampled = resample(data_train_c,replace=True,n_samples=self._num_least,random_state=self._seed)
                if data_resampled_all is None:
                    data_resampled_all = data_resampled
                else:
                    data_resampled_all = pd.concat([data_resampled_all,data_resampled],axis=0)
        if not data_resampled_all is None:
            print('resample {} training data'.format(len(data_resampled_all)))
        data_all = pd.concat([data,data_resampled_all],axis=0)
        return data_all

    def _readNpArrXYFromDisk(self, filepathToData, boolOneHot, dtypeStrX):
        # X: [samples, features]. 
        (X,Y,Y_generated) = self._timeReading(self.readIotFromDisk, filepathToData)
        # Y: [samples]
        dataX = self._preprocessX(X, dtypeStrX)
        labelsY = self._preprocessY(Y, boolOneHot, "uint8")

        return (dataX, labelsY, Y_generated)
    
    
    def _upsamplingF(self,data):
    
    # function:upsampling used to balance data. 
    # @parameters X: data features, Y: labels, num_least:least num a class must have, if the num of class exceed
    # num_least, data of the class will not change. otherwise more data will be added to the class. epsilon:
    # disturbance added to the replicated data. 
        data['generated_data'] = 0
        num_each_classes = [len(data[data['label']==label]) for label in range(int(data['label'].max())+1)]   
        
        for idx,num in enumerate(num_each_classes):         
            if num == 1:
                data_c = data[data['label']==idx]
                
                data_resampled = resample(data_c, 
                                        replace=True,     # sample with replacement
                                        n_samples=self._num_least,    # to match majority class
                                        random_state=self._seed) # reproducible results
                data_resampled['generated_data'] = 1
                data = pd.concat([data,data_resampled],axis = 0)
            elif num < self._lbl_samples_small and num > 0: # self._percent_lbl_samples_small
                # split first
                data_c = data[data['label']==idx]
                data.drop(data[data['label']==idx].index,inplace=True)
                # train_num = max(int(self._percent_lbl_samples_small*data_c.shape[0]),1)
                train_num = max(int(self._train_ratio*data_c.shape[0]),1)
                # data_c = shuffle(data_c)
                data_c_train = data_c[:train_num]
                data_c_test = data_c[train_num:]
                data_c_train['generated_data'] = 1
                if(num < self._num_least):
                    data_resampled = resample(data_c_train, 
                                            replace=True,     # sample with replacement
                                            n_samples=self._num_least,    # to match majority class
                                            random_state=self._seed) # reproducible results
                    # print(data_resampled['generated_data'])
                    data = pd.concat([data,data_resampled,data_c_test],axis = 0)

        return data 
    

    def _preprocessX(self, npArrX, dtypeStr, first_flag = 1):# first_flag=1 stands for scale
        # Convert from [0, 255] -> [0.0, 1.0].
        if "int" in dtypeStr:
            preprocX = npArrX.astype(dtypeStr)
        elif "float" in dtypeStr:
            npArrX = npArrX.astype(dtypeStr)
            # scaler_ins = MinMaxScaler(feature_range=(-1,1)),RobustScaler  
            if first_flag == 1:
                self._scaler_ins = self._scaler_ins.fit(npArrX)         
            preprocX = self._scaler_ins.transform(npArrX)
        else:
            raise NotImplementedError()
        return preprocX
    
    # change it to one hot 
    def _preprocessY(self, npArrY, boolOneHot, dtypeStr="uint8"):
        if not "int" in dtypeStr:
            raise NotImplementedError()
        npArrY = npArrY.astype(dtypeStr)
        kinds = len(np.unique(npArrY))
        if boolOneHot:
            npArrY = makeArrayOneHot(npArrY, kinds, 1) #!!!
        return npArrY
    
    # Adapted from tensorflow/contrib/learn/python/learn/datasets/mnist.py
    # def _read32(self, bytestream):
    #     dt = np.dtype(np.uint32).newbyteorder('>')
    #     return np.frombuffer(bytestream.read(4), dtype=dt)[0]
      
    def readIotFromDisk(self, filepath):
        """Extract the csv file into a 2D float np array [index, features].
        Returns:
        data: A 2D float np array [index, features].
        Raises:
        ValueError: 
        """
        with open(filepath, 'rb') as f:
            LOG.info('Extracting '+str(f.name))
            raw_data = pd.read_csv(filepath)
            self._column_names = raw_data.columns
            # rearrange labels
            if self._new_devices_list is not None:
                raw_data = self.renew_without_new_devices(raw_data)
            if(self._upsampling==True):
                raw_data_ = self._upsamplingF(raw_data)   
                raw_data = raw_data_            
            raw_data = raw_data.values
            cols_ = raw_data.shape[1]           
            if(self._upsampling==True):
                X = raw_data[:,1:cols_-1]
                Y_generated = raw_data[:,cols_-1]
            else:
                X = raw_data[:,1:]
                Y_generated = None
            Y = raw_data[:,0]
            # change Y to 0 or 1 for a small test!!!
            # for idx,item in enumerate(Y):
            #     if item!=24:
            #         Y[idx] = 1 # 1 stands for IoT
            #     else:
            #         Y[idx] = 0
            X = X.reshape(X.shape[0],-1)
            
            # Y = Y.reshape(-1,1)
            return (X,Y,Y_generated)      
            
    # delete new_devices and renew labels to continous number, input raw_data is all devices instance
    def renew_without_new_devices(self, raw_data):
        # store new devices data to csv
        new_devices_data = None
        for device_index in self._new_devices_list:
            data_ = raw_data[raw_data['label']==device_index]
            if new_devices_data is None:
                new_devices_data = data_
            else:
                new_devices_data = pd.concat([new_devices_data,data_],axis=0)
        
        # new_device_data replace label with minus value
        for device_idx in self._new_devices_list:
            if device_idx == 0:
                new_devices_data['label'] = new_devices_data['label'].replace(device_idx,-1*1000)
                self._old_new_device_idx_dict[int(device_idx)] = -1000
            else:
                new_devices_data['label'] = new_devices_data['label'].replace(device_idx,-1*device_idx)
                self._old_new_device_idx_dict[int(device_idx)] = int(-1*device_idx)
         
        for new_device in self._new_devices_list:
            raw_data.drop(raw_data[raw_data['label']==new_device].index, inplace=True)

        # 将niot label>self._niot_label的全替换成self._niot_label
        # judge_new_all_niot = True
        # for device_idx in self._new_devices_list:
        #     if device_idx < self._niot_label:
        #         judge_new_all_niot = False
        # if judge_new_all_niot == True: # 将剩下的大于等于niot_label的标签全替换成niot_label

        # rearrange labels of old data,replace label bigger than self._niot_label to self._niot_label
        label_max = max(np.unique(raw_data['label']))
        for device_idx in range(self._niot_label+1,label_max+1):
            raw_data['label'] = raw_data['label'].replace(device_idx,self._niot_label) #后面是替换后的值
            self._old_new_device_idx_dict[int(device_idx)] = int(self._niot_label)
        
        blank_list = copy.copy(list(filter(lambda x: x < self._niot_label,self._new_devices_list))) # 去掉>= self._niot_label的标签
        whole_list = list(range(max(raw_data['label'])+1))
        whole_list_np = np.array(whole_list)
        end_list = list(range(max(whole_list)+1-len(blank_list),max(whole_list)+1))
        while(set(blank_list) != set(end_list)):
            
            blank_now = min(blank_list)
            whole_list_filtered = list(filter(lambda x: x > blank_now, whole_list))
            whole_list_filtered_np = np.array(whole_list_filtered)
            if len(whole_list_filtered) == 0:
                break
            replaced_ele = min(whole_list_filtered_np[np.invert(np.isin(whole_list_filtered,blank_list))]) # 不在blank中的最小标签值
            raw_data['label'] = raw_data['label'].replace(replaced_ele,blank_now) #后面是替换后的值
            self._old_new_device_idx_dict[int(replaced_ele)] = int(blank_now)
            blank_list.append(replaced_ele)
            blank_list.remove(blank_now)
        # compute new niot_label
        self._niot_label = max(np.unique(raw_data['label']))

        # then merge the new_device_data with old data
        # each_old_category_number = 200 #-----------
        # merged_device_data = new_devices_data
        # for label in range(max(raw_data['label'])+1):
        #     data_ = raw_data[raw_data['label']==label].sample(frac=0.5)
        #     merged_device_data = pd.concat([merged_device_data, data_],axis = 0)

        # save self._old_new_device_idx_dict
        f = open(self._pathToDataFolder + self._new_devices_postfix + '/old_new_device_idx_dict.json','w')
        json.dump(self._old_new_device_idx_dict,f)
        f.close()
        self._new_devices_data_without_normalized = new_devices_data
        return raw_data
    
    #------------------------------------
    # delete new_devices and renew labels to continous number, input raw_data is all devices instance
    def renew_without_new_devices_backup(self, raw_data):
        # store new devices data to csv
        new_devices_data = None
        for device_index in self._new_devices_list:
            data_ = raw_data[raw_data['label']==device_index]
            if new_devices_data is None:
                new_devices_data = data_
            else:
                new_devices_data = pd.concat([new_devices_data,data_],axis=0)
        
        # new_device_data replace label with minus value
        for device_idx in self._new_devices_list:
            if device_idx == 0:
                new_devices_data['label'] = new_devices_data['label'].replace(device_idx,-1*1000)
                self._old_new_device_idx_dict[int(device_idx)] = -1000
            else:
                new_devices_data['label'] = new_devices_data['label'].replace(device_idx,-1*device_idx)
                self._old_new_device_idx_dict[int(device_idx)] = int(-1*device_idx)
         
        for new_device in self._new_devices_list:
            raw_data.drop(raw_data[raw_data['label']==new_device].index, inplace=True)

        # 将niot label>self._niot_label的全替换成self._niot_label
        # judge_new_all_niot = True
        # for device_idx in self._new_devices_list:
        #     if device_idx < self._niot_label:
        #         judge_new_all_niot = False
        # if judge_new_all_niot == True: # 将剩下的大于等于niot_label的标签全替换成niot_label

        # rearrange labels of old data,replace label bigger than self._niot_label to self._niot_label
        label_max = max(np.unique(raw_data['label']))
        for device_idx in range(self._niot_label+1,label_max+1):
            raw_data['label'] = raw_data['label'].replace(device_idx,self._niot_label) #后面是替换后的值
            self._old_new_device_idx_dict[int(device_idx)] = int(self._niot_label)
        
        blank_list = copy.copy(self._new_devices_list)
        whole_list = list(range(max(raw_data['label'])+1))
        whole_list_np = np.array(whole_list)
        end_list = list(range(max(whole_list)+1-len(blank_list),max(whole_list)+1))
        while(set(blank_list) != set(end_list)):
            
            blank_now = min(blank_list)
            whole_list_filtered = list(filter(lambda x: x > blank_now, whole_list))
            whole_list_filtered_np = np.array(whole_list_filtered)
            if len(whole_list_filtered) == 0:
                break
            replaced_ele = min(whole_list_filtered_np[np.invert(np.isin(whole_list_filtered,blank_list))]) # 不在blank中的最小标签值
            raw_data['label'] = raw_data['label'].replace(replaced_ele,blank_now) #后面是替换后的值
            self._old_new_device_idx_dict[int(replaced_ele)] = int(blank_now)
            blank_list.append(replaced_ele)
            blank_list.remove(blank_now)
        # compute new niot_label
        self._niot_label = max(np.unique(raw_data['label']))

        # then merge the new_device_data with old data
        # each_old_category_number = 200 #-----------
        # merged_device_data = new_devices_data
        # for label in range(max(raw_data['label'])+1):
        #     data_ = raw_data[raw_data['label']==label].sample(frac=0.5)
        #     merged_device_data = pd.concat([merged_device_data, data_],axis = 0)

        # save self._old_new_device_idx_dict
        f = open(self._pathToDataFolder + self._new_devices_postfix + '/old_new_device_idx_dict.json','w')
        json.dump(self._old_new_device_idx_dict,f)
        f.close()
        self._new_devices_data_without_normalized = new_devices_data
        return raw_data

    #------------------------------------

    def _normalize_and_store_merged_data(self,new_devices_list):
        raw_data = self._new_devices_data_without_normalized # pd format
        if self._record_data_index == True:
            raw_data['index'] = raw_data.index
            raw_data.to_csv(self.new_devices_dir + '/new_devices_with_index_before_normalization.csv',index=False)
        columns = list(raw_data.columns)
        raw_data = raw_data.values
        if self._record_data_index == True:
            cols_ = raw_data.shape[1] - 1
        else:
            cols_ = raw_data.shape[1]
        X = self._preprocessX(raw_data[:,1:cols_], self._dtypeStrX, first_flag=0)
        Y = raw_data[:,0].reshape(-1,1)
        index_col = raw_data[:,-1].reshape(-1,1)
        data = np.concatenate([Y,X],axis = 1)
        if self._record_data_index == True:
            data = np.concatenate([data,index_col],axis=1)
        data_pd = pd.DataFrame(data,columns=columns)
        train_data,test_data = train_test_split(data_pd,self._train_ratio)
        new_devices_str = '_'.join(str(item) for item in self._new_devices_list)
        train_data.to_csv(self.new_devices_dir + "/new_devices_train_data.csv",index=False,header = columns)
        test_data.to_csv(self.new_devices_dir + "/new_devices_test_data.csv",index=False,header = columns)
        print('new_devices_data has been stored in {}'.format(new_devices_str))
    # read_merged_data, do normalization and return numpy data
    def read_merged_devices_data(self):
        raw_data = pd.read_csv(self._pathToDataFolder + '/' + self._merged_devices_file)
        raw_data = raw_data.values()
        cols_ = raw_data.shape[1]
        return (raw_data[:,1:cols_], raw_data[:,0])
        # X = raw_data[]


    


    
        
        
        