#!/usr/bin/env python
# modified by Linna Fan
# Copyright (c) 2018, Konstantinos Kamnitsas
#
# This program is free software; you can redistribute and/or modify
# it under the terms of the Apache License, Version 2.0. See the 
# accompanying LICENSE file or read the terms at:
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import absolute_import, division, print_function

import logging
LOG = logging.getLogger('main')
import sys
import os
from functools import partial

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.training import saver as tf_saver

from cclp.sampling.preprocessing import normalize_tens
from cclp.sampling.augmentation import augment_tens
from cclp.data_utils import data_managers
from cclp.data_utils import plot_utils 
from cclp.neuralnet.models.feat_extractor_archs import get_feat_extractor_z_func
from cclp.embeddings.visualise_embeddings import plot_2d_emb,plot_tsne
from cclp.frontend.logging.utils import datetime_str as timestr
from cclp.frontend.logging import metrics
from cclp.sampling import samplers
from cclp.neuralnet.models import classifier
from cclp.neuralnet.trainers import trainers
from cclp.data_utils import compute_utils
import time
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score,f1_score,recall_score
from configs import local_dirs_to_data
from cclp.neuralnet.models.utils import EarlyStopping


import json

plot_tsne_flag = True

score_micro_flag = False
score_micro = []
score = {"accuracy_score":0,"precision_score":{},"recall_score":{},"f1_score":{}}

val_error_recorder = []
val_error_timer = []
new_devices_accuracy = {} # accuracy dict of new_devices in new_devices_list

cm_final = None
## compute last 2 value,if last value exceed a threshold,we assume it is a NoT then its label is 24, 
# else it is determined by argmax() of the 0-24 value, input is the a batch of logits result of neural network,
# result is the list of final lbls
def get_cm_str(cm):
    result = "["
    for idx,row in enumerate(cm):
        row = list(row)
        row_ = [str(item) for item in row]
        tmp = ','.join(list(row_))
        if idx == cm.shape[1]-1:
            tmp_ = "["+ tmp + "]"
        else:
            tmp_ = "["+ tmp + "],\n"
        result += tmp_
    result += "]"
    return result
def get_lbls_of_multi_task_model(data,num_classes=25,threshold=0.7):
    final_lbls = []
    rows = data.shape[0]
    cols = data.shape[1]
    for item in data:
        last = item[cols-1] #not
        last_sub_1 = item[cols-2] #iot
        M = max(last,last_sub_1)
        last = last-M # prevent overflow of softmax, last==1 stands for NoT
        last_sub_1 = last_sub_1-M 
        prob_IoT = math.exp(last_sub_1)*1.0/(math.exp(last_sub_1)+math.exp(last)) # softmax computation
        if prob_IoT>=threshold:
            lbl_ = item[0:num_classes].argmax(-1) 
            final_lbls.append(lbl_)
        else:
            final_lbls.append(num_classes-1)
    return final_lbls

def get_lbls_of_multi_task_model3(data,num_classes=25,threshold=0.7):
    final_lbls = []
    rows = data.shape[0]
    cols = data.shape[1]
    for item in data:
        last = item[cols-1] #iot
        last_sub_1 = item[cols-2] #not
        M = max(last,last_sub_1)
        last = last-M # prevent overflow of softmax, last==1 stands for NoT
        last_sub_1 = last_sub_1-M 
        prob_IoT = math.exp(last)*1.0/(math.exp(last_sub_1)+math.exp(last)) # softmax computation
        if prob_IoT>=threshold:
            lbl_ = item[0:num_classes].argmax(-1) 
            final_lbls.append(lbl_)
        else:
            final_lbls.append(num_classes-1)
    return final_lbls

def get_lbls_of_multi_task_model2(data,data2,num_classes=25,threshold=0.7):#data is logit(25 label), data2 is logit2(IoT/NoT)
    final_lbls = []
    rows = data.shape[0]
    cols = data.shape[1]
    for idx in range(rows):
        iot_ = data2[idx,0] #iot prob
        not_ = data2[idx,1] #not prob
        M = max(iot_,not_)
        iot_ = iot_-M # prevent overflow of softmax, last==1 stands for NoT
        not_ = not_-M 
        prob_IoT = math.exp(iot_)*1.0/(math.exp(iot_)+math.exp(not_)) # softmax computation
        if prob_IoT>=threshold:
            lbl_ = data[idx,0:num_classes].argmax(-1) 
            final_lbls.append(lbl_)
        else:
            final_lbls.append(num_classes-1)
    return final_lbls

def get_lbls_of_multi_task_model4(data,data2,num_classes=25,threshold=0.7):#data is logit(25 label), data2 is logit2(IoT/NoT)
    final_lbls = []
    rows = data.shape[0]
    cols = data.shape[1] #24
    print('logits1 shape',data.shape)
    for idx in range(rows):
        iot_ = data2[idx,0] #iot prob
        not_ = data2[idx,1] #not prob
        M = max(iot_,not_)
        iot_ = iot_-M # prevent overflow of softmax, last==1 stands for NoT
        not_ = not_-M 
        prob_IoT = math.exp(iot_)*1.0/(math.exp(iot_)+math.exp(not_)) # softmax computation
        if prob_IoT>=threshold:
            lbl_ = data[idx,0:num_classes-1].argmax(-1) 
            final_lbls.append(lbl_)
        else:
            final_lbls.append(num_classes-1)
    return final_lbls

# load data-> build model-> train model -> validation
def train(sessionNModelFlags, trainerFlags, retrain=False, tl=False, cnn_type=None):
    global score
    eval_emb_all = []
    val_error_rate = []
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN) # set to INFO for more exhaustive.
    config = tf.compat.v1.ConfigProto(log_device_placement = False) #don't print device assignment logs
    config.gpu_options.allow_growth = False # don't allow dynamicly request memory
    
    # Load data.
    if sessionNModelFlags["dataset"] == 'iot':
        db_manager = data_managers.IotManager(dtypeStrX="float",
                                            num_least=sessionNModelFlags['num_least'],
                                            upsampling=sessionNModelFlags['upsampling'],
                                            epsilon=sessionNModelFlags['epsilon'],
                                            seed=sessionNModelFlags['seed'],
                                            disturbe=sessionNModelFlags['disturbe'],
                                            prob=sessionNModelFlags['prob'],
                                            percent_lbl_samples=sessionNModelFlags['percent_lbl_samples'],
                                            new_devices_list = sessionNModelFlags['new_devices_list'],
                                            train_ratio = sessionNModelFlags['train_ratio'],
                                            retrain = retrain,
                                            dataset_name = sessionNModelFlags['dataset_name'],
                                            record_data_index = sessionNModelFlags['record_data_index'],
                                            niot_label = sessionNModelFlags['niot_label'],
                                            upsample_niot = sessionNModelFlags['upsample_niot'],
                                            percent_lbl_samples_small = sessionNModelFlags['percent_lbl_samples_small'],
                                            lbl_samples_small = sessionNModelFlags['lbl_samples_small'],
                                            cnn_type=cnn_type,
                    
                                            )
        
    LOG.info("")
    LOG.info("Before pre-augmentation normalization...")
    db_manager.print_characteristics_of_db(pause=False)
    # Data normalization
    db_manager.normalize_datasets(sessionNModelFlags["norm_imgs_pre_aug"])
    
    LOG.info("")
    LOG.info("After pre-augmentation normalization...")
    # train/val/unlab images in tensor of shape: [batch, x, y, channels]
    db_manager.print_characteristics_of_db(pause=False)
    
    # Sample Validation, labeled and unlabeled training data. all data
    seed = sessionNModelFlags["seed"]
    (train_samples_lbl_list_by_class,
    train_samples_unlbl,
    val_samples, val_labels) = db_manager.sample_folds(sessionNModelFlags['val_on_test'], 
                                                       sessionNModelFlags['num_val_samples'],
                                                       sessionNModelFlags['num_lbl_samples'],
                                                       sessionNModelFlags['num_unlbl_samples'],
                                                       sessionNModelFlags['unlbl_overlap_val'],
                                                       sessionNModelFlags['unlbl_overlap_lbl'],
                                                       seed=seed,
                                                       retrain=retrain
                                                       )
    
    num_classes = db_manager.datasetsDict['train'].num_classes
    image_shape = db_manager.datasetsDict['train'].getShapeOfAnImage() # [x, y, channs]. Without the batch size.
    str_val_or_test = "Test" if sessionNModelFlags['val_on_test'] else "Validation"
    #added
    if sessionNModelFlags['n_lbl_per_class_per_batch'] > sessionNModelFlags['num_lbl_samples']//num_classes:
        sessionNModelFlags['n_lbl_per_class_per_batch'] = sessionNModelFlags['num_lbl_samples']//num_classes
    
    # early stop
    stopper = EarlyStopping(patience=3)
    # Up until here, everything is numpy. Tensorflow starts below:
    graph = tf.Graph()
    with graph.as_default():
        with tf.device(tf.compat.v1.train.replica_device_setter(ps_tasks=0, merge_devices=True)):
            LOG.info("==========================================================")
            LOG.info("================== Creating the graph ====================")
            LOG.info("==========================================================\n")
            
            #!!! Get batch. Tensorflow batch-constructors. They return tensorflow tensors. Shape: (batch, H, W, Channs)
            t_sup_images, t_sup_labels = samplers.get_tf_batch_with_n_samples_per_class( train_samples_lbl_list_by_class, sessionNModelFlags["n_lbl_per_class_per_batch"], seed )
            t_unsup_images, _ = samplers.get_tf_batch(train_samples_unlbl, None, sessionNModelFlags["n_unlbl_per_batch"], seed)
            
            # Partially define augmentation, renormalization and the feature extractor functions...
            # ... So that they can later be applied on an input tensor straightforwardly.
            # Apply augmentation
            is_train_db_zero_centered = db_manager.datasetsDict['train'].get_db_stats() # to augment is accordingly.
            augment_tens_func = partial(augment_tens, params=sessionNModelFlags["augm_params"], db_zero_centered=is_train_db_zero_centered)
            # Re-normalize after augmentation if wanted.
            normalize_tens_func = partial(normalize_tens, norm_type=sessionNModelFlags["norm_tens_post_aug"])
            # Create function that the feature extractor z(.)
            embed_tens_func = partial( get_feat_extractor_z_func(sessionNModelFlags["feat_extractor_z"]), # function pointer.
                                       emb_z_size=sessionNModelFlags["emb_z_size"],
                                       emb_z_act=sessionNModelFlags["emb_z_act"],
                                       l2_weight=sessionNModelFlags["l2_reg_feat"],
                                       batch_norm_decay=sessionNModelFlags["batch_norm_decay"],
                                       seed=seed )
            
            # Initialize the model class (it does not yet build tf computation graph.)
            model = classifier.Classifier( augment_tens_func,
                                           normalize_tens_func,
                                           embed_tens_func,
                                           num_classes,
                                           image_shape,
                                           sessionNModelFlags["l2_reg_classif"],
                                           sessionNModelFlags["classifier_dropout"],
                                           use_multi_task_model=trainerFlags['use_multi_task_model'] )
            
            # Build the tf-graph for the forward pass through feature extractor z and classifier.
            model.forward_pass(input=None, tensor_family="eval", is_training=False, init_vars=True, seed=seed, image_summary=False)#variable initiallize
            summary_op_eval = tf.summary.merge_all(key="eval_summaries")
            model.forward_pass(input=t_sup_images, tensor_family="train_sup", is_training=True, init_vars=False, seed=seed, image_summary=False)
            model.forward_pass(input=t_unsup_images, tensor_family="train_unsup", is_training=True, init_vars=False, seed=seed, image_summary=False)
            # predict_eval_logits = model.tensor_families["eval"]["logits_tens"]
            
            # compute loss and do BP
            trainer = trainers.Trainer( params = trainerFlags, net_model=model, t_sup_labels=t_sup_labels)#!!!
            train_op = trainer.get_train_op()
            increase_model_step_op = trainer.get_increase_model_step_op()
            
            saver = tf_saver.Saver(max_to_keep=sessionNModelFlags["max_checkpoints_to_keep"])
            
            summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES) # This is what makes the tensorboard's metrics.
            summary_writer = tf.summary.FileWriter( sessionNModelFlags["logdir"]["summariesTf"], graph ) # creates the subfolder if not there already.
    
        if retrain == True:
            if tl==True:
                vars = tf.global_variables()     
                vars_to_restore = [var for var in vars if 'compute_logits_name_scope' not in var.name and 'global_step' not in var.name]   
                vars_to_init = [var for var in vars if 'compute_logits_name_scope' in var.name or 'global_step' in var.name]
                saver = tf.train.Saver(vars_to_restore)     
       
    with tf.Session(graph=graph, config=config) as sess:
        LOG.info(".......................................................................")
        if sessionNModelFlags['model_to_load'] is not None:#load previous model
            chkpt_fname = tf.train.latest_checkpoint( sessionNModelFlags['model_to_load'] ) if os.path.isdir( sessionNModelFlags['model_to_load'] ) else sessionNModelFlags['model_to_load'] 
            LOG.info("Loading model from: "+str(chkpt_fname))
            saver.restore(sess, chkpt_fname)
        else:# train a new model
            if retrain == True:
                
                LOG.info("load model given")
                main_model_path = './output/iot/trainTf/'
                new_devices_postfix = '_'.join(str(item) for item in sessionNModelFlags['new_devices_list'])
                if tl==True:
                    saver.restore(sess,tf.train.latest_checkpoint(main_model_path + new_devices_postfix))
                    tf.initialize_variables(vars_to_init).run() # must run
                else:
                    LOG.info("Initializing var of last layer...")
                    tf.global_variables_initializer().run() # must run
            else:
                LOG.info("Initializing all vars...")
                tf.global_variables_initializer().run() # must run
        LOG.info(".......................................................................\n")
        
        coordinator = tf.train.Coordinator() # Coordinates the threads. Starts and stops them.
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator) # Start the slicers/batchers registered in cclp.sampling.samplers.
        time_start = time.clock()
        LOG.info("====================================================================")
        LOG.info("================== Starting training iterations ====================")
        LOG.info("====================================================================\n")
        model_step = model.get_t_step().eval(session=sess)
        best_accuracy = 0
        while model_step < sessionNModelFlags["max_iters"] :
            
            (_,
            summaries,
            emb_train_sup,
            emb_train_unsup,
            nparr_t_sup_labels
            # np_t_sup_images,
            # np_t_unsup_images,
            # np_t_unsup_labels
            ) = sess.run([ train_op,
                        summary_op,
                        model.tensor_families["train_sup"]["emb_z_tens"],
                        model.tensor_families["train_unsup"]["emb_z_tens"],
                        t_sup_labels
                        # t_sup_images,
                        # t_unsup_images,
                        # t_unsup_labels 
                        ]) # This is the main training step.

            model_step = sess.run( increase_model_step_op )
            
            # The two following external if's are the same and could be merged.
            # Save summaries, create visualization of embedding if requested.
            if (model_step) % sessionNModelFlags["val_interval"] == 0 or model_step == 100:
                print(model_step)
            # if (model_step) == sessionNModelFlags["max_iters"]:
            if (model_step) % sessionNModelFlags["val_interval"] == 0 or model_step == 100:
            #     print(model_step)
                LOG.info('Step: %d' % model_step)
                summary_writer.add_summary(summaries, model_step) # log TRAINING metrics.
                              
                # PLOT AND SAVE EMBEDDING
                if sessionNModelFlags["plot_save_emb"] > 0 :
                    plot_2d_emb(emb_train_sup, emb_train_unsup, nparr_t_sup_labels, train_step=model_step,save_emb=sessionNModelFlags["plot_save_emb"] in [1,3], plot_emb=sessionNModelFlags["plot_save_emb"] in [2,3], output_folder=sessionNModelFlags["logdir"]["emb"] )               
                # ------------------------------------------------------------------    
                # ACCURACY ON TRAINING DATA.
                train_images_for_metrics = np.concatenate( train_samples_lbl_list_by_class, axis=0 )
                num_samples_per_class_train_to_eval = train_samples_lbl_list_by_class[0].shape[0]
                train_gt_lbls_for_metrics = [] # ground truth label 
                if(sessionNModelFlags['percent_lbl_samples']>0):# set train_gt_lbls_for_metrics
                    for c in range(0,num_classes):
                        train_gt_lbls_for_metrics += [c] * train_samples_lbl_list_by_class[c].shape[0]
                else:
                    for c in range(0, num_classes):
                        train_gt_lbls_for_metrics += [c] * num_samples_per_class_train_to_eval
                
                train_gt_lbls_for_metrics = np.asarray(train_gt_lbls_for_metrics)
                
                train_pred_lbls_for_metrics = []
                for i in range(0, len(train_images_for_metrics), sessionNModelFlags["eval_batch_size"]):#fetch a batch for evaluation
                    if trainerFlags['use_multi_task_model']==1 or trainerFlags['use_multi_task_model']==0 or trainerFlags['use_multi_task_model']==3:
                        [train_pred_logits_batch] =  sess.run( [ model.tensor_families["eval"]["logits_tens"] ],
                                                            { model.tensor_families["eval"]["inp_tens"] : train_images_for_metrics[ i : i+sessionNModelFlags["eval_batch_size"] ] } )
                        # print(train_pred_logits_batch)
                        if trainerFlags['use_multi_task_model']==1:#added
                            train_pred_lbls_for_metrics.append(get_lbls_of_multi_task_model(train_pred_logits_batch,num_classes,sessionNModelFlags['threshold'])) # compute last 2 value,if last value exceed a threshold,we assume it is a NoT then its label is 24, else it is determined by argmax() of the 0-24 value
                        elif trainerFlags['use_multi_task_model']==3:
                            train_pred_lbls_for_metrics.append(get_lbls_of_multi_task_model3(train_pred_logits_batch,num_classes,sessionNModelFlags['threshold']))
                        else:
                            train_pred_lbls_for_metrics.append( train_pred_logits_batch.argmax(-1) ) # !!!Take the classes with argmax.
                    else: #==2 or 4
                        [train_pred_logits_batch,train_pred_logits_batch2] =  sess.run( [ model.tensor_families["eval"]["logits_tens"],model.tensor_families["eval"]["logits_tens_2"]],
                                                            { model.tensor_families["eval"]["inp_tens"] : train_images_for_metrics[ i : i+sessionNModelFlags["eval_batch_size"] ] } )    
                        if trainerFlags['use_multi_task_model']==2:
                            train_pred_lbls_for_metrics.append(get_lbls_of_multi_task_model2(train_pred_logits_batch,train_pred_logits_batch2,num_classes,sessionNModelFlags['threshold'])) # compute last 2 value,if last value exceed a threshold,we assume it is a NoT then its label is 24, else it is determined by argmax() of the 0-24 value
                        else: #==4
                            train_pred_lbls_for_metrics.append(get_lbls_of_multi_task_model4(train_pred_logits_batch,train_pred_logits_batch2,num_classes,sessionNModelFlags['threshold'])) # compute last 2 value,if last value exceed a threshold,we assume it is a NoT then its label is 24, else it is determined by argmax() of the 0-24 value
                train_pred_lbls_for_metrics = np.concatenate(train_pred_lbls_for_metrics)
                if train_pred_lbls_for_metrics.shape[0] > train_gt_lbls_for_metrics.shape[0]: # can happen when superv data = -1. 59230 total data, batcher fills the last batches and the whole thing returns 60000? Not sure.
                    train_pred_lbls_for_metrics = train_pred_lbls_for_metrics[ : train_gt_lbls_for_metrics.shape[0] ]              
                train_err = (train_gt_lbls_for_metrics != train_pred_lbls_for_metrics).mean() * 100 # What is reported is percentage.
                train_summary = tf.Summary( value=[ tf.Summary.Value( tag='Train Err', simple_value=train_err) ] )
                summary_writer.add_summary(train_summary, model_step)
                
                confusion_mtx = metrics.confusion_mtx(train_gt_lbls_for_metrics, train_pred_lbls_for_metrics, num_classes)
                LOG.info("\n"+str(confusion_mtx))
                LOG.info('Mean training error: %.2f %% \n' % train_err)
                # add      
                error_percent = (np.sum(train_gt_lbls_for_metrics != train_pred_lbls_for_metrics))/len(train_gt_lbls_for_metrics)*100
                LOG.info('Total training error: %.2f %% \n' % error_percent)
                tf.summary.scalar('train_error',error_percent)
                
                # ACCURACY ON VALIDATION DATA.
                if sessionNModelFlags["val_during_train"]:
                    
                    eval_pred_lbls = [] # list of embeddings for each val batch: [ [batchSize, 10], .... [bs, 10] ]
                    
                    for i in range(0, len(val_samples), sessionNModelFlags["eval_batch_size"]):
                        if trainerFlags['use_multi_task_model']==1 or trainerFlags['use_multi_task_model']==0 or trainerFlags['use_multi_task_model']==3:
                            [eval_pred_logits_batch,
                            summaries_eval] =  sess.run( [ model.tensor_families["eval"]["logits_tens"], summary_op_eval ],
                                                        { model.tensor_families["eval"]["inp_tens"] : val_samples[ i : i+sessionNModelFlags["eval_batch_size"] ] } )
                            if trainerFlags['use_multi_task_model']==1:#added
                                eval_pred_lbls.append(get_lbls_of_multi_task_model(eval_pred_logits_batch,num_classes)) # compute last 2 value,if last value exceed a threshold,we assume it is a NoT then its label is 24, else it is determined by argmax() of the 0-24 value
                            else:                       
                                eval_pred_lbls.append( eval_pred_logits_batch.argmax(-1) ) #!!! Take the classes with argmax.
                        else:
                            if trainerFlags['use_multi_task_model']==2:
                                [eval_pred_logits_batch,eval_pred_logits_batch2,summaries_eval] =  sess.run( [ model.tensor_families["eval"]["logits_tens"],model.tensor_families["eval"]["logits_tens_2"],summary_op_eval],
                                                                { model.tensor_families["eval"]["inp_tens"] : val_samples[ i : i+sessionNModelFlags["eval_batch_size"] ] } )    
                                eval_pred_lbls.append(get_lbls_of_multi_task_model2(eval_pred_logits_batch,eval_pred_logits_batch2,num_classes,sessionNModelFlags['threshold'])) # compute last 2 value,if last value exceed a threshold,we assume it is a NoT then its label is 24, else it is determined by argmax() of the 0-24 value
                            else: # ==4
                                [eval_pred_logits_batch,eval_pred_logits_batch2,summaries_eval] =  sess.run( [ model.tensor_families["eval"]["logits_tens"],model.tensor_families["eval"]["logits_tens_2"],summary_op_eval],
                                                                { model.tensor_families["eval"]["inp_tens"] : val_samples[ i : i+sessionNModelFlags["eval_batch_size"] ] } )    
                                eval_pred_lbls.append(get_lbls_of_multi_task_model4(eval_pred_logits_batch,eval_pred_logits_batch2,num_classes,sessionNModelFlags['threshold'])) # compute last 2 value,if last value exceed a threshold,we assume it is a NoT then its label is 24, else it is determined by argmax() of the 0-24 value
                                #--------------------------------------------------------------------------------
                                if plot_tsne_flag == True and model_step == sessionNModelFlags["max_iters"]: #and model_step == sessionNModelFlags["max_iters"]//2:
                                    eval_emb = sess.run(model.tensor_families["eval"]["emb_z_tens"],{ model.tensor_families["eval"]["inp_tens"] : val_samples[ i : i+sessionNModelFlags["eval_batch_size"] ] })
                                    eval_emb_all.append(eval_emb)
                           

                    eval_pred_lbls = np.concatenate(eval_pred_lbls) # from list to array.
                    eval_err = (val_labels != eval_pred_lbls).mean() * 100 # report percentage.
                    eval_summary = tf.Summary( value=[ tf.Summary.Value( tag=str_val_or_test+' Err', simple_value=eval_err) ] )
                    summary_writer.add_summary(eval_summary, model_step)
                    summary_writer.add_summary(summaries_eval, model_step)
                    # if stopper.step(1-eval_err):
                    #     break
                    #------------------------------------score--------------------------------------
                    if score_micro_flag == False:
                        if sessionNModelFlags['record_best_score']:
                            score_current={"accuracy_score":0,"precision_score":{},"recall_score":{},"f1_score":{}}
                            metric_aver_list = ["macro","micro","weighted"]
                            metric_str_list = ["accuracy_score", "precision_score","f1_score","recall_score"]
                            metric_list = [accuracy_score, precision_score,f1_score,recall_score]
                            
                            for metric_str,metric in zip(metric_str_list,metric_list):
                                if metric == accuracy_score:
                                    score_current["accuracy_score"] = metric(val_labels,eval_pred_lbls)
                                else:
                                    for metric_aver in metric_aver_list:
                                        score_current[metric_str][metric_aver] = metric(val_labels,eval_pred_lbls,average=metric_aver)
                            if score_current["accuracy_score"] > best_accuracy:
                                score = score_current
                                best_accuracy = score_current["accuracy_score"]
                        else:
                            metric_aver_list = ["macro","micro","weighted"]
                            metric_str_list = ["accuracy_score", "precision_score","f1_score","recall_score"]
                            metric_list = [accuracy_score, precision_score,f1_score,recall_score]
                            if model_step == sessionNModelFlags["max_iters"]:
                                for metric_str,metric in zip(metric_str_list,metric_list):
                                    if metric == accuracy_score:
                                        score["accuracy_score"] = metric(val_labels,eval_pred_lbls)
                                    else:
                                        for metric_aver in metric_aver_list:
                                            score[metric_str][metric_aver] = metric(val_labels,eval_pred_lbls,average=metric_aver)
                    else:
                        score_micro.append(accuracy_score(val_labels,eval_pred_lbls))
                        score_micro.append(precision_score(val_labels,eval_pred_lbls,average='micro'))
                        score_micro.append(recall_score(val_labels,eval_pred_lbls,average='micro'))
                        score_micro.append(f1_score(val_labels,eval_pred_lbls,average='micro'))
                    #-------------------------------------------------------------------------------
                    confusion_mtx = metrics.confusion_mtx(val_labels, eval_pred_lbls, num_classes)
                    cm_final = confusion_mtx
                    LOG.info("\n"+str(repr(confusion_mtx)))
                    LOG.info(str_val_or_test+' error: %.2f %% \n' % eval_err)
                    # add      
                    error_percent_val = (np.sum(val_labels != eval_pred_lbls))/len(val_labels)*100
                    LOG.info('Total validation error: %.2f %% \n' % error_percent_val)
                    # if error_percent_val<10:
                    #     plot_utils.plot_confusion_matrix(confusion_mtx,normalize=True,save=True)
                    
                    time_now = time.clock()
                    val_error_recorder.append(round(error_percent_val,2))
                    val_error_timer.append(round(time_now-time_start,2))
                    print('time elapse:',time_now-time_start)
                    tf.summary.scalar('test_error',error_percent_val)
                    val_error_rate.append(error_percent_val)
                    if(len(val_error_rate)>=3):
                        if val_error_rate[-1]>val_error_rate[-2] or (abs(val_error_rate[-1]-val_error_rate[-2])<0.05 and abs(val_error_rate[-2]-val_error_rate[-3])<0.05):
                            trainer._params["cc_loss_on"]=True 
                    
                # SAVE MODEL changed by fln
                model_name = "model-"+str(model_step)+"-"+timestr()
                if (model_step) == sessionNModelFlags["max_iters"]:
                    new_devices_list = sessionNModelFlags['new_devices_list']
                    if new_devices_list is not None:
                        new_devices_model_save_postfix = '_'.join(str(item) for item in new_devices_list)
                        if retrain == True:
                            save_model_dir = sessionNModelFlags["logdir"]["trainTf"] + '/' + new_devices_model_save_postfix + '/retrain'
                            # compute new_device_1 and new_device_2's accuracy from cm_final, fln_test
                            old_new_label_dict_, new_rearange_label_dict_ = {},{}
                            with open(local_dirs_to_data.iot+new_devices_model_save_postfix+'/old_new_device_idx_dict.json') as f:
                                old_new_label_dict_ = json.load(f)
                                f.close()
                            with open(local_dirs_to_data.iot+new_devices_model_save_postfix+'/new_rearange_label_dict.json') as g:
                                new_rearange_label_dict_ = json.load(g)
                                g.close()
                            # compute old:now label dict
                            old_now_label_dict = {}
                            for old_label_ in new_devices_list:
                                tmp = old_new_label_dict_[str(old_label_)]
                                now_ = new_rearange_label_dict_[str(tmp) + '.0']
                                old_now_label_dict[old_label_] = now_
                            # compute accuracy of devices in new_devices_list
                            
                            for old_label_ in new_devices_list:
                                now_label_ = old_now_label_dict[old_label_]
                                accuracy_c_ = round(cm_final[now_label_,now_label_]/np.sum(cm_final[now_label_])*100,2)
                                new_devices_accuracy[old_label_] = accuracy_c_

                        else:
                            save_model_dir = sessionNModelFlags['logdir']['trainTf'] + '/' + new_devices_model_save_postfix
                        if not os.path.exists(save_model_dir):
                            os.makedirs(save_model_dir)
                    else:
                        save_model_dir = sessionNModelFlags["logdir"]["trainTf"]
                                       
                    filepath_to_save = os.path.join(save_model_dir, model_name)
                    LOG.info("[SAVE] Saving model at: "+str(filepath_to_save))
                    saver.save(sess, filepath_to_save, global_step=sessionNModelFlags["max_iters"], write_meta_graph=True, write_state=True ) # creates the subfolder if doesnt exist. Appends step if given.
    
                    with open(save_model_dir + '/model_name','w') as f:
                        f.write(model_name+'-'+str(sessionNModelFlags['max_iters']))
                        f.close()
        coordinator.request_stop() # Stop the threads.
        coordinator.join(threads)
    # output to csv 'val_error_rate.csv'
    # val_error_rate = pd.DataFrame(val_error_rate)
    # val_error_rate.to_csv(trainerFlags['out_path']+'val_error_rate.csv',index=False,header=None)
    # print(eval_emb_all)
    # eval_emb_all = np.concatenate(eval_emb_all)
    # plot_tsne(eval_emb_all,val_samples,val_labels) # plot_tsne
    # print(val_samples)
    # print(val_labels)

    with open(trainerFlags['out_path']+'/val_error_rate.txt','a') as f:
        f.write("===================================================================\n")
        f.write('multi label model = '+str(trainerFlags['use_multi_task_model'])+'\n')
        f.write('lbl percent = '+str(sessionNModelFlags['percent_lbl_samples'])+'\n')
        f.write('threshold = '+str(sessionNModelFlags['threshold'])+'\n')
        f.write('cc_loss_on ='+str(trainerFlags['cc_loss_on'])+'\n')
        f.write('dynamic_loss_weight ='+str(trainerFlags['dynamic_loss_weight'])+'\n')
        
        if sessionNModelFlags['new_devices_list'] is not None:
            new_devices_str = ','.join(str(item) for item in sessionNModelFlags['new_devices_list'])
            f.write('new_devices_list = {}\n'.format(new_devices_str))
        f.write('retrain = ' + str(sessionNModelFlags['retrain']) + '\n')
        f.write('transfer_learing = ' + str(tl) + '\n')
        f.write('use_cnn_layer_for_cluster:{}\n'.format(str(sessionNModelFlags['use_cnn_layer_for_cluster'])))
        f.write('cnn_type:{}\n'.format(cnn_type))
        for i in range(len(val_error_rate)):
            f.write(str(round(val_error_rate[i],10)))
            if i!=len(val_error_rate)-1:
                f.write(',')
        f.write('\n')
        #-----write score--------------------------------------------------------------------
        f.write("-------------score----------------------------------------------------------------\n")
        if score_micro_flag == False:
            metric_aver_list = ["macro","micro","weighted"]
            metric_str_list = ["accuracy_score", "precision_score","f1_score","recall_score"]
            metric_list = [accuracy_score, precision_score,f1_score,recall_score]
            for metric_str in metric_str_list:
                if metric_str == "accuracy_score":
                    f.write("accuracy_score = {}\n".format(score["accuracy_score"]))
                else:
                    for metric_aver in metric_aver_list:
                        f.write(metric_str + "({}) = {}\n".format(metric_aver, score[metric_str][metric_aver]))
        else:
            f.write("accuracy_score = {}\n".format(score_micro[0]))
            f.write("precision_score = {}\n".format(score_micro[1]))
            f.write("recall_score = {}\n".format(score_micro[2]))
            f.write("f1_score = {}\n".format(score_micro[3]))
        f.write("----------------------------------------------------------------------------------\n")
        f.write("------------------------new devices accuracy---------------------------------------\n")
        for key,value in new_devices_accuracy.items():
            print('new_device:{},accuracy:{}'.format(key,value))
            f.write('new_device:{},accuracy:{}\n'.format(key,value))
        #----------------------cm---------------------------------------------------------------------
        f.write("--------------------------------cm------------------------------------------------\n")
        f.write(get_cm_str(cm_final)+'\n')
        f.write("--------------------------------------------------------------------------------\n")
        f.write("test_error_recorder:\n")
        f.write(','.join(str(item) for item in val_error_recorder)+'\n')
        f.write("test_error_timer:\n")
        f.write(','.join(str(item) for item in val_error_timer)+'\n')
        f.write("--------------------------------------------------------------------------------\n")
        f.close()


                
               
            
            
    
    
    
    
    
    
    
    
