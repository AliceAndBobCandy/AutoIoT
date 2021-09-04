#!/usr/bin/env python
# modified by Linna Fan
# Copyright (c) 2018, Konstantinos Kamnitsas
# modified by Linna Fan
# This program is free software; you can redistribute and/or modify
# it under the terms of the Apache License, Version 2.0. See the 
# accompanying LICENSE file or read the terms at:
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import absolute_import, division, print_function

import logging
LOG = logging.getLogger('main')

from collections import OrderedDict
import os

# All logic of how to create the actual config from the parsed file+cmd should be in here. NOT earlier in run().
class ConfigFlags():
    def __init__(self, cfg_file_dict):
        self._params = OrderedDict() # dict with order
        self._fill_params_from_cfg_file_dict(cfg_file_dict)
        self._check_flags_ok()
        
    def __setitem__(self, key, value):
        self._params[key] = value
    def __getitem__(self, key):
        return self._params[key]
    
    def print_params(self):        
        LOG.info( self._str_to_print )
        for k in self._params:
            LOG.info( str(k) + ": " + str(self._params[k]) )
            
    # Abstracts
    def _fill_params_from_cfg_file_dict(self, cfg_file_dict):
        raise NotImplementedError() #sub-class should realize this method, or error will be raised
    
    def _check_flags_ok(self):
        raise NotImplementedError()
    
    
class TrainSessionNModelFlags(ConfigFlags):
        
    def _fill_params_from_cfg_file_dict(self, cfg_file_dict):
        self._str_to_print = "============ Parameters of Session and Model =============="
        
        # The below parameters could be broken down into two. Session (data load, loop etc) and the model.
        
        #===================== DATA ======================
        # --- Validation, labelled and unlabelled folds for training ---
        self._params['dataset'] = cfg_file_dict['dataset'] # Mnist/svhn/cifar
        self._params['val_on_test'] = cfg_file_dict['val_on_test'] # True or False. True: val on test data. False: Val on fold of train data.
        self._params['num_val_samples'] = cfg_file_dict['num_val_samples'] # How many samples to use for validation when training. -1 to use all.
        self._params['num_lbl_samples'] = cfg_file_dict['num_lbl_samples'] # Number of labelled samples from the database to use for supervision. -1 to use whole database.
        self._params['percent_lbl_samples'] = cfg_file_dict['percent_lbl_samples']
        self._params['num_unlbl_samples'] = cfg_file_dict['num_unlbl_samples'] # Number of unlabelled samples to use semi-supervised learning. -1 to use whole database.
        self._params['unlbl_overlap_val'] = cfg_file_dict['unlbl_overlap_val']  # If True and val_on_test=False, unlabelled samples may overlap with validation.
        self._params['unlbl_overlap_lbl'] = cfg_file_dict['unlbl_overlap_lbl'] # If True, unlabelled samples can overlap with labelled. If False, they are disjoint.
        self._params['new_devices_list'] = cfg_file_dict['new_devices_list']
        self._params['niot_label'] = cfg_file_dict['niot_label']
        self._params['dataset_name'] = cfg_file_dict['dataset_name']
        self._params['record_data_index'] = cfg_file_dict['record_data_index']
        self._params['record_best_score'] = cfg_file_dict['record_best_score']
        # --- Batch sampling, normalization, augmentation ---
        self._params['n_lbl_per_class_per_batch'] = cfg_file_dict['n_lbl_per_class_per_batch'] if 'n_lbl_per_class_per_batch' in cfg_file_dict else 10 # Number of LABELED samples PER CLASS PER BATCH. batch_size= this x Classes + unlabeled below.
        self._params['n_unlbl_per_batch'] = cfg_file_dict['n_unlbl_per_batch'] if 'n_unlbl_per_batch' in cfg_file_dict else 100 # Number of UNLABELED samples per batch.
        self._params['eval_batch_size'] = 100 # Number of samples per batch when doing evaluation (memory bounded)
        self._params['seed'] = cfg_file_dict['seed'] if "seed" in cfg_file_dict else None # Integer, or None for time.
        # --- upsampling ---
        self._params['upsampling'] = cfg_file_dict['upsampling']
        self._params['percent_lbl_samples_small'] = cfg_file_dict['percent_lbl_samples_small']
        self._params['lbl_samples_small'] = cfg_file_dict['lbl_samples_small']
        self._params['upsample_niot'] = cfg_file_dict['upsample_niot']
        self._params['num_least'] = cfg_file_dict['num_least']
        self._params['epsilon'] = cfg_file_dict['epsilon']
        self._params['prob'] = cfg_file_dict['prob']
        self._params['disturbe'] = cfg_file_dict['disturbe']
        self._params['threshold'] = cfg_file_dict['threshold']
        self._params['train_ratio'] = cfg_file_dict['train_ratio']
        #=================== DATA PRE-PROCESSING ===============
        self._params['norm_imgs_pre_aug'] = cfg_file_dict['norm_imgs_pre_aug'] # Type of data normalization. None, zscore, etc. This is NOT part of the graph. Outside. Session parameter.
        # Augmentation
        self._params["augm_params"] = cfg_file_dict["augm_params"]
        # Re-normalization of both training and testing, AFTER augmentation and right before the network input.
        self._params['norm_tens_post_aug'] = cfg_file_dict['norm_tens_post_aug'] if 'norm_tens_post_aug' in cfg_file_dict else None #Normalize after augmenting. None, zscorePerCase, ...
    
        #======================== MODEL ==========================
        self._params['feat_extractor_z'] = cfg_file_dict['model']['feat_extractor_z']
        self._params['emb_z_size'] = cfg_file_dict['model']['emb_z_size'] # Number of feature maps at latent layer before classifier, where CCLP will be performed.
        self._params['emb_z_act'] = cfg_file_dict['model']['emb_z_act'] # Activation function at the last hidden layer, where embedding is processed.
        self._params['batch_norm_decay'] = cfg_file_dict['model']['batch_norm_decay']
        self._params['l2_reg_feat'] = cfg_file_dict['model']['l2_reg_feat'] # L2 regularization for the feature extractor.
        self._params['l2_reg_classif'] = cfg_file_dict['model']['l2_reg_classif'] # L2 regularization for the classifier.
        self._params['classifier_dropout'] = cfg_file_dict['model']['classifier_dropout'] # Dropout at the classifier.
        
        # ================ TRAINING LOOP PARAMETERS ===================
        main_out_dir = cfg_file_dict['out_path']
        self._params['logdir'] = {"main": main_out_dir,
                                 "logs": os.path.join(main_out_dir, "logs"),
                                 "trainTf": os.path.join(main_out_dir, "trainTf"),
                                 "summariesTf": os.path.join(main_out_dir, "summariesTf"),
                                 "emb": os.path.join(main_out_dir, "emb"),
                                 "prob_fig": os.path.join(main_out_dir,"prob_fig") }
        self._params['max_iters'] = cfg_file_dict['max_iters'] # Number of SGD iterations till the end of training process.
        # Testing vars.
        self._params['val_during_train'] = cfg_file_dict['val_during_train'] # True to evaluate on test every eval_interval, otherwise only training metrics.
        self._params['val_interval'] = cfg_file_dict['val_interval'] # Number of steps between logging train and evaluating test metrics.
        # Plot Embedding:
        self._params['plot_save_emb'] = cfg_file_dict['plot_save_emb'] # Plot embedding of train batch: 0=dont plot dont save. 1=save. 2=plot. 3=save and plot.
        # Logging + Summaries.
        
        # Saver parameters.
        self._params['max_checkpoints_to_keep'] = 5 # Keep up to this many recent checkpoints.
        
        # Misc.
        self._params['model_to_load'] = cfg_file_dict['model_to_load'] if cfg_file_dict['model_to_load'] != "self" else self._params['logdir']['trainTf']
        self._params['device'] = cfg_file_dict['device'] # CPU or GPU
        #=================== 1st round modification =========================
        self._params['use_cnn_layer_for_cluster'] = cfg_file_dict['use_cnn_layer_for_cluster']
        self._params['merge_cnn_layer_methods'] = cfg_file_dict['merge_cnn_layer_methods']


    def _check_flags_ok(self):
        assert self._params['plot_save_emb'] == 0 or self._params['emb_z_size'] == 2
        assert self._params['device'] == -1 or self._params['device'] >=0
    
class TrainerFlags(ConfigFlags):
        
    def _fill_params_from_cfg_file_dict(self, cfg_file_dict):
        self._str_to_print = "============ Parameters of Trainer/Optimizer =============="
        
        ###################### TRAINER / OPTIMIZER PARAMETERS ####################
        
        # ================= SEMI SUPERVISION + LOSS PARAMETERS ================
        # added--------
        # --- multi task model ---
        self._params['use_multi_task_model'] = cfg_file_dict['use_multi_task_model']
        
        self._params['out_path'] = cfg_file_dict['out_path']
        #---------------------------------------------------------------------------
        # Supervised
        self._params['dynamic_loss_weight'] = cfg_file_dict['dynamic_loss_weight']
        self._params['logit_weight'] = 1.0
        # CCLP loss
        self._params['cc_loss_distance'] = cfg_file_dict['cc_loss_distance']
        self._params['cc_loss_on'] = cfg_file_dict['cc_loss_on'] # Boolean
        self._params['cc_weight'] = cfg_file_dict['cc_weight']
        self._params['cc_steps'] = cfg_file_dict['cc_steps']
        self._params['cc_sum_over_chains'] = cfg_file_dict['cc_sum_over_chains'] if 'cc_sum_over_chains' in cfg_file_dict else True
        # Similarity metric for creating the graph.
        self._params['cc_sim_metric'] = cfg_file_dict['cc_sim_metric'] #'dot' or 'L2'
        self._params['cc_l2_sigmas_init'] = cfg_file_dict['cc_l2_sigmas_init'] # for L2 only
        self._params['cc_l2_sigmas_trainable'] = cfg_file_dict['cc_l2_sigmas_trainable'] # for L2 only
        self._params['cc_l2_sigmas_lr_multipl'] = cfg_file_dict['cc_l2_sigmas_lr_multipl'] # for L2 only
        # cclp, less important configs.
        self._params['cc_e_smooth'] = cfg_file_dict['cc_e_smooth']
        self._params['cc_optim_smooth_mtx'] = cfg_file_dict['cc_optim_smooth_mtx']
        
        # Generics
        self._params['track_ratio_g_v'] = cfg_file_dict['track_ratio_g_v'] if 'track_ratio_g_v' in cfg_file_dict else False # Tracking ratio of grads to var. This is slow.
        
        #================ LEARNING RATE ============================
        self._params['lr_sched_type'] = cfg_file_dict['lr_sched_type'] # Type of lr schedule. expon_decay or piecewise.
        self._params['lr_expon_init'] = cfg_file_dict['lr_expon_init'] # LR-Exponential decay: Initial LR value.
        self._params['lr_expon_decay_factor'] = cfg_file_dict['lr_expon_decay_factor'] # LR-Exponential decay: Factor by how much to decrease.
        self._params['lr_expon_decay_steps'] = cfg_file_dict['lr_expon_decay_steps'] # LR-Exponential decay: How often to decrease.
        self._params['lr_piecewise_boundaries'] = cfg_file_dict['lr_piecewise_boundaries'] # LR-Piecewise schedule: boundaries where lr is changed.
        self._params['lr_piecewise_values'] = cfg_file_dict['lr_piecewise_values'] # LR-Piecewise schedule: Value of LR in the corresponding piecewise parts.
        self._params['lr_min_value'] = 1e-6 # Lowest value LR can take.
        #=================== original space similar ===============
        
        # self._params['original_space_similar_epsilon'] = cfg_file_dict['original_space_similar_epsilon']
        


    def _check_flags_ok(self):
        pass
        
        
        
        
        
        
